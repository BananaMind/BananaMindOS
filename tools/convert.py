#!/usr/bin/env python3
"""Convert supported BananaMind safetensors checkpoints to BM2NQ.

No PyTorch, NumPy, transformers, or safetensors package is required.  The
converter keeps a fixed tensor order for the supported decoder families,
which makes both the on-disk format and freestanding kernel parser tiny.
"""

from __future__ import annotations

import argparse
import array
import json
import mmap
import os
import struct
import sys
import tempfile
import zlib
from pathlib import Path

try:
    from .quantization import (
        SUPPORTED_BITS, pack_quantized, quantization_divisor, quantize_value,
    )
except ImportError:
    from quantization import (
        SUPPORTED_BITS, pack_quantized, quantization_divisor, quantize_value,
    )

MAGIC = b"BM2NQ\r\n\x1a"
VERSION = 1
HEADER = struct.Struct("<8s12I2f12I")
TOKEN = struct.Struct("<IHH")
MERGE = struct.Struct("<HHHH")


class SafeTensorFile:
    def __init__(self, path: Path):
        self.file = path.open("rb")
        self.map = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        header_len = struct.unpack_from("<Q", self.map, 0)[0]
        self.data_start = 8 + header_len
        self.meta = json.loads(self.map[8 : self.data_start])

    def close(self):
        self.map.close()
        self.file.close()

    def tensor(self, name: str) -> tuple[list[int], array.array]:
        info = self.meta[name]
        begin, end = info["data_offsets"]
        raw = self.map[self.data_start + begin : self.data_start + end]
        dtype = info["dtype"]
        if dtype == "F32":
            values = array.array("f")
            values.frombytes(raw)
            if sys.byteorder != "little":
                values.byteswap()
        elif dtype == "F16":
            values = array.array("f", (x[0] for x in struct.iter_unpack("<e", raw)))
        elif dtype == "BF16":
            values = array.array(
                "f",
                (struct.unpack("<f", b"\0\0" + x)[0] for (x,) in struct.iter_unpack("<2s", raw)),
            )
        else:
            raise ValueError(f"unsupported tensor dtype {dtype!r} for {name}")
        shape = list(info["shape"])
        expected = 1
        for dim in shape:
            expected *= dim
        if len(values) != expected:
            raise ValueError(f"bad element count for {name}: {len(values)} != {expected}")
        return shape, values


def choose_tensor(sf: SafeTensorFile, *names: str) -> str:
    for name in names:
        if name in sf.meta:
            return name
    raise KeyError(f"none of these tensors exists: {', '.join(names)}")


def write_vector(out, sf: SafeTensorFile, name: str, size: int):
    shape, values = sf.tensor(name)
    if shape != [size] and not (shape == [] and size == 1):
        raise ValueError(f"{name}: expected [{size}], got {shape}")
    if sys.byteorder != "little":
        values.byteswap()
    out.write(values.tobytes())


def write_matrix(out, sf: SafeTensorFile, name: str, rows: int, cols: int, bits: int):
    shape, values = sf.tensor(name)
    if shape != [rows, cols]:
        raise ValueError(f"{name}: expected [{rows}, {cols}], got {shape}")
    if bits == 32:
        if sys.byteorder != "little":
            values.byteswap()
        out.write(values.tobytes())
        return
    if bits == 16:
        for row in range(rows):
            packed = bytearray(cols * 2)
            start = row * cols
            for i in range(cols):
                struct.pack_into("<e", packed, i * 2, values[start + i])
            out.write(packed)
        return
    packed_size = (cols * bits + 7) // 8
    for row in range(rows):
        start = row * cols
        row_values = values[start : start + cols]
        maximum = max((abs(x) for x in row_values), default=0.0)
        divisor = float(quantization_divisor(bits))
        scale = maximum / divisor if maximum else 0.0
        out.write(struct.pack("<f", scale))
        packed = pack_quantized(row_values, scale, bits)
        if len(packed) != packed_size:
            raise AssertionError("packed quantized row has the wrong size")
        out.write(packed)


def gpt2_byte_decoder() -> dict[str, int]:
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(0xA1, 0xAC + 1))
    visible += list(range(0xAE, 0xFF + 1))
    chars = visible[:]
    extra = 0
    for byte in range(256):
        if byte not in visible:
            visible.append(byte)
            chars.append(256 + extra)
            extra += 1
    return {chr(codepoint): byte for byte, codepoint in zip(visible, chars)}


def load_token_bytes(path: Path, vocab_size: int, special_ids: set[int]) -> list[bytes]:
    tokenizer = json.loads(path.read_text(encoding="utf-8"))
    vocab = tokenizer.get("model", {}).get("vocab")
    if not isinstance(vocab, dict):
        raise ValueError("tokenizer.json does not contain model.vocab")
    tokens: list[bytes | None] = [None] * vocab_size
    decoder = gpt2_byte_decoder()
    for text, token_id in vocab.items():
        if not 0 <= token_id < vocab_size:
            continue
        if token_id in special_ids:
            tokens[token_id] = b""
            continue
        try:
            tokens[token_id] = bytes(decoder[ch] for ch in text)
        except KeyError:
            # This is only expected for an added non-ByteLevel token.
            tokens[token_id] = text.encode("utf-8")
    missing = [i for i, token in enumerate(tokens) if token is None]
    if missing:
        raise ValueError(f"tokenizer is missing IDs: {missing[:8]}")
    return [token or b"" for token in tokens]


def load_merges(path: Path) -> list[tuple[int, int, int, int]]:
    tokenizer = json.loads(path.read_text(encoding="utf-8"))
    model = tokenizer.get("model", {})
    vocab = model.get("vocab", {})
    result = []
    for rank, merge in enumerate(model.get("merges", [])):
        if isinstance(merge, str):
            left, right = merge.split(" ", 1)
        else:
            left, right = merge
        combined = left + right
        if left not in vocab or right not in vocab or combined not in vocab:
            raise ValueError(f"invalid BPE merge {merge!r}")
        ids = (vocab[left], vocab[right], vocab[combined], rank)
        if any(value > 0xFFFF for value in ids):
            raise ValueError("BM2NQ v1 supports at most 65536 tokens/merges")
        result.append(ids)
    return result


def tensor_plan(config: dict, sf: SafeTensorFile):
    h = config["hidden_size"]
    layers = config["num_hidden_layers"]
    head_dim = config.get("head_dim", h // config["num_attention_heads"])
    kv = config["num_key_value_heads"] * head_dim
    ff = config["intermediate_size"]
    vocab = config["vocab_size"]
    model_type = config.get("model_type", "")
    if model_type == "microbanana":
        yield "matrix", choose_tensor(sf, "embed_tokens.weight", "model.embed_tokens.weight"), vocab, h
        for i in range(layers):
            p = f"layers.{i}"
            yield "vector", f"{p}.input_layernorm.weight", h
            yield "matrix", f"{p}.self_attn.q_proj.weight", h, h
            yield "matrix", f"{p}.self_attn.k_proj.weight", kv, h
            yield "matrix", f"{p}.self_attn.v_proj.weight", kv, h
            yield "matrix", f"{p}.self_attn.o_proj.weight", h, h
            yield "vector", f"{p}.post_attention_layernorm.weight", h
            yield "matrix", f"{p}.mlp.gate_proj.weight", ff, h
            yield "matrix", f"{p}.mlp.up_proj.weight", ff, h
            yield "matrix", f"{p}.mlp.down_proj.weight", h, ff
        yield "vector", choose_tensor(sf, "norm.weight", "model.norm.weight"), h
        return

    yield "matrix", choose_tensor(sf, "transformer.wte.weight", "model.embed_tokens.weight"), vocab, h
    for i in range(layers):
        p = f"transformer.h.{i}"
        if model_type == "bananamind2_micro":
            yield "vector", f"{p}.input_norm.weight", h
            yield "matrix", f"{p}.attention.q_proj.weight", h, h
            yield "matrix", f"{p}.attention.k_proj.weight", kv, h
            yield "matrix", f"{p}.attention.v_proj.weight", kv, h
            yield "matrix", f"{p}.attention.o_proj.weight", h, h
            yield "vector", f"{p}.attention.q_norm.weight", head_dim
            yield "vector", f"{p}.attention.k_norm.weight", head_dim
            yield "vector", f"{p}.refresh.attention_norm.weight", h
            yield "vector", f"{p}.refresh.embedding_norm.weight", h
            yield "vector", f"{p}.refresh.output_norm.weight", h
            yield "matrix", f"{p}.refresh.gate_proj.weight", h, h
            yield "matrix", f"{p}.refresh.value_proj.weight", h, h
            yield "matrix", f"{p}.refresh.out_proj.weight", h, h
            yield "matrix", f"{p}.refresh.depthwise_kernel", h, config["refresh_kernel_size"]
            yield "vector", f"{p}.refresh.alpha", 1
            yield "vector", f"{p}.post_attention_norm.weight", h
            yield "matrix", f"{p}.mlp.gate_proj.weight", ff, h
            yield "matrix", f"{p}.mlp.up_proj.weight", ff, h
            yield "matrix", f"{p}.mlp.down_proj.weight", h, ff
            continue
        yield "vector", f"{p}.ln_1.weight", h
        yield "matrix", f"{p}.attn.q_proj.weight", h, h
        yield "matrix", f"{p}.attn.k_proj.weight", kv, h
        yield "matrix", f"{p}.attn.v_proj.weight", kv, h
        yield "matrix", f"{p}.attn.o_proj.weight", h, h
        yield "vector", f"{p}.attn.q_norm.weight", head_dim
        yield "vector", f"{p}.attn.k_norm.weight", head_dim
        yield "vector", f"{p}.ln_2.weight", h
        yield "matrix", f"{p}.mlp.w_gate.weight", ff, h
        yield "matrix", f"{p}.mlp.w_up.weight", ff, h
        yield "matrix", f"{p}.mlp.w_down.weight", h, ff
    yield "vector", choose_tensor(sf, "transformer.ln_f.weight", "model.norm.weight"), h


def convert(model: Path, tokenizer: Path, config_path: Path, output: Path, bits: int):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {
        "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "intermediate_size",
        "max_position_embeddings", "rope_theta", "rms_norm_eps",
    }
    absent = required - config.keys()
    if absent:
        raise ValueError(f"config is missing {sorted(absent)}")
    if bits not in SUPPORTED_BITS:
        raise ValueError("bits must be Q1-Q8, FP16, or FP32")
    config.setdefault("head_dim", config["hidden_size"] // config["num_attention_heads"])

    sf = SafeTensorFile(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=output.name + ".", dir=output.parent)
    try:
        with os.fdopen(fd, "w+b") as out:
            out.write(bytes(HEADER.size))
            weights_offset = out.tell()
            plan = list(tensor_plan(config, sf))
            for number, item in enumerate(plan, 1):
                kind, name, *dims = item
                print(f"[{number:02}/{len(plan)}] {name}", file=sys.stderr)
                if kind == "vector":
                    write_vector(out, sf, name, dims[0])
                else:
                    write_matrix(out, sf, name, dims[0], dims[1], bits)
            weights_size = out.tell() - weights_offset

            special = {
                config.get("bos_token_id", 1), config.get("eos_token_id", 2),
                config.get("pad_token_id", 0), config.get("unk_token_id", 3),
            }
            tokens = load_token_bytes(tokenizer, config["vocab_size"], special)
            token_index_offset = out.tell()
            token_offset = 0
            for token in tokens:
                if len(token) > 0xFFFF:
                    raise ValueError("token is too large")
                out.write(TOKEN.pack(token_offset, len(token), 0))
                token_offset += len(token)
            token_data_offset = out.tell()
            for token in tokens:
                out.write(token)
            token_data_size = out.tell() - token_data_offset
            merges = load_merges(tokenizer)
            merges_offset = out.tell()
            for merge in merges:
                out.write(MERGE.pack(*merge))
            file_size = out.tell()

            out.flush()
            out.seek(weights_offset)
            crc = 0
            remaining = file_size - weights_offset
            while remaining:
                block = out.read(min(1024 * 1024, remaining))
                crc = zlib.crc32(block, crc)
                remaining -= len(block)

            header = HEADER.pack(
                MAGIC, VERSION, HEADER.size, file_size, bits,
                config["vocab_size"], config["hidden_size"], config["num_hidden_layers"],
                config["num_attention_heads"], config["num_key_value_heads"],
                config["head_dim"], config["intermediate_size"],
                config["max_position_embeddings"], float(config["rope_theta"]),
                float(config["rms_norm_eps"]), weights_offset, weights_size,
                token_index_offset, token_data_offset, token_data_size,
                merges_offset, len(merges),
                config.get("bos_token_id", 1), config.get("eos_token_id", 2),
                config.get("pad_token_id", 0), config.get("unk_token_id", 3), crc,
            )
            out.seek(0)
            out.write(header)
        os.replace(temporary_name, output)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    finally:
        sf.close()
    precision = f"Q{bits}" if bits <= 8 else f"FP{bits}"
    print(f"wrote {output} ({output.stat().st_size / (1024 * 1024):.2f} MiB, {precision})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--bits", required=True, type=int, choices=SUPPORTED_BITS)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    convert(args.model, args.tokenizer, args.config, args.output, args.bits)


if __name__ == "__main__":
    main()
