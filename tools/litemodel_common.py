#!/usr/bin/env python3
"""Shared .litemodel container, quantizer, and tokenizer helpers."""

from __future__ import annotations

import array
import json
import mmap
import os
import struct
import sys
import tempfile
import zlib
from pathlib import Path

from tools.quantization import (
    SUPPORTED_BITS, pack_quantized, quantization_divisor, quantize_value,
)

MAGIC = b"LITEMDL\x1a"
VERSION = 2
HEADER = struct.Struct("<8s14I2f14I")
TOKEN = struct.Struct("<IHH")
MERGE = struct.Struct("<IIII")
TOKEN_SPECIAL = 1 << 0

ARCH_BANANA = 1
ARCH_LLAMA = 2
ARCH_GPTX2 = 3
ARCH_ROSE_X1 = 4
ARCH_MINSPARK = 5
ARCH_LFM2 = 6
ARCH_GEMMA3 = 7
ARCH_QWEN35 = 8

FLAG_CHAT = 1 << 0
FLAG_TIED_EMBEDDING = 1 << 1
FLAG_SPACE_TO_MARKER = 1 << 2
FLAG_NO_BOS = 1 << 3
FLAG_CHAT_STYLE_SHIFT = 8

CHAT_STYLES = {
    "banana": 0,
    "chatml": 1,
    "smollm": 2,
    "qwen35": 3,
    "lfm2": 4,
}


def chat_style_flag(name: str) -> int:
    try:
        return CHAT_STYLES[name] << FLAG_CHAT_STYLE_SHIFT
    except KeyError as error:
        raise ValueError(f"unsupported chat style {name!r}") from error


class SafeTensorFile:
    """Tiny mmap safetensors reader; conversion needs no ML framework."""

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
            values = array.array("f", (value[0] for value in struct.iter_unpack("<e", raw)))
        elif dtype == "BF16":
            values = array.array(
                "f",
                (struct.unpack("<f", b"\0\0" + value)[0]
                 for (value,) in struct.iter_unpack("<2s", raw)),
            )
        else:
            raise ValueError(f"unsupported tensor dtype {dtype!r} for {name}")
        shape = list(info["shape"])
        expected = 1
        for dimension in shape:
            expected *= dimension
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
    count = 1
    for dimension in shape:
        count *= dimension
    if count != size:
        raise ValueError(f"{name}: expected {size} values, got shape {shape}")
    if sys.byteorder != "little":
        values.byteswap()
    out.write(values.tobytes())


def write_matrix(out, sf: SafeTensorFile, name: str, rows: int, cols: int, bits: int):
    shape, values = sf.tensor(name)
    count = 1
    for dimension in shape:
        count *= dimension
    if count != rows * cols or not shape or shape[0] != rows:
        raise ValueError(f"{name}: expected [{rows}, ..., {cols}] ({rows * cols} values), got {shape}")
    if bits == 32:
        if sys.byteorder != "little":
            values.byteswap()
        out.write(values.tobytes())
        return
    if bits == 16:
        for value in values:
            out.write(struct.pack("<e", value))
        return
    packed_size = (cols * bits + 7) // 8
    for row in range(rows):
        start = row * cols
        row_values = values[start : start + cols]
        maximum = max((abs(value) for value in row_values), default=0.0)
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
    characters = visible[:]
    extra = 0
    for byte in range(256):
        if byte not in visible:
            visible.append(byte)
            characters.append(256 + extra)
            extra += 1
    return {chr(codepoint): byte for byte, codepoint in zip(visible, characters)}


def _token_bytes(text: str, decoder: dict[str, int]) -> bytes:
    if len(text) == 6 and text.startswith("<0x") and text.endswith(">"):
        try:
            return bytes((int(text[3:5], 16),))
        except ValueError:
            pass
    try:
        return bytes(decoder[character] for character in text)
    except KeyError:
        return text.encode("utf-8")


def load_token_bytes(path: Path, vocab_size: int, special_ids: set[int]) -> list[bytes]:
    tokenizer = json.loads(path.read_text(encoding="utf-8"))
    vocab = tokenizer.get("model", {}).get("vocab")
    if not isinstance(vocab, dict):
        raise ValueError("tokenizer.json does not contain model.vocab")
    tokens: list[bytes | None] = [None] * vocab_size
    decoder = gpt2_byte_decoder()
    for text, token_id in vocab.items():
        if 0 <= token_id < vocab_size:
            tokens[token_id] = _token_bytes(text, decoder)
    for added in tokenizer.get("added_tokens", []):
        token_id = added.get("id")
        if isinstance(token_id, int) and 0 <= token_id < vocab_size:
            text = str(added.get("content", ""))
            tokens[token_id] = text.encode("utf-8")
    # Some checkpoints reserve embedding rows that deliberately have no
    # tokenizer piece (LFM2.5 leaves part of its 65,536-row vocabulary inert).
    # Keep those IDs non-encodable and non-printing instead of rejecting an
    # otherwise valid checkpoint.
    return [token or b"" for token in tokens]


def load_special_token_ids(path: Path, special_ids: set[int]) -> set[int]:
    tokenizer = json.loads(path.read_text(encoding="utf-8"))
    result = set(special_ids)
    for added in tokenizer.get("added_tokens", []):
        token_id = added.get("id")
        if added.get("special") and isinstance(token_id, int):
            result.add(token_id)
    return result


def load_merges(path: Path) -> list[tuple[int, int, int, int]]:
    tokenizer = json.loads(path.read_text(encoding="utf-8"))
    model = tokenizer.get("model", {})
    vocab = dict(model.get("vocab", {}))
    for added in tokenizer.get("added_tokens", []):
        if isinstance(added.get("id"), int):
            vocab.setdefault(str(added.get("content", "")), added["id"])
    result = []
    for rank, merge in enumerate(model.get("merges", [])):
        left, right = merge.split(" ", 1) if isinstance(merge, str) else merge
        combined = left + right
        if left not in vocab or right not in vocab or combined not in vocab:
            raise ValueError(f"invalid BPE merge {merge!r}")
        ids = (vocab[left], vocab[right], vocab[combined], rank)
        if any(value < 0 or value > 0xFFFFFFFF for value in ids):
            raise ValueError(".litemodel tokenizer IDs must fit in 32 bits")
        result.append(ids)
    return result


def _special_id(config: dict, name: str, fallback: int) -> int:
    value = config.get(name, fallback)
    if isinstance(value, list):
        value = value[0] if value else fallback
    return fallback if value is None else int(value)


def _configured_special_ids(config: dict) -> set[int]:
    result = set()
    for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        value = config.get(name)
        if isinstance(value, list):
            result.update(int(item) for item in value if item is not None)
        elif value is not None:
            result.add(int(value))
    return result


def write_litemodel(*, model: Path, tokenizer: Path, config: dict, output: Path,
                    bits: int, architecture: int, flags: int,
                    architecture_data: bytes, plan: list[tuple]):
    if bits not in SUPPORTED_BITS:
        raise ValueError("bits must be Q1-Q8, FP16, or FP32")
    sf = SafeTensorFile(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=output.name + ".", dir=output.parent)
    try:
        with os.fdopen(fd, "w+b") as out:
            out.write(bytes(HEADER.size))
            architecture_data_offset = out.tell()
            out.write(architecture_data)
            while out.tell() & 3:
                out.write(b"\0")
            weights_offset = out.tell()
            for number, item in enumerate(plan, 1):
                kind, name, *dimensions = item
                print(f"[{number:03}/{len(plan)}] {name}", file=sys.stderr)
                if kind == "vector":
                    write_vector(out, sf, name, dimensions[0])
                else:
                    write_matrix(out, sf, name, dimensions[0], dimensions[1], bits)
            weights_size = out.tell() - weights_offset

            bos = _special_id(config, "bos_token_id", 1)
            eos = _special_id(config, "eos_token_id", 2)
            pad = _special_id(config, "pad_token_id", 0)
            unk = _special_id(config, "unk_token_id", 3)
            special = _configured_special_ids(config)
            tokens = load_token_bytes(tokenizer, int(config["vocab_size"]), special)
            special_tokens = load_special_token_ids(tokenizer, special)
            token_index_offset = out.tell()
            token_offset = 0
            for token_id, token in enumerate(tokens):
                if len(token) > 0xFFFF:
                    raise ValueError("token is too large")
                token_flags = TOKEN_SPECIAL if token_id in special_tokens else 0
                out.write(TOKEN.pack(token_offset, len(token), token_flags))
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
                MAGIC, VERSION, HEADER.size, file_size, architecture, bits, flags,
                int(config["vocab_size"]), int(config["hidden_size"]),
                int(config["num_hidden_layers"]), int(config["num_attention_heads"]),
                int(config["num_key_value_heads"]), int(config["head_dim"]),
                int(config["intermediate_size"]), int(config["max_position_embeddings"]),
                float(config["rope_theta"]), float(config["rms_norm_eps"]),
                weights_offset, weights_size, token_index_offset, token_data_offset,
                token_data_size, merges_offset, len(merges), bos, eos, pad, unk, crc,
                architecture_data_offset, len(architecture_data),
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
    print(f"wrote {output} ({output.stat().st_size / 1048576:.2f} MiB, {precision})")
