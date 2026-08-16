#!/usr/bin/env python3
"""Run a BM2NQ model directly on a normal x86-64 host with NumPy/BLAS."""

from __future__ import annotations

import argparse
import mmap
import struct
import sys
import time
from pathlib import Path

import numpy as np

from convert import HEADER, MAGIC, MERGE, TOKEN


class BM2NRunner:
    def __init__(self, path: Path, context: int = 256, kernel_math: bool = False):
        self.file = path.open("rb")
        self.data = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        values = HEADER.unpack_from(self.data)
        names = (
            "magic version header_size file_size bits vocab hidden layers heads kv_heads "
            "head_dim intermediate max_seq rope_theta rms_eps weights_offset weights_size "
            "token_index_offset token_data_offset token_data_size merges_offset merges_count "
            "bos eos pad unk crc"
        ).split()
        self.h = dict(zip(names, values))
        if self.h["magic"] != MAGIC or self.h["version"] != 1:
            raise ValueError("not a BM2NQ v1 model")
        if self.h["file_size"] != len(self.data):
            raise ValueError("truncated BM2NQ model")
        self.context = min(context, self.h["max_seq"])
        self.kernel_math = kernel_math
        self.kind = self._model_kind()
        self._parse_tokenizer()
        self._parse_weights()
        self.keys = np.empty(
            (self.h["layers"], self.context, self.h["kv_heads"], self.h["head_dim"]),
            dtype=np.float32,
        )
        self.values = np.empty_like(self.keys)
        self.refresh_history = np.zeros(
            (self.h["layers"], self.h["hidden"], 8), dtype=np.float32
        ) if self.kind == "micro2" else None

    def _model_kind(self):
        signature = (self.h["vocab"], self.h["hidden"], self.h["layers"])
        kinds = {
            (8192, 256, 10): "banana2",
            (8192, 384, 14): "banana2",
            (2048, 128, 9): "micro2",
            (1536, 128, 4): "microv1",
        }
        try:
            return kinds[signature]
        except KeyError as error:
            raise ValueError(f"unsupported BM2NQ architecture {signature}") from error

    def close(self):
        self.data.close()
        self.file.close()

    def _vector(self, offset: int, size: int):
        result = np.frombuffer(self.data, dtype="<f4", count=size, offset=offset).copy()
        return result, offset + size * 4

    def _matrix(self, offset: int, rows: int, cols: int):
        bits = self.h["bits"]
        if bits == 32:
            count = rows * cols
            matrix = np.frombuffer(self.data, dtype="<f4", count=count, offset=offset).copy()
            return matrix.reshape(rows, cols), offset + count * 4
        if bits == 16:
            count = rows * cols
            matrix = np.frombuffer(self.data, dtype="<f2", count=count, offset=offset).astype(np.float32)
            return matrix.reshape(rows, cols), offset + count * 2
        packed_size = (cols * bits + 7) // 8
        stride = packed_size + 4
        raw = np.frombuffer(self.data, dtype=np.uint8, count=rows * stride, offset=offset)
        records = raw.reshape(rows, stride)
        scales = records[:, :4].copy().reshape(-1).view("<f4")
        packed = records[:, 4:]
        if bits == 8:
            quantized = packed[:, :cols].view(np.int8)
        elif bits == 4:
            quantized = np.empty((rows, cols), dtype=np.int8)
            quantized[:, 0::2] = packed & 15
            odd = quantized[:, 1::2]
            odd[:] = (packed >> 4)[:, : odd.shape[1]]
            quantized[quantized >= 8] -= 16
        else:
            quantized = np.empty((rows, cols), dtype=np.int8)
            for shift in range(4):
                quantized[:, shift::4] = ((packed >> (shift * 2)) & 3)[:, : quantized[:, shift::4].shape[1]] - 1
        matrix = quantized.astype(np.float32) * scales[:, None]
        return matrix, offset + rows * stride

    def _parse_weights(self):
        h = self.h
        p = h["weights_offset"]
        self.embedding, p = self._matrix(p, h["vocab"], h["hidden"])
        self.blocks = []
        kv = h["kv_heads"] * h["head_dim"]
        for _ in range(h["layers"]):
            block = {}
            block["ln1"], p = self._vector(p, h["hidden"])
            block["q"], p = self._matrix(p, h["hidden"], h["hidden"])
            block["k"], p = self._matrix(p, kv, h["hidden"])
            block["v"], p = self._matrix(p, kv, h["hidden"])
            block["o"], p = self._matrix(p, h["hidden"], h["hidden"])
            if self.kind != "microv1":
                block["qnorm"], p = self._vector(p, h["head_dim"])
                block["knorm"], p = self._vector(p, h["head_dim"])
            if self.kind == "micro2":
                block["refresh_att_norm"], p = self._vector(p, h["hidden"])
                block["refresh_emb_norm"], p = self._vector(p, h["hidden"])
                block["refresh_out_norm"], p = self._vector(p, h["hidden"])
                block["refresh_gate"], p = self._matrix(p, h["hidden"], h["hidden"])
                block["refresh_value"], p = self._matrix(p, h["hidden"], h["hidden"])
                block["refresh_out"], p = self._matrix(p, h["hidden"], h["hidden"])
                block["refresh_kernel"], p = self._matrix(p, h["hidden"], 9)
                block["refresh_alpha"], p = self._vector(p, 1)
            block["ln2"], p = self._vector(p, h["hidden"])
            block["gate"], p = self._matrix(p, h["intermediate"], h["hidden"])
            block["up"], p = self._matrix(p, h["intermediate"], h["hidden"])
            block["down"], p = self._matrix(p, h["hidden"], h["intermediate"])
            self.blocks.append(block)
        self.final_norm, p = self._vector(p, h["hidden"])
        if p != h["weights_offset"] + h["weights_size"]:
            raise ValueError("weight layout does not match BM2NQ header")

    def _parse_tokenizer(self):
        h = self.h
        self.tokens = []
        for token_id in range(h["vocab"]):
            entry = h["token_index_offset"] + token_id * TOKEN.size
            offset, length, _ = TOKEN.unpack_from(self.data, entry)
            begin = h["token_data_offset"] + offset
            self.tokens.append(bytes(self.data[begin : begin + length]))
        self.byte_tokens = {token[0]: i for i, token in enumerate(self.tokens) if len(token) == 1}
        self.merge_map = {}
        for rank in range(h["merges_count"]):
            left, right, result, stored_rank = MERGE.unpack_from(
                self.data, h["merges_offset"] + rank * MERGE.size
            )
            self.merge_map[(left, right)] = (stored_rank, result)

    def tokenize(self, text: str, chat: bool = False):
        if chat:
            text = f"<|user|>\n{text}\n<|assistant|>\n"
        pieces = [self.byte_tokens.get(byte, self.h["unk"]) for byte in text.encode("utf-8")]
        while len(pieces) > 1:
            best_position = -1
            best_rank = 1 << 30
            best_result = 0
            for position, pair in enumerate(zip(pieces, pieces[1:])):
                merge = self.merge_map.get(pair)
                if merge is not None and merge[0] < best_rank:
                    best_rank, best_result = merge
                    best_position = position
            if best_position < 0:
                break
            pieces[best_position : best_position + 2] = [best_result]
        capacity = self.context - 1
        return [self.h["bos"]] + pieces[-capacity:]

    def _rms(self, values, weights):
        return values * (1.0 / np.sqrt(np.mean(values * values) + self.h["rms_eps"])) * weights

    def _rope(self, values, position):
        head_dim = self.h["head_dim"]
        frequencies = 1.0 / (self.h["rope_theta"] ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
        angles = position * frequencies
        if self.kernel_math:
            angles = (angles + np.pi) % (2.0 * np.pi) - np.pi
            squared = angles * angles
            sine = angles * (1.0 + squared * (-0.16666667 + squared * (0.00833333 + squared * -0.00019841)))
            cosine = 1.0 + squared * (-0.5 + squared * (0.04166667 + squared * -0.00138889))
        else:
            cosine, sine = np.cos(angles), np.sin(angles)
        even = values[:, 0::2].copy()
        odd = values[:, 1::2].copy()
        values[:, 0::2] = even * cosine - odd * sine
        values[:, 1::2] = even * sine + odd * cosine

    def forward(self, token: int, position: int):
        h = self.h
        heads, kv_heads, head_dim = h["heads"], h["kv_heads"], h["head_dim"]
        original_embedding = self.embedding[token].copy()
        x = original_embedding.copy()
        if self.kind != "microv1":
            x *= np.sqrt(np.float32(h["hidden"]))
            original_embedding = x.copy()
        for layer_index, block in enumerate(self.blocks):
            normalized = self._rms(x, block["ln1"])
            q = (block["q"] @ normalized).reshape(heads, head_dim)
            k = (block["k"] @ normalized).reshape(kv_heads, head_dim)
            v = (block["v"] @ normalized).reshape(kv_heads, head_dim)
            if self.kind != "microv1":
                for head in range(heads):
                    q[head] = self._rms(q[head], block["qnorm"])
                for head in range(kv_heads):
                    k[head] = self._rms(k[head], block["knorm"])
            self._rope(q, position)
            self._rope(k, position)
            self.keys[layer_index, position] = k
            self.values[layer_index, position] = v
            attention = np.empty((heads, head_dim), dtype=np.float32)
            repeats = heads // kv_heads
            for head in range(heads):
                kv_head = head // repeats
                scores = self.keys[layer_index, : position + 1, kv_head] @ q[head]
                scores *= 1.0 / np.sqrt(np.float32(head_dim))
                scores -= np.max(scores)
                probabilities = self._exp(scores)
                probabilities /= np.sum(probabilities)
                attention[head] = probabilities @ self.values[layer_index, : position + 1, kv_head]
            attention_output = block["o"] @ attention.reshape(-1)
            x += attention_output
            if self.kind == "micro2":
                signal = self._rms(attention_output, block["refresh_att_norm"])
                history = self.refresh_history[layer_index]
                convolution_input = np.concatenate((history, signal[:, None]), axis=1)
                convolved = np.sum(convolution_input * block["refresh_kernel"], axis=1)
                self.refresh_history[layer_index] = convolution_input[:, 1:]
                gate = block["refresh_gate"] @ signal + convolved
                value = block["refresh_value"] @ self._rms(
                    original_embedding, block["refresh_emb_norm"]
                )
                refreshed = block["refresh_out"] @ (gate / (1.0 + self._exp(-gate)) * value)
                x += block["refresh_alpha"][0] * self._rms(
                    refreshed, block["refresh_out_norm"]
                )
            normalized = self._rms(x, block["ln2"])
            gate = block["gate"] @ normalized
            up = block["up"] @ normalized
            silu = gate / (1.0 + self._exp(-gate))
            x += block["down"] @ (silu * up)
        normalized = self._rms(x, self.final_norm)
        return self.embedding @ normalized

    def _exp(self, values):
        values = np.asarray(values, dtype=np.float32)
        if not self.kernel_math:
            return np.exp(np.clip(values, -30.0, 30.0))
        values = np.clip(values, -16.0, 16.0)
        z = values * np.float32(1.4426950409)
        power = np.floor(z).astype(np.int32)
        fraction = z - power
        poly = 1.0 + fraction * (
            0.69314718 + fraction * (0.24022651 + fraction * (0.05550411 + fraction * 0.00961813))
        )
        return (poly * np.exp2(power)).astype(np.float32)

    @staticmethod
    def _select(logits, seen, penalty):
        if penalty != 1.0:
            logits = logits.copy()
            ids = np.fromiter(set(seen), dtype=np.int32)
            selected = logits[ids]
            logits[ids] = np.where(selected < 0, selected * penalty, selected / penalty)
        return int(np.argmax(logits))

    def generate(self, prompt: str, max_tokens: int = 16, chat: bool = False,
                 repetition_penalty: float = 1.0, stream: bool = False):
        token_ids = self.tokenize(prompt, chat=chat)
        if self.refresh_history is not None:
            self.refresh_history.fill(0)
        if len(token_ids) + max_tokens > self.context:
            token_ids = [token_ids[0]] + token_ids[-(self.context - max_tokens - 1) :]
        logits = None
        for position, token in enumerate(token_ids):
            logits = self.forward(token, position)
        generated = []
        started = time.perf_counter()
        while len(generated) < max_tokens:
            token = self._select(logits, token_ids + generated, repetition_penalty)
            if token == self.h["eos"]:
                break
            generated.append(token)
            if stream:
                sys.stdout.buffer.write(self.tokens[token])
                sys.stdout.buffer.flush()
            if len(token_ids) + len(generated) >= self.context:
                break
            logits = self.forward(token, len(token_ids) + len(generated) - 1)
        elapsed = time.perf_counter() - started
        output = b"".join(self.tokens[token] for token in generated).decode("utf-8", errors="replace")
        return output, len(generated) / elapsed if elapsed else float("inf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--chat", action="store_true", help="apply the Nano-Chat user/assistant template")
    parser.add_argument("--prompt", help="generate once instead of entering interactive mode")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--kernel-math", action="store_true", help="match the 486 kernel's fast math approximations")
    args = parser.parse_args()
    runner = BM2NRunner(args.model, context=args.context, kernel_math=args.kernel_math)
    try:
        prompts = [args.prompt] if args.prompt is not None else iter(lambda: input("\nprompt> "), "")
        for prompt in prompts:
            if prompt is None:
                break
            print("response> ", end="", flush=True)
            output, tps = runner.generate(
                prompt, args.max_tokens, args.chat, args.repetition_penalty, stream=True
            )
            del output
            print(f"\n[{tps:.1f} tokens/s]")
    finally:
        runner.close()


if __name__ == "__main__":
    main()
