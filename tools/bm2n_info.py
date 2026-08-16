#!/usr/bin/env python3
"""Inspect and checksum a BM2NQ model without shadowing Python's inspect module."""

import argparse
import mmap
import struct
import zlib
from pathlib import Path

from convert import HEADER, MAGIC

FIELDS = (
    "magic version header_size file_size bits vocab_size hidden_size num_layers "
    "num_heads num_kv_heads head_dim intermediate_size max_seq_len rope_theta "
    "rms_eps weights_offset weights_size token_index_offset token_data_offset "
    "token_data_size merges_offset merges_count bos_id eos_id pad_id unk_id payload_crc32"
).split()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    with args.model.open("rb") as file:
        data = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        values = HEADER.unpack_from(data)
        info = dict(zip(FIELDS, values))
        if info["magic"] != MAGIC:
            raise SystemExit("bad BM2NQ magic")
        if info["file_size"] != len(data):
            raise SystemExit(f"file size mismatch: header={info['file_size']} actual={len(data)}")
        actual_crc = zlib.crc32(data[info["weights_offset"] :])
        valid = actual_crc == info["payload_crc32"]
        precision = f"Q{info['bits']}" if info["bits"] <= 8 else f"FP{info['bits']}"
        print(f"BM2NQ v{info['version']} {precision}")
        print(f"size: {len(data) / (1024 * 1024):.2f} MiB")
        print(f"architecture: {info['num_layers']}L, d={info['hidden_size']}, "
              f"heads={info['num_heads']}/{info['num_kv_heads']}, ff={info['intermediate_size']}")
        print(f"tokenizer: {info['vocab_size']} tokens, {info['merges_count']} merges")
        print(f"payload CRC-32: {info['payload_crc32']:08x} ({'ok' if valid else 'FAILED'})")
        data.close()
        if not valid:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
