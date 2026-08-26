#!/usr/bin/env python3
"""Convert supported checkpoints into the portable .litemodel container."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.litemodel_common import FLAG_CHAT, SafeTensorFile, write_litemodel

ARCHITECTURES = {
    "banana": "tools.architectures.banana",
    "llama": "tools.architectures.llama",
    "gptx": "tools.architectures.gptx",
    "rose": "tools.architectures.rose",
    "minspark": "tools.architectures.minspark",
}


def architecture_name(config: dict) -> str:
    model_type = str(config.get("model_type", ""))
    if model_type.startswith("bananamind") or model_type == "microbanana":
        return "banana"
    if model_type in {"llama", "smollm", "smollm2"}:
        return "llama"
    if model_type == "gptx2":
        return "gptx"
    if model_type == "rose_x1":
        return "rose"
    if model_type == "minspark":
        return "minspark"
    raise ValueError(f"unsupported model_type {model_type!r}")


def normalize_config(config: dict, architecture: str) -> dict:
    config = dict(config)
    if architecture == "minspark":
        config.update({
            "vocab_size": config["vocab_size"],
            "hidden_size": config["dim"],
            "num_attention_heads": config["n_heads"],
            "num_key_value_heads": config.get("n_kv_heads") or config["n_heads"],
            "intermediate_size": config["ffn_hidden"],
            "num_hidden_layers": (
                config["prelude_layers"] + config["body_blocks"] + config["coda_layers"]
            ),
            "head_dim": config["dim"] // config["n_heads"],
            "rope_theta": config["rope_base"],
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": config.get("max_seq_len", 512),
            "bos_token_id": config.get("bos_token_id", config.get("doc_mask_eos", 2)),
            "eos_token_id": config.get("eos_token_id", config.get("doc_mask_eos", 2)),
            "pad_token_id": config.get("pad_token_id", 0),
            "unk_token_id": config.get("unk_token_id", 1),
        })
    else:
        config.setdefault(
            "head_dim", config["hidden_size"] // config["num_attention_heads"]
        )
    required = {
        "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "head_dim", "intermediate_size",
        "max_position_embeddings", "rope_theta", "rms_norm_eps",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"config is missing {sorted(missing)}")
    return config


def convert(model: Path, tokenizer: Path, config_path: Path, output: Path,
            bits: int, architecture: str = "auto", chat: bool = False):
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    selected = architecture_name(raw_config) if architecture == "auto" else architecture
    config = normalize_config(raw_config, selected)
    module = importlib.import_module(ARCHITECTURES[selected])
    sf = SafeTensorFile(model)
    try:
        arch_id, flags, architecture_data, plan = module.prepare(config, sf)
    finally:
        sf.close()
    if chat:
        flags |= FLAG_CHAT
    write_litemodel(
        model=model, tokenizer=tokenizer, config=config, output=output, bits=bits,
        architecture=arch_id, flags=flags, architecture_data=architecture_data,
        plan=plan,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--bits", required=True, type=int, choices=(2, 4, 8, 16, 32))
    parser.add_argument("--architecture", choices=("auto", *ARCHITECTURES), default="auto")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    convert(
        args.model, args.tokenizer, args.config, args.output, args.bits,
        args.architecture, args.chat,
    )


if __name__ == "__main__":
    main()
