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

from tools.litemodel_common import (
    CHAT_STYLES, FLAG_CHAT, FLAG_NO_BOS, SafeTensorFile, chat_style_flag,
    write_litemodel,
)
from tools.quantization import SUPPORTED_BITS

ARCHITECTURES = {
    "banana": "tools.architectures.banana",
    "llama": "tools.architectures.llama",
    "gptx": "tools.architectures.gptx",
    "rose": "tools.architectures.rose",
    "minspark": "tools.architectures.minspark",
    "lfm2": "tools.architectures.lfm2",
    "gemma3": "tools.architectures.gemma3",
    "qwen35": "tools.architectures.qwen35",
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
    if model_type == "lfm2":
        return "lfm2"
    if model_type in {"gemma3", "gemma3_text"}:
        return "gemma3"
    if model_type in {"qwen3_5", "qwen3_5_text"}:
        return "qwen35"
    raise ValueError(f"unsupported model_type {model_type!r}")


def normalize_config(config: dict, architecture: str) -> dict:
    config = dict(config)
    if architecture == "qwen35" and "text_config" in config:
        config = dict(config["text_config"])
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
        if "rope_theta" not in config:
            rope = config.get("rope_parameters", {})
            if architecture == "gemma3" and "full_attention" in rope:
                rope = rope["full_attention"]
            config["rope_theta"] = rope.get("rope_theta", 10000.0)
        if "rms_norm_eps" not in config and "norm_eps" in config:
            config["rms_norm_eps"] = config["norm_eps"]
        if architecture == "lfm2" and config.get("block_auto_adjust_ff_dim"):
            intermediate = int(2 * int(config["intermediate_size"]) / 3)
            intermediate = int(config.get("block_ffn_dim_multiplier", 1.0) * intermediate)
            multiple = int(config.get("block_multiple_of", 256))
            config["intermediate_size"] = multiple * ((intermediate + multiple - 1) // multiple)
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
            bits: int, architecture: str = "auto", chat: bool = False,
            chat_style: str = "banana"):
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
        flags |= FLAG_CHAT | chat_style_flag(chat_style)
        if chat_style in {"smollm", "qwen35"}:
            flags |= FLAG_NO_BOS
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
    parser.add_argument("--bits", required=True, type=int, choices=SUPPORTED_BITS)
    parser.add_argument("--architecture", choices=("auto", *ARCHITECTURES), default="auto")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--chat-style", choices=tuple(CHAT_STYLES), default="banana")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    convert(
        args.model, args.tokenizer, args.config, args.output, args.bits,
        args.architecture, args.chat, args.chat_style,
    )


if __name__ == "__main__":
    main()
