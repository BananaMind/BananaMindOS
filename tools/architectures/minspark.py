"""min-spark Meiosis tensor layout."""

from __future__ import annotations

import struct

from tools.litemodel_common import ARCH_MINSPARK, FLAG_TIED_EMBEDDING


def _plain_block(plan: list, prefix: str, hidden: int, qkv: int, intermediate: int):
    plan.extend([
        ("vector", f"{prefix}.attn_norm.weight", hidden),
        ("matrix", f"{prefix}.attn.qkv.weight", qkv, hidden),
        ("matrix", f"{prefix}.attn.out.weight", hidden, hidden),
        ("vector", f"{prefix}.ffn_norm.weight", hidden),
        ("matrix", f"{prefix}.ffn.gate_up.weight", intermediate * 2, hidden),
        ("matrix", f"{prefix}.ffn.down.weight", hidden, intermediate),
    ])


def prepare(config: dict, sf):
    del sf
    hidden = config["hidden_size"]
    kv_dim = config["num_key_value_heads"] * config["head_dim"]
    qkv = hidden + 2 * kv_dim
    intermediate = config["intermediate_size"]
    prelude = int(config["prelude_layers"])
    body = int(config["body_blocks"])
    coda = int(config["coda_layers"])
    loops = int(config["max_loops"])
    rank = int(config["lora_rank"])
    plan = [("matrix", "embed.weight", config["vocab_size"], hidden)]
    for index in range(prelude):
        _plain_block(plan, f"prelude.{index}", hidden, qkv, intermediate)
    for index in range(body):
        prefix = f"body.{index}"
        _plain_block(plan, prefix, hidden, qkv, intermediate)
        for residual in ("ddl_attn", "ddl_ffn"):
            plan.extend([
                ("matrix", f"{prefix}.{residual}.beta.weight", 1, hidden),
                ("vector", f"{prefix}.{residual}.beta.bias", 1),
                ("matrix", f"{prefix}.{residual}.v_proj.weight", 1, hidden),
                ("vector", f"{prefix}.{residual}.v_proj.bias", 1),
            ])
        for loop in range(loops):
            plan.extend([
                ("matrix", f"loop_lora.{index}.down.{loop}.weight", rank, hidden),
                ("matrix", f"loop_lora.{index}.up.{loop}.weight", qkv, rank),
            ])
    plan.append(("matrix", "loop_embed.weight", loops, hidden))
    for index in range(coda):
        _plain_block(plan, f"coda.{index}", hidden, qkv, intermediate)
    plan.append(("vector", "final_norm.weight", hidden))
    architecture_data = struct.pack(
        "<6I2fI", prelude, body, coda, loops, int(config["train_loops"]), rank,
        float(config.get("ddl_k_eps", 0.01)), float(config.get("ddl_v_sigmoid_scale", 4.0)),
        int(config.get("doc_mask_eos", 0xFFFFFFFF) if config.get("doc_mask_eos") is not None else 0xFFFFFFFF),
    )
    return ARCH_MINSPARK, FLAG_TIED_EMBEDDING, architecture_data, plan
