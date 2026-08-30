"""Google Gemma 3 text tensor layout."""

from __future__ import annotations

import struct

from tools.litemodel_common import (
    ARCH_GEMMA3, FLAG_SPACE_TO_MARKER, FLAG_TIED_EMBEDDING,
)


def prepare(config: dict, sf):
    del sf
    hidden = int(config["hidden_size"])
    layers = int(config["num_hidden_layers"])
    head_dim = int(config["head_dim"])
    query = int(config["num_attention_heads"]) * head_dim
    kv = int(config["num_key_value_heads"]) * head_dim
    intermediate = int(config["intermediate_size"])
    layer_types = list(config["layer_types"])
    if len(layer_types) != layers:
        raise ValueError("Gemma 3 layer_types does not match num_hidden_layers")
    full_mask = 0
    plan = [("matrix", "model.embed_tokens.weight", int(config["vocab_size"]), hidden)]
    for index, layer_type in enumerate(layer_types):
        if layer_type == "full_attention":
            full_mask |= 1 << index
        elif layer_type != "sliding_attention":
            raise ValueError(f"unsupported Gemma 3 layer type {layer_type!r}")
        prefix = f"model.layers.{index}"
        plan.extend([
            ("vector", f"{prefix}.input_layernorm.weight", hidden),
            ("matrix", f"{prefix}.self_attn.q_proj.weight", query, hidden),
            ("matrix", f"{prefix}.self_attn.k_proj.weight", kv, hidden),
            ("matrix", f"{prefix}.self_attn.v_proj.weight", kv, hidden),
            ("matrix", f"{prefix}.self_attn.o_proj.weight", hidden, query),
            ("vector", f"{prefix}.self_attn.q_norm.weight", head_dim),
            ("vector", f"{prefix}.self_attn.k_norm.weight", head_dim),
            ("vector", f"{prefix}.post_attention_layernorm.weight", hidden),
            ("vector", f"{prefix}.pre_feedforward_layernorm.weight", hidden),
            ("matrix", f"{prefix}.mlp.gate_proj.weight", intermediate, hidden),
            ("matrix", f"{prefix}.mlp.up_proj.weight", intermediate, hidden),
            ("matrix", f"{prefix}.mlp.down_proj.weight", hidden, intermediate),
            ("vector", f"{prefix}.post_feedforward_layernorm.weight", hidden),
        ])
    plan.append(("vector", "model.norm.weight", hidden))
    rope = config.get("rope_parameters", {})
    full_theta = float(
        rope.get("full_attention", {}).get("rope_theta", config.get("rope_theta", 1_000_000.0))
    )
    sliding_theta = float(
        rope.get("sliding_attention", {}).get(
            "rope_theta", config.get("rope_local_base_freq", 10_000.0)
        )
    )
    architecture_data = struct.pack(
        "<IIfff", full_mask, int(config.get("sliding_window", 512)),
        full_theta ** (-2.0 / head_dim), sliding_theta ** (-2.0 / head_dim),
        float(config.get("query_pre_attn_scalar", head_dim)) ** -0.5,
    )
    flags = FLAG_SPACE_TO_MARKER
    if config.get("tie_word_embeddings", True):
        flags |= FLAG_TIED_EMBEDDING
    return ARCH_GEMMA3, flags, architecture_data, plan
