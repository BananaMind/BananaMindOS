"""LiquidAI LFM2 hybrid convolution/attention tensor layout."""

from __future__ import annotations

import struct

from tools.litemodel_common import ARCH_LFM2, FLAG_TIED_EMBEDDING


def prepare(config: dict, sf):
    del sf
    hidden = int(config["hidden_size"])
    layers = int(config["num_hidden_layers"])
    head_dim = int(config["head_dim"])
    kv = int(config["num_key_value_heads"]) * head_dim
    intermediate = int(config["intermediate_size"])
    layer_types = list(config["layer_types"])
    if len(layer_types) != layers:
        raise ValueError("LFM2 layer_types does not match num_hidden_layers")
    attention_mask = 0
    plan = [("matrix", "model.embed_tokens.weight", int(config["vocab_size"]), hidden)]
    for index, layer_type in enumerate(layer_types):
        prefix = f"model.layers.{index}"
        plan.append(("vector", f"{prefix}.operator_norm.weight", hidden))
        if layer_type == "full_attention":
            attention_mask |= 1 << index
            plan.extend([
                ("matrix", f"{prefix}.self_attn.q_proj.weight", hidden, hidden),
                ("matrix", f"{prefix}.self_attn.k_proj.weight", kv, hidden),
                ("matrix", f"{prefix}.self_attn.v_proj.weight", kv, hidden),
                ("matrix", f"{prefix}.self_attn.out_proj.weight", hidden, hidden),
                ("vector", f"{prefix}.self_attn.q_layernorm.weight", head_dim),
                ("vector", f"{prefix}.self_attn.k_layernorm.weight", head_dim),
            ])
        elif layer_type == "conv":
            kernel = int(config["conv_L_cache"])
            plan.extend([
                ("matrix", f"{prefix}.conv.in_proj.weight", hidden * 3, hidden),
                ("matrix", f"{prefix}.conv.conv.weight", hidden, kernel),
                ("matrix", f"{prefix}.conv.out_proj.weight", hidden, hidden),
            ])
        else:
            raise ValueError(f"unsupported LFM2 layer type {layer_type!r}")
        plan.extend([
            ("vector", f"{prefix}.ffn_norm.weight", hidden),
            ("matrix", f"{prefix}.feed_forward.w1.weight", intermediate, hidden),
            ("matrix", f"{prefix}.feed_forward.w3.weight", intermediate, hidden),
            ("matrix", f"{prefix}.feed_forward.w2.weight", hidden, intermediate),
        ])
    plan.append(("vector", "model.embedding_norm.weight", hidden))
    theta = float(config["rope_theta"])
    rope_ratio = theta ** (-2.0 / head_dim)
    architecture_data = struct.pack(
        "<IIf", attention_mask, int(config["conv_L_cache"]), rope_ratio
    )
    flags = FLAG_TIED_EMBEDDING if config.get("tie_word_embeddings", True) else 0
    return ARCH_LFM2, flags, architecture_data, plan
