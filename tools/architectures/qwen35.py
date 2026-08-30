"""Qwen 3.5 text-only hybrid gated-delta/attention tensor layout."""

from __future__ import annotations

import struct

from tools.litemodel_common import ARCH_QWEN35, FLAG_TIED_EMBEDDING, choose_tensor


def prepare(config: dict, sf):
    hidden = int(config["hidden_size"])
    layers = int(config["num_hidden_layers"])
    head_dim = int(config["head_dim"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    intermediate = int(config["intermediate_size"])
    key_heads = int(config["linear_num_key_heads"])
    value_heads = int(config["linear_num_value_heads"])
    key_head_dim = int(config["linear_key_head_dim"])
    value_head_dim = int(config["linear_value_head_dim"])
    key_dim = key_heads * key_head_dim
    value_dim = value_heads * value_head_dim
    conv_dim = key_dim * 2 + value_dim
    kernel = int(config["linear_conv_kernel_dim"])
    layer_types = list(config["layer_types"])
    if len(layer_types) != layers:
        raise ValueError("Qwen 3.5 layer_types does not match num_hidden_layers")
    root = "model.language_model" if "model.language_model.embed_tokens.weight" in sf.meta else "model"
    full_mask = 0
    plan = [("matrix", f"{root}.embed_tokens.weight", int(config["vocab_size"]), hidden)]
    for index, layer_type in enumerate(layer_types):
        prefix = f"{root}.layers.{index}"
        plan.append(("vector", f"{prefix}.input_layernorm.weight", hidden))
        if layer_type == "linear_attention":
            linear = f"{prefix}.linear_attn"
            plan.extend([
                ("matrix", f"{linear}.in_proj_qkv.weight", conv_dim, hidden),
                ("matrix", f"{linear}.in_proj_z.weight", value_dim, hidden),
                ("matrix", f"{linear}.in_proj_b.weight", value_heads, hidden),
                ("matrix", f"{linear}.in_proj_a.weight", value_heads, hidden),
                ("matrix", f"{linear}.conv1d.weight", conv_dim, kernel),
                ("vector", f"{linear}.dt_bias", value_heads),
                ("vector", f"{linear}.A_log", value_heads),
                ("vector", f"{linear}.norm.weight", value_head_dim),
                ("matrix", f"{linear}.out_proj.weight", hidden, value_dim),
            ])
        elif layer_type == "full_attention":
            full_mask |= 1 << index
            attention = f"{prefix}.self_attn"
            plan.extend([
                ("matrix", f"{attention}.q_proj.weight", heads * head_dim * 2, hidden),
                ("matrix", f"{attention}.k_proj.weight", kv_heads * head_dim, hidden),
                ("matrix", f"{attention}.v_proj.weight", kv_heads * head_dim, hidden),
                ("matrix", f"{attention}.o_proj.weight", hidden, heads * head_dim),
                ("vector", f"{attention}.q_norm.weight", head_dim),
                ("vector", f"{attention}.k_norm.weight", head_dim),
            ])
        else:
            raise ValueError(f"unsupported Qwen 3.5 layer type {layer_type!r}")
        plan.extend([
            ("vector", f"{prefix}.post_attention_layernorm.weight", hidden),
            ("matrix", f"{prefix}.mlp.gate_proj.weight", intermediate, hidden),
            ("matrix", f"{prefix}.mlp.up_proj.weight", intermediate, hidden),
            ("matrix", f"{prefix}.mlp.down_proj.weight", hidden, intermediate),
        ])
    plan.append(("vector", choose_tensor(sf, f"{root}.norm.weight"), hidden))
    partial = float(config.get("rope_parameters", {}).get("partial_rotary_factor", 1.0))
    rotary_dim = int(head_dim * partial)
    theta = float(config["rope_theta"])
    architecture_data = struct.pack(
        "<7If", full_mask, kernel, key_heads, value_heads, key_head_dim,
        value_head_dim, rotary_dim, theta ** (-2.0 / rotary_dim),
    )
    flags = FLAG_TIED_EMBEDDING if config.get("tie_word_embeddings", True) else 0
    return ARCH_QWEN35, flags, architecture_data, plan
