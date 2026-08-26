"""Llama/SmolLM tensor layout."""

from __future__ import annotations

import struct

from tools.litemodel_common import ARCH_LLAMA, FLAG_TIED_EMBEDDING, choose_tensor


def prepare(config: dict, sf):
    hidden = config["hidden_size"]
    layers = config["num_hidden_layers"]
    head_dim = config["head_dim"]
    kv = config["num_key_value_heads"] * head_dim
    intermediate = config["intermediate_size"]
    vocab = config["vocab_size"]
    plan = [("matrix", choose_tensor(sf, "model.embed_tokens.weight"), vocab, hidden)]
    for index in range(layers):
        prefix = f"model.layers.{index}"
        plan.extend([
            ("vector", f"{prefix}.input_layernorm.weight", hidden),
            ("matrix", f"{prefix}.self_attn.q_proj.weight", hidden, hidden),
            ("matrix", f"{prefix}.self_attn.k_proj.weight", kv, hidden),
            ("matrix", f"{prefix}.self_attn.v_proj.weight", kv, hidden),
            ("matrix", f"{prefix}.self_attn.o_proj.weight", hidden, hidden),
            ("vector", f"{prefix}.post_attention_layernorm.weight", hidden),
            ("matrix", f"{prefix}.mlp.gate_proj.weight", intermediate, hidden),
            ("matrix", f"{prefix}.mlp.up_proj.weight", intermediate, hidden),
            ("matrix", f"{prefix}.mlp.down_proj.weight", hidden, intermediate),
        ])
    plan.append(("vector", "model.norm.weight", hidden))
    # 1 = half-split rotary layout used by Hugging Face Llama.
    architecture_data = struct.pack("<I", 1)
    flags = FLAG_TIED_EMBEDDING if config.get("tie_word_embeddings", True) else 0
    return ARCH_LLAMA, flags, architecture_data, plan
