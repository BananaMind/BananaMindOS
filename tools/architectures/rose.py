"""Rose X1 tensor layout with selected-layer refresh gates."""

from __future__ import annotations

import struct

from tools.litemodel_common import ARCH_ROSE_X1, FLAG_TIED_EMBEDDING


def prepare(config: dict, sf):
    del sf
    hidden = config["hidden_size"]
    layers = config["num_hidden_layers"]
    head_dim = config["head_dim"]
    kv = config["num_key_value_heads"] * head_dim
    intermediate = config["intermediate_size"]
    vocab = config["vocab_size"]
    refresh_layers = {int(index) for index in config.get("refresh_gate_inject_layers", [])}
    refresh_mask = sum(1 << index for index in refresh_layers)
    kernel = int(config.get("refresh_gate_kernel_size", 9))
    plan = [("matrix", "model.embed_tokens.weight", vocab, hidden)]
    for index in range(layers):
        prefix = f"model.layers.{index}"
        plan.extend([
            ("vector", f"{prefix}.input_layernorm.weight", hidden),
            ("matrix", f"{prefix}.self_attn.q_proj.weight", hidden, hidden),
            ("matrix", f"{prefix}.self_attn.k_proj.weight", kv, hidden),
            ("matrix", f"{prefix}.self_attn.v_proj.weight", kv, hidden),
            ("matrix", f"{prefix}.self_attn.o_proj.weight", hidden, hidden),
            ("vector", f"{prefix}.self_attn.q_norm.weight", head_dim),
            ("vector", f"{prefix}.self_attn.k_norm.weight", head_dim),
        ])
        if index in refresh_layers:
            refresh = f"{prefix}.refresh_gate"
            plan.extend([
                ("vector", f"{refresh}.attn_norm.weight", hidden),
                ("vector", f"{refresh}.emb_norm.weight", hidden),
                ("vector", f"{refresh}.out_norm.weight", hidden),
                ("matrix", f"{refresh}.gate_proj.weight", hidden, hidden),
                ("matrix", f"{refresh}.value_proj.weight", hidden, hidden),
                ("matrix", f"{refresh}.out_proj.weight", hidden, hidden),
                ("matrix", f"{refresh}.causal_conv.weight", hidden, kernel),
                ("vector", f"{refresh}.alpha", 1),
            ])
        plan.extend([
            ("vector", f"{prefix}.post_attention_layernorm.weight", hidden),
            ("matrix", f"{prefix}.mlp.gate_proj.weight", intermediate, hidden),
            ("matrix", f"{prefix}.mlp.up_proj.weight", intermediate, hidden),
            ("matrix", f"{prefix}.mlp.down_proj.weight", hidden, intermediate),
        ])
    plan.append(("vector", "model.norm.weight", hidden))
    # Rose uses half-split RoPE. Refresh mask and kernel are Rose-owned data.
    architecture_data = struct.pack("<4I", 1, int(config.get("use_qk_norm", True)), refresh_mask, kernel)
    flags = FLAG_TIED_EMBEDDING if config.get("tie_word_embeddings", True) else 0
    return ARCH_ROSE_X1, flags, architecture_data, plan
