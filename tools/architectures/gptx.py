"""GPT-X2.5 tensor layout with XSA attention."""

from __future__ import annotations

import struct

from tools.litemodel_common import ARCH_GPTX2, FLAG_TIED_EMBEDDING, choose_tensor


def prepare(config: dict, sf):
    hidden = config["hidden_size"]
    layers = config["num_hidden_layers"]
    head_dim = config["head_dim"]
    kv = config["num_key_value_heads"] * head_dim
    intermediate = config["intermediate_size"]
    vocab = config["vocab_size"]
    plan = [("matrix", choose_tensor(sf, "transformer.wte.weight"), vocab, hidden)]
    for index in range(layers):
        prefix = f"transformer.h.{index}"
        plan.extend([
            ("vector", f"{prefix}.ln_1.weight", hidden),
            ("matrix", f"{prefix}.attn.q_proj.weight", hidden, hidden),
            ("matrix", f"{prefix}.attn.k_proj.weight", kv, hidden),
            ("matrix", f"{prefix}.attn.v_proj.weight", kv, hidden),
            ("matrix", f"{prefix}.attn.o_proj.weight", hidden, hidden),
            ("vector", f"{prefix}.ln_2.weight", hidden),
            ("matrix", f"{prefix}.mlp.w_gate.weight", intermediate, hidden),
            ("matrix", f"{prefix}.mlp.w_up.weight", intermediate, hidden),
            ("matrix", f"{prefix}.mlp.w_down.weight", hidden, intermediate),
        ])
    plan.append(("vector", "transformer.ln_f.weight", hidden))
    architecture_data = struct.pack(
        "<3I", int(config.get("xsa_projection", True)),
        int(config.get("qk_norm", False)), int(config.get("embedding_scale", False))
    )
    flags = FLAG_TIED_EMBEDDING if config.get("tie_word_embeddings", True) else 0
    return ARCH_GPTX2, flags, architecture_data, plan
