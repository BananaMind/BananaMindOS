"""BananaMind tensor layout.  Kept separate from other runtimes by design."""

from __future__ import annotations

import struct

from tools.litemodel_common import ARCH_BANANA, FLAG_TIED_EMBEDDING, choose_tensor

KIND_STANDARD = 1
KIND_MICRO2 = 2
KIND_MICRO_V1 = 3


def prepare(config: dict, sf):
    model_type = config.get("model_type", "")
    hidden = config["hidden_size"]
    layers = config["num_hidden_layers"]
    head_dim = config["head_dim"]
    kv = config["num_key_value_heads"] * head_dim
    intermediate = config["intermediate_size"]
    vocab = config["vocab_size"]
    plan = []

    if model_type == "microbanana":
        kind, qk_norm, embedding_scale, refresh_mask, refresh_kernel = (
            KIND_MICRO_V1, 0, 0, 0, 0
        )
        plan.append(("matrix", choose_tensor(sf, "embed_tokens.weight", "model.embed_tokens.weight"), vocab, hidden))
        for index in range(layers):
            prefix = f"layers.{index}"
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
        plan.append(("vector", choose_tensor(sf, "norm.weight", "model.norm.weight"), hidden))
    else:
        micro = model_type == "bananamind2_micro"
        kind = KIND_MICRO2 if micro else KIND_STANDARD
        qk_norm = 1
        embedding_scale = 1
        refresh_mask = (1 << layers) - 1 if micro else 0
        refresh_kernel = int(config.get("refresh_kernel_size", 0)) if micro else 0
        plan.append(("matrix", choose_tensor(sf, "transformer.wte.weight", "model.embed_tokens.weight"), vocab, hidden))
        for index in range(layers):
            prefix = f"transformer.h.{index}"
            if micro:
                plan.extend([
                    ("vector", f"{prefix}.input_norm.weight", hidden),
                    ("matrix", f"{prefix}.attention.q_proj.weight", hidden, hidden),
                    ("matrix", f"{prefix}.attention.k_proj.weight", kv, hidden),
                    ("matrix", f"{prefix}.attention.v_proj.weight", kv, hidden),
                    ("matrix", f"{prefix}.attention.o_proj.weight", hidden, hidden),
                    ("vector", f"{prefix}.attention.q_norm.weight", head_dim),
                    ("vector", f"{prefix}.attention.k_norm.weight", head_dim),
                    ("vector", f"{prefix}.refresh.attention_norm.weight", hidden),
                    ("vector", f"{prefix}.refresh.embedding_norm.weight", hidden),
                    ("vector", f"{prefix}.refresh.output_norm.weight", hidden),
                    ("matrix", f"{prefix}.refresh.gate_proj.weight", hidden, hidden),
                    ("matrix", f"{prefix}.refresh.value_proj.weight", hidden, hidden),
                    ("matrix", f"{prefix}.refresh.out_proj.weight", hidden, hidden),
                    ("matrix", f"{prefix}.refresh.depthwise_kernel", hidden, refresh_kernel),
                    ("vector", f"{prefix}.refresh.alpha", 1),
                    ("vector", f"{prefix}.post_attention_norm.weight", hidden),
                    ("matrix", f"{prefix}.mlp.gate_proj.weight", intermediate, hidden),
                    ("matrix", f"{prefix}.mlp.up_proj.weight", intermediate, hidden),
                    ("matrix", f"{prefix}.mlp.down_proj.weight", hidden, intermediate),
                ])
            else:
                plan.extend([
                    ("vector", f"{prefix}.ln_1.weight", hidden),
                    ("matrix", f"{prefix}.attn.q_proj.weight", hidden, hidden),
                    ("matrix", f"{prefix}.attn.k_proj.weight", kv, hidden),
                    ("matrix", f"{prefix}.attn.v_proj.weight", kv, hidden),
                    ("matrix", f"{prefix}.attn.o_proj.weight", hidden, hidden),
                    ("vector", f"{prefix}.attn.q_norm.weight", head_dim),
                    ("vector", f"{prefix}.attn.k_norm.weight", head_dim),
                    ("vector", f"{prefix}.ln_2.weight", hidden),
                    ("matrix", f"{prefix}.mlp.w_gate.weight", intermediate, hidden),
                    ("matrix", f"{prefix}.mlp.w_up.weight", intermediate, hidden),
                    ("matrix", f"{prefix}.mlp.w_down.weight", hidden, intermediate),
                ])
        plan.append(("vector", choose_tensor(sf, "transformer.ln_f.weight", "model.norm.weight"), hidden))

    architecture_data = struct.pack(
        "<5I", kind, qk_norm, embedding_scale, refresh_mask, refresh_kernel
    )
    flags = FLAG_TIED_EMBEDDING if config.get("tie_word_embeddings", True) else 0
    return ARCH_BANANA, flags, architecture_data, plan
