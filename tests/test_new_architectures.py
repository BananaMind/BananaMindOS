import json
import struct
import tempfile
import unittest
from pathlib import Path

from tools.architectures import gemma3, lfm2, qwen35
from tools.convert_litemodel import normalize_config
from tools.litemodel_common import (
    ARCH_GEMMA3,
    ARCH_LFM2,
    ARCH_QWEN35,
    FLAG_CHAT,
    FLAG_SPACE_TO_MARKER,
    FLAG_TIED_EMBEDDING,
    HEADER,
    load_merges,
    load_special_token_ids,
    load_token_bytes,
    _configured_special_ids,
    write_litemodel,
)


class FakeSafeTensor:
    def __init__(self, names=()):
        self.meta = {name: {} for name in names}


def common_config(**updates):
    config = {
        "vocab_size": 64,
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "intermediate_size": 16,
        "max_position_embeddings": 256,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
    }
    config.update(updates)
    return config


class NewArchitectureTests(unittest.TestCase):
    def test_lfm350_uses_the_checkpoint_actual_ffn_width(self):
        raw = common_config(
            model_type="lfm2",
            intermediate_size=6656,
            block_auto_adjust_ff_dim=True,
            block_multiple_of=256,
            block_ffn_dim_multiplier=1.0,
            layer_types=["conv", "full_attention"],
            conv_L_cache=3,
        )
        config = normalize_config(raw, "lfm2")
        self.assertEqual(config["intermediate_size"], 4608)
        architecture, flags, data, plan = lfm2.prepare(config, FakeSafeTensor())
        self.assertEqual(architecture, ARCH_LFM2)
        self.assertEqual(flags, FLAG_TIED_EMBEDDING)
        attention_mask, kernel, _ratio = struct.unpack("<IIf", data)
        self.assertEqual((attention_mask, kernel), (2, 3))
        self.assertIn(
            ("matrix", "model.layers.1.feed_forward.w1.weight", 4608, 8),
            plan,
        )

    def test_gemma_plan_keeps_local_and_global_attention_separate(self):
        config = common_config(
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=512,
            rope_theta=1_000_000.0,
            rope_local_base_freq=10_000.0,
            query_pre_attn_scalar=256,
        )
        architecture, flags, data, plan = gemma3.prepare(config, FakeSafeTensor())
        self.assertEqual(architecture, ARCH_GEMMA3)
        self.assertEqual(flags, FLAG_TIED_EMBEDDING | FLAG_SPACE_TO_MARKER)
        full_mask, window, *_ = struct.unpack("<IIfff", data)
        self.assertEqual((full_mask, window), (2, 512))
        self.assertIn(
            ("vector", "model.layers.0.post_feedforward_layernorm.weight", 8),
            plan,
        )

    def test_qwen_plan_selects_language_stack_and_omits_auxiliary_tensors(self):
        raw = {
            "model_type": "qwen3_5",
            "text_config": common_config(
                model_type="qwen3_5_text",
                num_hidden_layers=4,
                layer_types=[
                    "linear_attention", "linear_attention",
                    "linear_attention", "full_attention",
                ],
                linear_num_key_heads=2,
                linear_num_value_heads=2,
                linear_key_head_dim=4,
                linear_value_head_dim=4,
                linear_conv_kernel_dim=4,
                rope_parameters={
                    "rope_theta": 10_000_000,
                    "partial_rotary_factor": 0.5,
                },
            ),
        }
        config = normalize_config(raw, "qwen35")
        root = "model.language_model"
        sf = FakeSafeTensor((f"{root}.embed_tokens.weight", f"{root}.norm.weight"))
        architecture, flags, data, plan = qwen35.prepare(config, sf)
        self.assertEqual(architecture, ARCH_QWEN35)
        self.assertEqual(flags, FLAG_TIED_EMBEDDING)
        full_mask, kernel, *_ = struct.unpack("<7If", data)
        self.assertEqual((full_mask, kernel), (8, 4))
        names = [item[1] for item in plan]
        self.assertTrue(all(name.startswith(root) for name in names))
        self.assertFalse(any("vision" in name or "mtp" in name for name in names))


class VersionTwoTokenizerTests(unittest.TestCase):
    def test_token_flags_do_not_overwrite_container_flags(self):
        tokenizer = {
            "model": {
                "vocab": {"<pad>": 0, "<bos>": 1, "<eos>": 2, "a": 3},
                "merges": [],
            },
            "added_tokens": [
                {"id": 1, "content": "<bos>", "special": True},
                {"id": 2, "content": "<eos>", "special": True},
            ],
        }
        config = common_config(
            vocab_size=4, bos_token_id=1, eos_token_id=2,
            pad_token_id=0, unk_token_id=3,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.safetensors"
            model.write_bytes(struct.pack("<Q", 2) + b"{}")
            tokenizer_path = root / "tokenizer.json"
            tokenizer_path.write_text(json.dumps(tokenizer), encoding="utf-8")
            output = root / "test.litemodel"
            write_litemodel(
                model=model, tokenizer=tokenizer_path, config=config, output=output,
                bits=4, architecture=ARCH_LFM2, flags=FLAG_CHAT,
                architecture_data=b"", plan=[],
            )
            header = HEADER.unpack_from(output.read_bytes())
            self.assertEqual(header[6], FLAG_CHAT)

    def test_missing_special_ids_do_not_turn_normal_tokens_into_controls(self):
        self.assertEqual(
            _configured_special_ids({"eos_token_id": 248044}),
            {248044},
        )

    def test_sparse_32_bit_ids_merges_and_byte_fallback(self):
        tokenizer = {
            "model": {
                "vocab": {
                    "a": 1,
                    "b": 70000,
                    "ab": 70001,
                    "<0x20>": 70002,
                },
                "merges": [["a", "b"]],
            },
            "added_tokens": [
                {"id": 70003, "content": "<|im_end|>", "special": True}
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            path.write_text(json.dumps(tokenizer), encoding="utf-8")
            tokens = load_token_bytes(path, 70004, set())
            self.assertEqual(tokens[70002], b" ")
            self.assertEqual(tokens[69999], b"")
            self.assertEqual(load_merges(path), [(1, 70000, 70001, 0)])
            self.assertEqual(load_special_token_ids(path, set()), {70003})


if __name__ == "__main__":
    unittest.main()
