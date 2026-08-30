import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "build_litemodel_iso", ROOT / "tools" / "build_litemodel_iso.py"
)
BUILDER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BUILDER)


class IsoBuilderSelectionTests(unittest.TestCase):
    def setUp(self):
        self.nano = {
            "id": "nano", "name": "Nano",
            "quants": {str(bits): bits + 4 for bits in range(1, 9)},
        }
        self.mini = {
            "id": "mini", "name": "Mini", "quants": {"4": 20, "8": 34}
        }
        self.models = {"nano": self.nano, "mini": self.mini}

    def compact(self, text):
        return [
            (model["id"], bits)
            for model, bits in BUILDER.parse_selections(text, self.models)
        ]

    def test_multiple_quantizations_after_one_model_id(self):
        self.assertEqual(
            self.compact("nano:2,4,8"),
            [("nano", 2), ("nano", 4), ("nano", 8)],
        )

    def test_all_integer_quantizations_can_be_selected(self):
        self.assertEqual(
            self.compact("nano:1,2,3,4,5,6,7,8"),
            [("nano", bits) for bits in range(1, 9)],
        )

    def test_registry_exposes_q1_through_q8_for_every_model(self):
        registry, _ = BUILDER.load_registry()
        expected = {str(bits) for bits in range(1, 9)}
        for model in registry["models"]:
            self.assertTrue(expected.issubset(model["quants"]), model["id"])

    def test_compact_list_can_switch_models(self):
        self.assertEqual(
            self.compact("nano:2,4,mini:4,8"),
            [("nano", 2), ("nano", 4), ("mini", 4), ("mini", 8)],
        )

    def test_explicit_repeated_model_syntax_still_works(self):
        self.assertEqual(
            self.compact("nano:2,nano:4,nano:8"),
            [("nano", 2), ("nano", 4), ("nano", 8)],
        )

    def test_duplicate_model_quantization_is_removed(self):
        self.assertEqual(self.compact("nano:4,4,nano:4"), [("nano", 4)])

    def test_invalid_quantization_is_rejected(self):
        with self.assertRaises(SystemExit):
            BUILDER.parse_selections("nano:16", self.models)

    def test_interactive_precision_prompt_accepts_commas(self):
        registry = {
            "presets": {"10": [], "25": [], "100": [], "250": []}
        }
        with patch("builtins.input", side_effect=["c", "1", "2,4,8"]):
            selections, label = BUILDER.interactive_selection(registry, self.models)
        self.assertEqual(label, "custom")
        self.assertEqual(
            [(model["id"], bits) for model, bits in selections],
            [("nano", 2), ("nano", 4), ("nano", 8)],
        )


if __name__ == "__main__":
    unittest.main()
