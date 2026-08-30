import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "build_dosboxx", ROOT / "tools" / "build_dosboxx.py"
)
BUILDER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BUILDER)


class DosboxXBuilderTests(unittest.TestCase):
    def catalog(self, text: str) -> Path:
        temporary = tempfile.NamedTemporaryFile("w", encoding="ascii", delete=False)
        temporary.write(text)
        temporary.close()
        self.addCleanup(Path(temporary.name).unlink, missing_ok=True)
        return Path(temporary.name)

    def test_selects_only_micro_q4_catalog_entry(self):
        path = self.catalog(
            "# header\n"
            "micro2|MIC4.LITEMODEL|BananaMind 2 Micro|Q4|4|base|legacy|Tiny\n"
            "micro2|MIC8.LITEMODEL|BananaMind 2 Micro|Q8|5|base|legacy|Tiny\n"
        )
        self.assertEqual(
            BUILDER.micro_q4_catalog_line(path),
            "micro2|MIC4.LITEMODEL|BananaMind 2 Micro|Q4|4|base|legacy|Tiny",
        )

    def test_missing_micro_q4_is_rejected(self):
        path = self.catalog(
            "micro2|MIC8.LITEMODEL|BananaMind 2 Micro|Q8|5|base|legacy|Tiny\n"
        )
        with self.assertRaises(SystemExit):
            BUILDER.micro_q4_catalog_line(path)

    def test_duplicate_micro_q4_is_rejected(self):
        line = "micro2|MIC4.LITEMODEL|BananaMind 2 Micro|Q4|4|base|legacy|Tiny\n"
        with self.assertRaises(SystemExit):
            BUILDER.micro_q4_catalog_line(self.catalog(line + line))

    def test_selects_nano_chat_q4_catalog_entry(self):
        path = self.catalog(
            "nano-chat|NNC2.LITEMODEL|BananaMind 2 Nano Chat|Q2|6|chat|legacy|Fast\n"
            "nano-chat|NNC4.LITEMODEL|BananaMind 2 Nano Chat|Q4|8|chat|legacy|Fast\n"
        )
        self.assertEqual(
            BUILDER.catalog_line(path, "nano-chat", "Q4"),
            "nano-chat|NNC4.LITEMODEL|BananaMind 2 Nano Chat|Q4|8|chat|legacy|Fast",
        )


if __name__ == "__main__":
    unittest.main()
