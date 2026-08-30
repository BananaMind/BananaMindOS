import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "update_build_number", ROOT / "tools" / "update_build_number.py"
)
UPDATER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(UPDATER)


class BuildNumberTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "include").mkdir()
        (self.root / "BUILD_NUMBER").write_text("134\n", encoding="ascii")

    def tearDown(self):
        self.temporary.cleanup()

    def test_user_build_keeps_number(self):
        self.assertEqual(UPDATER.update_build_number(self.root), 134)
        self.assertEqual((self.root / "BUILD_NUMBER").read_text(), "134\n")

    def test_each_developer_build_increments_number(self):
        (self.root / "dev_folder").mkdir()
        self.assertEqual(UPDATER.update_build_number(self.root), 135)
        self.assertEqual(UPDATER.update_build_number(self.root), 136)
        self.assertIn(
            "BANANAMIND_BUILD_NUMBER 136u",
            (self.root / "include" / "build_number.h").read_text(),
        )


if __name__ == "__main__":
    unittest.main()
