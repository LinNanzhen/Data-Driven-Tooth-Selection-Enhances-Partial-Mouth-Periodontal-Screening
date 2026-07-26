import hashlib
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = REPO_ROOT / "Stage1"
VARIANT = REPO_ROOT / "Stage1_survey_weight_only"


def source_digest(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.name != ".DS_Store"
    }


class SurveyWeightVariantTests(unittest.TestCase):
    def test_variant_exists_with_same_initial_file_set(self):
        self.assertTrue(VARIANT.is_dir())
        self.assertEqual(set(source_digest(ORIGINAL)), set(source_digest(VARIANT)))


if __name__ == "__main__":
    unittest.main()
