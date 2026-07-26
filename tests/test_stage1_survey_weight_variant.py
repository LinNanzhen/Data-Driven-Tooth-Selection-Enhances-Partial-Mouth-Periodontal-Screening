import ast
import hashlib
import importlib.util
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


def parse_variant_module(relative_path):
    return ast.parse(
        (VARIANT / relative_path).read_text(encoding="utf-8-sig")
    )


def imported_names(module):
    names = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
    return names


class SurveyWeightVariantTests(unittest.TestCase):
    def test_variant_exists_with_same_initial_file_set(self):
        self.assertTrue(VARIANT.is_dir())
        self.assertEqual(set(source_digest(ORIGINAL)), set(source_digest(VARIANT)))

    def test_tuner_defines_weighted_scorer_without_class_weight_code(self):
        module = parse_variant_module("core/hyperopt.py")
        class_names = {
            node.name for node in module.body if isinstance(node, ast.ClassDef)
        }
        names = {
            node.id for node in ast.walk(module) if isinstance(node, ast.Name)
        }

        self.assertIn("SurveyWeightedOvRMacroAUC", class_names)
        self.assertNotIn("compute_sample_weight", imported_names(module))
        self.assertNotIn("compute_sample_weight", names)

    def test_weighted_scorer_aligns_validation_weights_by_index(self):
        try:
            import numpy as np
            import pandas as pd
        except ImportError:
            self.skipTest("scientific Python dependencies are not installed")

        module_path = VARIANT / "core" / "hyperopt.py"
        spec = importlib.util.spec_from_file_location(
            "survey_weight_hyperopt_for_test",
            module_path,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        captured = {}

        def fake_roc_auc_score(y_true, y_score, **kwargs):
            captured["sample_weight"] = kwargs["sample_weight"].tolist()
            captured["labels"] = kwargs["labels"].tolist()
            return 0.75

        class FakeEstimator:
            classes_ = np.array([0, 1, 2])

            @staticmethod
            def predict_proba(X):
                return np.full((len(X), 3), 1 / 3)

        module.roc_auc_score = fake_roc_auc_score
        scorer = module.SurveyWeightedOvRMacroAUC(
            pd.Series([1.0, 4.0, 2.0], index=[10, 20, 30])
        )
        X_validation = pd.DataFrame({"x": [1, 2]}, index=[20, 10])

        score = scorer(FakeEstimator(), X_validation, pd.Series([0, 1]))

        self.assertEqual(score, 0.75)
        self.assertEqual(captured["sample_weight"], [4.0, 1.0])
        self.assertEqual(captured["labels"], [0, 1, 2])

        with self.assertRaisesRegex(ValueError, "index must be unique"):
            module.SurveyWeightedOvRMacroAUC(
                pd.Series([1.0, 2.0], index=[10, 10])
            )


if __name__ == "__main__":
    unittest.main()
