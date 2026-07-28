import ast
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE1_CORE = REPO_ROOT / "Stage1" / "core"


def parse_module(filename):
    return ast.parse((STAGE1_CORE / filename).read_text(encoding="utf-8-sig"))


def class_definition(module, class_name):
    return next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )


def annotated_fields(class_node):
    return [
        node.target.id
        for node in class_node.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ]


class Stage1SelectionOnlyTests(unittest.TestCase):
    def test_stage1_result_contract_contains_selection_outputs_only(self):
        config_module = parse_module("config.py")

        output_fields = annotated_fields(class_definition(config_module, "OutputSpec"))
        result_fields = annotated_fields(class_definition(config_module, "RunResult"))

        self.assertEqual(
            output_fields,
            ["metrics_cv", "summary_final", "legacy_map"],
        )
        self.assertEqual(
            result_fields,
            ["cv_results", "summary", "stability_results", "extra"],
        )

    def test_stage1_has_no_holdout_model_evaluation_api_or_call(self):
        evaluation_module = parse_module("evaluation.py")
        pipeline_module = parse_module("pipeline.py")

        model_pipeline = class_definition(evaluation_module, "ModelPipeline")
        method_names = {
            node.name for node in model_pipeline.body if isinstance(node, ast.FunctionDef)
        }
        module_function_names = {
            node.name for node in evaluation_module.body if isinstance(node, ast.FunctionDef)
        }
        called_attributes = {
            node.func.attr
            for node in ast.walk(pipeline_module)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }

        self.assertNotIn("evaluate_test_set", method_names)
        self.assertNotIn("create_test_performance_summary_plot", module_function_names)
        self.assertNotIn("evaluate_test_set", called_attributes)

    def test_stage1_still_exports_reserved_holdout_rows(self):
        pipeline_module = parse_module("pipeline.py")
        module_function_names = {
            node.name for node in pipeline_module.body if isinstance(node, ast.FunctionDef)
        }
        save_holdout_calls = [
            node
            for node in ast.walk(pipeline_module)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_save_test_set"
        ]

        self.assertIn("_save_test_set", module_function_names)
        self.assertGreaterEqual(len(save_holdout_calls), 2)


if __name__ == "__main__":
    unittest.main()
