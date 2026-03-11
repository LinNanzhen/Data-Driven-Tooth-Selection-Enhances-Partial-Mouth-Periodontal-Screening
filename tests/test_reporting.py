import pandas as pd

from core.config import OutputSpec, RunResult
from core.reporting import export_results


def test_export_results_writes_standard_and_legacy(tmp_path):
    run_result = RunResult(
        cv_results=pd.DataFrame([{"a": 1}]),
        test_results=pd.DataFrame([{"b": 2}]),
        comparison=pd.DataFrame([{"c": 3}]),
        summary=pd.DataFrame([{"d": 4}]),
        stability_results={},
    )

    output_spec = OutputSpec(
        metrics_cv="metrics_cv.csv",
        metrics_test="metrics_test.csv",
        comparison_cv_test="comparison_cv_test.csv",
        summary_final="summary_final.csv",
        legacy_map={
            "metrics_cv.csv": "cross_validation_results.csv",
            "metrics_test.csv": "final_test_results.csv",
        },
    )

    export_results(run_result, tmp_path, output_spec, keep_legacy=True)

    assert (tmp_path / "metrics_cv.csv").exists()
    assert (tmp_path / "metrics_test.csv").exists()
    assert (tmp_path / "comparison_cv_test.csv").exists()
    assert (tmp_path / "summary_final.csv").exists()
    assert (tmp_path / "cross_validation_results.csv").exists()
    assert (tmp_path / "final_test_results.csv").exists()
