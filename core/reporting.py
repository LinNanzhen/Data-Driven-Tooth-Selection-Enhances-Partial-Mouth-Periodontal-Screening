from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from core.config import OutputSpec, RunResult


def _safe_to_csv(df, path: Path):
    if df is None:
        return
    if isinstance(df, pd.DataFrame):
        if df.empty:
            df.to_csv(path, index=False)
            return
        df.to_csv(path, index=False)


def export_results(run_result: RunResult, output_dir: Path, output_spec: OutputSpec, keep_legacy: bool = True):
    output_dir.mkdir(parents=True, exist_ok=True)

    std_paths = {
        "metrics_cv.csv": output_dir / output_spec.metrics_cv,
        "metrics_test.csv": output_dir / output_spec.metrics_test,
        "comparison_cv_test.csv": output_dir / output_spec.comparison_cv_test,
        "summary_final.csv": output_dir / output_spec.summary_final,
    }

    _safe_to_csv(run_result.cv_results, std_paths["metrics_cv.csv"])
    _safe_to_csv(run_result.test_results, std_paths["metrics_test.csv"])
    _safe_to_csv(run_result.comparison, std_paths["comparison_cv_test.csv"])
    _safe_to_csv(run_result.summary, std_paths["summary_final.csv"])

    if not keep_legacy:
        return std_paths

    for standard_name, legacy_name in output_spec.legacy_map.items():
        src = std_paths.get(standard_name)
        if src is None or not src.exists():
            continue
        dst = output_dir / legacy_name
        if src.resolve() == dst.resolve():
            continue
        shutil.copy2(src, dst)

    return std_paths
