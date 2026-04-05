from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _epi_val(epi: dict, key: str) -> float:
    return epi.get(key, np.nan)


def _epi_ci(epi_ci: dict, key: str, bound: str) -> float:
    return epi_ci.get(key, {}).get(bound, np.nan)


def _build_metric_row(res: dict) -> dict:
    md = res.get("metadata", {})
    epi = res.get("epi_metrics", {})
    epi_ci = res.get("epi_metrics_ci", {})
    return {
        "Strategy": md.get("name", ""),
        "Teeth_FDI": ", ".join(md.get("teeth_fdi", [])),
        "N_Teeth": len(md.get("teeth_fdi", [])),
        "Kappa_Quadratic": res["kappa_quadratic"]["point_estimate"],
        "Kappa_CI_Lower": res["kappa_quadratic"]["ci_lower"],
        "Kappa_CI_Upper": res["kappa_quadratic"]["ci_upper"],
        "F1_Macro": res["f1_macro"]["point_estimate"],
        "F1_CI_Lower": res["f1_macro"]["ci_lower"],
        "F1_CI_Upper": res["f1_macro"]["ci_upper"],
        "Specificity": _epi_val(epi, "specificity"),
        "Specificity_CI_Lower": _epi_ci(epi_ci, "specificity", "ci_lower"),
        "Specificity_CI_Upper": _epi_ci(epi_ci, "specificity", "ci_upper"),
        "Sensitivity": _epi_val(epi, "sensitivity"),
        "Sensitivity_CI_Lower": _epi_ci(epi_ci, "sensitivity", "ci_lower"),
        "Sensitivity_CI_Upper": _epi_ci(epi_ci, "sensitivity", "ci_upper"),
        "Prev_True": _epi_val(epi, "prev_true"),
        "Prev_True_CI_Lower": _epi_ci(epi_ci, "prev_true", "ci_lower"),
        "Prev_True_CI_Upper": _epi_ci(epi_ci, "prev_true", "ci_upper"),
        "Prev_Pred": _epi_val(epi, "prev_pred"),
        "Prev_Pred_CI_Lower": _epi_ci(epi_ci, "prev_pred", "ci_lower"),
        "Prev_Pred_CI_Upper": _epi_ci(epi_ci, "prev_pred", "ci_upper"),
        "Abs_Bias": _epi_val(epi, "abs_bias"),
        "Abs_Bias_CI_Lower": _epi_ci(epi_ci, "abs_bias", "ci_lower"),
        "Abs_Bias_CI_Upper": _epi_ci(epi_ci, "abs_bias", "ci_upper"),
        "Rel_Bias": _epi_val(epi, "rel_bias"),
        "Rel_Bias_CI_Lower": _epi_ci(epi_ci, "rel_bias", "ci_lower"),
        "Rel_Bias_CI_Upper": _epi_ci(epi_ci, "rel_bias", "ci_upper"),
        "Inflation_Factor": _epi_val(epi, "infl_factor"),
        "Inflation_Factor_CI_Lower": _epi_ci(epi_ci, "infl_factor", "ci_lower"),
        "Inflation_Factor_CI_Upper": _epi_ci(epi_ci, "infl_factor", "ci_upper"),
    }


def make_overall_summary(results: Dict[str, dict]) -> pd.DataFrame:
    rows = [_build_metric_row(res) for res in results.values()]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Kappa_Quadratic", ascending=False).reset_index(drop=True)
    return df


def make_subgroup_summary(all_results: Dict[str, Dict[str, dict]]) -> pd.DataFrame:
    rows: List[dict] = []
    for subgroup, group_results in all_results.items():
        for strategy, res in group_results.items():
            row = _build_metric_row(res)
            row["Subgroup"] = subgroup
            row["Strategy_Type"] = res.get("metadata", {}).get("strategy_type", "unknown")
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        cols = ["Subgroup", "Strategy_Type"] + [c for c in df.columns if c not in ("Subgroup", "Strategy_Type")]
        df = df[cols].sort_values(["Subgroup", "Kappa_Quadratic"], ascending=[True, False]).reset_index(drop=True)
    return df


def write_overall_outputs(
    output_dir: Path,
    results: Dict[str, dict],
    comparisons: List[dict],
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, str] = {}

    summary_df = make_overall_summary(results)
    summary_path = output_dir / "metrics_summary.xlsx"
    summary_df.to_excel(summary_path, index=False)
    artifacts["metrics_summary"] = str(summary_path)

    comp_df = pd.DataFrame(comparisons)
    comp_path = output_dir / "pairwise_significance.xlsx"
    comp_df.to_excel(comp_path, index=False)
    artifacts["pairwise_significance"] = str(comp_path)

    return summary_df, artifacts


def write_subgroup_outputs(
    output_dir: Path,
    all_results: Dict[str, Dict[str, dict]],
    all_comparisons: Dict[str, List[dict]],
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, str] = {}

    summary_df = make_subgroup_summary(all_results)
    summary_path = output_dir / "metrics_summary.xlsx"
    summary_df.to_excel(summary_path, index=False)
    artifacts["metrics_summary"] = str(summary_path)

    flat_comp = []
    for subgroup, comps in all_comparisons.items():
        for c in comps:
            item = dict(c)
            item["subgroup"] = subgroup
            flat_comp.append(item)
    comp_df = pd.DataFrame(flat_comp)
    comp_path = output_dir / "pairwise_significance.xlsx"
    comp_df.to_excel(comp_path, index=False)
    artifacts["pairwise_significance"] = str(comp_path)

    return summary_df, artifacts
