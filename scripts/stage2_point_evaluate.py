"""Evaluate fixed tooth sets with the original Stage 2 case rule and survey weights."""

import json
from pathlib import Path
import sys

import pandas as pd

from stage1_data import DEVELOPMENT_CSV, EXTERNAL_CSV, STAGE1_OUTPUT_DIR

# Load the original Stage 2 package in a process dedicated to this evaluation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Stage2"))
from core.data_processing import DataProcessor
from core.metrics import compute_screening_metrics, weighted_kappa_quadratic
from scenarios.overall import build_config


def main():
    shap_dir = STAGE1_OUTPUT_DIR / "shap"
    summary = json.loads((shap_dir / "summary.json").read_text())
    top10 = pd.read_csv(shap_dir / "top10.csv", dtype={"tooth_fdi": str})
    output_dir = STAGE1_OUTPUT_DIR / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    analyses = [(summary["primary_aggregation"], "fixed_tooth_screening_metrics.csv")]
    if "class_prior_sensitivity_aggregation" in summary:
        analyses.append((summary["class_prior_sensitivity_aggregation"], "fixed_tooth_class_prior_sensitivity.csv"))
    config = build_config()
    for aggregation, filename in analyses:
        schemes = {
            f"{line}:{model}": group.sort_values("rank")["tooth_fdi"].tolist()
            for (line, model), group in top10.loc[top10["shap_aggregation"].eq(aggregation)].groupby(
                ["training_weighting", "model"]
            )
        }
        if not schemes:
            raise ValueError(f"No fixed tooth schemes for SHAP aggregation {aggregation}")
        schemes.update({name: config.tooth_selection_strategies[name] for name in ("XGB10", "LGB10")})
        results = evaluate_schemes(schemes)
        results.to_csv(output_dir / filename, index=False)
        print(f"SHAP aggregation: {aggregation}")
        print(results.drop(columns="fdi").round(4).to_string(index=False))


def evaluate_schemes(schemes):
    config = build_config()
    if any(len(teeth) != 10 or len(set(teeth)) != 10 for teeth in schemes.values()):
        raise ValueError("Each fixed scheme must contain 10 distinct teeth")

    manifest = pd.read_csv(STAGE1_OUTPUT_DIR / "split_assignments.csv")
    internal = manifest.loc[manifest["split"].eq("internal_test"), "SEQN"]
    train = manifest.loc[manifest["split"].eq("train"), "SEQN"]
    if not manifest["SEQN"].is_unique:
        raise ValueError("Stage 1 split must assign each participant once")
    rows = []
    for cohort_name, path in (("internal_test", DEVELOPMENT_CSV), ("external_2013_2014", EXTERNAL_CSV)):
        processor = DataProcessor(config)
        data = processor.load_and_preprocess(path).set_index("SEQN")
        if not data.index.is_unique:
            raise ValueError(f"Duplicate participant IDs in {path}")
        if cohort_name == "internal_test":
            data = data.loc[internal].copy()
        elif len(data.index.intersection(train)) or len(data.index.intersection(internal)):
            raise ValueError("The external cohort overlaps the development split")

        labels = data[config.target_column].to_numpy()
        weights = data[config.weight_column].to_numpy()
        for name, fdi in schemes.items():
            nhanes = processor.get_tooth_measurements(fdi)
            if len(nhanes) != 10:
                raise ValueError(f"Could not map all 10 teeth for {name}")
            data["partial_label"] = processor.classify_with_tooth_selection(data, nhanes)
            total = compute_screening_metrics(data, config.weight_column, config.target_column, "partial_label", [1, 2])
            severe = compute_screening_metrics(data, config.weight_column, config.target_column, "partial_label", [2])
            rows.append({
                "cohort": cohort_name, "scheme": name, "fdi": " ".join(fdi), "n": len(data),
                "qwk": weighted_kappa_quadratic(labels, data["partial_label"].to_numpy(), weights),
                "full28_prevalence_pct": total["prev_true"], "top10_prevalence_pct": total["prev_pred"],
                "inflation_factor_pct": total["infl_factor"], "sensitivity_pct": total["sensitivity"],
                "specificity_pct": total["specificity"], "severe_sensitivity_pct": severe["sensitivity"],
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
