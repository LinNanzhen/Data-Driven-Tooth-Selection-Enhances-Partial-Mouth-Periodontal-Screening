"""Compare the NHANES-only XGBoost and original XGB10 sets on the external cycle."""

import json

import numpy as np
import pandas as pd

from stage2_point_evaluate import (
    EXTERNAL_CSV, STAGE1_OUTPUT_DIR, DataProcessor, build_config,
    weighted_kappa_quadratic,
)
from core.bootstrap import PSUBootstrapSampler


def main(*, smoke=False):
    output_root = STAGE1_OUTPUT_DIR / "smoke" if smoke else STAGE1_OUTPUT_DIR
    shap_dir = output_root / "shap"
    aggregation = json.loads((shap_dir / "summary.json").read_text())["primary_aggregation"]
    rankings = pd.read_csv(shap_dir / "top10.csv", dtype={"tooth_fdi": str})
    selected = rankings.loc[
        rankings["shap_aggregation"].eq(aggregation)
        & rankings["training_weighting"].eq("nhanes_only")
        & rankings["model"].eq("xgboost")
    ].sort_values("rank")["tooth_fdi"].tolist()
    if len(set(selected)) != 10 or len(selected) != 10:
        raise ValueError("Expected ten distinct teeth for NHANES-only XGBoost")

    config = build_config()
    if smoke:
        config.n_bootstrap = 20
    processor = DataProcessor(config)
    data = processor.load_and_preprocess(EXTERNAL_CSV)
    original = config.tooth_selection_strategies["XGB10"]
    predictions = [
        processor.classify_with_tooth_selection(data, processor.get_tooth_measurements(teeth))
        for teeth in (selected, original)
    ]
    truth = data[config.target_column].to_numpy()
    weights = data[config.weight_column].to_numpy()

    def metrics(y, predictions, w):
        any_case = y > 0
        severe = y == 2
        return np.array([
            weighted_kappa_quadratic(y, predictions, w),
            100 * w[any_case & (predictions > 0)].sum() / w[any_case].sum(),
            100 * w[severe & (predictions == 2)].sum() / w[severe].sum(),
        ])

    def difference(index):
        y, w = truth[index], weights[index]
        return metrics(y, predictions[0][index], w) - metrics(y, predictions[1][index], w)

    sampler = PSUBootstrapSampler(data, config.stratum_column, config.psu_column, config.random_seed)
    differences = np.array([difference(sampler.generate_indices()) for _ in range(config.n_bootstrap)])
    lower, upper = np.percentile(differences, [2.5, 97.5], axis=0)
    results = pd.DataFrame({
        "metric": ["qwk", "any_periodontitis_sensitivity_pp", "severe_sensitivity_pp"],
        "difference": difference(slice(None)), "ci_lower": lower, "ci_upper": upper,
        "n": len(data), "bootstrap_replicates": config.n_bootstrap, "seed": config.random_seed,
        "nhanes_only_xgboost_fdi": " ".join(selected), "original_xgb10_fdi": " ".join(original),
    })
    output = output_root / "evaluation" / "external_paired_bootstrap.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
