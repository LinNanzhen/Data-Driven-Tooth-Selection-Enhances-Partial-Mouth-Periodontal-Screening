"""Compare full-mouth and SHAP Top 10 models on the fixed Stage 1 internal test set."""

import json

from joblib import load
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score, roc_auc_score

from stage1_data import MAX_FEATURE, STAGE1_OUTPUT_DIR, TARGET_COLUMN, nhanes_to_fdi, prepare_stage1_data
from stage1_train import balance_survey_weights, load_training_data, overall_models_and_spaces


def weighted_metrics(model, features, labels, weights):
    if tuple(model.classes_) != (0, 1, 2):
        raise ValueError(f"Unexpected model class order: {model.classes_}")
    probabilities = model.predict_proba(features)
    predicted = model.predict(features)
    severe = labels.eq(2)
    predicted_severe = predicted == 2
    return {
        "auc_macro_ovr": roc_auc_score(labels, probabilities, multi_class="ovr", average="macro", sample_weight=weights),
        "accuracy": accuracy_score(labels, predicted, sample_weight=weights),
        "f1_macro": f1_score(labels, predicted, average="macro", sample_weight=weights),
        "qwk": cohen_kappa_score(labels, predicted, weights="quadratic", sample_weight=weights),
        "severe_auc_ovr": roc_auc_score(severe, probabilities[:, 2], sample_weight=weights),
        "severe_recall": recall_score(severe, predicted_severe, sample_weight=weights),
        "severe_precision": precision_score(severe, predicted_severe, sample_weight=weights, zero_division=0),
        "severe_false_positive_rate": np.average(predicted_severe[~severe], weights=weights[~severe]),
    }


def main(*, smoke=False):
    output_root = STAGE1_OUTPUT_DIR / "smoke" if smoke else STAGE1_OUTPUT_DIR
    x_train, y_train, w_train = load_training_data()
    cohort, features = prepare_stage1_data()
    cohort = cohort.set_index("SEQN")
    features.index = cohort.index
    manifest = pd.read_csv(STAGE1_OUTPUT_DIR / "split_assignments.csv")
    test_ids = manifest.loc[manifest["split"].eq("internal_test"), "SEQN"]
    if test_ids.empty or not x_train.index.intersection(test_ids).empty:
        raise ValueError("Expected a nonempty internal test split disjoint from training")
    x_test = features.loc[test_ids]
    y_test = cohort.loc[test_ids, TARGET_COLUMN]
    w_test = cohort.loc[test_ids, "WTMEC2YR"]

    shap_dir = output_root / "shap"
    shap_summary = json.loads((shap_dir / "summary.json").read_text())
    aggregation = shap_summary["primary_aggregation"]
    top10 = pd.read_csv(shap_dir / "top10.csv", dtype={"tooth_fdi": str})
    top10 = top10.loc[top10["shap_aggregation"].eq(aggregation)].sort_values("rank")
    tooth_by_feature = {feature: nhanes_to_fdi(MAX_FEATURE.fullmatch(feature)[1]) for feature in x_train}

    rows = []
    for line in ("nhanes_only", "nhanes_balanced"):
        fit_weights = balance_survey_weights(y_train, w_train) if line == "nhanes_balanced" else w_train
        training_dir = output_root / "training" / line
        tuning = json.loads((training_dir / "summary.json").read_text())
        for name, (reduced_model, _) in overall_models_and_spaces().items():
            params = tuning["models"][name]["best_params"]
            if params != shap_summary["best_params"][line][name]:
                raise ValueError(f"SHAP and tuning parameters differ for {line}/{name}")
            selected = top10.loc[top10["training_weighting"].eq(line) & top10["model"].eq(name), "tooth_fdi"]
            columns = [feature for feature, tooth in tooth_by_feature.items() if tooth in set(selected)]
            if len(selected) != 10 or selected.nunique() != 10 or len(columns) != 20:
                raise ValueError(f"Expected 10 distinct Top 10 teeth and 20 features for {line}/{name}")

            full_model = load(training_dir / f"{name}.joblib")
            if any(full_model.get_params()[key] != value for key, value in params.items()):
                raise ValueError(f"Saved full-mouth model and tuning parameters differ for {line}/{name}")
            reduced_model.set_params(**params).fit(x_train[columns], y_train, sample_weight=fit_weights)
            for feature_set, model, test_features in (
                ("full28", full_model, x_test), ("shap_top10", reduced_model, x_test[columns]),
            ):
                rows.append({
                    "training_weighting": line, "model": name, "feature_set": feature_set,
                    "shap_aggregation": aggregation if feature_set == "shap_top10" else "",
                    "selected_fdi": " ".join(selected) if feature_set == "shap_top10" else "",
                    "n_features": test_features.shape[1], "n_test": len(y_test),
                    **weighted_metrics(model, test_features, y_test, w_test),
                })

    output_dir = output_root / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "internal_test_metrics.csv", index=False)
    print(y_test.value_counts().sort_index().to_string())
    print(results.drop(columns=["selected_fdi", "shap_aggregation"]).to_string(index=False))


if __name__ == "__main__":
    main()
