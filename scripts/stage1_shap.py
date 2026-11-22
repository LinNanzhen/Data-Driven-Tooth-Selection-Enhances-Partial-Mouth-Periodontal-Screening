"""Rank Stage 1 teeth using repeated, survey-weighted multiclass Tree SHAP."""

import argparse
from importlib.metadata import version
import json

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from stage1_data import MAX_FEATURE, nhanes_to_fdi
from stage1_train import CV_FOLDS, STAGE1_OUTPUT_DIR, balance_survey_weights, load_training_data, overall_models_and_spaces


SEEDS = (42, 123, 456, 999, 2025)
TRAINING_LINES = ("nhanes_only", "nhanes_balanced")
CLASS_LABELS = (0, 1, 2)
EQUAL_CLASS_WEIGHTS = np.full(len(CLASS_LABELS), 1 / len(CLASS_LABELS))
PRIMARY_AGGREGATION = "survey_full_equal_class"
CLASS_PRIOR_AGGREGATION = "survey_full"
AGGREGATION_KEYS = ["training_weighting", "model", "shap_aggregation"]


def shap_tensor(values, expected_shape):
    tensor = np.stack(values, axis=-1) if isinstance(values, list) else np.asarray(values)
    if tensor.shape != expected_shape or not np.isfinite(tensor).all():
        raise ValueError(f"Expected finite SHAP values with shape {expected_shape}; got {tensor.shape}")
    return tensor


def weighted_mean_abs_shap(values, weights, class_weights):
    return np.average(np.abs(values), axis=0, weights=weights) @ class_weights


def explain_fold(
    model, model_name, weighting, X, y, survey_weights, seed, fold, *,
    fit_weights=None, include_survey_sample=False, include_equal_classes=False,
):
    survey_weights = survey_weights.to_numpy()
    if fit_weights is None:
        fit_weights = balance_survey_weights(y, survey_weights) if weighting == "nhanes_balanced" else survey_weights
    model = clone(model).set_params(random_state=seed)
    model.fit(X, y, sample_weight=fit_weights)
    if tuple(model.classes_) != CLASS_LABELS:
        raise ValueError(f"Unexpected model class order: {model.classes_}")

    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", model_output="raw")
    values = shap_tensor(explainer.shap_values(X, check_additivity=False), (*X.shape, len(CLASS_LABELS)))
    raw = model.predict(X, output_margin=True) if model_name == "xgboost" else model.predict(X, raw_score=True)
    reconstructed = values.sum(axis=1) + np.asarray(explainer.expected_value)
    np.testing.assert_allclose(reconstructed, raw, rtol=1e-4, atol=1e-4, err_msg=f"{weighting}/{model_name}/{seed}/{fold}")

    survey_totals = np.bincount(y, weights=survey_weights)
    class_priors = survey_totals / survey_totals.sum()
    sample = np.random.RandomState(seed + fold).choice(
        len(X), size=min(1000, len(X)), replace=False, p=fit_weights / fit_weights.sum(),
    )
    full_shap_by_class = np.average(np.abs(values), axis=0, weights=survey_weights)
    scores = {
        CLASS_PRIOR_AGGREGATION: full_shap_by_class @ class_priors,
        "fit_weighted_sample": weighted_mean_abs_shap(values[sample], fit_weights[sample], class_priors),
    }
    if include_equal_classes:
        scores[PRIMARY_AGGREGATION] = full_shap_by_class @ EQUAL_CLASS_WEIGHTS
    if include_survey_sample:
        scores["survey_same_sample"] = weighted_mean_abs_shap(values[sample], survey_weights[sample], class_priors)
    details = {
        "training_weighting": weighting,
        "model": model_name,
        "seed": seed,
        "fold": fold,
        "n_train": len(X),
        "n_sample": len(sample),
        "class_priors": class_priors.tolist(),
        "survey_weight_sum": float(survey_totals.sum()),
        "fit_weight_sum": float(fit_weights.sum()),
        "sample_weight_sum": float(fit_weights[sample].sum()),
        "fit_weight_by_class": np.bincount(y, weights=fit_weights).tolist(),
        "max_additivity_error": float(np.max(np.abs(reconstructed - raw))),
    }
    return scores, details


def rank_teeth(fold_features):
    fold_keys = [*AGGREGATION_KEYS, "seed", "fold"]
    tooth_keys = [*AGGREGATION_KEYS, "tooth_fdi"]
    fold_teeth = (
        fold_features.groupby([*fold_keys, "tooth_fdi"], as_index=False)["mean_abs_shap"].sum()
        .sort_values([*fold_keys, "mean_abs_shap", "tooth_fdi"], ascending=[True] * len(fold_keys) + [False, True])
    )
    fold_teeth["in_fold_top10"] = fold_teeth.groupby(fold_keys).cumcount().lt(10)
    ranks = fold_teeth.groupby(tooth_keys).agg(
        mean_shap=("mean_abs_shap", "mean"),
        std_shap=("mean_abs_shap", lambda scores: scores.std(ddof=0)),
        fold_top10_count=("in_fold_top10", "sum"),
        n_folds=("mean_abs_shap", "size"),
    )
    feature_std = fold_features.groupby([*tooth_keys, "feature"])["mean_abs_shap"].std(ddof=0)
    ranks["shap_error"] = feature_std.groupby(level=tooth_keys).mean()
    ranks = ranks.reset_index().sort_values(
        [*AGGREGATION_KEYS, "mean_shap", "tooth_fdi"], ascending=[True, True, True, False, True],
    )
    ranks["rank"] = ranks.groupby(AGGREGATION_KEYS).cumcount() + 1
    return ranks


def compare_rankings(rankings, baseline, comparison):
    def fdi_ranks(key):
        return rankings.set_index(AGGREGATION_KEYS).loc[key].set_index("tooth_fdi")["rank"]

    left, right = fdi_ranks(baseline), fdi_ranks(comparison)
    overlap = sorted(set(left[left.le(10)].index) & set(right[right.le(10)].index))
    return {
        "baseline": baseline, "comparison": comparison,
        "top10_overlap": overlap, "n_overlap": len(overlap),
        "rank_change_by_fdi": (right - left).to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Use one seed, two folds and smoke tuning parameters.")
    parser.add_argument("--jobs", type=int, default=4, help="Number of concurrent fold fits.")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")

    training_dir = STAGE1_OUTPUT_DIR / ("training_smoke" if args.smoke else "training")
    output_dir = STAGE1_OUTPUT_DIR / ("shap_smoke" if args.smoke else "shap")
    features, labels, weights = load_training_data()
    if not (features.index.equals(labels.index) and labels.index.equals(weights.index)):
        raise ValueError("Feature, label and survey weight participant indices differ")
    if not np.isfinite(weights).all() or not weights.gt(0).all():
        raise ValueError("SHAP requires finite positive survey weights")
    teeth = [nhanes_to_fdi(MAX_FEATURE.fullmatch(feature)[1]) for feature in features]
    if len(teeth) != 56 or len(set(teeth)) != 28 or not pd.Series(teeth).value_counts().eq(2).all():
        raise ValueError("Expected exactly two features for each of the 28 teeth")

    seeds, folds = ((42,), 2) if args.smoke else (SEEDS, CV_FOLDS)
    splits = [
        (seed, fold, train)
        for seed in seeds
        for fold, (train, _) in enumerate(
            StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(features, labels)
        )
    ]
    frames, fold_details, best_params = [], [], {}
    for weighting in TRAINING_LINES:
        tuning_path = training_dir / weighting / "summary.json"
        tuning = json.loads(tuning_path.read_text())
        if tuning["n_train"] != len(features) or tuning["training_weighting"] != weighting:
            raise ValueError(f"Tuning summary does not match the current training data and weighting: {tuning_path}")
        best_params[weighting] = {}
        for name, (model, _) in overall_models_and_spaces().items():
            params = tuning["models"][name]["best_params"]
            best_params[weighting][name] = params
            model.set_params(**params)
            print(f"{weighting}/{name}: calculating SHAP for {len(splits)} folds", flush=True)
            results = Parallel(n_jobs=args.jobs, prefer="threads")(
                delayed(explain_fold)(
                    model, name, weighting, features.iloc[train], labels.iloc[train], weights.iloc[train], seed, fold,
                    include_equal_classes=True,
                )
                for seed, fold, train in splits
            )
            for scores, details in results:
                fold_details.append(details)
                for aggregation, scores_by_feature in scores.items():
                    frames.append(pd.DataFrame({
                        "training_weighting": weighting, "model": name, "shap_aggregation": aggregation,
                        "seed": details["seed"], "fold": details["fold"],
                        "feature": features.columns, "tooth_fdi": teeth, "mean_abs_shap": scores_by_feature,
                    }))

    fold_features = pd.concat(frames, ignore_index=True)
    rankings = rank_teeth(fold_features)
    top10 = rankings.loc[rankings["rank"].le(10)]
    model_names = tuple(overall_models_and_spaces())
    comparisons = [
        compare_rankings(rankings, ("nhanes_only", model, aggregation), ("nhanes_balanced", model, aggregation))
        for model in model_names for aggregation in (PRIMARY_AGGREGATION, CLASS_PRIOR_AGGREGATION)
    ] + [
        compare_rankings(rankings, (line, model, CLASS_PRIOR_AGGREGATION), (line, model, aggregation))
        for line in TRAINING_LINES for model in model_names
        for aggregation in (PRIMARY_AGGREGATION, "fit_weighted_sample")
    ]
    summary = {
        "n_development_train": len(features), "seeds": seeds, "folds_per_seed": folds,
        "model_output": "raw", "feature_perturbation": "tree_path_dependent", "classes": CLASS_LABELS,
        "primary_aggregation": PRIMARY_AGGREGATION,
        "primary_class_weights": dict(zip(CLASS_LABELS, EQUAL_CLASS_WEIGHTS.tolist())),
        "class_prior_sensitivity_aggregation": CLASS_PRIOR_AGGREGATION,
        "versions": {
            package: version(package) for package in ("shap", "xgboost", "lightgbm", "scikit-learn", "numpy")
        },
        "best_params": best_params, "comparisons": comparisons, "fold_details": fold_details,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_features.to_csv(output_dir / "fold_feature_importance.csv", index=False)
    rankings.to_csv(output_dir / "tooth_rankings.csv", index=False)
    top10.to_csv(output_dir / "top10.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote SHAP rankings to {output_dir}")


if __name__ == "__main__":
    main()
