"""Tune and fit the Stage 1 models with and without survey-weighted class balancing."""

import argparse
import json

import numpy as np
import pandas as pd
from joblib import dump
from lightgbm import LGBMClassifier
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from xgboost import XGBClassifier

from stage1_data import SPLIT_SEED, STAGE1_OUTPUT_DIR, TARGET_COLUMN, prepare_stage1_data

CV_FOLDS = 5
SEARCH_ITERATIONS = 30
SCORING = "roc_auc_ovr"


def unweighted_cv_score(estimator, X, y, sample_weight=None):
    # Some scikit-learn versions also route fit weights to the scorer.
    return get_scorer(SCORING)(estimator, X, y)


def balance_survey_weights(labels, survey_weights):
    """Equalize class weight totals while preserving the overall weight total."""
    survey_weights = np.asarray(survey_weights, dtype=float)
    classes, class_index = np.unique(labels, return_inverse=True)
    class_totals = np.bincount(class_index, weights=survey_weights)
    class_factors = (class_totals.sum() / len(classes)) / class_totals
    return survey_weights * class_factors[class_index]


class SurveyClassBalanceMixin:
    def fit(self, X, y, *, sample_weight=None, **kwargs):
        # BayesSearchCV slices these weights before calling fit on a training fold.
        weights = balance_survey_weights(y, sample_weight)
        return super().fit(X, y, sample_weight=weights, **kwargs)


class BalancedXGBClassifier(SurveyClassBalanceMixin, XGBClassifier):
    pass


class BalancedLGBMClassifier(SurveyClassBalanceMixin, LGBMClassifier):
    pass


def load_training_data():
    cohort, features = prepare_stage1_data()
    assignments = pd.read_csv(STAGE1_OUTPUT_DIR / "split_assignments.csv").set_index("SEQN")
    labels_by_id = cohort.set_index("SEQN")[TARGET_COLUMN]
    if (
        not assignments.index.is_unique
        or set(assignments.index) != set(labels_by_id.index)
        or set(assignments["split"]) != {"train", "internal_test"}
        or not assignments[TARGET_COLUMN].eq(labels_by_id.reindex(assignments.index)).all()
    ):
        raise ValueError("Split manifest does not match the Stage 1 cohort. Rerun stage1_data.py.")

    # Use the saved training order for reproducible CV fold membership.
    features.index = cohort["SEQN"]
    train_ids = assignments.index[assignments["split"].eq("train")]
    train = cohort.set_index("SEQN").loc[train_ids]
    return features.loc[train_ids], train[TARGET_COLUMN], train["WTMEC2YR"]


def overall_models_and_spaces():
    # Search ranges from the overall scenario in the pinned upstream repository.
    return {
        "xgboost": (
            XGBClassifier(
                objective="multi:softprob", eval_metric="mlogloss",
                random_state=SPLIT_SEED, n_jobs=1, verbosity=0,
            ),
            {
                "n_estimators": Integer(1000, 1300),
                "max_depth": Integer(2, 4),
                "learning_rate": Real(0.05, 0.15, prior="log-uniform"),
                "subsample": Real(0.7, 0.9),
                "colsample_bytree": Real(0.85, 0.95),
                "gamma": Real(0.2, 0.8),
                "reg_alpha": Real(1e-3, 1e-2, prior="log-uniform"),
                "reg_lambda": Real(1e-3, 1e-2, prior="log-uniform"),
                "min_child_weight": Integer(8, 12),
            },
        ),
        "lightgbm": (
            LGBMClassifier(
                objective="multiclass", random_state=SPLIT_SEED, n_jobs=1, verbosity=-1,
            ),
            {
                "n_estimators": Integer(800, 1200),
                "max_depth": Integer(1, 3),
                "num_leaves": Integer(80, 150),
                "learning_rate": Real(0.08, 0.20, prior="log-uniform"),
                "subsample": Real(0.7, 0.9),
                "colsample_bytree": Real(0.6, 0.8),
                "reg_alpha": Real(1e-5, 1e-3, prior="log-uniform"),
                "reg_lambda": Real(5.0, 15.0, prior="log-uniform"),
                "min_child_samples": Integer(10, 25),
            },
        ),
    }


def train_weighting_line(features, labels, survey_weights, output_dir, *, folds, iterations, jobs, smoke, balanced):
    output_dir.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SPLIT_SEED)
    fit_weights = balance_survey_weights(labels, survey_weights) if balanced else survey_weights
    balanced_models = {"xgboost": BalancedXGBClassifier, "lightgbm": BalancedLGBMClassifier}
    summary = {}

    for name, (model, space) in overall_models_and_spaces().items():
        if smoke:
            space["n_estimators"] = Integer(12, 24)
        search_model = balanced_models[name](**model.get_params()) if balanced else model
        print(f"{output_dir.name}/{name}: {len(labels):,} training rows, {iterations} candidates, {folds} folds", flush=True)
        search = BayesSearchCV(
            search_model, space, n_iter=iterations, cv=cv, scoring=unweighted_cv_score, refit=False,
            n_jobs=jobs, random_state=SPLIT_SEED, error_score="raise",
        )
        search.fit(features, labels, sample_weight=survey_weights)
        model.set_params(**search.best_params_).fit(features, labels, sample_weight=fit_weights)
        pd.DataFrame(search.cv_results_).to_csv(output_dir / f"{name}_cv.csv", index=False)
        dump(model, output_dir / f"{name}.joblib")
        summary[name] = {"best_params": search.best_params_, "mean_cv_auc": search.best_score_}
        print(f"{output_dir.name}/{name}: mean CV macro OvR AUC = {search.best_score_:.6f}", flush=True)

    report = {
        "n_train": len(labels), "cv_folds": folds, "search_iterations": iterations,
        "seed": SPLIT_SEED, "scoring": SCORING, "scoring_weighted": False,
        "training_weighting": output_dir.name,
        "survey_weight_totals_by_class": survey_weights.groupby(labels).sum().to_dict(),
        "final_fit_weight_totals_by_class": pd.Series(fit_weights, index=labels.index).groupby(labels).sum().to_dict(),
        "models": summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote trained models and tuning results to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run 3 quick candidates on 2 folds.")
    parser.add_argument("--jobs", type=int, default=4, help="Number of parallel CV fits.")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")

    features, labels, survey_weights = load_training_data()
    folds = 2 if args.smoke else CV_FOLDS
    iterations = 3 if args.smoke else SEARCH_ITERATIONS
    output_dir = STAGE1_OUTPUT_DIR / ("training_smoke" if args.smoke else "training")
    for name, balanced in (("nhanes_only", False), ("nhanes_balanced", True)):
        train_weighting_line(
            features, labels, survey_weights, output_dir / name,
            folds=folds, iterations=iterations, jobs=args.jobs, smoke=args.smoke, balanced=balanced,
        )


if __name__ == "__main__":
    main()
