from __future__ import annotations

import contextlib
import gc
import io
import re
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from core.config import ScenarioConfig
from core.model_factory import create_model

try:
    import shap

    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False
    shap = None

warnings.filterwarnings(
    "ignore",
    message=(
        r"In the future, passing feature_perturbation='interventional' "
        r"without providing a background dataset will raise an error\..*"
    ),
    category=FutureWarning,
    module=r"shap\.explainers\._tree",
)


class MultiSeedStabilityAnalyzer:
    def __init__(self, data_processor, config: ScenarioConfig, random_seeds=None):
        self.data_processor = data_processor
        self.config = config
        self.random_seeds = random_seeds or config.random_seeds
        self.stability_results = {}
        self.all_run_shap_values = defaultdict(list)

    def analyze_top_teeth_stability(
        self,
        x_train,
        y_train,
        sample_weights,
        model_name,
        best_params,
        current_output_dir,
    ):
        print("Analyzing feature stability for {0}".format(model_name))
        if not SHAP_AVAILABLE:
            print("SHAP is not available, skipping stability analysis")
            return {}

        self.all_run_shap_values.clear()

        for seed in self.random_seeds:
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=seed)
            for fold_idx, (train_idx, _) in enumerate(cv.split(x_train, y_train)):
                x_train_sub = x_train.iloc[train_idx]
                y_train_sub = y_train.iloc[train_idx]
                sw_train_sub = sample_weights.iloc[train_idx] if sample_weights is not None else None

                model = create_model(model_name, random_state=seed)
                model.set_params(**best_params)

                fit_params = {}
                if sw_train_sub is not None:
                    fit_params["sample_weight"] = sw_train_sub

                if hasattr(model, "set_params"):
                    try:
                        model.set_params(verbosity=0)
                    except Exception:
                        pass

                with contextlib.redirect_stdout(io.StringIO()):
                    model.fit(x_train_sub, y_train_sub, **fit_params)

                bg_weights = (
                    sw_train_sub.to_numpy()
                    if sw_train_sub is not None
                    else np.ones(len(x_train_sub))
                )
                x_sample, x_sample_weights = self._weighted_sample(
                    x_train_sub,
                    bg_weights,
                    self.config.shap_sample_size,
                    random_state=seed + fold_idx,
                )

                try:
                    explainer = shap.TreeExplainer(
                        model,
                        model_output="probability",
                        feature_perturbation="interventional",
                    )
                except Exception:
                    explainer = shap.TreeExplainer(model)

                shap_values = explainer.shap_values(x_sample)
                class_priors = self._compute_weighted_class_priors(y_train_sub, bg_weights)
                mean_shap = self._compute_weighted_global_shap(shap_values, class_priors, x_sample_weights)

                for feature_name, shap_val in zip(x_train.columns, mean_shap):
                    self.all_run_shap_values[feature_name].append(shap_val)

                del model, explainer, shap_values
                gc.collect()

        stability_analysis = self._calculate_and_visualize_stability_results(
            model_name, current_output_dir
        )
        self.stability_results[model_name] = stability_analysis
        return stability_analysis

    @staticmethod
    def _weighted_sample(x, weights, n, random_state):
        n = min(len(x), n)
        if n <= 0:
            return x.iloc[:0], np.array([])
        weights = np.asarray(weights)
        weights_norm = weights / weights.sum()
        rng = np.random.RandomState(random_state)
        indices = rng.choice(len(x), size=n, replace=False, p=weights_norm)
        return x.iloc[indices], weights[indices]

    @staticmethod
    def _compute_weighted_class_priors(y, weights):
        y = np.asarray(y)
        weights = np.asarray(weights) if weights is not None else np.ones(len(y))
        classes = np.unique(y)
        priors = {}
        total_weight = weights.sum()
        for c in classes:
            priors[c] = weights[y == c].sum() / total_weight if total_weight > 0 else 1.0 / len(classes)
        return priors

    @staticmethod
    def _compute_weighted_global_shap(shap_values, class_priors, bg_weights):
        if isinstance(shap_values, list):
            n_classes = len(shap_values)
            n_features = shap_values[0].shape[1]
            global_imp = np.zeros(n_features)
            for c_idx in range(n_classes):
                sv = shap_values[c_idx]
                weighted_mean_abs = np.average(np.abs(sv), axis=0, weights=bg_weights)
                prior = class_priors.get(c_idx, 1.0 / n_classes)
                global_imp += prior * weighted_mean_abs
            return global_imp
        else:
            return np.average(np.abs(shap_values), axis=0, weights=bg_weights)

    def _calculate_and_visualize_stability_results(self, model_name, current_output_dir):
        feature_stability_data = [
            {
                "feature": feature,
                "mean_shap": np.mean(values),
                "std_shap": np.std(values),
            }
            for feature, values in self.all_run_shap_values.items()
        ]

        if not feature_stability_data:
            return {}

        feature_df = pd.DataFrame(feature_stability_data).sort_values("mean_shap", ascending=False)

        detailed_importance_data = []
        for _, row in feature_df.iterrows():
            match = re.search(r"OHX(\d{2})(PC|LA)_max", row["feature"])
            if not match:
                continue

            nhanes_num = match.group(1)
            measurement_type = "PD" if match.group(2) == "PC" else "CAL"
            fdi_num = self.data_processor.nhanes_to_fdi(nhanes_num)
            detailed_importance_data.append(
                {
                    "tooth_fdi": fdi_num,
                    "measurement_type": measurement_type,
                    "feature": row["feature"],
                    "mean_shap": row["mean_shap"],
                    "shap_error": row["std_shap"],
                }
            )

        detailed_importance_df = pd.DataFrame(detailed_importance_data)
        if detailed_importance_df.empty:
            return {"feature_level_stability": feature_df}

        tooth_df_aggregated = (
            detailed_importance_df.groupby("tooth_fdi")
            .agg(total_mean_shap=("mean_shap", "sum"), shap_error=("shap_error", "mean"))
            .sort_values("total_mean_shap", ascending=False)
            .reset_index()
        )

        detailed_importance_df.to_csv(
            current_output_dir / "{0}_detailed_feature_importance.csv".format(model_name),
            index=False,
        )
        tooth_df_aggregated.to_csv(
            current_output_dir / "{0}_aggregated_tooth_importance.csv".format(model_name),
            index=False,
        )

        return {
            "tooth_level_importance": tooth_df_aggregated,
            "feature_level_stability": feature_df,
            "detailed_importance": detailed_importance_df,
        }
