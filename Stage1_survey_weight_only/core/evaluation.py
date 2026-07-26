from __future__ import annotations

import gc
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, auc, cohen_kappa_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize

from core.config import ScenarioConfig
from core.hyperopt import SKOPT_AVAILABLE, HyperparameterTuner
from core.model_factory import create_model


class ModelPipeline:
    def __init__(self, config: ScenarioConfig, random_seeds=None, cv_folds=None):
        self.config = config
        self.random_seeds = random_seeds or config.random_seeds
        self.cv_folds = cv_folds or config.cv_folds
        self.tuner = HyperparameterTuner(
            cv_folds=self.cv_folds,
            random_state=config.random_state,
            n_iter=config.bayesian_n_iter,
        )
        self.best_params_cache = {}

    def _calculate_weighted_metrics(self, y_true, y_pred, y_pred_proba, sample_weight=None):
        auc_macro = roc_auc_score(
            y_true,
            y_pred_proba,
            multi_class="ovr",
            average="macro",
            sample_weight=sample_weight,
        )
        accuracy = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
        f1_macro = f1_score(y_true, y_pred, average="macro", sample_weight=sample_weight)
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic", sample_weight=sample_weight)
        return {
            "auc_macro": auc_macro,
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "qwk": qwk,
        }

    def tune_hyperparameters_once(
        self,
        model_name,
        x,
        y,
        sample_weight,
        feature_set_name="Combined",
        subgroup_name="Overall",
    ):
        param_key = "{0}_{1}_{2}".format(subgroup_name, model_name, feature_set_name)
        if param_key in self.best_params_cache:
            return self.best_params_cache[param_key]

        if model_name in self.config.bayesian_hyperparameter_spaces and SKOPT_AVAILABLE:
            base_model = create_model(model_name)
            _, best_params = self.tuner.tune_model_bayes(
                model_name,
                base_model,
                x,
                y,
                sample_weight,
                self.config.bayesian_hyperparameter_spaces[model_name],
            )
            self.best_params_cache[param_key] = best_params
            del base_model
            gc.collect()
            return best_params

        self.best_params_cache[param_key] = {}
        return {}

    def evaluate_feature_sets_multiseed(
        self,
        x,
        y,
        sample_weights,
        model_name,
        best_params,
        stability_analyzer,
        current_output_dir,
    ):
        all_results = []

        stability_results = stability_analyzer.analyze_top_teeth_stability(
            x,
            y,
            sample_weights,
            model_name,
            best_params,
            current_output_dir,
        )

        feature_sets = {"Combined": x}

        if "tooth_level_importance" in stability_results:
            consensus_features = self._create_consensus_feature_sets(
                x, stability_results, model_name
            )
            if (
                "SHAP_Consensus" in consensus_features
                and not consensus_features["SHAP_Consensus"].empty
            ):
                feature_sets["SHAP_Consensus_{0}".format(model_name)] = consensus_features[
                    "SHAP_Consensus"
                ]

        roc_data = {}
        for fs_name, x_fs in feature_sets.items():
            if x_fs.empty:
                continue

            cache_name = "SHAP_Consensus" if fs_name.startswith("SHAP_Consensus_") else fs_name
            best_params_fs = self.tune_hyperparameters_once(
                model_name,
                x_fs,
                y,
                sample_weights,
                feature_set_name=cache_name,
                subgroup_name=current_output_dir.name,
            )

            metrics_across_seeds, roc_curves = self._evaluate_with_multiseed_cv(
                x_fs, y, sample_weights, model_name, best_params_fs
            )

            for metric, vals in metrics_across_seeds.items():
                all_results.append(
                    {
                        "Model": model_name,
                        "Feature_Set": fs_name,
                        "Metric": metric,
                        "Mean": np.mean(vals),
                        "Std": np.std(vals),
                        "N_Features": x_fs.shape[1],
                    }
                )

            roc_data[fs_name] = roc_curves

        if roc_data:
            self._plot_train_roc_curves(roc_data, model_name, current_output_dir)

        return all_results, stability_results

    def _plot_train_roc_curves(self, roc_data, model_name, output_dir):
        fig, axes = plt.subplots(1, 3, figsize=(36, 12))
        axes = axes.flatten()

        for idx, (_, class_label) in enumerate(self.config.class_labels.items()):
            ax = axes[idx]
            for fs_name, curves in roc_data.items():
                fpr_tpr_list = curves.get(class_label, [])
                if not fpr_tpr_list:
                    continue

                all_fpr = np.unique(np.concatenate([fpr for fpr, _ in fpr_tpr_list]))
                mean_tpr = np.zeros_like(all_fpr)
                for fpr, tpr in fpr_tpr_list:
                    mean_tpr += np.interp(all_fpr, fpr, tpr)
                mean_tpr /= len(fpr_tpr_list)
                mean_auc = auc(all_fpr, mean_tpr)
                ax.plot(all_fpr, mean_tpr, label="{0} (AUC={1:.2f})".format(fs_name, mean_auc))

            ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
            ax.set_title("{0} - {1}".format(class_label, model_name))
            ax.legend(loc="lower right", fontsize="small")

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(output_dir / "roc_group_{0}_train.png".format(model_name), dpi=300)
        plt.close(fig)

    def _create_consensus_feature_sets(self, x_full, stability_results, model_name):
        feature_sets = {}

        if "tooth_level_importance" in stability_results:
            tooth_df = stability_results["tooth_level_importance"]
            top_teeth_fdi = tooth_df.head(self.config.top_features_count)["tooth_fdi"].tolist()
            shap_features = []
            for tooth_fdi in top_teeth_fdi:
                nhanes_tooth = next(
                    (
                        nhanes
                        for nhanes, fdi in self.config.nhanes_to_fdi_mapping.items()
                        if fdi == tooth_fdi
                    ),
                    None,
                )
                if not nhanes_tooth:
                    continue
                for p_type in ["PC", "LA"]:
                    feature = "OHX{0}{1}_max".format(nhanes_tooth, p_type)
                    if feature in x_full.columns:
                        shap_features.append(feature)

            if shap_features:
                feature_sets["SHAP_Consensus"] = x_full[shap_features]

        feature_sets["CPI_Reference"] = self._get_clinical_index_features(x_full, "CPI")
        feature_sets["Ramfjord_Reference"] = self._get_clinical_index_features(
            x_full, "Ramfjord"
        )
        return feature_sets

    def _get_clinical_index_features(self, x_combined, method_name):
        fdi_numbers = self.config.cpi_ramfjord_tooth_numbers[method_name]
        selected_features = []
        for fdi_num in fdi_numbers:
            nhanes_num = next(
                (
                    nhanes
                    for nhanes, fdi in self.config.nhanes_to_fdi_mapping.items()
                    if fdi == fdi_num
                ),
                None,
            )
            if not nhanes_num:
                continue
            for p_type in ["PC", "LA"]:
                feature = "OHX{0}{1}_max".format(nhanes_num, p_type)
                if feature in x_combined.columns:
                    selected_features.append(feature)

        return x_combined[selected_features] if selected_features else pd.DataFrame()

    def _evaluate_with_multiseed_cv(self, x, y, sample_weights, model_name, best_params):
        all_metrics = defaultdict(list)
        roc_curves = {cls: [] for cls in self.config.class_labels.values()}

        classes = list(self.config.class_labels.keys())
        y_bin_full = label_binarize(y, classes=classes)

        for seed in self.random_seeds:
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=seed)
            for train_idx, val_idx in cv.split(x, y):
                x_train = x.iloc[train_idx]
                x_val = x.iloc[val_idx]
                y_train = y.iloc[train_idx]
                y_val = y.iloc[val_idx]
                w_train = sample_weights.iloc[train_idx] if sample_weights is not None else None
                w_val = sample_weights.iloc[val_idx] if sample_weights is not None else None

                model = create_model(model_name, random_state=seed)
                model.set_params(**best_params)

                fit_params = {}
                if w_train is not None:
                    fit_params["sample_weight"] = w_train
                model.fit(x_train, y_train, **fit_params)

                y_score = model.predict_proba(x_val)
                y_val_bin = y_bin_full[val_idx, :]

                for i, class_name in enumerate(self.config.class_labels.values()):
                    fpr_i, tpr_i, _ = roc_curve(y_val_bin[:, i], y_score[:, i], sample_weight=w_val)
                    roc_curves[class_name].append((fpr_i, tpr_i))

                metrics = self._calculate_weighted_metrics(
                    y_val, model.predict(x_val), y_score, w_val
                )
                for name, value in metrics.items():
                    all_metrics[name].append(value)

        return all_metrics, roc_curves
