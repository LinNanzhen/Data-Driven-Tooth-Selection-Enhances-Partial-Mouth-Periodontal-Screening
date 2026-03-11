from __future__ import annotations

import contextlib
import gc
import io
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from core.config import ScenarioConfig
from core.model_factory import create_model

try:
    import shap

    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False
    shap = None


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
        x_sample = x_train.sample(min(len(x_train), self.config.shap_sample_size), random_state=42)

        for seed in self.random_seeds:
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=seed)
            for train_idx, _ in cv.split(x_train, y_train):
                x_train_sub = x_train.iloc[train_idx]
                y_train_sub = y_train.iloc[train_idx]
                sw_train_sub = sample_weights.iloc[train_idx] if sample_weights is not None else None

                model = create_model(model_name, random_state=seed)
                model.set_params(**best_params)

                class_sample_weights = compute_sample_weight("balanced", y=y_train_sub)
                fit_params = {
                    "sample_weight": sw_train_sub * class_sample_weights
                    if sw_train_sub is not None
                    else class_sample_weights
                }

                if hasattr(model, "set_params"):
                    try:
                        model.set_params(verbosity=0)
                    except Exception:
                        pass

                with contextlib.redirect_stdout(io.StringIO()):
                    model.fit(x_train_sub, y_train_sub, **fit_params)

                try:
                    explainer = shap.TreeExplainer(
                        model,
                        model_output="probability",
                        feature_perturbation="interventional",
                    )
                except Exception:
                    explainer = shap.TreeExplainer(model)

                shap_values = explainer.shap_values(x_sample)
                if isinstance(shap_values, list):
                    mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
                else:
                    mean_shap = np.abs(shap_values).mean(axis=0)

                for feature_name, shap_val in zip(x_train.columns, mean_shap):
                    self.all_run_shap_values[feature_name].append(shap_val)

                del model, explainer, shap_values
                gc.collect()

        stability_analysis = self._calculate_and_visualize_stability_results(
            model_name, current_output_dir
        )
        self.stability_results[model_name] = stability_analysis
        return stability_analysis

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

        self.visualize_detailed_importance(
            detailed_importance_df, tooth_df_aggregated, model_name, current_output_dir
        )

        return {
            "tooth_level_importance": tooth_df_aggregated,
            "feature_level_stability": feature_df,
            "detailed_importance": detailed_importance_df,
        }

    def visualize_detailed_importance(
        self, detailed_df, aggregated_df, model_name, current_output_dir
    ):
        plt.style.use("seaborn-v0_8-whitegrid")

        top_teeth = aggregated_df.head(10)["tooth_fdi"].tolist()
        plot_data = detailed_df[detailed_df["tooth_fdi"].isin(top_teeth)]
        if plot_data.empty:
            return

        pivot_df = (
            plot_data.pivot_table(index="tooth_fdi", columns="measurement_type", values="mean_shap")
            .fillna(0)
            .reindex(top_teeth)
        )

        if "PD" not in pivot_df.columns:
            pivot_df["PD"] = 0.0
        if "CAL" not in pivot_df.columns:
            pivot_df["CAL"] = 0.0

        err_df = (
            plot_data.pivot_table(
                index="tooth_fdi",
                columns="measurement_type",
                values="shap_error",
                aggfunc="mean",
            )
            .fillna(0)
            .reindex(top_teeth)
        )
        if "PD" not in err_df.columns:
            err_df["PD"] = 0.0
        if "CAL" not in err_df.columns:
            err_df["CAL"] = 0.0

        fig, ax = plt.subplots(figsize=(18, 9))
        bar_width = 0.35
        index = np.arange(len(pivot_df.index))

        pd_means = pivot_df["PD"]
        pd_std = err_df["PD"]
        cal_means = pivot_df["CAL"]
        cal_std = err_df["CAL"]

        ax.bar(
            index - bar_width / 2,
            pd_means,
            bar_width,
            yerr=[np.minimum(pd_std, pd_means).to_numpy(), pd_std.to_numpy()],
            capsize=4,
            label="PD",
            color="royalblue",
        )
        ax.bar(
            index + bar_width / 2,
            cal_means,
            bar_width,
            yerr=[np.minimum(cal_std, cal_means).to_numpy(), cal_std.to_numpy()],
            capsize=4,
            label="CAL",
            color="skyblue",
        )

        ax.set_ylabel("Robust Mean(|SHAP|)")
        ax.set_xlabel("Tooth (FDI)")
        ax.set_title("PD vs CAL importance for top teeth - {0}".format(model_name))
        ax.set_xticks(index)
        ax.set_xticklabels(pivot_df.index, rotation=45, ha="right")
        ax.legend()
        fig.tight_layout()
        plt.savefig(
            current_output_dir / "{0}_paired_importance_barchart.png".format(model_name),
            dpi=300,
        )
        plt.close(fig)

        fdi_layout = {
            "18": (0, 0),
            "17": (0, 1),
            "16": (0, 2),
            "15": (0, 3),
            "14": (0, 4),
            "13": (0, 5),
            "12": (0, 6),
            "11": (0, 7),
            "21": (0, 8),
            "22": (0, 9),
            "23": (0, 10),
            "24": (0, 11),
            "25": (0, 12),
            "26": (0, 13),
            "27": (0, 14),
            "28": (0, 15),
            "48": (1, 0),
            "47": (1, 1),
            "46": (1, 2),
            "45": (1, 3),
            "44": (1, 4),
            "43": (1, 5),
            "42": (1, 6),
            "41": (1, 7),
            "31": (1, 8),
            "32": (1, 9),
            "33": (1, 10),
            "34": (1, 11),
            "35": (1, 12),
            "36": (1, 13),
            "37": (1, 14),
            "38": (1, 15),
        }

        heatmap_data = np.full((2, 16), np.nan)
        importance_dict = aggregated_df.set_index("tooth_fdi")["total_mean_shap"].to_dict()
        for fdi, pos in fdi_layout.items():
            heatmap_data[pos] = importance_dict.get(fdi, np.nan)

        fig, ax = plt.subplots(figsize=(16, 4))
        im = ax.imshow(heatmap_data, cmap="Reds", interpolation="nearest", aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
        ax.set_title("Aggregated tooth importance heatmap - {0}".format(model_name))
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(
            current_output_dir / "{0}_aggregated_importance_heatmap.png".format(model_name),
            dpi=300,
        )
        plt.close(fig)
