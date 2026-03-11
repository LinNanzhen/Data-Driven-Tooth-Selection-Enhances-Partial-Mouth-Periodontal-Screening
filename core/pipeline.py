from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split

from core.config import GroupFilter, RunResult, ScenarioConfig, make_output_spec
from core.data_processing import DataProcessor
from core.evaluation import ModelPipeline, create_test_performance_summary_plot
from core.model_factory import available_models
from core.reporting import export_results
from core.stability import MultiSeedStabilityAnalyzer


def _save_test_set(df, x_test, config: ScenarioConfig, out_path: Path):
    test_idx = x_test.index
    test_full = df.loc[test_idx].copy()
    test_full["label_name"] = test_full[config.target_column].map(config.class_labels)

    front = [config.target_column, "label_name", config.weight_column, config.age_column]
    if config.gender_column:
        front.append(config.gender_column)

    front_exist = [c for c in front if c in test_full.columns]
    test_full = test_full[front_exist + [c for c in test_full.columns if c not in front_exist]]
    test_full.to_csv(out_path, index=False)
    return test_full


def _make_comparison(cv_df: pd.DataFrame, test_df: pd.DataFrame, aggregate_mean: bool) -> pd.DataFrame:
    if cv_df is None or cv_df.empty or test_df is None or test_df.empty:
        return pd.DataFrame()

    if aggregate_mean:
        cv_summary = (
            cv_df.groupby(["Model", "Feature_Set", "Metric"])["Mean"]
            .mean()
            .reset_index()
            .rename(columns={"Mean": "CV_Score_Avg"})
        )
        test_summary = (
            test_df.groupby(["Model", "Feature_Set", "Metric"])["Test_Score"]
            .mean()
            .reset_index()
            .rename(columns={"Test_Score": "Test_Score_Avg"})
        )
        comparison_df = pd.merge(
            cv_summary,
            test_summary,
            on=["Model", "Feature_Set", "Metric"],
            how="outer",
        )
        comparison_df["Performance_Gap"] = (
            comparison_df["Test_Score_Avg"] - comparison_df["CV_Score_Avg"]
        )
        comparison_df["Potential_Overfitting"] = comparison_df["Performance_Gap"] < -0.02
        return comparison_df

    cv_summary = (
        cv_df.groupby(["Model", "Feature_Set", "Metric"])["Mean"]
        .first()
        .reset_index()
        .rename(columns={"Mean": "CV_Score"})
    )
    comparison_df = pd.merge(
        cv_summary,
        test_df,
        on=["Model", "Feature_Set", "Metric"],
        how="outer",
    )
    comparison_df["Difference"] = comparison_df["Test_Score"] - comparison_df["CV_Score"]
    comparison_df["Overfitting"] = comparison_df["Difference"] < -0.01
    return comparison_df


def _build_overall_summary(all_stability_results, pipeline, final_test_results, config: ScenarioConfig):
    rows = []
    test_df = pd.DataFrame(final_test_results) if final_test_results else pd.DataFrame()

    for model_name, stability in all_stability_results.items():
        if "tooth_level_importance" not in stability:
            continue

        tooth_ranking_df = stability["tooth_level_importance"]
        consensus_teeth = tooth_ranking_df.head(config.top_features_count)["tooth_fdi"].tolist()

        key = "Overall_{0}_Combined".format(model_name)
        params = pipeline.best_params_cache.get(key, {})

        test_perf = {}
        if not test_df.empty:
            combined_test = test_df[
                (test_df["Model"] == model_name) & (test_df["Feature_Set"] == "Combined")
            ]
            for _, row in combined_test.iterrows():
                test_perf["Test_{0}".format(row["Metric"])] = "{0:.3f}".format(row["Test_Score"])

        shap_test_perf = {}
        if not test_df.empty:
            shap_test = test_df[
                (test_df["Model"] == model_name)
                & (test_df["Feature_Set"] == "SHAP_Consensus_{0}".format(model_name))
            ]
            for _, row in shap_test.iterrows():
                shap_test_perf["SHAP_Test_{0}".format(row["Metric"])] = "{0:.3f}".format(
                    row["Test_Score"]
                )

        summary_row = {
            "Model": model_name,
            "Consensus_Teeth_FDI": ", ".join(map(str, sorted(consensus_teeth))) if consensus_teeth else "None",
            "N_Consensus_Teeth": len(consensus_teeth),
        }
        summary_row.update(params)
        summary_row.update(test_perf)
        summary_row.update(shap_test_perf)
        rows.append(summary_row)

    if not rows:
        return pd.DataFrame()

    summary_df = pd.DataFrame(rows).fillna("N/A")
    base_cols = ["Model", "Consensus_Teeth_FDI", "N_Consensus_Teeth"]
    test_cols = sorted(
        [c for c in summary_df.columns if c.startswith("Test_") or c.startswith("SHAP_Test_")]
    )
    param_cols = sorted([c for c in summary_df.columns if c not in base_cols + test_cols])
    return summary_df[base_cols + test_cols + param_cols]


def _build_subgroup_summary(all_stability_results, pipeline, final_test_results, config: ScenarioConfig):
    rows = []
    test_df = pd.DataFrame(final_test_results) if final_test_results else pd.DataFrame()

    for stability_key, stability in all_stability_results.items():
        if "tooth_level_importance" not in stability:
            continue

        parts = stability_key.split("_")
        if len(parts) < 2:
            continue
        model_name = parts[-1]
        subgroup_name = "_".join(parts[:-1])

        tooth_ranking_df = stability["tooth_level_importance"]
        consensus_teeth = tooth_ranking_df.head(config.top_features_count)["tooth_fdi"].tolist()

        key = "{0}_{1}_Combined".format(subgroup_name, model_name)
        params = pipeline.best_params_cache.get(key, {})

        test_perf = {}
        if not test_df.empty:
            combined_test = test_df[
                (test_df["Model"] == model_name)
                & (test_df["Feature_Set"] == "Combined")
                & (test_df["Subgroup"] == subgroup_name)
            ]
            for _, row in combined_test.iterrows():
                test_perf["Test_{0}".format(row["Metric"])] = "{0:.3f}".format(row["Test_Score"])

        shap_test_perf = {}
        if not test_df.empty:
            shap_test = test_df[
                (test_df["Model"] == model_name)
                & (test_df["Feature_Set"] == "SHAP_Consensus_{0}".format(model_name))
                & (test_df["Subgroup"] == subgroup_name)
            ]
            for _, row in shap_test.iterrows():
                shap_test_perf["SHAP_Test_{0}".format(row["Metric"])] = "{0:.3f}".format(
                    row["Test_Score"]
                )

        summary_row = {
            "Subgroup": subgroup_name,
            "Model": model_name,
            "Consensus_Teeth_FDI": ", ".join(map(str, sorted(consensus_teeth))) if consensus_teeth else "None",
            "N_Consensus_Teeth": len(consensus_teeth),
        }
        summary_row.update(params)
        summary_row.update(test_perf)
        summary_row.update(shap_test_perf)
        rows.append(summary_row)

    if not rows:
        return pd.DataFrame()

    summary_df = pd.DataFrame(rows).fillna("N/A")
    base_cols = ["Subgroup", "Model", "Consensus_Teeth_FDI", "N_Consensus_Teeth"]
    test_cols = sorted(
        [c for c in summary_df.columns if c.startswith("Test_") or c.startswith("SHAP_Test_")]
    )
    param_cols = sorted([c for c in summary_df.columns if c not in base_cols + test_cols])
    return summary_df[base_cols + test_cols + param_cols]


def run_overall(config: ScenarioConfig) -> RunResult:
    config.prepare_output_dir()

    model_names = available_models(config.models)
    if not model_names:
        raise RuntimeError("Neither XGBoost nor LightGBM is available")

    data_processor = DataProcessor(config)
    df = data_processor.load_and_preprocess(config.data_path)
    if df.empty:
        raise RuntimeError("No data available after preprocessing")

    x_combined, y, sample_weights = data_processor.get_combined_feature_set(df)

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x_combined,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    sw_train_full, sw_test = (None, None)
    if sample_weights is not None:
        sw_train_full, sw_test = train_test_split(
            sample_weights,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=y,
        )

    _save_test_set(df, x_test, config, config.output_dir / "test_set_full.csv")

    pipeline = ModelPipeline(config)
    stability_analyzer = MultiSeedStabilityAnalyzer(data_processor, config)

    all_results = []
    all_stability_results = {}
    final_test_results = []

    for model_name in model_names:
        best_params = pipeline.tune_hyperparameters_once(
            model_name,
            x_train_full,
            y_train_full,
            sw_train_full,
            subgroup_name="Overall",
        )
        model_results, stability_results = pipeline.evaluate_feature_sets_multiseed(
            x_train_full,
            y_train_full,
            sw_train_full,
            model_name,
            best_params,
            stability_analyzer,
            config.output_dir,
        )
        all_results.extend(model_results)
        all_stability_results[model_name] = stability_results

        test_results = pipeline.evaluate_test_set(
            x_train_full,
            y_train_full,
            x_test,
            y_test,
            sw_train_full,
            sw_test,
            model_name,
            best_params,
            stability_results,
            config.output_dir,
        )
        final_test_results.extend(test_results)

    cv_df = pd.DataFrame(all_results)
    test_df = pd.DataFrame(final_test_results)
    comparison_df = _make_comparison(cv_df, test_df, aggregate_mean=False)
    summary_df = _build_overall_summary(all_stability_results, pipeline, final_test_results, config)

    result = RunResult(
        cv_results=cv_df,
        test_results=test_df,
        comparison=comparison_df,
        summary=summary_df,
        stability_results=all_stability_results,
    )

    export_results(result, config.output_dir, make_output_spec("overall"), config.keep_legacy_outputs)
    return result


def _normalize_groups(groups: Dict[str, GroupFilter | dict]) -> Dict[str, GroupFilter]:
    normalized = {}
    for name, value in groups.items():
        if isinstance(value, GroupFilter):
            normalized[name] = value
        else:
            normalized[name] = GroupFilter(
                age_min=value["age_min"],
                age_max=value["age_max"],
                gender=value.get("gender"),
            )
    return normalized


def run_subgroup(config: ScenarioConfig, groups: Dict[str, GroupFilter | dict]) -> RunResult:
    config.prepare_output_dir()

    model_names = available_models(config.models)
    if not model_names:
        raise RuntimeError("Neither XGBoost nor LightGBM is available")

    data_processor = DataProcessor(config)
    df = data_processor.load_and_preprocess(config.data_path)
    if df.empty:
        raise RuntimeError("No data available after preprocessing")

    groups = _normalize_groups(groups)

    pipeline = ModelPipeline(config)
    all_results = []
    final_test_results = []
    all_stability_results = {}
    all_test_full = []
    skipped_groups = []

    for group_name, group_filter in groups.items():
        age_filter = (df[config.age_column] >= group_filter.age_min) & (
            df[config.age_column] <= group_filter.age_max
        )
        group_df = df[age_filter]

        if group_filter.gender is not None and config.gender_column:
            group_df = group_df[group_df[config.gender_column] == group_filter.gender]

        group_df = group_df.copy()
        n_samples = len(group_df)
        if n_samples < config.min_samples_per_group:
            skipped_groups.append((group_name, n_samples))
            continue

        group_output_dir = config.output_dir / group_name
        group_output_dir.mkdir(parents=True, exist_ok=True)

        x, y, sample_weights = data_processor.get_combined_feature_set(group_df)

        try:
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=config.test_size,
                random_state=config.random_state,
                stratify=y,
            )

            sw_train, sw_test = (None, None)
            if sample_weights is not None:
                sw_train, sw_test = train_test_split(
                    sample_weights,
                    test_size=config.test_size,
                    random_state=config.random_state,
                    stratify=y,
                )
        except ValueError:
            skipped_groups.append((group_name, n_samples))
            continue

        test_full = _save_test_set(group_df, x_test, config, group_output_dir / "test_set_full.csv")
        test_full["Subgroup"] = group_name
        all_test_full.append(test_full)

        stability_analyzer = MultiSeedStabilityAnalyzer(data_processor, config)

        group_results = []
        group_test_results = []
        group_stability_results = {}

        for model_name in model_names:
            best_params = pipeline.tune_hyperparameters_once(
                model_name,
                x_train,
                y_train,
                sw_train,
                subgroup_name=group_name,
            )

            model_results, stability_results = pipeline.evaluate_feature_sets_multiseed(
                x_train,
                y_train,
                sw_train,
                model_name,
                best_params,
                stability_analyzer,
                group_output_dir,
            )
            group_results.extend(model_results)
            group_stability_results[model_name] = stability_results

            test_results = pipeline.evaluate_test_set(
                x_train,
                y_train,
                x_test,
                y_test,
                sw_train,
                sw_test,
                model_name,
                best_params,
                stability_results,
                group_output_dir,
            )
            for row in test_results:
                row["Subgroup"] = group_name
            group_test_results.extend(test_results)

        group_cv_df = pd.DataFrame(group_results)
        group_test_df = pd.DataFrame(group_test_results)
        group_cv_df.to_csv(group_output_dir / "metrics_cv.csv", index=False)
        group_test_df.to_csv(group_output_dir / "metrics_test.csv", index=False)

        if config.keep_legacy_outputs:
            group_cv_df.to_csv(group_output_dir / "cross_validation_results.csv", index=False)
            group_test_df.to_csv(group_output_dir / "test_results.csv", index=False)

        all_results.extend(group_results)
        final_test_results.extend(group_test_results)
        all_stability_results.update(
            {"{0}_{1}".format(group_name, model): value for model, value in group_stability_results.items()}
        )

    cv_df = pd.DataFrame(all_results)
    test_df = pd.DataFrame(final_test_results)
    comparison_df = _make_comparison(cv_df, test_df, aggregate_mean=True)
    summary_df = _build_subgroup_summary(
        all_stability_results, pipeline, final_test_results, config
    )

    if not test_df.empty:
        create_test_performance_summary_plot(test_df, config.output_dir)

    if all_test_full:
        pd.concat(all_test_full).to_csv(
            config.output_dir / "all_subgroups_test_set_full.csv", index=False
        )

    result = RunResult(
        cv_results=cv_df,
        test_results=test_df,
        comparison=comparison_df,
        summary=summary_df,
        stability_results=all_stability_results,
        extra={"skipped_groups": skipped_groups},
    )

    export_results(result, config.output_dir, make_output_spec("subgroup"), config.keep_legacy_outputs)
    return result
