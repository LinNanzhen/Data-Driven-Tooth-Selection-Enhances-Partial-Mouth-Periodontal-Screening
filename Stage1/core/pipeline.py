from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split

from core.config import GroupFilter, RunResult, ScenarioConfig, make_output_spec
from core.data_processing import DataProcessor
from core.evaluation import ModelPipeline
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


def _build_summary_rows(all_stability_results, pipeline, config: ScenarioConfig, *, parse_key):
    """Build rows for a stability summary table.

    Parameters
    ----------
    parse_key : callable
        Receives (stability_key, stability_value) and returns
        (subgroup_name_or_None, model_name).
    """
    rows = []

    for stability_key, stability in all_stability_results.items():
        if "tooth_level_importance" not in stability:
            continue

        subgroup_name, model_name = parse_key(stability_key, stability)
        if model_name is None:
            continue

        tooth_ranking_df = stability["tooth_level_importance"]
        consensus_teeth = tooth_ranking_df.head(config.top_features_count)["tooth_fdi"].tolist()

        if subgroup_name:
            cache_key = "{0}_{1}_Combined".format(subgroup_name, model_name)
        else:
            cache_key = "Overall_{0}_Combined".format(model_name)
        params = pipeline.best_params_cache.get(cache_key, {})

        summary_row = {
            "Model": model_name,
            "Consensus_Teeth_FDI": ", ".join(map(str, sorted(consensus_teeth))) if consensus_teeth else "None",
            "N_Consensus_Teeth": len(consensus_teeth),
        }
        if subgroup_name:
            summary_row["Subgroup"] = subgroup_name
        summary_row.update(params)
        rows.append(summary_row)

    return rows


def _build_overall_summary(all_stability_results, pipeline, config: ScenarioConfig):
    def _parse(stability_key, stability):
        return None, stability_key

    rows = _build_summary_rows(
        all_stability_results, pipeline, config, parse_key=_parse,
    )

    if not rows:
        return pd.DataFrame()

    summary_df = pd.DataFrame(rows).fillna("N/A")
    base_cols = ["Model", "Consensus_Teeth_FDI", "N_Consensus_Teeth"]
    param_cols = sorted([c for c in summary_df.columns if c not in base_cols])
    return summary_df[base_cols + param_cols]


def _build_subgroup_summary(all_stability_results, pipeline, config: ScenarioConfig):
    def _parse(stability_key, _stability):
        parts = stability_key.split("_")
        if len(parts) < 2:
            return None, None
        return "_".join(parts[:-1]), parts[-1]

    rows = _build_summary_rows(
        all_stability_results, pipeline, config, parse_key=_parse,
    )

    if not rows:
        return pd.DataFrame()

    summary_df = pd.DataFrame(rows).fillna("N/A")
    base_cols = ["Subgroup", "Model", "Consensus_Teeth_FDI", "N_Consensus_Teeth"]
    param_cols = sorted([c for c in summary_df.columns if c not in base_cols])
    return summary_df[base_cols + param_cols]


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

    x_train_full, x_test, y_train_full, _ = train_test_split(
        x_combined,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    sw_train_full = None
    if sample_weights is not None:
        sw_train_full, _ = train_test_split(
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

    cv_df = pd.DataFrame(all_results)
    summary_df = _build_overall_summary(all_stability_results, pipeline, config)

    result = RunResult(
        cv_results=cv_df,
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
            x_train, x_test, y_train, _ = train_test_split(
                x,
                y,
                test_size=config.test_size,
                random_state=config.random_state,
                stratify=y,
            )

            sw_train = None
            if sample_weights is not None:
                sw_train, _ = train_test_split(
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

        group_cv_df = pd.DataFrame(group_results)
        group_cv_df.to_csv(group_output_dir / "metrics_cv.csv", index=False)

        if config.keep_legacy_outputs:
            group_cv_df.to_csv(group_output_dir / "cross_validation_results.csv", index=False)

        all_results.extend(group_results)
        all_stability_results.update(
            {"{0}_{1}".format(group_name, model): value for model, value in group_stability_results.items()}
        )

    cv_df = pd.DataFrame(all_results)
    summary_df = _build_subgroup_summary(all_stability_results, pipeline, config)

    if all_test_full:
        pd.concat(all_test_full).to_csv(
            config.output_dir / "all_subgroups_test_set_full.csv", index=False
        )

    result = RunResult(
        cv_results=cv_df,
        summary=summary_df,
        stability_results=all_stability_results,
        extra={"skipped_groups": skipped_groups},
    )

    export_results(result, config.output_dir, make_output_spec("subgroup"), config.keep_legacy_outputs)
    return result
