from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from core.classification import CDCClassifier
from core.data_processing import DataProcessor
from core.evaluation import StrategyEvaluator, make_overall_strategies, make_subgroup_strategies
from core.export import write_overall_outputs, write_subgroup_outputs
from core.types import AnalysisResult, ConfigSpec, ScenarioName


def run_analysis(
    config: ConfigSpec,
    n_bootstrap: Optional[int] = None,
    seed: Optional[int] = None,
) -> AnalysisResult:
    scenario = config.scenario
    if n_bootstrap is not None:
        config.n_bootstrap = int(n_bootstrap)
    if seed is not None:
        config.random_seed = int(seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_processor = DataProcessor(config)
    df = data_processor.load_and_preprocess(config.data_path)
    if df.empty:
        raise RuntimeError("No data available after preprocessing")
    df["reference_true"] = df[config.target_column]

    classifier = CDCClassifier(data_processor)
    evaluator = StrategyEvaluator(config, data_processor, classifier)

    if scenario == "overall":
        strategy_specs = make_overall_strategies(config)
        group_results, comparisons, _ = evaluator.evaluate_group(df, strategy_specs)
        summary_df, artifacts = write_overall_outputs(output_dir, group_results, comparisons)
        return AnalysisResult(
            scenario=scenario,
            output_dir=str(output_dir),
            results=group_results,
            comparisons={"overall": comparisons},
            artifacts=artifacts,
            summary_records=summary_df.to_dict(orient="records"),
        )

    all_results: Dict[str, Dict[str, dict]] = {}
    all_comparisons: Dict[str, List[dict]] = {}

    if scenario == "s1":
        subgroup_candidates = sorted([x for x in config.subgroup_teeth.keys() if x.startswith("Age_")]) + sorted(
            [x for x in config.subgroup_teeth.keys() if x.startswith("Gender_")]
        )
        for subgroup_name in subgroup_candidates:
            if subgroup_name.startswith("Age_"):
                subgroup_df = df[df["age_subgroup"] == subgroup_name].copy()
            else:
                subgroup_df = df[df["gender_subgroup"] == subgroup_name].copy()
            if len(subgroup_df) < config.min_subgroup_size:
                continue
            specs = make_subgroup_strategies(config, subgroup_name)
            results, comparisons, _ = evaluator.evaluate_group(subgroup_df, specs)
            if results:
                all_results[subgroup_name] = results
                all_comparisons[subgroup_name] = comparisons

        overall_specs = {
            name: spec
            for name, spec in make_subgroup_strategies(config, "Overall").items()
            if spec.strategy_type == "overall_on_subgroup"
        }
        overall_results, overall_comp, _ = evaluator.evaluate_group(df, overall_specs)
        if overall_results:
            all_results["Overall"] = overall_results
            all_comparisons["Overall"] = overall_comp

    elif scenario == "s2":
        available = set(df["age_gender_subgroup"].dropna().unique().tolist())
        defined = set(config.subgroup_teeth.keys())
        subgroup_candidates = sorted(available & defined)
        for subgroup_name in subgroup_candidates:
            subgroup_df = df[df["age_gender_subgroup"] == subgroup_name].copy()
            if len(subgroup_df) < config.min_subgroup_size:
                continue
            specs = make_subgroup_strategies(config, subgroup_name)
            results, comparisons, _ = evaluator.evaluate_group(subgroup_df, specs)
            if results:
                all_results[subgroup_name] = results
                all_comparisons[subgroup_name] = comparisons
    else:
        raise ValueError("Unsupported scenario: {0}".format(scenario))

    summary_df, artifacts = write_subgroup_outputs(output_dir, all_results, all_comparisons)

    return AnalysisResult(
        scenario=scenario,
        output_dir=str(output_dir),
        results=all_results,
        comparisons=all_comparisons,
        artifacts=artifacts,
        summary_records=summary_df.to_dict(orient="records"),
    )
