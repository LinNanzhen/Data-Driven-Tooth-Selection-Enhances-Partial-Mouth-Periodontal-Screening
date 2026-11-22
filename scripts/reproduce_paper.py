"""Run the published Stage 1 and Stage 2 analyses."""

import argparse
from importlib import import_module
from pathlib import Path
import sys

import pandas as pd

from prepare_data import OUT_DIR

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ("overall", "subgroup_single_factor", "subgroup_age_gender")
MODEL_KEYS = {"XGBoost": "XGB10", "LightGBM": "LGB10"}


def run_stage1(module, name, data_dir, output, smoke):
    config = module.build_config(data_dir)
    config.output_dir = output / "stage1" / name
    config.keep_legacy_outputs = False
    if smoke:
        from skopt.space import Integer

        config.cv_folds = 2
        config.bayesian_n_iter = 1
        config.random_seeds = [config.random_state]
        for space in config.bayesian_hyperparameter_spaces.values():
            space["n_estimators"] = Integer(12, 24)

    result = module.run(config)
    if result.extra.get("skipped_groups"):
        raise RuntimeError(f"Stage 1 skipped groups in {name}: {result.extra['skipped_groups']}")
    print(f"Stage 1 {name}: {len(result.summary)} model summaries")


def read_selected_teeth(stage1_dir, scenario):
    summary = pd.read_csv(stage1_dir / scenario / "summary_final.csv")
    rows = {}
    for row in summary.itertuples(index=False):
        teeth = row.Consensus_Teeth_FDI.split(", ")
        if len(set(teeth)) != 10 or len(teeth) != 10:
            raise ValueError(f"Expected ten distinct teeth for {row.Model} in {scenario}")
        rows.setdefault(getattr(row, "Subgroup", "overall"), {})[row.Model] = teeth
    return rows


def use_stage1_teeth(config, scenario, stage1_dir):
    overall = read_selected_teeth(stage1_dir, "overall")["overall"]
    if scenario == "overall":
        config.tooth_selection_strategies.update({MODEL_KEYS[model]: overall[model] for model in MODEL_KEYS})
        return
    config.overall_tooth_selection.update(
        {f"{model}_SHAP_Consensus": overall[model] for model in MODEL_KEYS}
    )
    selected = read_selected_teeth(stage1_dir, scenario)
    if set(selected) != set(config.subgroup_teeth) or any(set(group) != set(MODEL_KEYS) for group in selected.values()):
        raise ValueError(f"Stage 1 summaries do not cover every group and model in {scenario}")
    config.subgroup_teeth = selected


def evaluate_internal(config, scenario, development_csv, stage1_dir):
    from core.classification import CDCClassifier
    from core.data_processing import DataProcessor
    from core.evaluation import StrategyEvaluator, make_overall_strategies, make_subgroup_strategies
    from core.export import write_overall_outputs, write_subgroup_outputs

    processor = DataProcessor(config)
    data = processor.load_and_preprocess(development_csv).set_index("SEQN")
    if not data.index.is_unique:
        raise ValueError(f"Duplicate participant IDs in {development_csv}")
    data["reference_true"] = data[config.target_column]
    evaluator = StrategyEvaluator(config, processor, CDCClassifier(processor))
    output = Path(config.output_dir)

    def evaluate(holdout, strategies, min_size=1):
        ids = pd.read_csv(holdout, usecols=["SEQN"])["SEQN"]
        if not ids.is_unique:
            raise ValueError(f"Duplicate participant IDs in {holdout}")
        group = data.loc[ids]
        if len(group) < min_size:
            print(f"Skipped {holdout.parent.name}: {len(group)} participants (minimum {min_size})")
            return {}, []
        results, comparisons, _ = evaluator.evaluate_group(group, strategies)
        return results, comparisons

    overall_holdout = stage1_dir / "overall" / "test_set_full.csv"
    if scenario == "overall":
        results, comparisons = evaluate(overall_holdout, make_overall_strategies(config))
        write_overall_outputs(output, results, comparisons)
        return

    all_results, all_comparisons = {}, {}
    for group in config.subgroup_teeth:
        holdout = stage1_dir / scenario / group / "test_set_full.csv"
        results, comparisons = evaluate(holdout, make_subgroup_strategies(config, group), config.min_subgroup_size)
        if results:
            all_results[group], all_comparisons[group] = results, comparisons
    if scenario == "subgroup_single_factor":
        all_results["Overall"], all_comparisons["Overall"] = evaluate(
            overall_holdout, make_subgroup_strategies(config, "Overall")
        )
    write_subgroup_outputs(output, all_results, all_comparisons)


def run_stage2(module, name, data_dir, output, args):
    config = module.build_config(data_dir)
    if args.smoke:
        config.n_bootstrap = 20
    if args.tooth_sets == "stage1":
        use_stage1_teeth(config, name, output / "stage1")

    stage2 = output / f"stage2_{args.tooth_sets}"
    if args.cohort in ("external", "both"):
        config.output_dir = str(stage2 / "external" / name)
        module.run(config)
        print(f"Stage 2 external {name}: {config.output_dir}")
    if args.cohort in ("internal", "both"):
        config.output_dir = str(stage2 / "internal" / name)
        evaluate_internal(config, name, data_dir / "cleaned_data.csv", output / "stage1")
        print(f"Stage 2 internal {name}: {config.output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("stage1", "stage2"))
    parser.add_argument("--scenario", choices=("all", *SCENARIOS), default="all")
    parser.add_argument("--rebuilt-data", action="store_true", help="Use CSVs from prepare_data.py.")
    parser.add_argument("--smoke", action="store_true", help="Use short fits or 20 bootstrap replicates.")
    parser.add_argument("--cohort", choices=("external", "internal", "both"), default="external")
    parser.add_argument("--tooth-sets", choices=("published", "stage1"), default="published")
    args = parser.parse_args()

    data_dir = OUT_DIR if args.rebuilt_data else ROOT
    output = OUT_DIR / ("paper_rebuilt" if args.rebuilt_data else "paper_published")
    if args.smoke:
        output /= "smoke"
    sys.path.insert(0, str(ROOT / ("Stage1" if args.stage == "stage1" else "Stage2")))
    for name in SCENARIOS if args.scenario == "all" else (args.scenario,):
        module = import_module(f"scenarios.{name}")
        if args.stage == "stage1":
            run_stage1(module, name, data_dir, output, args.smoke)
        else:
            run_stage2(module, name, data_dir, output, args)


if __name__ == "__main__":
    main()
