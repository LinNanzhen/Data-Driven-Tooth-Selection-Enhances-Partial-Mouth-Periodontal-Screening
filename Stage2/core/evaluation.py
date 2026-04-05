from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from core.bootstrap import BootstrapEngine
from core.classification import CDCClassifier
from core.data_processing import DataProcessor
from core.types import ConfigSpec, StrategySpec


class StrategyEvaluator:
    def __init__(self, config: ConfigSpec, data_processor: DataProcessor, classifier: CDCClassifier):
        self.config = config
        self.data_processor = data_processor
        self.classifier = classifier
        self.bootstrap = BootstrapEngine(config)

    def apply_strategies(
        self,
        df: pd.DataFrame,
        strategy_specs: Dict[str, StrategySpec],
    ) -> Tuple[pd.DataFrame, Dict[str, StrategySpec]]:
        applied: Dict[str, StrategySpec] = {}
        work_df = df.copy()

        for strategy_name, spec in strategy_specs.items():
            selected_teeth = self.data_processor.get_tooth_measurements(spec.teeth_fdi)
            if not selected_teeth:
                continue
            work_df["{0}_pred".format(strategy_name)] = self.classifier.classify_with_tooth_selection(
                work_df, selected_teeth
            )
            applied[strategy_name] = spec

        return work_df, applied

    def evaluate_group(
        self,
        df: pd.DataFrame,
        strategy_specs: Dict[str, StrategySpec],
    ) -> Tuple[Dict[str, dict], List[dict], Dict[str, StrategySpec]]:
        prepared_df, applied = self.apply_strategies(df, strategy_specs)
        if not applied:
            return {}, [], {}
        results, comparisons = self.bootstrap.run(prepared_df, applied)
        return results, comparisons, applied


def make_overall_strategies(config: ConfigSpec) -> Dict[str, StrategySpec]:
    return {
        name: StrategySpec(name=name, teeth_fdi=list(teeth), strategy_type="overall")
        for name, teeth in config.tooth_selection_strategies.items()
    }


def make_subgroup_strategies(
    config: ConfigSpec,
    subgroup_name: str,
) -> Dict[str, StrategySpec]:
    strategy_specs: Dict[str, StrategySpec] = {}

    subgroup_map = config.subgroup_teeth.get(subgroup_name, {})
    for model_name, teeth_fdi in subgroup_map.items():
        strategy_name = "{0}_SHAP_{1}".format(model_name, subgroup_name)
        strategy_specs[strategy_name] = StrategySpec(
            name=strategy_name,
            teeth_fdi=list(teeth_fdi),
            strategy_type="subgroup_specific",
            subgroup=subgroup_name,
        )

    for name, teeth_fdi in config.overall_tooth_selection.items():
        strategy_specs[name] = StrategySpec(
            name=name,
            teeth_fdi=list(teeth_fdi),
            strategy_type="overall_on_subgroup",
            subgroup=subgroup_name,
        )

    return strategy_specs
