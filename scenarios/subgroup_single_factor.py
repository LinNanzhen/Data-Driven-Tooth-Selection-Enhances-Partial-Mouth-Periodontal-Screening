from __future__ import annotations

from pathlib import Path

from core.config import GroupFilter, ScenarioConfig
from core.pipeline import run_subgroup


def build_config(base_dir: Path | None = None) -> ScenarioConfig:
    base = Path(".") if base_dir is None else Path(base_dir)
    return ScenarioConfig(
        name="subgroup_single_factor",
        data_path=base / "cleaned_data.csv",
        output_dir=base / "Gender_or_age_analysis",
        top_features_count=10,
        min_samples_per_group=100,
    )


def build_groups():
    return {
        "Age_35-44": GroupFilter(age_min=35, age_max=44, gender=None),
        "Age_45-54": GroupFilter(age_min=45, age_max=54, gender=None),
        "Age_55-64": GroupFilter(age_min=55, age_max=64, gender=None),
        "Age_65-74": GroupFilter(age_min=65, age_max=74, gender=None),
        "Age_75+": GroupFilter(age_min=75, age_max=999, gender=None),
        "Gender_Male": GroupFilter(age_min=35, age_max=999, gender=1),
        "Gender_Female": GroupFilter(age_min=35, age_max=999, gender=2),
    }


def run(config: ScenarioConfig | None = None, groups=None):
    cfg = config or build_config()
    return run_subgroup(cfg, groups or build_groups())


if __name__ == "__main__":
    run()
