from __future__ import annotations

from pathlib import Path

from core.config import GroupFilter, ScenarioConfig
from core.pipeline import run_subgroup


def build_config(base_dir: Path | None = None) -> ScenarioConfig:
    base = Path(".") if base_dir is None else Path(base_dir)
    return ScenarioConfig(
        name="subgroup_age_gender",
        data_path=base / "cleaned_data.csv",
        output_dir=base / "Gender_and_age_analysis",
        top_features_count=10,
        min_samples_per_group=100,
    )


def build_groups():
    return {
        "35-44_Male": GroupFilter(age_min=35, age_max=44, gender=1),
        "35-44_Female": GroupFilter(age_min=35, age_max=44, gender=2),
        "45-54_Male": GroupFilter(age_min=45, age_max=54, gender=1),
        "45-54_Female": GroupFilter(age_min=45, age_max=54, gender=2),
        "55-64_Male": GroupFilter(age_min=55, age_max=64, gender=1),
        "55-64_Female": GroupFilter(age_min=55, age_max=64, gender=2),
        "65-74_Male": GroupFilter(age_min=65, age_max=74, gender=1),
        "65-74_Female": GroupFilter(age_min=65, age_max=74, gender=2),
        "75+_Male": GroupFilter(age_min=75, age_max=999, gender=1),
        "75+_Female": GroupFilter(age_min=75, age_max=999, gender=2),
    }


def run(config: ScenarioConfig | None = None, groups=None):
    cfg = config or build_config()
    return run_subgroup(cfg, groups or build_groups())


if __name__ == "__main__":
    run()
