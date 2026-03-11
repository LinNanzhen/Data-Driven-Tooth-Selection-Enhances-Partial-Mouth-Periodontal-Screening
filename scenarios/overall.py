from __future__ import annotations

from pathlib import Path

from core.config import ScenarioConfig
from core.pipeline import run_overall


def build_config(base_dir: Path | None = None) -> ScenarioConfig:
    base = Path(".") if base_dir is None else Path(base_dir)
    return ScenarioConfig(
        name="overall",
        data_path=base / "cleaned_data.csv",
        output_dir=base / "Overall_after_impute",
        gender_column=None,
        top_features_count=10,
        min_samples_per_group=0,
    )


def run(config: ScenarioConfig | None = None):
    cfg = config or build_config()
    return run_overall(cfg)


if __name__ == "__main__":
    run()
