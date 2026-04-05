from __future__ import annotations

from pathlib import Path

from core.config import DEFAULT_NHANES_TO_FDI
from core.pipeline import run_analysis
from core.types import ConfigSpec


def build_config(base_dir: Path | None = None) -> ConfigSpec:
    base = Path(".") if base_dir is None else Path(base_dir)
    nhanes_to_fdi = DEFAULT_NHANES_TO_FDI
    fdi_to_nhanes = {v: k for k, v in nhanes_to_fdi.items()}
    return ConfigSpec(
        scenario="overall",
        data_path=str(base / "test_data.csv"),
        output_dir=str(base /  "Global_teeth_test"),
        target_column="perio_label_cdc",
        age_column="RIDAGEYR",
        gender_column=None,
        weight_column="WTMEC2YR",
        stratum_column="SDMVSTRA",
        psu_column="SDMVPSU",
        min_age=35,
        n_bootstrap=8000,
        random_seed=42,
        min_subgroup_size=50,
        interproximal_sites=["D", "S", "P", "A"],
        class_labels={0: "Healthy", 1: "Other", 2: "Severe"},
        nhanes_to_fdi_mapping=nhanes_to_fdi,
        fdi_to_nhanes_mapping=fdi_to_nhanes,
        tooth_selection_strategies={
            "XGB10": ["47", "27", "17", "26", "37", "16", "42", "46", "36", "41"],
            "LGB10": ["47", "27", "17", "26", "37", "16", "42", "46", "36", "25"],
            "CPITN_Reference": ["11", "16", "17", "26", "27", "31", "36", "37", "46", "47"],
            "Ramfjord_Reference": ["14", "16", "21", "24", "26", "34", "36", "41", "44", "46"],
        },
    )


def run(config: ConfigSpec | None = None):
    cfg = config or build_config()
    return run_analysis(cfg)


if __name__ == "__main__":
    run()
