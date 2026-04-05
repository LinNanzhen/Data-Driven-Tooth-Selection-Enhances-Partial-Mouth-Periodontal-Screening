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
        scenario="s2",
        data_path=str(base / "test_data.csv"),
        output_dir=str(base / "Subgroup_gender_and_age_teeth_test"),
        target_column="perio_label_cdc",
        age_column="RIDAGEYR",
        gender_column="RIAGENDR",
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
        overall_tooth_selection={
            "XGBoost_SHAP_Consensus": ["16", "17", "26", "27", "37", "47", "41", "42", "46", "36"],
            "LightGBM_SHAP_Consensus": ["16", "17", "26", "27", "37", "47", "25", "42", "46", "36"],
            "CPITN_Reference": ["11", "16", "17", "26", "27", "31", "36", "37", "46", "47"],
            "Ramfjord_Reference": ["14", "16", "21", "24", "26", "34", "36", "41", "44", "46"],
        },
        subgroup_teeth={
            "35-44_Male": {
                "XGBoost": ["11", "16", "17", "26", "27", "37", "41", "45", "46", "47"],
                "LightGBM": ["11", "16", "17", "26", "27", "37", "41", "45", "46", "47"],
            },
            "35-44_Female": {
                "XGBoost": ["12", "17", "22", "23", "26", "27", "31", "37", "43", "47"],
                "LightGBM": ["12", "16", "17", "22", "26", "27", "37", "43", "46", "47"],
            },
            "45-54_Male": {
                "XGBoost": ["17", "16", "27", "26", "47", "37", "42", "46", "15", "35"],
                "LightGBM": ["16", "27", "17", "26", "47", "37", "15", "42", "46", "23"],
            },
            "45-54_Female": {
                "XGBoost": ["27", "26", "46", "17", "34", "42", "47", "43", "15", "14"],
                "LightGBM": ["27", "26", "46", "17", "34", "47", "42", "16", "15", "43"],
            },
            "55-64_Male": {
                "XGBoost": ["17", "16", "27", "47", "26", "33", "45", "37", "43", "25"],
                "LightGBM": ["17", "16", "26", "47", "27", "37", "42", "45", "25", "33"],
            },
            "55-64_Female": {
                "XGBoost": ["27", "43", "17", "26", "37", "46", "16", "33", "24", "25"],
                "LightGBM": ["27", "46", "37", "26", "17", "43", "33", "25", "24", "45"],
            },
            "65-74_Male": {
                "XGBoost": ["27", "45", "16", "47", "26", "34", "44", "21", "37", "42"],
                "LightGBM": ["27", "45", "16", "26", "11", "42", "34", "37", "47", "25"],
            },
            "65-74_Female": {
                "XGBoost": ["26", "42", "27", "17", "43", "34", "41", "31", "35", "44"],
                "LightGBM": ["26", "42", "27", "43", "17", "34", "35", "13", "32", "44"],
            },
            "75+_Male": {
                "XGBoost": ["47", "33", "32", "43", "31", "21", "37", "45", "16", "27"],
                "LightGBM": ["47", "33", "31", "37", "21", "43", "27", "16", "32", "45"],
            },
            "75+_Female": {
                "XGBoost": ["32", "33", "41", "43", "26", "23", "31", "27", "46", "16"],
                "LightGBM": ["32", "33", "41", "26", "23", "31", "43", "27", "46", "16"],
            },
        },
    )


def run(config: ConfigSpec | None = None):
    cfg = config or build_config()
    return run_analysis(cfg)


if __name__ == "__main__":
    run()
