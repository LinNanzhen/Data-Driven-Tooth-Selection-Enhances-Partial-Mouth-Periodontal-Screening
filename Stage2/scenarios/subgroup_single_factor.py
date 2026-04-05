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
        scenario="s1",
        data_path=str(base / "test_data.csv"),
        output_dir=str(base / "Subgroup_gender_or_age_teeth_test"),
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
            "Age_35-44": {
                "XGBoost": ["11", "16", "17", "26", "27", "37", "41", "43", "46", "47"],
                "LightGBM": ["16", "17", "26", "27", "36", "37", "41", "43", "46", "47"],
            },
            "Age_45-54": {
                "XGBoost": ["16", "17", "26", "27", "31", "37", "42", "43", "46", "47"],
                "LightGBM": ["16", "17", "26", "27", "31", "37", "42", "45", "46", "47"],
            },
            "Age_55-64": {
                "XGBoost": ["16", "17", "26", "27", "33", "36", "37", "42", "46", "47"],
                "LightGBM": ["16", "17", "26", "27", "33", "36", "42", "43", "46", "47"],
            },
            "Age_65-74": {
                "XGBoost": ["13", "16", "17", "25", "26", "27", "31", "37", "43", "47"],
                "LightGBM": ["13", "16", "17", "25", "26", "27", "31", "37", "43", "47"],
            },
            "Age_75+": {
                "XGBoost": ["14", "16", "17", "23", "26", "27", "32", "42", "46", "47"],
                "LightGBM": ["14", "16", "17", "23", "27", "32", "42", "44", "46", "47"],
            },
            "Gender_Male": {
                "XGBoost": ["16", "17", "25", "26", "27", "36", "37", "42", "46", "47"],
                "LightGBM": ["16", "17", "26", "27", "32", "36", "37", "42", "46", "47"],
            },
            "Gender_Female": {
                "XGBoost": ["16", "17", "26", "27", "33", "37", "41", "42", "46", "47"],
                "LightGBM": ["16", "17", "26", "27", "33", "37", "41", "42", "46", "47"],
            },
        },
    )


def run(config: ConfigSpec | None = None):
    cfg = config or build_config()
    return run_analysis(cfg)


if __name__ == "__main__":
    run()
