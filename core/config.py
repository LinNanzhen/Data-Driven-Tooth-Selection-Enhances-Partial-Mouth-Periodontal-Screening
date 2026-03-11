from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def default_hyperparameter_spaces() -> Dict[str, Dict[str, Any]]:
    try:
        from skopt.space import Integer, Real
    except Exception:
        return {}

    return {
        "XGBoost": {
            "n_estimators": Integer(1000, 1300),
            "max_depth": Integer(2, 4),
            "learning_rate": Real(0.05, 0.15, "log-uniform"),
            "subsample": Real(0.7, 0.9, "uniform"),
            "colsample_bytree": Real(0.85, 0.95, "uniform"),
            "gamma": Real(0.2, 0.8, "uniform"),
            "reg_alpha": Real(1e-3, 1e-2, "log-uniform"),
            "reg_lambda": Real(1e-3, 1e-2, "log-uniform"),
            "min_child_weight": Integer(8, 12),
        },
        "LightGBM": {
            "n_estimators": Integer(800, 1200),
            "max_depth": Integer(1, 3),
            "num_leaves": Integer(80, 150),
            "learning_rate": Real(0.08, 0.20, "log-uniform"),
            "subsample": Real(0.7, 0.9, "uniform"),
            "colsample_bytree": Real(0.6, 0.8, "uniform"),
            "reg_alpha": Real(1e-5, 1e-3, "log-uniform"),
            "reg_lambda": Real(5.0, 15.0, "log-uniform"),
            "min_child_samples": Integer(10, 25),
        },
    }


def default_nhanes_to_fdi_mapping() -> Dict[str, str]:
    return {
        "01": "18",
        "02": "17",
        "03": "16",
        "04": "15",
        "05": "14",
        "06": "13",
        "07": "12",
        "08": "11",
        "09": "21",
        "10": "22",
        "11": "23",
        "12": "24",
        "13": "25",
        "14": "26",
        "15": "27",
        "16": "28",
        "17": "38",
        "18": "37",
        "19": "36",
        "20": "35",
        "21": "34",
        "22": "33",
        "23": "32",
        "24": "31",
        "25": "41",
        "26": "42",
        "27": "43",
        "28": "44",
        "29": "45",
        "30": "46",
        "31": "47",
        "32": "48",
    }


def default_reference_tooth_sets() -> Dict[str, List[str]]:
    return {
        "Ramfjord": ["16", "14", "21", "24", "26", "36", "34", "41", "44", "46"],
        "CPI": ["11", "16", "17", "26", "27", "31", "36", "37", "46", "47"],
    }


@dataclass(frozen=True)
class GroupFilter:
    age_min: int
    age_max: int
    gender: Optional[int] = None


@dataclass
class ScenarioConfig:
    name: str
    data_path: Path
    output_dir: Path
    target_column: str = "perio_label_cdc"
    age_column: str = "RIDAGEYR"
    gender_column: Optional[str] = "RIAGENDR"
    weight_column: str = "WTMEC2YR"
    min_age: int = 35
    min_samples_per_group: int = 100
    random_state: int = 42
    test_size: float = 0.2
    random_seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 999, 2025])
    cv_folds: int = 5
    interproximal_sites: set = field(default_factory=lambda: {"D", "S", "P", "A"})
    class_labels: Dict[int, str] = field(
        default_factory=lambda: {0: "Healthy", 1: "Other", 2: "Severe"}
    )
    top_features_count: int = 10
    shap_sample_size: int = 1000
    shap_plot_max_display: int = 20
    cpi_ramfjord_tooth_numbers: Dict[str, List[str]] = field(
        default_factory=default_reference_tooth_sets
    )
    nhanes_to_fdi_mapping: Dict[str, str] = field(default_factory=default_nhanes_to_fdi_mapping)
    bayesian_hyperparameter_spaces: Dict[str, Dict[str, Any]] = field(
        default_factory=default_hyperparameter_spaces
    )
    bayesian_n_iter: int = 30
    analysis_groups: Dict[str, GroupFilter] = field(default_factory=dict)
    models: List[str] = field(default_factory=lambda: ["XGBoost", "LightGBM"])
    keep_legacy_outputs: bool = True

    def prepare_output_dir(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir


@dataclass
class OutputSpec:
    metrics_cv: str = "metrics_cv.csv"
    metrics_test: str = "metrics_test.csv"
    comparison_cv_test: str = "comparison_cv_test.csv"
    summary_final: str = "summary_final.csv"
    legacy_map: Dict[str, str] = field(default_factory=dict)


@dataclass
class RunResult:
    cv_results: Any
    test_results: Any
    comparison: Any
    summary: Any
    stability_results: Dict[str, Any]
    extra: Dict[str, Any] = field(default_factory=dict)


def make_output_spec(mode: str) -> OutputSpec:
    if mode == "overall":
        legacy = {
            "metrics_cv.csv": "cross_validation_results.csv",
            "metrics_test.csv": "final_test_results.csv",
            "comparison_cv_test.csv": "cv_vs_test_comparison.csv",
            "summary_final.csv": "final_consolidated_summary.csv",
        }
        return OutputSpec(legacy_map=legacy)

    legacy = {
        "metrics_cv.csv": "all_subgroups_cv_results.csv",
        "metrics_test.csv": "all_subgroups_test_results.csv",
        "comparison_cv_test.csv": "cv_vs_test_comparison_by_subgroup.csv",
        "summary_final.csv": "final_consolidated_summary.csv",
    }
    return OutputSpec(legacy_map=legacy)
