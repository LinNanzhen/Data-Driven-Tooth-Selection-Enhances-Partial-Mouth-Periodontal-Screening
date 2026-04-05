from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


ScenarioName = Literal["overall", "s1", "s2"]


@dataclass
class StrategySpec:
    name: str
    teeth_fdi: List[str]
    strategy_type: str = "generic"
    subgroup: Optional[str] = None


@dataclass
class MetricBundle:
    point_estimate: float
    ci_lower: float
    ci_upper: float
    bootstrap_dist: List[float] = field(default_factory=list)


@dataclass
class ConfigSpec:
    scenario: ScenarioName
    data_path: str
    output_dir: str
    target_column: str
    age_column: str
    gender_column: Optional[str]
    weight_column: str
    stratum_column: str
    psu_column: str
    min_age: int
    n_bootstrap: int
    random_seed: int
    min_subgroup_size: int
    interproximal_sites: List[str]
    class_labels: Dict[int, str]
    nhanes_to_fdi_mapping: Dict[str, str]
    fdi_to_nhanes_mapping: Dict[str, str]
    tooth_selection_strategies: Dict[str, List[str]] = field(default_factory=dict)
    overall_tooth_selection: Dict[str, List[str]] = field(default_factory=dict)
    subgroup_teeth: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    scenario: ScenarioName
    output_dir: str
    results: Dict[str, Any]
    comparisons: Dict[str, Any]
    artifacts: Dict[str, str]
    summary_records: List[Dict[str, Any]]
