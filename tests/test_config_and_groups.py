import pandas as pd

from core.config import GroupFilter, ScenarioConfig
from core.model_factory import available_models
from scenarios.subgroup_age_gender import build_groups as build_age_gender_groups
from scenarios.subgroup_single_factor import build_groups as build_single_factor_groups


def test_group_filter_dataclass():
    group = GroupFilter(age_min=35, age_max=44, gender=1)
    assert group.age_min == 35
    assert group.age_max == 44
    assert group.gender == 1


def test_scenario_config_defaults(tmp_path):
    cfg = ScenarioConfig(name="x", data_path=tmp_path / "a.csv", output_dir=tmp_path / "out")
    cfg.prepare_output_dir()
    assert cfg.output_dir.exists()
    assert cfg.cv_folds == 5
    assert cfg.random_seeds


def test_group_builders_shape():
    s1_groups = build_single_factor_groups()
    s2_groups = build_age_gender_groups()
    assert len(s1_groups) == 7
    assert len(s2_groups) == 10
    assert any(v.gender is None for v in s1_groups.values())
    assert all(v.gender in (1, 2) for v in s2_groups.values())


def test_available_models_contract():
    models = available_models(["XGBoost", "LightGBM"])
    assert isinstance(models, list)
    assert set(models).issubset({"XGBoost", "LightGBM"})
