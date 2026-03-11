import pandas as pd

from core.config import ScenarioConfig
from core.data_processing import DataProcessor


def _make_base_df():
    rows = []
    for idx in range(3):
        rows.append({"RIDAGEYR": 50 + idx, "WTMEC2YR": 1.0, "RIAGENDR": 1})
    df = pd.DataFrame(rows)

    for tooth in ["01", "02"]:
        for site in ["A", "D", "S", "P"]:
            df[f"OHX{tooth}PC{site}"] = [1, 4, 5 if tooth == "01" else 1]
            df[f"OHX{tooth}LA{site}"] = [1, 3, 6]

    return df


def test_preprocess_and_cdc_labels(tmp_path):
    raw_df = _make_base_df()
    csv_path = tmp_path / "mock.csv"
    raw_df.to_csv(csv_path, index=False)

    cfg = ScenarioConfig(name="t", data_path=csv_path, output_dir=tmp_path / "out")
    processor = DataProcessor(cfg)
    df = processor.load_and_preprocess(csv_path)

    assert cfg.target_column in df.columns
    assert list(df[cfg.target_column]) == [0, 1, 2]


def test_combined_feature_generation(tmp_path):
    raw_df = _make_base_df()
    csv_path = tmp_path / "mock.csv"
    raw_df.to_csv(csv_path, index=False)

    cfg = ScenarioConfig(name="t", data_path=csv_path, output_dir=tmp_path / "out")
    processor = DataProcessor(cfg)
    df = processor.load_and_preprocess(csv_path)
    x, y, sw = processor.get_combined_feature_set(df)

    assert not x.empty
    assert len(y) == len(df)
    assert sw is not None
    assert any(col.endswith("PC_max") for col in x.columns)
    assert any(col.endswith("LA_max") for col in x.columns)
