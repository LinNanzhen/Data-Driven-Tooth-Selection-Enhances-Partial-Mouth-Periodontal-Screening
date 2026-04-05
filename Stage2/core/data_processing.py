from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from core.types import ConfigSpec


class DataProcessor:
    def __init__(self, config: ConfigSpec):
        self.config = config
        self.pd_columns: List[str] = []
        self.cal_columns: List[str] = []
        self.tooth_site_mapping: Dict[str, Dict[str, Dict[str, str]]] = {}

    def load_and_preprocess(self, filepath: Union[str, Path]) -> pd.DataFrame:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError("Data file not found: {0}".format(filepath))

        df = pd.read_csv(filepath, low_memory=False)
        self._ensure_required_columns(df)

        df = df[df[self.config.age_column] >= self.config.min_age].copy()
        df = df[(df[self.config.weight_column].notna()) & (df[self.config.weight_column] > 0)].copy()

        df = self._identify_periodontal_measurements(df)
        df = self._apply_cdc_classification_fullmouth(df)
        df = self._create_subgroup_columns(df)
        return df

    def _ensure_required_columns(self, df: pd.DataFrame) -> None:
        required = [self.config.age_column, self.config.weight_column]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError("Missing required columns: {0}".format(missing))

        if self.config.stratum_column not in df.columns:
            df[self.config.stratum_column] = 1
        if self.config.psu_column not in df.columns:
            df[self.config.psu_column] = np.arange(len(df))

    def _identify_periodontal_measurements(self, df: pd.DataFrame) -> pd.DataFrame:
        pd_pattern = r"^OHX\d{2}PC[ADSP]$"
        cal_pattern = r"^OHX\d{2}LA[ADSP]$"

        self.pd_columns = sorted([c for c in df.columns if re.search(pd_pattern, c)])
        self.cal_columns = sorted([c for c in df.columns if re.search(cal_pattern, c)])

        for col in self.pd_columns + self.cal_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace(99, np.nan)

            tooth_num = col[3:5]
            site = col[-1]
            measurement_type = "PD" if "PC" in col else "CAL"
            self.tooth_site_mapping.setdefault(tooth_num, {}).setdefault(measurement_type, {})[site] = col

        return df.copy()

    def _apply_cdc_classification_fullmouth(self, df: pd.DataFrame) -> pd.DataFrame:
        all_teeth = sorted(self.tooth_site_mapping.keys())
        labels = self.classify_with_tooth_selection(df, all_teeth)

        df[self.config.target_column] = labels
        df["perio_other"] = ((labels == 1) | (labels == 2)).astype(int)
        df["perio_total"] = (labels >= 1).astype(int)
        return df

    def _create_subgroup_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        age_col = self.config.age_column
        gender_col = self.config.gender_column

        def _age_bucket(v: float) -> Optional[str]:
            if pd.isna(v):
                return None
            v = float(v)
            if 35 <= v < 45:
                return "Age_35-44"
            if 45 <= v < 55:
                return "Age_45-54"
            if 55 <= v < 65:
                return "Age_55-64"
            if 65 <= v < 75:
                return "Age_65-74"
            if v >= 75:
                return "Age_75+"
            return None

        def _sex_bucket(v: Union[float, str]) -> Optional[str]:
            if pd.isna(v):
                return None
            v = str(int(v)) if not isinstance(v, str) else v
            if v == "1":
                return "Gender_Male"
            if v == "2":
                return "Gender_Female"
            return None

        df["age_subgroup"] = df[age_col].apply(_age_bucket)
        if gender_col and gender_col in df.columns:
            df["gender_subgroup"] = df[gender_col].apply(_sex_bucket)
            df["age_gender_subgroup"] = df["age_subgroup"].str.replace("Age_", "", regex=False) + "_" + df[
                "gender_subgroup"
            ].str.replace("Gender_", "", regex=False)
            df.loc[df["age_subgroup"].isna() | df["gender_subgroup"].isna(), "age_gender_subgroup"] = None
        else:
            df["gender_subgroup"] = None
            df["age_gender_subgroup"] = None
        return df

    def get_tooth_measurements(self, tooth_strategy_fdi: List[str]) -> List[str]:
        selected_teeth: List[str] = []
        for fdi in tooth_strategy_fdi:
            nhanes_tooth = self.config.fdi_to_nhanes_mapping.get(str(fdi))
            if nhanes_tooth and nhanes_tooth in self.tooth_site_mapping:
                selected_teeth.append(nhanes_tooth)
        return selected_teeth

    def _build_tooth_max_frames(
        self,
        df: pd.DataFrame,
        selected_teeth_nhanes: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        sites = set(self.config.interproximal_sites)
        tooth_pd_max: Dict[str, pd.Series] = {}
        tooth_cal_max: Dict[str, pd.Series] = {}

        for tooth in selected_teeth_nhanes:
            mapping = self.tooth_site_mapping.get(tooth, {})
            pd_cols = [mapping.get("PD", {}).get(site) for site in sites]
            cal_cols = [mapping.get("CAL", {}).get(site) for site in sites]
            pd_cols = [c for c in pd_cols if c is not None]
            cal_cols = [c for c in cal_cols if c is not None]

            if pd_cols:
                tooth_pd_max[tooth] = df[pd_cols].max(axis=1, skipna=True)
            else:
                tooth_pd_max[tooth] = pd.Series(np.nan, index=df.index)

            if cal_cols:
                tooth_cal_max[tooth] = df[cal_cols].max(axis=1, skipna=True)
            else:
                tooth_cal_max[tooth] = pd.Series(np.nan, index=df.index)

        return pd.DataFrame(tooth_pd_max, index=df.index), pd.DataFrame(tooth_cal_max, index=df.index)

    @staticmethod
    def _labels_from_tooth_max(tooth_pd_max_df: pd.DataFrame, tooth_cal_max_df: pd.DataFrame) -> np.ndarray:
        n_cal3 = (tooth_cal_max_df >= 3.0).sum(axis=1)
        n_cal6 = (tooth_cal_max_df >= 6.0).sum(axis=1)
        n_ppd4 = (tooth_pd_max_df >= 4.0).sum(axis=1)
        n_ppd5 = (tooth_pd_max_df >= 5.0).sum(axis=1)

        severe = (n_cal6 >= 2) & (n_ppd5 >= 1)
        other = (~severe) & (n_cal3 >= 2) & ((n_ppd4 >= 2) | (n_ppd5 >= 1))

        return np.where(severe, 2, np.where(other, 1, 0)).astype(int)

    def classify_with_tooth_selection(self, df: pd.DataFrame, selected_teeth_nhanes: List[str]) -> np.ndarray:
        if not selected_teeth_nhanes:
            return np.zeros(len(df), dtype=int)
        tooth_pd_max_df, tooth_cal_max_df = self._build_tooth_max_frames(df, selected_teeth_nhanes)
        return self._labels_from_tooth_max(tooth_pd_max_df, tooth_cal_max_df)
