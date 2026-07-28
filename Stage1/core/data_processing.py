from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd

from core.config import ScenarioConfig


class DataProcessor:
    def __init__(self, config: ScenarioConfig):
        self.config = config
        self.pd_columns = []
        self.cal_columns = []

    def load_and_preprocess(self, filepath):
        print("Loading data from: {0}".format(filepath))
        try:
            df = pd.read_csv(filepath, low_memory=False)
        except FileNotFoundError:
            print("Error: data file not found at {0}".format(filepath))
            return pd.DataFrame()

        required_cols = [self.config.age_column, self.config.weight_column]
        if self.config.gender_column:
            required_cols.append(self.config.gender_column)

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print("Warning: missing required columns: {0}".format(missing_cols))

        print("Initial sample size: {0}".format(len(df)))
        df = df[df[self.config.age_column] >= self.config.min_age].copy()
        print(
            "Sample size after age filtering (>= {0}): {1}".format(
                self.config.min_age, len(df)
            )
        )

        weighted_size = len(df)
        df = df[(df[self.config.weight_column].notna()) & (df[self.config.weight_column] > 0)].copy()
        print(
            "Sample size after weight filtering: {0} (removed {1} records)".format(
                len(df), weighted_size - len(df)
            )
        )

        df = self._identify_periodontal_features(df)
        df = self._handle_missing_values(df)
        if self.config.weight_column in df.columns:
            df[self.config.weight_column] = df[self.config.weight_column] * self.config.weight_scale_factor
        df = self._apply_cdc_classification(df)
        print("Data preprocessing complete")
        return df

    def _identify_periodontal_features(self, df):
        pd_pattern = r"^OHX\d{2}PC[ADSP]$"
        cal_pattern = r"^OHX\d{2}LA[ADSP]$"
        self.pd_columns = sorted([c for c in df.columns if re.search(pd_pattern, c)])
        self.cal_columns = sorted([c for c in df.columns if re.search(cal_pattern, c)])
        for col in self.pd_columns + self.cal_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _handle_missing_values(self, df):
        cols = self.pd_columns + self.cal_columns
        df[cols] = df[cols].replace(99, np.nan)
        return df

    def _apply_cdc_classification(self, df):
        pd_cols = self.pd_columns
        cal_cols = self.cal_columns
        impute_cols = pd_cols + cal_cols

        if not impute_cols:
            df[self.config.target_column] = 0
            return df

        df_imputed = df[impute_cols].copy()

        tooth_prefixes = sorted({c[:5] for c in impute_cols})
        tooth_pd_cols = {tp: [] for tp in tooth_prefixes}
        tooth_cal_cols = {tp: [] for tp in tooth_prefixes}

        for col in pd_cols:
            if col[-1] in self.config.interproximal_sites:
                tooth_pd_cols[col[:5]].append(col)
        for col in cal_cols:
            if col[-1] in self.config.interproximal_sites:
                tooth_cal_cols[col[:5]].append(col)

        tooth_pd_max = {}
        tooth_cal_max = {}
        for tp in tooth_prefixes:
            cols_pd = tooth_pd_cols[tp]
            cols_cal = tooth_cal_cols[tp]
            tooth_pd_max[tp] = (
                df_imputed[cols_pd].max(axis=1)
                if cols_pd
                else pd.Series(np.nan, index=df.index)
            )
            tooth_cal_max[tp] = (
                df_imputed[cols_cal].max(axis=1)
                if cols_cal
                else pd.Series(np.nan, index=df.index)
            )

        tooth_pd_max_df = pd.DataFrame(tooth_pd_max)
        tooth_cal_max_df = pd.DataFrame(tooth_cal_max)

        def count_teeth_ge(mat: pd.DataFrame, thr: float) -> pd.Series:
            return (mat >= thr).sum(axis=1)

        n_cal3 = count_teeth_ge(tooth_cal_max_df, 3.0)
        n_cal6 = count_teeth_ge(tooth_cal_max_df, 6.0)
        n_ppd4 = count_teeth_ge(tooth_pd_max_df, 4.0)
        n_ppd5 = count_teeth_ge(tooth_pd_max_df, 5.0)

        severe = (n_cal6 >= 2) & (n_ppd5 >= 1)
        other = ((~severe) & (n_cal3 >= 2) & ((n_ppd4 >= 2) | (n_ppd5 >= 1)))

        label = np.where(severe, 2, np.where(other, 1, 0)).astype(int)
        df[self.config.target_column] = label
        return df

    def _create_max_features(self, df, column_list):
        if not column_list:
            return pd.DataFrame(index=df.index)

        data_for_agg = df[column_list]
        grouped_cols = defaultdict(list)
        for col in column_list:
            grouped_cols[col[:-1]].append(col)

        feature_series_list = []
        for prefix, columns in grouped_cols.items():
            max_values = data_for_agg[columns].max(axis=1)
            max_values.name = "{0}_max".format(prefix)
            feature_series_list.append(max_values)

        return pd.concat(feature_series_list, axis=1) if feature_series_list else pd.DataFrame(index=df.index)

    def get_combined_feature_set(self, df):
        x_pd_max = self._create_max_features(df, self.pd_columns)
        x_cal_max = self._create_max_features(df, self.cal_columns)

        if x_pd_max.empty and x_cal_max.empty:
            x_combined = pd.DataFrame(index=df.index)
        elif x_pd_max.empty:
            x_combined = x_cal_max.copy()
        elif x_cal_max.empty:
            x_combined = x_pd_max.copy()
        else:
            x_combined = pd.concat([x_pd_max.reindex(df.index), x_cal_max.reindex(df.index)], axis=1)

        y = df[self.config.target_column]
        sample_weights = df[self.config.weight_column] if self.config.weight_column in df.columns else None
        return x_combined, y, sample_weights

    def nhanes_to_fdi(self, nhanes_tooth):
        return self.config.nhanes_to_fdi_mapping.get(nhanes_tooth, nhanes_tooth)

    def extract_tooth_numbers_from_features(self, feature_list):
        tooth_numbers_fdi = set()
        for feature in feature_list:
            if "OHX" in feature and ("PC_max" in feature or "LA_max" in feature):
                tooth_num = feature.replace("OHX", "").replace("PC_max", "").replace("LA_max", "")
                if tooth_num.isdigit():
                    tooth_numbers_fdi.add(self.nhanes_to_fdi(tooth_num))
        return sorted(list(tooth_numbers_fdi))
