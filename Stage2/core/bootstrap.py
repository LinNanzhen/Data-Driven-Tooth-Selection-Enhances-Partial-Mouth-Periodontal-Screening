from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from core.metrics import (
    bootstrap_pairwise_test,
    compute_screening_metrics,
    weighted_f1_macro,
    weighted_kappa_quadratic,
    weighted_kappa_simple,
)
from core.types import ConfigSpec, StrategySpec


class PSUBootstrapSampler:
    def __init__(self, df: pd.DataFrame, stratum_col: str, psu_col: str, seed: int):
        self.df = df
        self.stratum_col = stratum_col
        self.psu_col = psu_col
        self.rng = np.random.default_rng(seed)
        self.psu_structure = self._prepare_psu_structure()

    def _prepare_psu_structure(self) -> Dict[object, np.ndarray]:
        psu_structure: Dict[object, np.ndarray] = {}
        for stratum in self.df[self.stratum_col].dropna().unique():
            mask = self.df[self.stratum_col] == stratum
            psus = self.df.loc[mask, self.psu_col].dropna().unique()
            psu_structure[stratum] = psus
        return psu_structure

    def generate_indices(self) -> List[int]:
        bootstrap_indices: List[int] = []
        for stratum, psus in self.psu_structure.items():
            if len(psus) == 0:
                continue
            sampled_psus = self.rng.choice(psus, size=len(psus), replace=True)
            for psu in sampled_psus:
                mask = (self.df[self.stratum_col] == stratum) & (self.df[self.psu_col] == psu)
                idx = np.where(mask.to_numpy())[0].tolist()
                bootstrap_indices.extend(idx)
        return bootstrap_indices


class BootstrapEngine:
    def __init__(self, config: ConfigSpec):
        self.config = config
        self.epi_keys = [
            "prev_true",
            "prev_pred",
            "abs_bias",
            "rel_bias",
            "sensitivity",
            "specificity",
            "infl_factor",
        ]

    @staticmethod
    def _aggregate_distribution(dist: List[float]) -> Dict[str, object]:
        arr = np.asarray(dist, dtype=float)
        if arr.size == 0:
            return {"ci_lower": np.nan, "ci_upper": np.nan, "bootstrap_dist": []}
        return {
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5)),
            "bootstrap_dist": arr.tolist(),
        }

    def run(
        self,
        df: pd.DataFrame,
        strategies: Dict[str, StrategySpec],
        y_true_col: str = "reference_true",
    ) -> Tuple[Dict[str, dict], List[dict]]:
        sampler = PSUBootstrapSampler(
            df=df,
            stratum_col=self.config.stratum_column,
            psu_col=self.config.psu_column,
            seed=self.config.random_seed,
        )

        strategy_names = list(strategies.keys())
        distributions = defaultdict(lambda: defaultdict(list))

        for _ in range(self.config.n_bootstrap):
            boot_idx = sampler.generate_indices()
            if not boot_idx:
                continue
            boot_df = df.iloc[boot_idx].copy()

            boot_w = boot_df[self.config.weight_column].to_numpy(dtype=float)
            if np.sum(boot_w) > 0:
                boot_w = boot_w / np.sum(boot_w) * len(boot_w)

            for strategy_name in strategy_names:
                y_pred_col = "{0}_pred".format(strategy_name)
                y_true = boot_df[y_true_col].to_numpy()
                y_pred = boot_df[y_pred_col].to_numpy()

                distributions["kappa_simple"][strategy_name].append(
                    weighted_kappa_simple(y_true, y_pred, boot_w)
                )
                distributions["kappa_quadratic"][strategy_name].append(
                    weighted_kappa_quadratic(y_true, y_pred, boot_w)
                )
                distributions["f1_macro"][strategy_name].append(weighted_f1_macro(y_true, y_pred, boot_w))

                epi_boot = compute_screening_metrics(
                    boot_df, self.config.weight_column, y_true_col, y_pred_col, positive_classes=[1, 2]
                )
                for key in self.epi_keys:
                    val = epi_boot[key]
                    if not np.isnan(val):
                        distributions["epi_{0}".format(key)][strategy_name].append(val)

        results: Dict[str, dict] = {}
        orig_w = df[self.config.weight_column].to_numpy(dtype=float)
        for strategy_name in strategy_names:
            y_pred_col = "{0}_pred".format(strategy_name)
            y_true = df[y_true_col].to_numpy()
            y_pred = df[y_pred_col].to_numpy()
            entry: Dict[str, object] = {}
            entry["metadata"] = {
                "name": strategy_name,
                "teeth_fdi": strategies[strategy_name].teeth_fdi,
                "strategy_type": strategies[strategy_name].strategy_type,
                "subgroup": strategies[strategy_name].subgroup,
            }

            kappa_simple_bundle = self._aggregate_distribution(distributions["kappa_simple"][strategy_name])
            kappa_simple_bundle["point_estimate"] = weighted_kappa_simple(y_true, y_pred, orig_w)
            entry["kappa_simple"] = kappa_simple_bundle

            kappa_quad_bundle = self._aggregate_distribution(distributions["kappa_quadratic"][strategy_name])
            kappa_quad_bundle["point_estimate"] = weighted_kappa_quadratic(y_true, y_pred, orig_w)
            entry["kappa_quadratic"] = kappa_quad_bundle

            f1_bundle = self._aggregate_distribution(distributions["f1_macro"][strategy_name])
            f1_bundle["point_estimate"] = weighted_f1_macro(y_true, y_pred, orig_w)
            entry["f1_macro"] = f1_bundle

            epi_point = compute_screening_metrics(
                df, self.config.weight_column, y_true_col, y_pred_col, positive_classes=[1, 2]
            )
            epi_ci = {}
            for key in self.epi_keys:
                dist = distributions["epi_{0}".format(key)][strategy_name]
                if dist:
                    epi_ci[key] = {
                        "ci_lower": float(np.percentile(dist, 2.5)),
                        "ci_upper": float(np.percentile(dist, 97.5)),
                        "bootstrap_dist": list(np.asarray(dist, dtype=float)),
                    }
                else:
                    epi_ci[key] = {"ci_lower": np.nan, "ci_upper": np.nan, "bootstrap_dist": []}
            entry["epi_metrics"] = epi_point
            entry["epi_metrics_ci"] = epi_ci
            results[strategy_name] = entry

        comparisons = self._pairwise_comparisons(results)
        return results, comparisons

    def _pairwise_comparisons(self, results: Dict[str, dict]) -> List[dict]:
        comparisons: List[dict] = []
        for strategy_1, strategy_2 in combinations(results.keys(), 2):
            r1 = results[strategy_1]
            r2 = results[strategy_2]
            test_kappa = bootstrap_pairwise_test(
                np.asarray(r1["kappa_quadratic"]["bootstrap_dist"], dtype=float),
                np.asarray(r2["kappa_quadratic"]["bootstrap_dist"], dtype=float),
            )
            test_if = bootstrap_pairwise_test(
                np.asarray(r1["epi_metrics_ci"]["infl_factor"]["bootstrap_dist"], dtype=float),
                np.asarray(r2["epi_metrics_ci"]["infl_factor"]["bootstrap_dist"], dtype=float),
            )

            comparisons.append(
                {
                    "strategy_1": strategy_1,
                    "strategy_2": strategy_2,
                    "kappa_quadratic_diff": test_kappa["difference"],
                    "kappa_quadratic_ci_lower": test_kappa["ci_lower"],
                    "kappa_quadratic_ci_upper": test_kappa["ci_upper"],
                    "kappa_quadratic_p": test_kappa["p_value"],
                    "kappa_quadratic_significant": test_kappa["significant"],
                    "infl_factor_diff": test_if["difference"],
                    "infl_factor_ci_lower": test_if["ci_lower"],
                    "infl_factor_ci_upper": test_if["ci_upper"],
                    "infl_factor_p": test_if["p_value"],
                    "infl_factor_significant": test_if["significant"],
                }
            )
        return comparisons
