from __future__ import annotations

from typing import Dict, Iterable, List, Union

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


LABELS = [0, 1, 2]


def weighted_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    labels: Iterable[int] = LABELS,
) -> np.ndarray:
    labels = list(labels)
    cm = np.zeros((len(labels), len(labels)), dtype=float)
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            mask = (y_true == true_label) & (y_pred == pred_label)
            cm[i, j] = np.sum(weights[mask])
    return cm


def weighted_kappa_simple(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    cm = weighted_confusion_matrix(y_true, y_pred, weights, LABELS)
    denom = cm.sum()
    if denom <= 0:
        return 0.0
    po = np.trace(cm) / denom
    marginal_true = cm.sum(axis=1) / denom
    marginal_pred = cm.sum(axis=0) / denom
    pe = np.sum(marginal_true * marginal_pred)
    return float((po - pe) / (1 - pe)) if (1 - pe) > 0 else 0.0


def weighted_kappa_quadratic(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    cm = weighted_confusion_matrix(y_true, y_pred, weights, LABELS)
    denom = cm.sum()
    if denom <= 0:
        return 0.0

    n = len(LABELS)
    weight_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            weight_matrix[i, j] = 1 - ((i - j) ** 2) / ((n - 1) ** 2)

    weighted_cm = cm * weight_matrix
    po_weighted = weighted_cm.sum() / denom

    marginal_true = cm.sum(axis=1) / denom
    marginal_pred = cm.sum(axis=0) / denom
    expected_matrix = np.outer(marginal_true, marginal_pred)
    weighted_expected = expected_matrix * weight_matrix
    pe_weighted = weighted_expected.sum()

    return float((po_weighted - pe_weighted) / (1 - pe_weighted)) if (1 - pe_weighted) > 0 else 0.0


def weighted_f1_macro(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", sample_weight=weights, zero_division=0))


def compute_screening_metrics(
    df: pd.DataFrame,
    weight_col: str,
    y_true_col: str,
    y_pred_col: str,
    positive_classes: List[int],
) -> Dict[str, float]:
    w = df[weight_col].to_numpy(dtype=float)
    y_true = df[y_true_col].to_numpy()
    y_pred = df[y_pred_col].to_numpy()

    pos_true = np.isin(y_true, positive_classes)
    pos_pred = np.isin(y_pred, positive_classes)

    prev_true = np.average(pos_true.astype(int), weights=w) * 100.0
    prev_pred = np.average(pos_pred.astype(int), weights=w) * 100.0
    abs_bias = abs(prev_pred - prev_true)
    rel_bias = ((prev_pred - prev_true) / prev_true * 100.0) if prev_true != 0 else 0.0

    tp = w[(pos_true) & (pos_pred)].sum()
    fn = w[(pos_true) & (~pos_pred)].sum()
    tn = w[(~pos_true) & (~pos_pred)].sum()
    fp = w[(~pos_true) & (pos_pred)].sum()

    sensitivity = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    specificity = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0

    if prev_pred > 0:
        infl_factor = prev_true / prev_pred * 100.0
    else:
        infl_factor = 9999.0 if prev_true > 0 else 100.0

    return {
        "prev_true": float(prev_true),
        "prev_pred": float(prev_pred),
        "abs_bias": float(abs_bias),
        "rel_bias": float(rel_bias),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "infl_factor": float(infl_factor),
    }


def bootstrap_pairwise_test(dist_a: np.ndarray, dist_b: np.ndarray, alpha: float = 0.05) -> Dict[str, Union[float, bool]]:
    dist_a = np.asarray(dist_a, dtype=float)
    dist_b = np.asarray(dist_b, dtype=float)
    n = min(len(dist_a), len(dist_b))
    if n == 0:
        return {
            "difference": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
            "significant": False,
        }

    diff = dist_a[:n] - dist_b[:n]
    mean_diff = np.mean(diff)
    ci_lower, ci_upper = np.percentile(diff, [2.5, 97.5])
    p_value = 2 * min(np.mean(diff > 0), np.mean(diff < 0))
    return {
        "difference": float(mean_diff),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_value": float(p_value),
        "significant": bool(ci_lower > 0 or ci_upper < 0),
    }
