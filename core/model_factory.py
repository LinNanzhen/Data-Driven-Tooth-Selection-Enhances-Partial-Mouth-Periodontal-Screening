from __future__ import annotations

from typing import Dict, List

XGB_AVAILABLE = True
LGB_AVAILABLE = True

try:
    import xgboost as xgb
except Exception:
    XGB_AVAILABLE = False
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    LGB_AVAILABLE = False
    lgb = None


def availability_flags() -> Dict[str, bool]:
    return {
        "XGBoost": XGB_AVAILABLE,
        "LightGBM": LGB_AVAILABLE,
    }


def available_models(preferred: List[str] | None = None) -> List[str]:
    selected = preferred or ["XGBoost", "LightGBM"]
    flags = availability_flags()
    return [name for name in selected if flags.get(name, False)]


def create_model(model_name: str, random_state: int = 42, **kwargs):
    if model_name == "XGBoost":
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost is not available")
        return xgb.XGBClassifier(
            objective="multi:softprob",
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="mlogloss",
            n_jobs=1,
            verbosity=0,
            **kwargs,
        )

    if model_name == "LightGBM":
        if not LGB_AVAILABLE:
            raise ImportError("LightGBM is not available")
        return lgb.LGBMClassifier(
            objective="multiclass",
            random_state=random_state,
            n_jobs=-1,
            class_weight=None,
            verbose=-1,
            **kwargs,
        )

    raise ValueError("Unknown model: {0}".format(model_name))
