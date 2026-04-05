from __future__ import annotations

import gc
from typing import Dict, Tuple

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

try:
    from skopt import BayesSearchCV

    SKOPT_AVAILABLE = True
except Exception:
    SKOPT_AVAILABLE = False
    BayesSearchCV = None


class HyperparameterTuner:
    def __init__(self, cv_folds: int = 5, random_state: int = 42, n_iter: int = 30):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_iter = n_iter

    def tune_model_bayes(
        self,
        model_name: str,
        base_model,
        X,
        y,
        sample_weight,
        param_space: Dict,
        scoring: str = "roc_auc_ovr",
    ) -> Tuple[object, Dict]:
        if not SKOPT_AVAILABLE:
            return base_model, {}

        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )
        bayes_search = BayesSearchCV(
            base_model,
            param_space,
            n_iter=self.n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=self.random_state,
            verbose=0,
        )

        class_sample_weights = compute_sample_weight(class_weight="balanced", y=y)
        fit_params = {
            "sample_weight": sample_weight * class_sample_weights
            if sample_weight is not None
            else class_sample_weights
        }

        bayes_search.fit(X, y, **fit_params)
        tuned_model = base_model.set_params(**bayes_search.best_params_)
        gc.collect()
        return tuned_model, bayes_search.best_params_
