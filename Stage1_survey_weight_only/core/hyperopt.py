from __future__ import annotations

import gc
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from skopt import BayesSearchCV

    SKOPT_AVAILABLE = True
except Exception:
    SKOPT_AVAILABLE = False
    BayesSearchCV = None


class SurveyWeightedOvRMacroAUC:
    def __init__(self, sample_weight: pd.Series):
        if not isinstance(sample_weight, pd.Series):
            raise TypeError("sample_weight must be a pandas Series")
        if not sample_weight.index.is_unique:
            raise ValueError("sample_weight index must be unique")
        if sample_weight.isna().any() or not np.isfinite(sample_weight.to_numpy()).all():
            raise ValueError("sample_weight must contain only finite values")
        if (sample_weight < 0).any():
            raise ValueError("sample_weight must be non-negative")
        self.sample_weight = sample_weight.copy()

    def __call__(self, estimator, X, y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("weighted scoring requires X to be a pandas DataFrame")
        if not X.index.is_unique:
            raise ValueError("validation index must be unique")
        missing = X.index.difference(self.sample_weight.index)
        if not missing.empty:
            raise ValueError("validation rows are missing survey weights")

        validation_weight = self.sample_weight.loc[X.index].to_numpy()
        probabilities = estimator.predict_proba(X)
        return roc_auc_score(
            y,
            probabilities,
            labels=estimator.classes_,
            multi_class="ovr",
            average="macro",
            sample_weight=validation_weight,
        )


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
        search_scoring = (
            SurveyWeightedOvRMacroAUC(sample_weight)
            if sample_weight is not None
            else scoring
        )
        base_model.set_params(n_jobs=1)
        bayes_search = BayesSearchCV(
            base_model,
            param_space,
            n_iter=self.n_iter,
            cv=cv,
            scoring=search_scoring,
            n_jobs=-1,
            random_state=self.random_state,
            verbose=0,
        )

        fit_params = {}
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight

        bayes_search.fit(X, y, **fit_params)
        tuned_model = base_model.set_params(**bayes_search.best_params_)
        gc.collect()
        return tuned_model, bayes_search.best_params_
