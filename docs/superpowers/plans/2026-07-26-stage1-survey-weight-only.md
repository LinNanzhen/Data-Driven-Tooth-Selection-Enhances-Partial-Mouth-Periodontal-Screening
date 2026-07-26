# Stage 1 Survey-Weight-Only Variant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `Stage1_survey_weight_only/`, an isolated Stage 1 variant whose model fitting and Bayesian validation scoring use NHANES survey weights without class-balancing weights.

**Architecture:** Mechanically copy the revised selection-only `Stage1/` tree, then make three bounded changes inside the copy: add an index-aligned survey-weighted multiclass AUC scorer to the tuner, pass only survey weights to model fits, and remove all class-weight computation from CV and SHAP stability. Source-level tests protect the original tree and work without scientific Python dependencies; full model execution is deferred to the configured Windows environment.

**Tech Stack:** Python 3, pandas, NumPy, scikit-learn, scikit-optimize, XGBoost, LightGBM, SHAP, standard-library `unittest` and `ast`.

## Global Constraints

- Create the variant at exactly `Stage1_survey_weight_only/`.
- Do not modify `Stage1/` or `Stage2/` during variant implementation.
- Keep `weight_scale_factor=0.5` unchanged.
- Do not calculate or multiply `class_weight="balanced"` anywhere in the variant.
- Use NHANES survey weights for both model fitting and Bayesian validation macro-AUC.
- Keep the revised Stage 1 selection-only output contract and reserved hold-out export.
- Do not use Stage 2 hold-out results to choose between weighting strategies.
- Keep Bayesian-search fold-level parallelism and force each searched estimator to one thread.

---

### Task 1: Create and protect the isolated Stage 1 copy

**Files:**
- Create: `Stage1_survey_weight_only/` as a mechanical copy of `Stage1/`
- Create: `tests/test_stage1_survey_weight_variant.py`

**Interfaces:**
- Consumes: the current revised selection-only `Stage1/` tree.
- Produces: an isolated source tree at `Stage1_survey_weight_only/` and a test helper `source_digest(root: Path) -> dict[str, str]`.

- [ ] **Step 1: Write the failing isolation test**

```python
import hashlib
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = REPO_ROOT / "Stage1"
VARIANT = REPO_ROOT / "Stage1_survey_weight_only"


def source_digest(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


class SurveyWeightVariantTests(unittest.TestCase):
    def test_variant_exists_with_same_initial_file_set(self):
        self.assertTrue(VARIANT.is_dir())
        self.assertEqual(set(source_digest(ORIGINAL)), set(source_digest(VARIANT)))
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests/test_stage1_survey_weight_variant.py -v
```

Expected: FAIL because `Stage1_survey_weight_only/` does not exist.

- [ ] **Step 3: Copy Stage 1 mechanically**

Run:

```bash
cp -R Stage1 Stage1_survey_weight_only
```

Do not copy `.DS_Store` or `__pycache__` if either appears during execution.

- [ ] **Step 4: Run the isolation test**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests/test_stage1_survey_weight_variant.py -v
```

Expected: PASS before variant-specific source modifications.

- [ ] **Step 5: Record the original source digest for later protection**

Extend the test module with:

```python
ORIGINAL_DIGEST_BEFORE_VARIANT_CHANGES = source_digest(ORIGINAL)
```

Later tests compare the live original digest with this value inside the same test process.

- [ ] **Step 6: Commit the isolated copy and test**

```bash
git add Stage1_survey_weight_only tests/test_stage1_survey_weight_variant.py
git commit -m "test: scaffold survey-weight-only Stage1 variant"
```

---

### Task 2: Add a survey-weighted Bayesian macro-AUC scorer

**Files:**
- Modify: `Stage1_survey_weight_only/core/hyperopt.py`
- Modify: `tests/test_stage1_survey_weight_variant.py`

**Interfaces:**
- Produces: `SurveyWeightedOvRMacroAUC(sample_weight: pd.Series)`.
- Produces: `SurveyWeightedOvRMacroAUC.__call__(estimator, X: pd.DataFrame, y) -> float`.
- Preserves: `HyperparameterTuner.tune_model_bayes(...) -> Tuple[object, Dict]`.

- [ ] **Step 1: Add failing source and behavior tests**

Add an AST test asserting that variant `hyperopt.py` imports neither
`compute_sample_weight` nor `sklearn.utils.class_weight`.

Add a dependency-free behavior test that loads only the scorer definition from
the AST-compiled module with stubbed `numpy`, `pandas`, and `roc_auc_score`
objects. The fake metric must record `sample_weight`; assert that validation
rows `[20, 10]` receive weights `[4.0, 1.0]` in that order.

Also assert that duplicate survey-weight indices raise `ValueError`.

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests/test_stage1_survey_weight_variant.py -v
```

Expected: FAIL because the scorer does not exist and class-weight code remains.

- [ ] **Step 3: Implement the scorer**

Add these imports:

```python
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
```

Implement:

```python
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
```

- [ ] **Step 4: Wire the scorer and survey-only fit weights into Bayesian search**

Inside `tune_model_bayes`:

```python
search_scoring = (
    SurveyWeightedOvRMacroAUC(sample_weight)
    if sample_weight is not None
    else scoring
)
base_model.set_params(n_jobs=1)
```

Pass `search_scoring` to `BayesSearchCV`.

Replace class-balanced fit parameters with:

```python
fit_params = {}
if sample_weight is not None:
    fit_params["sample_weight"] = sample_weight
```

Keep:

```python
bayes_search.fit(X, y, **fit_params)
```

- [ ] **Step 5: Run the tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests/test_stage1_survey_weight_variant.py -v
```

Expected: all Task 1 and Task 2 tests PASS.

- [ ] **Step 6: Commit the weighted tuner**

```bash
git add Stage1_survey_weight_only/core/hyperopt.py tests/test_stage1_survey_weight_variant.py
git commit -m "feat: tune Stage1 variant with survey-weighted AUC"
```

---

### Task 3: Remove class balancing from CV and SHAP model fits

**Files:**
- Modify: `Stage1_survey_weight_only/core/evaluation.py`
- Modify: `Stage1_survey_weight_only/core/stability.py`
- Modify: `tests/test_stage1_survey_weight_variant.py`

**Interfaces:**
- Preserves all public `ModelPipeline` and `MultiSeedStabilityAnalyzer` method signatures.
- Changes fit semantics to `sample_weight=survey_weight` only.

- [ ] **Step 1: Add failing AST tests**

For every Python file below, assert that neither an import nor a call references
`compute_sample_weight`:

```python
[
    "core/hyperopt.py",
    "core/evaluation.py",
    "core/stability.py",
]
```

Assert that `evaluation.py` and `stability.py` still contain model `fit` calls
and still reference the fold-specific survey-weight variables.

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests/test_stage1_survey_weight_variant.py -v
```

Expected: FAIL because evaluation and stability still call
`compute_sample_weight`.

- [ ] **Step 3: Change repeated CV fitting**

In `evaluation.py`, remove the class-weight import and replace:

```python
class_sample_weights = compute_sample_weight(...)
fit_params = {"sample_weight": ...}
model.fit(x_train, y_train, **fit_params)
```

with:

```python
fit_params = {}
if w_train is not None:
    fit_params["sample_weight"] = w_train
model.fit(x_train, y_train, **fit_params)
```

- [ ] **Step 4: Change SHAP stability fitting and sampling**

In `stability.py`, remove the class-weight import. Replace model fit weighting
with:

```python
fit_params = {}
if sw_train_sub is not None:
    fit_params["sample_weight"] = sw_train_sub
```

Use survey weights alone for SHAP background sampling:

```python
bg_weights = (
    sw_train_sub.to_numpy()
    if sw_train_sub is not None
    else np.ones(len(x_train_sub))
)
x_sample, x_sample_weights = self._weighted_sample(
    x_train_sub,
    bg_weights,
    self.config.shap_sample_size,
    random_state=seed + fold_idx,
)
```

- [ ] **Step 5: Run all source-level tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit survey-only fitting**

```bash
git add Stage1_survey_weight_only/core/evaluation.py Stage1_survey_weight_only/core/stability.py tests/test_stage1_survey_weight_variant.py
git commit -m "feat: fit Stage1 variant with survey weights only"
```

---

### Task 4: Distinguish the variant and verify output boundaries

**Files:**
- Create: `Stage1_survey_weight_only/README.md`
- Modify: `tests/test_stage1_survey_weight_variant.py`

**Interfaces:**
- Produces operator instructions for local and Windows execution.
- Preserves output filenames while requiring execution from the variant
  directory or an explicitly distinct `base_dir`.

- [ ] **Step 1: Add failing documentation and output-boundary tests**

Assert that the variant README contains:

```text
survey weights only
Stage2 hold-out
do not compare variants on the hold-out
```

Parse every variant scenario and assert that it builds paths relative to
`base_dir`, preventing writes into `Stage1/` when run from the variant directory.

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests/test_stage1_survey_weight_variant.py -v
```

Expected: FAIL because the variant README does not exist.

- [ ] **Step 3: Write the variant README**

Document:

```text
Purpose: survey weights only
Run location: Stage1_survey_weight_only/
Input: cleaned_data.csv
Outputs: variant-local scenario directories
Stage2 hold-out: exported but not used to choose the weighting strategy
Windows command: python -m scenarios.overall
```

- [ ] **Step 4: Run all tests and compile changed Python files**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
python3 -c 'from pathlib import Path; files=list(Path("Stage1_survey_weight_only").rglob("*.py")); [compile(p.read_text(encoding="utf-8-sig"), str(p), "exec") for p in files]'
git diff --check
```

Expected: tests PASS, compilation exits 0, and `git diff --check` emits no
errors.

- [ ] **Step 5: Verify the original source has no variant diff**

Run:

```bash
git diff -- Stage1
```

Expected: only pre-existing Stage1 changes are shown; no line in that diff was
introduced by Tasks 1–4. Confirm with the source digest captured before copying.

- [ ] **Step 6: Commit variant documentation**

```bash
git add Stage1_survey_weight_only/README.md tests/test_stage1_survey_weight_variant.py
git commit -m "docs: explain survey-weight-only Stage1 workflow"
```

---

### Task 5: Run full scientific-stack verification on Windows

**Files:**
- No source changes expected.
- Inspect generated variant outputs and captured logs.

**Interfaces:**
- Consumes: committed `Stage1_survey_weight_only/`.
- Produces: Windows test logs, CV summaries, and tooth-stability artifacts.

- [ ] **Step 1: Synchronize the repository**

Preferred Git workflow on Windows:

```powershell
git fetch
git checkout 4.5
git pull
```

If work is placed on a new branch before synchronization, replace `4.5` with
that exact branch name.

- [ ] **Step 2: Create and activate an isolated Windows environment**

```powershell
py -m venv .venv-survey
.\.venv-survey\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

- [ ] **Step 3: Run tests with the scientific dependencies installed**

```powershell
python -m unittest discover -s tests -v
```

Expected: all tests PASS.

- [ ] **Step 4: Run a reduced smoke test**

Temporarily construct the scenario config in an interactive command with:

```python
config.bayesian_n_iter = 2
config.random_seeds = [42]
config.cv_folds = 2
```

Run the overall scenario against a small verified data extract. Confirm that:

- Bayesian search completes;
- no class-weight call appears in logs or source;
- weighted scorer receives validation weights;
- SHAP output is generated;
- the reserved hold-out file is exported;
- no Stage 1 model hold-out metrics are produced.

- [ ] **Step 5: Run the planned full analysis**

From `Stage1_survey_weight_only/`:

```powershell
python -m scenarios.overall
```

Then run the required subgroup scenarios using their existing module entry
points.

- [ ] **Step 6: Compare development-set stability only**

Compare original versus survey-only variants using:

- top-10 tooth overlap;
- rank correlation;
- per-tooth cross-seed selection frequency;
- CV metric sensitivity.

Do not inspect Stage 2 hold-out results until the variant and analysis protocol
are locked.
