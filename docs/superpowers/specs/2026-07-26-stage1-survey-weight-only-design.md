# Stage 1 Survey-Weight-Only Variant Design

## Purpose

Create an isolated Stage 1 variant that estimates tooth importance for the US
target population using NHANES survey weights only. The existing `Stage1/`
directory remains unchanged.

## Scope

The new variant will be created at:

```text
Stage1_survey_weight_only/
```

It will preserve the existing Stage 1 scenarios, preprocessing, model
definitions, SHAP stability analysis, output structure, and reserved hold-out
export. Only weight handling and the Bayesian-search scorer will change.

Stage 2 will not be copied or modified.

## Weighting Rules

### Model fitting

Every XGBoost and LightGBM fit in the variant will receive only the corresponding
NHANES survey weight:

```text
fit sample weight = NHANES survey weight
```

No `class_weight="balanced"` weight will be calculated or multiplied into the
survey weight. When no survey weight is available, the estimator will be fitted
without `sample_weight`.

The existing `weight_scale_factor=0.5` remains unchanged because the current
configuration treats the input as two combined 2-year NHANES cycles. Changing
that assumption is outside this variant's scope.

### Bayesian-search validation score

The Bayesian search will maximize survey-weighted one-vs-rest macro-AUC on each
validation fold. The scorer will:

1. predict class probabilities for the validation fold;
2. align probability columns with `estimator.classes_`;
3. select validation survey weights by the preserved pandas row index;
4. call multiclass `roc_auc_score` with those validation weights.

The scorer will be a top-level callable object rather than a nested closure so
that joblib can serialize it during parallel search.

Inputs to weighted Bayesian search must therefore use:

- a pandas DataFrame with a unique index for `X`;
- a target Series aligned to that index;
- a survey-weight Series aligned to that index.

The tuner will raise a clear error when weighted scoring is requested with
misaligned or duplicate indices. It will not silently fall back to unweighted
scoring.

### Cross-validation and SHAP stability

The repeated CV evaluator and SHAP stability analyzer will fit models with the
training fold's survey weights only. Their validation metrics will continue to
use the validation fold's survey weights.

## Dependency and Parallelism Boundaries

The variant will keep `scikit-optimize` as the Bayesian-search implementation.
If it is unavailable, the existing fallback behavior will remain unchanged in
this change; making missing tuning dependencies fatal is a separate decision.

To prevent nested parallelism, model-level training during Bayesian search will
use one thread per fit while `BayesSearchCV` controls fold-level parallelism.

## Outputs and Interpretation

The variant will keep the same filenames and result schema as the revised
selection-only Stage 1 pipeline:

- development-set CV metrics;
- tooth-level SHAP stability outputs;
- consensus tooth lists;
- reserved hold-out rows for Stage 2.

Outputs must be written to directories distinct from the original Stage 1
outputs so that the two weighting strategies cannot overwrite each other.

The Stage 2 hold-out must not be used to choose between the original and
survey-weight-only variants. The comparison between variants will use only
development-set tooth-ranking stability, such as top-10 overlap, rank
correlation, and cross-seed selection frequency.

## Verification

Automated checks will verify that:

1. the original `Stage1/` source remains unchanged by the variant work;
2. no variant training path imports or calls `compute_sample_weight`;
3. training receives survey weights without class-weight multiplication;
4. the Bayesian scorer changes when validation survey weights change;
5. the scorer aligns probabilities through `estimator.classes_`;
6. misaligned or duplicate weight indices raise clear errors;
7. the variant retains reserved hold-out export but has no Stage 1 hold-out
   model-evaluation API;
8. all changed Python files compile.

Because the current shell Python lacks the project's scientific dependencies,
full XGBoost, LightGBM, and scikit-optimize execution will be run on the user's
configured Windows environment after the source-level and dependency-free tests
pass locally.

## Windows Test Workflow

The preferred workflow is Git synchronization:

1. make the variant available on a dedicated branch;
2. pull that branch on Windows;
3. run the variant in the Windows scientific Python environment;
4. return logs and generated summaries for review.

Direct remote execution is optional. It requires Windows OpenSSH Server, key
authentication, and network reachability through the local network or Tailscale.
Passwords will not be requested or stored.
