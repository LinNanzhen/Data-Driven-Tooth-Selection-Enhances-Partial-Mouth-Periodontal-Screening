# Stage 1: Survey Weights Only

This directory is an isolated variant of `Stage1/`. It estimates model
parameters and tooth importance for the US target population using NHANES
survey weights only. It does not calculate or multiply class-balancing weights.

## Weighting

- Bayesian-search model fitting uses the NHANES survey weights.
- Bayesian-search validation uses survey-weighted one-vs-rest macro-AUC.
- Repeated CV and SHAP stability fits use the training fold's survey weights.
- CV metrics and SHAP aggregation use the corresponding survey weights.
- `weight_scale_factor=0.5` remains unchanged for two combined 2-year cycles.

## Running

Place the prepared input at:

```text
Stage1_survey_weight_only/cleaned_data.csv
```

Run from this directory so scenario outputs remain separate from the original
Stage 1 outputs:

```bash
cd Stage1_survey_weight_only
python -m scenarios.overall
python -m scenarios.subgroup_single_factor
python -m scenarios.subgroup_age_gender
```

On Windows PowerShell, use the same commands after activating the project's
scientific Python environment.

## Stage2 hold-out

The pipeline still reserves and exports hold-out rows for the locked Stage2
CDC/AAP rule-based evaluation. Stage 1 does not fit or evaluate a model on those
rows.

Do not compare variants on the hold-out to decide whether the original or
survey-weight-only Stage 1 method should be used. Compare their tooth rankings
and stability on development data, lock the analysis choice, and only then run
the Stage2 hold-out evaluation.
