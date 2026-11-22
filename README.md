# Data-Driven Tooth Selection Enhances Partial-Mouth Periodontal Screening

This repository implements a two-stage (discovery and evaluation) pipeline for optimising partial-mouth periodontal examination (PMPE) schemes using NHANES data.

- **Stage 1** uses gradient-boosting models (XGBoost / LightGBM) as a data-mining tool to identify candidate tooth subsets via SHAP feature importance.
- **Stage 2** evaluates these subsets with the deterministic CDC/AAP case-definition rules -- the same rules used to generate the gold-standard labels -- rather than with the ML models themselves.

## Repository Structure

```
├── requirements.txt                 # Dependencies for both stages
├── Stage1/                          # Tooth selection (SHAP-based)
│   ├── core/                        # Library modules
│   │   ├── config.py                #   ScenarioConfig, hyperparameter spaces
│   │   ├── data_processing.py       #   NHANES→FDI mapping, feature extraction
│   │   ├── evaluation.py            #   ModelPipeline (CV, test-set evaluation)
│   │   ├── hyperopt.py              #   Bayesian optimisation wrapper
│   │   ├── model_factory.py         #   XGBoost / LightGBM factory
│   │   ├── pipeline.py              #   run_overall(), run_subgroup()
│   │   ├── reporting.py             #   CSV / plot export helpers
│   │   └── stability.py             #   Multi-seed SHAP aggregation
│   ├── scenarios/                   # Thin scenario wrappers
│   │   ├── overall.py               #   Global (all ages/genders)
│   │   ├── subgroup_single_factor.py#   Age-only or gender-only
│   │   └── subgroup_age_gender.py   #   Age × gender cross subgroups
│   ├── Global_teeth_selection.ipynb
│   ├── Subgroup_gender_or_age_teeth_selection.ipynb
│   └── Subgroup_gender_and_age_teeth_selection.ipynb
│
├── Stage2/                          # CDC/AAP rule-based evaluation
│   ├── core/                        # Library modules
│   │   ├── types.py                 #   Dataclasses (AnalysisResult, ConfigSpec, …)
│   │   ├── config.py                #   NHANES→FDI mapping constants
│   │   ├── data_processing.py       #   CDC classification, subgroup creation
│   │   ├── classification.py        #   CDCClassifier wrapper
│   │   ├── metrics.py               #   QWK, weighted F1, confusion matrix
│   │   ├── bootstrap.py             #   PSU-stratified bootstrap engine
│   │   ├── evaluation.py            #   StrategyEvaluator, strategy factories
│   │   ├── export.py                #   Excel output writers (metrics + significance)
│   │   └── pipeline.py              #   run_analysis() main entry point
│   ├── scenarios/                   # Thin scenario wrappers
│   │   ├── overall.py               #   Global evaluation
│   │   ├── subgroup_single_factor.py#   Age-only or gender-only
│   │   └── subgroup_age_gender.py   #   Age × gender cross subgroups
│   ├── Global_teeth_test.ipynb
│   ├── Subgroup_gender_or_age_teeth_test.ipynb
│   └── Subgroup_gender_and_age_teeth_test.ipynb
```

## Stage 1 -- SHAP-Based Tooth Selection

1. **Data loading & preprocessing** -- read NHANES periodontal measurements, map NHANES tooth to FDI notation, extract interproximal-site features per tooth.
2. **Bayesian hyperparameter optimisation** -- tune XGBoost and LightGBM with `scikit-optimize` over scenario-specific search spaces.
3. **Multi-seed SHAP extraction** -- train each model across multiple random seeds, compute SHAP values, aggregate site-level importance into tooth-level rankings.
4. **Consensus tooth ranking** -- derive a stable top-10 tooth set from the multi-seed results.
5. **Stratified CV + hold-out evaluation** -- report CV and test-set ML metrics (primary reported results come from Stage 2).

### Key Outputs

- Consensus tooth lists per model and subgroup
- SHAP importance rankings
- CV and test-set metrics (accuracy, weighted F1, QWK)

## Stage 2 -- CDC/AAP Rule-Based Evaluation

1. **Strategy construction** -- build full-mouth and partial-mouth examination strategies from Stage 1 tooth subsets, plus established reference protocols (CPI, Ramfjord).
2. **CDC/AAP classification** -- apply deterministic case-definition rules to each strategy's tooth subset.
3. **PSU-stratified bootstrap** -- survey-weighted resampling to obtain confidence intervals for all metrics.
4. **Pairwise statistical comparison** -- bootstrap-based hypothesis tests comparing each partial-mouth strategy against the full-mouth gold standard.
5. **Excel export** -- two spreadsheets per scenario: metric summary and pairwise significance tests.

### Key Outputs

- `metrics_summary.xlsx` -- QWK,  sensitivity, specificity, prevalence (true/predicted), absolute & relative bias, inflation factor -- each with 95% bootstrap CI
- `pairwise_significance.xlsx` -- bootstrap *p*-values and CIs for QWK and IF differences between every strategy pair


## Notebooks

| Stage | Notebook | Description |
|-------|----------|-------------|
| 1 | `Global_teeth_selection.ipynb` | Full-population tooth ranking |
| 1 | `Subgroup_gender_or_age_teeth_selection.ipynb` | Rankings by age or gender |
| 1 | `Subgroup_gender_and_age_teeth_selection.ipynb` | Cross-stratified rankings (e.g. 35-44 Male) |
| 2 | `Global_teeth_test.ipynb` | Full-population CDC evaluation |
| 2 | `Subgroup_gender_or_age_teeth_test.ipynb` | Evaluation by age or gender |
| 2 | `Subgroup_gender_and_age_teeth_test.ipynb` | Cross-stratified CDC evaluation |

## Data

This project uses public NHANES (National Health and Nutrition Examination Survey) periodontal examination data. 

1. Download oral health examination data (OHX periodontal) and demographics from [NHANES](https://wwwn.cdc.gov/nchs/nhanes/).
2. Stage 1 expects `Stage1/cleaned_data.csv`; Stage 2 expects `Stage2/test_data.csv`.
3. Run Stage 1 to generate the holdout records used by internal Stage 2.

## Reproduce the Stage 1 and Stage 2 analyses

Run from the repository root with Python 3.12. On macOS, install the LightGBM OpenMP runtime with `brew install libomp` if needed. [LightGBM installation guide](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html#macos)

```bash
python -m pip install -r scripts/requirements-weighting.txt
python scripts/reproduce_paper.py stage1
python scripts/reproduce_paper.py stage2 --cohort both
```

Both commands run the overall, age or sex, and age-by-sex scenarios. Stage 1 exports tuning summaries, SHAP rankings, model metrics, plots and holdout records. Stage 2 applies the CDC/AAP rule and exports screening metrics, PSU bootstrap confidence intervals and pairwise comparisons for the published tooth sets. For internal evaluation, it reads the matching Stage 1 holdout for each group.

Use `--scenario overall`, `--scenario subgroup_single_factor` or `--scenario subgroup_age_gender` to run one scenario. Stage 2 defaults to the external cohort and accepts `--tooth-sets stage1` to evaluate the regenerated tooth sets. Run Stage 1 overall before evaluating regenerated sets for subgroups. Outputs are written to `data/processed/paper_published/`. The `--smoke` option writes a separate short run under `smoke/`. Internal subgroups below the existing Stage 2 minimum sample size are omitted.

To rebuild the cohorts and export the pooled baseline characteristics from public NHANES files:

```bash
python scripts/download_nhanes.py
python scripts/prepare_data.py
python scripts/baseline_statistics.py
```

Pass `--rebuilt-data` to both analysis commands to use these cohorts and write results to `data/processed/paper_rebuilt/`.

With the distributed data and the default 8,000 bootstrap replicates, the overall external XGB10 run returns QWK 0.84537 and IF 127.294%. The pairwise IF comparison with CPI returns `P = 0.0035`. The article reports `P = 0.003`. [Published article](https://doi.org/10.1177/00220345261432570)

## Limitations

### 1. Greedy top-k approximation

The consensus teeth are selected by ranking individual tooth-level SHAP values and taking the top-10 tooth. Exhaustive subset search was not pursued because the combinatorial space ($2^{28}-1$) is prohibitive and the paper evaluates performance across varying *k*, making a single "optimal *k*" search less directly relevant. However, this approach constitutes a heuristic rather than a globally optimal subset selection. Future work could explore more principled subset selection strategies (e.g., wrapper-based optimisation or embedded feature selection methods such as regularisation-based approaches or all-relevant selection frameworks) to better align feature selection with downstream epidemiological objectives.

### 2. Objective inconsistency between Stage 1 and Stage 2

Stage 1 optimises macro-AUC (via Bayesian hyperparameter search), while Stage 2 evaluates with quadratic weighted kappa (QWK) and inflation factor (IF). The SHAP-ranked teeth are therefore a heuristic solution rather than a guaranteed optimum under CDC/AAP metrics. AUC optimisation was chosen because it yields stable, well-calibrated probability estimates that facilitate interpretable SHAP attribution, while QWK/IF are better suited for measuring partial-mouth vs. full-mouth agreement.

### 3. Internal test-set reuse

The 20% hold-out split from NHANES 2009--2012 is used to report both ML predictive metrics (Stage 1) and CDC-rule metrics (Stage 2). The released Stage 1 code derives SHAP consensus teeth within the training split, and internal Stage 2 evaluates within the same NHANES cycles. The primary generalisation evidence in the manuscript comes from the external temporal validation on NHANES 2013--2014.

### 4. SHAP sensitivity to collinearity

Periodontal measurements exhibit strong spatial correlation between neighbouring teeth. SHAP may distribute importance across correlated sites, meaning a highly ranked tooth could partly reflect an adjacent tooth's contribution. The multi-seed stability analysis (25 runs) and cross-model consensus (XGBoost + LightGBM) mitigate but do not eliminate this issue.

### 5. Data availability and generalisability

The study is constrained by the limited availability of large-scale periodontal datasets with tooth-level PD and CAL measurements. Subgroup analyses (e.g., by age and sex) are subject to sample size and reduced statistical power, which may affect the stability of CDC/AAP classification and agreement metrics within strata.


## Supplementary analysis

See the [NHANES weighting sensitivity and participant flow](docs/nhanes-weighting-and-cohort-flow.md) for the overall-cohort weighting comparisons, external validation, and the zero-MEC-weight exclusion counts.
