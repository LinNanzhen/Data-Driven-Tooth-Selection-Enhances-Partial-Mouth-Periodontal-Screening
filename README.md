# Data-Driven Tooth Selection Enhances Partial-Mouth Periodontal Screening

Reproduction code for the [published study](https://doi.org/10.1177/00220345261432570) using NHANES 2009–2014. Stage 1 ranks teeth with XGBoost, LightGBM and SHAP. Stage 2 applies the CDC/AAP case definition to fixed tooth subsets and the 28-tooth reference.

## Run

Use Python 3.12. On macOS, the tree model packages require an OpenMP runtime. Use `brew install libomp` if the runtime is missing.

```bash
python -m pip install -r requirements.txt
python scripts/reproduce_paper.py stage1
python scripts/reproduce_paper.py stage2 --cohort both
```

The commands use the distributed `cleaned_data.csv` and `test_data.csv`. They run the overall, age or sex, and age-by-sex scenarios. Stage 1 writes CV and holdout metrics, SHAP rankings, figures and holdout records. Stage 2 writes screening metrics and pairwise comparisons with 8,000 stratified PSU bootstrap replicates per group. Internal evaluation uses each group's Stage 1 holdout. External evaluation uses NHANES 2013–2014 and includes CPI and modified Ramfjord as reference protocols.

Use `--scenario overall`, `--scenario subgroup_single_factor` or `--scenario subgroup_age_gender` to run one scenario. Stage 2 defaults to external validation with published tooth sets. Use `--tooth-sets stage1` to evaluate regenerated sets after running Stage 1 overall and the corresponding subgroup scenario. Outputs are written to `data/processed/paper_published/`. Add `--smoke` for a short run with separate outputs.

### Rebuild the data and run the sensitivity analysis

```bash
python scripts/reproduce_paper.py prepare
python scripts/reproduce_paper.py sensitivity
```

`prepare` downloads the public NHANES XPT files and writes the cohort flow, cleaned datasets and pooled baseline table. Use `--rebuilt-data` on both paper analysis commands to use those datasets. Those results go to `data/processed/paper_rebuilt/`. The overall-only `sensitivity` command uses the rebuilt data and writes its results under `data/processed/stage1/`.

## Reproduction statements

- **Participant flow.** Of the 16,397 participants excluded at the three-table linkage, 1,064 had zero MEC examination weight. These were 284, 418 and 362 participants in the three successive NHANES cycles. The final counts are 6,116 for development and 3,103 for external validation. The detailed flow and wording for the manuscript figure are given below. [Participant flow and weighting](docs/nhanes-weighting-and-cohort-flow.md#participant-flow-clarification)
- **Weighting.** The paper workflow retains the released fitting procedure and fixed protocols. The supplementary workflow compares NHANES-only and NHANES-plus-class-balance fitting, with separate SHAP aggregation choices. The external paired comparison did not establish superiority of the 44-based set over the published 41-based XGB10 set. [Sensitivity results](docs/nhanes-weighting-and-cohort-flow.md#stage-2-external-validation)
- **Reference checks.** The released external XGB10 returns QWK 0.84537 and IF 127.294%. With the pinned environment and 8,000 bootstrap draws, the CPI comparison returns `P = 0.0035` for IF. The article reports `P = 0.003`. [Published article](https://doi.org/10.1177/00220345261432570)
- **Validation scope.** Formal external Stage 2 runs were checked for the overall cohort and all 17 subgroups. The 55–64-year-old male XGBoost tooth set matches the abstract. The complete Stage 1 subgroup retraining and exact agreement with every published table and figure remain unverified.

The current command-line workflow succeeds the six notebook entry points archived at [the original 4.5 commit](https://github.com/LinNanzhen/Data-Driven-Tooth-Selection-Enhances-Partial-Mouth-Periodontal-Screening/tree/8faa8e156d497dd59d54b525eca25099ff902d1a).
