# NHANES weighting sensitivity and participant flow

This supplementary analysis covers the overall cohort. It reproduces the published XGB10 and LGB10 sets as reference protocols and compares them with tooth sets derived under alternative Stage 1 weighting choices.

## Participant flow clarification

The published final counts are 6,116 participants for development (2009–2012) and 3,103 for external validation (2013–2014). The omitted flow detail concerns 1,064 participants with a zero MEC examination weight. None of these participants entered the three-table merge by participant ID. They were already included in the original reduction from 30,468 demographic records to 14,071 matched records.

| NHANES cycle | Demographic records | Zero MEC weight | Remaining after weight check | Additional records missing at least one input table | Three-table match | Final cohort |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2009–2010 | 10,537 | 284 | 10,253 | 5,216 | 5,037 | 3,250 |
| 2011–2012 | 9,756 | 418 | 9,338 | 4,973 | 4,365 | 2,866 |
| 2013–2014 | 10,175 | 362 | 9,813 | 5,144 | 4,669 | 3,103 |
| **Total** | **30,468** | **1,064** | **29,404** | **15,333** | **14,071** | **9,219** |

Suggested figure annotation for the existing merge step:

> Of the 16,397 participants who did not enter the linked demographic, dentition and periodontal dataset, 1,064 had a zero MEC examination weight (284 in 2009–2010, 418 in 2011–2012 and 362 in 2013–2014).

A figure with sequential boxes can show 30,468 → 29,404 → 14,071, with exclusions of 1,064 for zero MEC examination weight and 15,333 for the remaining unmatched records. The downstream box counts follow the original eligibility rules. The check on the source files matched the zero weights to interview-only status in each cycle, including the SAS XPORT zero encoding that pandas can decode as a tiny positive number. [2009–2010 codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/DEMO_F.htm) · [2011–2012 codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/DEMO_G.htm) · [2013–2014 codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/DEMO_H.htm)

## Stage 1 weighting

Both analysis arms use the same development split, search ranges, and unweighted five-fold macro one-vs-rest AUC for tuning. Each arm is tuned independently. The pooled development MEC weights are half the two-year weights, and both arms share the additional Stage 1 scale factor of 0.5 from the reproduction. [NHANES survey weighting guidance](https://wwwn.cdc.gov/nchs/nhanes/tutorials/Weighting.aspx)

- **NHANES only** uses the survey weight for each participant's contribution to the fitting loss.
- **NHANES plus class balance** multiplies the survey weight by a class factor computed within each fitting fold. The factor is total survey weight divided by the number of classes and by that class's survey-weight sum. The total fitting weight stays constant and the three classes contribute equal total weight. The original implementation instead derives its class factors from unweighted participant counts.

| Model | NHANES only CV AUC | NHANES plus balance CV AUC | NHANES only 10-tooth QWK | NHANES plus balance 10-tooth QWK |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | 0.99165 | 0.99167 | 0.86142 | 0.81818 |
| LightGBM | 0.99133 | 0.99103 | 0.85100 | 0.81707 |

The last two columns are survey-weighted QWK on the 1,224-person internal holdout for models refitted using their respective 10-tooth sets. Tuning AUC was similar between arms, while the NHANES-only 10-tooth models had higher holdout QWK. The XGBoost arms selected the same 10 teeth, and the LightGBM arms selected different tenth teeth. These are comparisons of independently tuned fitting policies, including any changes in selected hyperparameters. In an earlier XGBoost ablation with count-based class balancing and full-fold class-prior SHAP, switching from weighted to unweighted CV scoring retained all ten selected teeth, including tooth 45.

## SHAP weighting and tooth ranks

For each of 25 training folds (five seeds times five folds), the primary sensitivity analysis explains all rows used to fit that fold's model. For each feature and model output, it averages absolute raw-margin SHAP values with NHANES participant weights. It then gives each of the three output classes one-third of the feature score, sums PPD and CAL contributions by tooth, and averages the tooth scores across folds.

The class-prior sensitivity uses the same fitted models and explained rows with output classes weighted by their survey-weighted frequencies. For NHANES-only XGBoost this alone moves tooth 45 from rank 13 to 10 and tooth 44 from rank 10 to 16. With equal class output weights, tooth 44 narrowly outranked tooth 41 (0.36906 versus 0.36879). A reconstruction of the original Stage 1 with SHAP 0.54.1 recovered tooth 41 for XGB10. In that run, the original code gave output classes equal weight through an array-wide mean, and explained a sample of up to 1,000 fitting rows drawn with probabilities proportional to the combined fitting weights. Those same weights also entered the mean over the sampled rows, so higher-weight participants influenced both inclusion and aggregation. Its list-based SHAP branch applies class priors instead. In a controlled check of the NHANES-only XGBoost fit, the 1,000-row sampled analysis with equal class output weighting placed 41 tenth and 44 eleventh.

Under the full-fold, equal-class method, both XGBoost arms and the balanced LightGBM arm selected tooth 44 as the tenth tooth. NHANES-only LightGBM selected 25. The released XGB10 and LGB10 sets selected 41 and 25, respectively. All sets share FDI teeth 16, 17, 26, 27, 36, 37, 42, 46 and 47.

## Stage 2 external validation

The external evaluation applies the same CDC/AAP rule to measurements from each 10-tooth set and the 28-tooth reference in the 3,103-person NHANES 2013–2014 cohort. All point estimates use MEC survey weights.

| Tenth FDI tooth | How selected | QWK | Any periodontitis sensitivity | Severe periodontitis sensitivity |
| --- | --- | ---: | ---: | ---: |
| 41 | Original XGB10 | 0.84537 | 78.56% | 63.60% |
| 44 | Both XGBoost arms and balanced LightGBM, equal-class SHAP | 0.84666 | 78.14% | 65.12% |
| 25 | Original LGB10 and NHANES-only LightGBM | 0.82553 | 77.32% | 60.20% |
| 45 | Both XGBoost arms and balanced LightGBM, class-prior SHAP | 0.8369 | 77.85% | 62.81% |

With 8,000 paired bootstrap draws resampling PSUs within strata, the 44-minus-41 difference was +0.00129 QWK (95% CI −0.00976 to +0.01389), −0.42 percentage points for any-periodontitis sensitivity (−1.39 to +0.54), and +1.52 percentage points for severe-periodontitis sensitivity (−2.51 to +6.21). Each interval includes zero. The candidate rankings are sensitive near the tenth-tooth cutoff, and the paired external comparison does not establish superiority for replacing 41 with 44.

## Reproducing the analysis

Run from the repository root. The reported analysis used Python 3.12, NumPy 2.5.5, pandas 3.1.1, SciPy 1.18.1, scikit-learn 1.9.1, scikit-optimize 0.10.2, XGBoost 3.4.1, LightGBM 4.7.0 and SHAP 0.54.1.

```bash
python -m pip install -r requirements.txt
python scripts/reproduce_paper.py prepare
python scripts/reproduce_paper.py sensitivity
```

The cohort ledger is written to `data/processed/inclusion_flow.csv`. Outputs under `data/processed/stage1/` include tuning summaries, SHAP rankings, the internal model metrics, fixed-tooth screening metrics and the external paired bootstrap results. The analysis uses the repository's existing Stage 2 code and keeps the published protocols as references.
