import re

import numpy as np
import pandas as pd

from prepare_data import MIN_AGE, OUT_DIR, TOOTH_NUMBERS

DEVELOPMENT_CSV = OUT_DIR / "cleaned_data.csv"
EXTERNAL_CSV = OUT_DIR / "test_data.csv"
STAGE1_OUTPUT_DIR = OUT_DIR / "stage1"
TARGET_COLUMN = "perio_label_cdc"
LABEL_NAMES = {0: "Healthy", 1: "Other", 2: "Severe"}
STAGE1_WEIGHT_FACTOR = 0.5  # Retain the original Stage 1 reproduction setting.
INTERNAL_TEST_SIZE = 0.2
SPLIT_SEED = 42
INTERPROXIMAL_SITES = "ADPS"
METRICS = ("PC", "LA")
FEATURE_SITES = {
    f"OHX{tooth:02d}{metric}_max": [
        f"OHX{tooth:02d}{metric}{site}" for site in INTERPROXIMAL_SITES
    ]
    for metric in METRICS
    for tooth in TOOTH_NUMBERS
}
MEASUREMENT_COLUMNS = [column for sites in FEATURE_SITES.values() for column in sites]
MAX_FEATURE = re.compile(r"OHX([0-9]{2})(?:PC|LA)_max")
# Permanent teeth in NHANES order from 01 to 32. The converter also covers third molars.
FDI_IN_NHANES_ORDER = (
    "18", "17", "16", "15", "14", "13", "12", "11",
    "21", "22", "23", "24", "25", "26", "27", "28",
    "38", "37", "36", "35", "34", "33", "32", "31",
    "41", "42", "43", "44", "45", "46", "47", "48",
)
NHANES_TO_FDI = {
    f"{tooth:02d}": fdi for tooth, fdi in enumerate(FDI_IN_NHANES_ORDER, start=1)
}


def load_data(filepath=DEVELOPMENT_CSV):
    """Read the cleaned development cohort and apply the Stage 1 filters."""
    data = pd.read_csv(filepath, low_memory=False)
    required = ["SEQN", "RIDAGEYR", "WTMEC2YR", *MEASUREMENT_COLUMNS]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"{filepath}: missing required columns: {', '.join(missing)}")
    if data["SEQN"].isna().any() or data["SEQN"].duplicated().any():
        raise ValueError(f"{filepath}: SEQN must be present and unique")

    ages = pd.to_numeric(data["RIDAGEYR"])
    data = data.loc[np.isfinite(ages) & ages.ge(MIN_AGE)]
    weights = pd.to_numeric(data["WTMEC2YR"])
    data = data.loc[np.isfinite(weights) & weights.gt(0)]
    measurements = data[MEASUREMENT_COLUMNS].apply(pd.to_numeric, errors="coerce")
    cleaned = pd.concat(
        [data.drop(columns=MEASUREMENT_COLUMNS), measurements.replace(99, np.nan)],
        axis=1,
    )[data.columns].copy()
    cleaned["WTMEC2YR"] = weights.loc[data.index] * STAGE1_WEIGHT_FACTOR
    return cleaned


def build_features(data):
    """Return one PPD and one CAL maximum per tooth across the four interproximal sites."""
    return pd.DataFrame(
        {feature: data[sites].max(axis=1) for feature, sites in FEATURE_SITES.items()},
        index=data.index,
    )


def classify_periodontitis(features):
    """Apply the original Stage 1 rule to the 56 features from build_features."""
    ppd = features.filter(regex=r"PC_max$")
    cal = features.filter(regex=r"LA_max$")
    deep_pocket = ppd.ge(5).any(axis=1)
    severe = cal.ge(6).sum(axis=1).ge(2) & deep_pocket
    other = cal.ge(3).sum(axis=1).ge(2) & (ppd.ge(4).sum(axis=1).ge(2) | deep_pocket)
    labels = np.select([severe, other], [2, 1], default=0)
    return pd.Series(labels, index=features.index, name=TARGET_COLUMN)


def nhanes_to_fdi(nhanes_tooth):
    """Convert a permanent NHANES tooth number to a two-character FDI code."""
    key = str(nhanes_tooth).strip().zfill(2)
    if key not in NHANES_TO_FDI:
        raise ValueError(f"Unknown NHANES permanent tooth number: {nhanes_tooth!r}")
    return NHANES_TO_FDI[key]


def extract_tooth_numbers_from_features(feature_names):
    teeth = set()
    for feature in feature_names:
        match = MAX_FEATURE.fullmatch(feature)
        if match:
            teeth.add(nhanes_to_fdi(match.group(1)))
    return sorted(teeth)


def prepare_stage1_data(filepath=DEVELOPMENT_CSV):
    """Return the labeled cohort and its aligned feature matrix."""
    cohort = load_data(filepath)
    features = build_features(cohort)
    cohort[TARGET_COLUMN] = classify_periodontitis(features)
    return cohort, features


def build_quality_reports(cohort, features):
    """Summarize label prevalence and interproximal measurement coverage."""
    labels = cohort.groupby(TARGET_COLUMN)["WTMEC2YR"].agg(n="size", weight_sum="sum")
    labels = labels.reindex(LABEL_NAMES, fill_value=0)
    labels.insert(0, "label_name", labels.index.map(LABEL_NAMES))
    labels["unweighted_pct"] = 100 * labels["n"] / len(cohort)
    labels["weighted_pct"] = 100 * labels["weight_sum"] / labels["weight_sum"].sum()

    tooth_rows = []
    for tooth in TOOTH_NUMBERS:
        ppd_missing = features[f"OHX{tooth:02d}PC_max"].isna()
        cal_missing = features[f"OHX{tooth:02d}LA_max"].isna()
        both_missing = ppd_missing & cal_missing
        tooth_rows.append({
            "nhanes_tooth": f"{tooth:02d}",
            "fdi_tooth": nhanes_to_fdi(tooth),
            "n_participants": len(cohort),
            "ppd_missing_n": ppd_missing.sum(),
            "ppd_missing_pct": 100 * ppd_missing.mean(),
            "cal_missing_n": cal_missing.sum(),
            "cal_missing_pct": 100 * cal_missing.mean(),
            "both_missing_n": both_missing.sum(),
            "both_missing_pct": 100 * both_missing.mean(),
        })

    flagged = cohort.loc[features.isna().all(axis=1), ["SEQN", TARGET_COLUMN]]
    return labels.reset_index(), pd.DataFrame(tooth_rows), flagged


def split_cohort(cohort, external_csv=EXTERNAL_CSV):
    """Return the stratified split in training order after checking external IDs."""
    from sklearn.model_selection import train_test_split

    external_ids = pd.read_csv(external_csv, usecols=["SEQN"])["SEQN"]
    overlap = cohort["SEQN"].isin(external_ids).sum()
    if overlap:
        raise ValueError(f"Development and external validation cohorts share {overlap:,} SEQN")

    participants = cohort[["SEQN", TARGET_COLUMN]]
    train, internal_test = train_test_split(
        participants,
        test_size=INTERNAL_TEST_SIZE,
        random_state=SPLIT_SEED,
        stratify=participants[TARGET_COLUMN],
    )
    return pd.concat(
        [train.assign(split="train"), internal_test.assign(split="internal_test")],
        ignore_index=True,
    )


def main():
    cohort, features = prepare_stage1_data()
    split = split_cohort(cohort)
    labels, teeth, flagged = build_quality_reports(cohort, features)

    STAGE1_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels.to_csv(STAGE1_OUTPUT_DIR / "label_distribution.csv", index=False)
    teeth.to_csv(STAGE1_OUTPUT_DIR / "tooth_missingness.csv", index=False)
    flagged.to_csv(STAGE1_OUTPUT_DIR / "all_missing_participants.csv", index=False)
    split.to_csv(STAGE1_OUTPUT_DIR / "split_assignments.csv", index=False)

    print(f"Stage 1 development cohort: {len(cohort):,} participants, {features.shape[1]} features")
    for label in labels.itertuples(index=False):
        print(f"{label.perio_label_cdc} {label.label_name}: {label.n:,}")
    print(f"Participants with all interproximal measurements missing: {len(flagged):,}")
    print(f"Training: {split['split'].eq('train').sum():,}")
    print(f"Internal test: {split['split'].eq('internal_test').sum():,}")
    print("SEQN overlap with external validation: 0")
    print(f"Wrote Stage 1 reports to {STAGE1_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
