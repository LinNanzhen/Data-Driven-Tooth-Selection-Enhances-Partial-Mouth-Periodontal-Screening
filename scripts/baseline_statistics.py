"""Export unweighted counts and survey-weighted cohort characteristics."""

import pandas as pd

from prepare_data import CYCLES, OUT_DIR, POOLED_FILE
from stage1_data import LABEL_NAMES, build_features, classify_periodontitis

AGE_LABELS = ["35-44", "45-54", "55-64", "65-74", "75+"]
RACE = {
    1: "Mexican American",
    2: "Other Hispanic",
    3: "Non-Hispanic White",
    4: "Non-Hispanic Black",
    5: "Other Race - Including Multi-Racial",
}
EDUCATION = {
    1: "Less than high school",
    2: "Less than high school",
    3: "High school or equivalent",
    4: "Some college or higher",
    5: "Some college or higher",
}


def main():
    data = pd.read_csv(OUT_DIR / POOLED_FILE)
    labels = classify_periodontitis(build_features(data)).map(LABEL_NAMES)
    age = pd.cut(data["RIDAGEYR"], [35, 45, 55, 65, 75, float("inf")], labels=AGE_LABELS, right=False)
    gender = {2: "Female", 1: "Male"}
    groups = (
        ("Age, y", age, AGE_LABELS),
        ("Gender", data["RIAGENDR"].map(gender), gender.values()),
        ("Race/Ethnicity", data["RIDRETH1"].map(RACE), RACE.values()),
        ("Education level", data["DMDEDUC2"].map(EDUCATION).fillna("Other"), [*dict.fromkeys(EDUCATION.values()), "Other"]),
        ("Prevalence", labels, LABEL_NAMES.values()),
    )
    weights = data["WTMEC2YR"] / len(CYCLES)
    total = weights.sum()
    rows = [("Total", "", len(data), 100, total, 100)]
    for characteristic, values, categories in groups:
        for category in categories:
            selected = values.eq(category)
            n = int(selected.sum())
            weighted_n = weights[selected].sum()
            rows.append((characteristic, category, n, 100 * n / len(data), weighted_n, 100 * weighted_n / total))

    columns = ["Characteristic", "Category", "Unweighted n", "Unweighted %",
               "Weighted population estimate", "Weighted %"]
    output = OUT_DIR / "baseline_statistics.csv"
    pd.DataFrame(rows, columns=columns).to_csv(output, index=False, float_format="%.6f")
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
