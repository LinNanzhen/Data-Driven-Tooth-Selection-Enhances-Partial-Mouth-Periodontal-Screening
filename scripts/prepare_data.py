from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "processed"
CYCLES = ("F", "G", "H")
COHORTS = (
    ("cleaned_data.csv", ("F", "G"), 2),
    ("test_data.csv", ("H",), 1),
)
EXAM_STATUS = 1
EXAM_NOT_DONE = 3
EXCLUSION_STATUS = 2
MIN_TEETH = 6
MIN_AGE = 35
POOLED_FILE = "pooled_data.csv"
FLOW_FILE = "inclusion_flow.csv"
# Pandas decodes the NHANES XPORT representation of SAS zero to this value.
XPORT_ZERO = 5.397605346934028e-79
FLOW_COLUMNS = ["cycle", "stage", "before", "excluded", "remaining"]
TOOTH_CODES = (1, 2, 3, 5)
TOOTH_NUMBERS = [number for number in range(2, 32) if number not in (16, 17)]
TOOTH_COLUMNS = [f"OHX{number:02d}TC" for number in TOOTH_NUMBERS]
MEASUREMENT_COLUMNS = [
    f"OHX{number:02d}{measure}{site}"
    for number in TOOTH_NUMBERS
    for measure in ("LA", "PC")
    for site in "ADLMPS"
]
DEMO_COLUMNS = [
    "SEQN", "SDMVPSU", "SDMVSTRA", "DMDEDUC2", "RIDRETH1",
    "RIDAGEYR", "RIAGENDR", "WTMEC2YR",
]
OUTPUT_COLUMNS = DEMO_COLUMNS[:5] + ["num_teeth"] + DEMO_COLUMNS[5:] + MEASUREMENT_COLUMNS


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged_cycles = []
    baseline = {}
    positive_weight = {}

    for cycle in CYCLES:
        periodontal = pd.read_sas(RAW_DIR / f"OHXPER_{cycle}.xpt", format="xport")
        dentition = pd.read_sas(RAW_DIR / f"OHXDEN_{cycle}.xpt", format="xport")
        demographics = pd.read_sas(RAW_DIR / f"DEMO_{cycle}.xpt", format="xport")
        baseline[cycle] = len(demographics)
        weights = demographics["WTMEC2YR"].replace(XPORT_ZERO, 0)
        demographics = demographics.loc[np.isfinite(weights) & weights.gt(0)]
        positive_weight[cycle] = len(demographics)
        dentition["num_teeth"] = dentition[TOOTH_COLUMNS].isin(TOOTH_CODES).sum(axis=1)

        merged = demographics[DEMO_COLUMNS].merge(
            periodontal[["SEQN", "OHDPDSTS", "OHDEXCLU", *MEASUREMENT_COLUMNS]],
            on="SEQN", validate="one_to_one",
        )
        merged = merged.merge(
            dentition[["SEQN", "num_teeth"]], on="SEQN", validate="one_to_one"
        )
        merged_cycles.append(merged.assign(cycle=cycle))

    eligible = pd.concat(merged_cycles, ignore_index=True)
    stages = (
        ("Excluded for medical reasons", eligible["OHDEXCLU"].eq(EXCLUSION_STATUS)),
        ("Periodontal exam not done", eligible["OHDPDSTS"].ne(EXAM_NOT_DONE)),
        ("Periodontal exam partially completed", eligible["OHDPDSTS"].eq(EXAM_STATUS)),
        ("No countable teeth", eligible["num_teeth"].gt(0)),
        (f"Fewer than {MIN_TEETH} countable teeth", eligible["num_teeth"].ge(MIN_TEETH)),
        (f"Age under {MIN_AGE} years", eligible["RIDAGEYR"].ge(MIN_AGE)),
    )
    counts = [pd.Series(baseline), pd.Series(positive_weight), eligible["cycle"].value_counts()]
    for _, keep in stages:
        eligible = eligible.loc[keep.loc[eligible.index]]
        counts.append(eligible["cycle"].value_counts())

    flow = []
    stage_names = [
        "MEC exam weight not positive or not finite",
        "Records in all three tables",
        *[stage for stage, _ in stages],
    ]
    for stage, before, remaining in zip(stage_names, counts, counts[1:]):
        for cycle in CYCLES:
            start = before.get(cycle, 0)
            end = remaining.get(cycle, 0)
            flow.append((cycle, stage, start, start - end, end))
        start = before.sum()
        end = remaining.sum()
        flow.append(("total", stage, start, start - end, end))
    pd.DataFrame(flow, columns=FLOW_COLUMNS).to_csv(OUT_DIR / FLOW_FILE, index=False)

    eligible[MEASUREMENT_COLUMNS] = (
        eligible[MEASUREMENT_COLUMNS].round().astype("Int64").replace([0, 99], pd.NA)
    )
    eligible["SEQN"] = eligible["SEQN"].astype(int)
    eligible = eligible.sort_values("SEQN")
    eligible[["cycle", *OUTPUT_COLUMNS]].to_csv(OUT_DIR / POOLED_FILE, index=False)
    print(f"Wrote {len(eligible):,} participants to {POOLED_FILE}")

    for filename, cycles, weight_divisor in COHORTS:
        result = eligible.loc[eligible["cycle"].isin(cycles), OUTPUT_COLUMNS].copy()
        result["WTMEC2YR"] = result["WTMEC2YR"] / weight_divisor
        result.to_csv(OUT_DIR / filename, index=False)
        print(f"Wrote {len(result):,} participants to {filename}")


if __name__ == "__main__":
    main()
