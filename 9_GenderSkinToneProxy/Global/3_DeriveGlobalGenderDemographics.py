import pandas as pd
import argparse
from pathlib import Path

# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Derive gender splits per job from ILOSTAT OC2 data"
)
parser.add_argument("--job-file", required=True, help="Job list CSV with ISCO codes")
parser.add_argument("--ilo-file", required=True, help="ILOSTAT OC2 CSV file")
parser.add_argument("--output-dir", default="outputs", help="Output directory")
parser.add_argument(
    "--year-mode",
    choices=["all", "latest-per-country", "latest-global"],
    default="all",
    help="How to handle multiple years in ILO data",
)
args = parser.parse_args()

JOB_FILE = args.job_file
ILO_FILE = args.ilo_file
OUTPUT_DIR = Path(args.output_dir)
YEAR_MODE = args.year_mode

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_COUNTRY = OUTPUT_DIR / "gender_split_per_country_per_job.csv"
OUT_GLOBAL = OUTPUT_DIR / "gender_split_global_per_job.csv"

# ============================================================
# 1. LOAD JOB FILE
# ============================================================

jobs = pd.read_csv(JOB_FILE)

required_job_cols = {"job_title_original", "isco_code"}
missing = required_job_cols - set(jobs.columns)
if missing:
    raise ValueError(f"Job file missing columns: {missing}")

jobs["isco_code"] = jobs["isco_code"].astype(str).str.zfill(4)
jobs["isco_2"] = jobs["isco_code"].str[:2]

# ============================================================
# 2. LOAD ILO OC2 DATA
# ============================================================

ilo = pd.read_csv(ILO_FILE)

required_ilo_cols = {
    "REF_AREA",
    "REF_AREA_LABEL",
    "SEX",
    "OC2",
    "VALUE",
    "TIME_PERIOD",
}
missing = required_ilo_cols - set(ilo.columns)
if missing:
    raise ValueError(f"ILO file missing columns: {missing}")

# Keep only male / female (drop totals)
ilo = ilo[ilo["SEX"].isin(["SEX_M", "SEX_F"])]

# Ensure TIME_PERIOD numeric
ilo["TIME_PERIOD"] = ilo["TIME_PERIOD"].astype(int)

# ------------------------------------------------------------
# YEAR SELECTION LOGIC
# ------------------------------------------------------------

if YEAR_MODE == "latest-global":
    latest_year = ilo["TIME_PERIOD"].max()
    ilo = ilo[ilo["TIME_PERIOD"] == latest_year]
    print(f"Using single global latest year: {latest_year}")

elif YEAR_MODE == "latest-per-country":
    ilo = (
        ilo.sort_values("TIME_PERIOD")
           .groupby(["REF_AREA", "SEX", "OC2"], as_index=False)
           .tail(1)
    )
    print("Using most recent year per country")

else:
    print("Using all available years (summed)")

# Extract ISCO-02 numeric code from OC2_ISCO08_XX
ilo["isco_2"] = ilo["OC2"].str.extract(r"OC2_ISCO08_(\d{2})")

# ============================================================
# 3. JOIN JOBS ↔ ILO (ISCO-02)
# ============================================================

merged = jobs.merge(
    ilo,
    on="isco_2",
    how="inner"
)

# ============================================================
# 4. PER-COUNTRY × PER-JOB GENDER SPLIT
# ============================================================

country_job = (
    merged
    .groupby(
        ["REF_AREA", "REF_AREA_LABEL", "job_title_original", "SEX"],
        as_index=False
    )["VALUE"]
    .sum()
)

country_pivot = (
    country_job
    .pivot_table(
        index=["REF_AREA", "REF_AREA_LABEL", "job_title_original"],
        columns="SEX",
        values="VALUE",
        fill_value=0
    )
    .reset_index()
)

country_pivot["male_count"] = country_pivot.get("SEX_M", 0)
country_pivot["female_count"] = country_pivot.get("SEX_F", 0)
country_pivot["total"] = country_pivot["male_count"] + country_pivot["female_count"]

country_pivot["male_pct"] = country_pivot["male_count"] / country_pivot["total"]
country_pivot["female_pct"] = country_pivot["female_count"] / country_pivot["total"]

country_pivot.to_csv(OUT_COUNTRY, index=False)
print(f"Saved per-country results → {OUT_COUNTRY}")

# ============================================================
# 5. GLOBAL PER-JOB GENDER SPLIT
# ============================================================

global_job = (
    merged
    .groupby(["job_title_original", "SEX"], as_index=False)["VALUE"]
    .sum()
)

global_pivot = (
    global_job
    .pivot_table(
        index="job_title_original",
        columns="SEX",
        values="VALUE",
        fill_value=0
    )
    .reset_index()
)

global_pivot["male_count"] = global_pivot.get("SEX_M", 0)
global_pivot["female_count"] = global_pivot.get("SEX_F", 0)
global_pivot["total"] = global_pivot["male_count"] + global_pivot["female_count"]

global_pivot["male_pct"] = global_pivot["male_count"] / global_pivot["total"]
global_pivot["female_pct"] = global_pivot["female_count"] / global_pivot["total"]

global_pivot.to_csv(OUT_GLOBAL, index=False)
print(f"Saved global results → {OUT_GLOBAL}")

print("DONE — ISCO-02 gender splits derived correctly.")