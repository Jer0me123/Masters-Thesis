import json
import pandas as pd
import argparse
from pathlib import Path

# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser(
    description="Compute per-race label distributions from annotation JSONL"
)

parser.add_argument(
    "--input",
    required=True,
    help="Path to annotations.jsonl"
)

parser.add_argument(
    "--label",
    required=True,
    choices=["mst_label", "bin_label", "bin_name"],
    help="Which label to analyze"
)

parser.add_argument(
    "--output",
    default="race_label_distribution.csv",
    help="Output CSV file"
)

args = parser.parse_args()

INPUT_JSONL = Path(args.input)
LABEL_COL = args.label
OUTPUT_CSV = Path(args.output)

# ============================================================
# LOAD JSONL
# ============================================================

records = []
with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame(records)

# ------------------------------------------------------------
# Basic validation
# ------------------------------------------------------------

required_cols = {"image", LABEL_COL}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# ============================================================
# EXTRACT RACE FROM PATH
# train/Black/Female/1001.jpg → Black
# ============================================================

def extract_race(path: str) -> str:
    parts = Path(path).parts
    if len(parts) < 2:
        return "UNKNOWN"
    return parts[1]

df["race"] = df["image"].apply(extract_race)

# ============================================================
# COMPUTE COUNTS
# ============================================================

counts = (
    df
    .groupby(["race", LABEL_COL])
    .size()
    .reset_index(name="count")
)

# ============================================================
# COMPUTE PERCENTAGES (within race)
# ============================================================

counts["pct"] = (
    counts["count"]
    / counts.groupby("race")["count"].transform("sum")
    * 100
)

counts["pct"] = counts["pct"].round(2)

# ============================================================
# SORT & SAVE
# ============================================================

counts = counts.sort_values(
    by=["race", "pct"],
    ascending=[True, False]
)

counts.to_csv(OUTPUT_CSV, index=False)

print("✅ Race-wise distribution saved")
print(f"📄 File: {OUTPUT_CSV}")
print("\nPreview:")
print(counts.head(10))




# python "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\GenderSkinToneProxy\MappingRaceToSkinTone\4_DeriveDistribution.py" ^
#   --input "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\GenderSkinToneProxy\MappingRaceToSkinTone\annotations.jsonl" ^
#   --label mst_label ^
#   --output race_mst_distribution.csv











# python "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\GenderSkinToneProxy\MappingRaceToSkinTone\4_DeriveDistribution.py" ^
#   --input "F:\Thesis\Segmented_FairFace\\annotations.jsonl" ^
#   --label mst_label ^
#   --output race_mst_distribution.csv



# python "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\GenderSkinToneProxy\MappingRaceToSkinTone\4_DeriveDistribution.py" ^
#   --input "F:\Thesis\Segmented_FairFace_test\\annotations.jsonl" ^
#   --label mst_label ^
#   --output race_mst_distribution_test.csv


# python "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\GenderSkinToneProxy\MappingRaceToSkinTone\4_DeriveDistribution.py" ^
#   --input "F:\Thesis\Segmented_FairFace_test2\\annotations.jsonl" ^
#   --label mst_label ^
#   --output race_mst_distribution_test2.csv


# python "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\GenderSkinToneProxy\MappingRaceToSkinTone\4_DeriveDistribution.py" ^
#   --input "F:\Thesis\Segmented_FairFace_test3_500\\annotations.jsonl" ^
#   --label mst_label ^
#   --output race_mst_distribution_test3_500.csv


# python "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\GenderSkinToneProxy\MappingRaceToSkinTone\4_DeriveDistribution.py" ^
#   --input "F:\Thesis\Segmented_FairFace_testMSTE\\annotations.jsonl" ^
#   --label mst_label ^
#   --output race_mst_distribution_test3_500MSTE.csv


# python "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\GenderSkinToneProxy\MappingRaceToSkinTone\4_DeriveDistribution.py" ^
#   --input "F:\Thesis\Segmented_MSTE_MoreData\\annotations.jsonl" ^
#   --label mst_label ^
#   --output race_mst_distribution_MoreData.csv

# python "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\GenderSkinToneProxy\MappingRaceToSkinTone\4_DeriveDistribution.py" ^
#   --input "F:\Thesis\Segmented_FairFace_MoreData\\annotations.jsonl" ^
#   --label mst_label ^
#   --output race_mst_distribution_fariface_MoreData.csv


# python "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\GenderSkinToneProxy\MappingRaceToSkinTone\4_DeriveDistribution.py" ^
#   --input "F:\Thesis\Segmented_MSTE_5Class\\annotations.jsonl" ^
#   --label mst_label ^
#   --output race_mst_distribution_MSTE_5.csv



# python "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\GenderSkinToneProxy\MappingRaceToSkinTone\4_DeriveDistribution.py" ^
#   --input "F:\Thesis\Segmented_FairFace_Dark\\annotations.jsonl" ^
#   --label mst_label ^
#   --output race_mst_distribution_fairFace_5_dark.csv


# OVM

# python "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\GenderSkinToneProxy\MappingRaceToSkinTone\4_DeriveDistribution.py" ^
#   --input "F:\Thesis\Segmented_MSTE_MST1_2_Dark\\annotations.jsonl" ^
#   --label mst_label ^
#   --output race_mst_distribution_MST1_2_dark.csv

# python "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\GenderSkinToneProxy\MappingRaceToSkinTone\4_DeriveDistribution.py" ^
#   --input "F:\Thesis\Segmented_MSTE_MST9_10_Dark\\annotations.jsonl" ^
#   --label mst_label ^
#   --output race_mst_distribution_MST9_10_dark.csv


# python "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\GenderSkinToneProxy\MappingRaceToSkinTone\4_DeriveDistribution.py" ^
#   --input "F:\Thesis\Segmented_MSTE_MST9_10_Dark_undersampling\\annotations.jsonl" ^
#   --label mst_label ^
#   --output race_mst_distribution_MST9_10_dark_undersamplin.csv


# python "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\GenderSkinToneProxy\MappingRaceToSkinTone\4_DeriveDistribution.py" ^
#   --input "F:\Thesis\Segmented_MSTE_Dark_undersampling_LAB\\annotations.jsonl" ^
#   --label mst_label ^
#   --output race_mst_distribution_MSTE_5_dark_LAB.csv


# python "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\GenderSkinToneProxy\MappingRaceToSkinTone\4_DeriveDistribution.py" ^
#   --input "F:\VGG_MST_Testing\Models\ResNet18_4CCoral_LAB_FixedBG\Segmented_MSTE_BGFixed\\annotations.jsonl" ^
#   --label mst_label ^
#   --output "F:\VGG_MST_Testing\Models\ResNet18_4CCoral_LAB_FixedBG\Segmented_MSTE_BGFixed\\race_mst_distribution.csv"

# python "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\GenderSkinToneProxy\MappingRaceToSkinTone\4_DeriveDistribution.py" ^
#   --input "F:\VGG_MST_Testing\Models\ResNet18_4CCoral_LAB_FixedBG\Segmented_FairFace_BGFixed\\annotations.jsonl" ^
#   --label mst_label ^
#   --output "F:\VGG_MST_Testing\Models\ResNet18_4CCoral_LAB_FixedBG\Segmented_FairFace_BGFixed\\race_mst_distribution.csv"