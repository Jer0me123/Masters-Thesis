"""
Job Title → ISCO-08 Mapper (WageIndicator WISCO)
Batch-only, argument-driven version.

- Downloads WISCO file automatically if not provided
- Applies hard-coded job remapping (e.g. DJ → Disc Jockey)
- Maps job lists from CSV or TXT
"""

import argparse
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher
import re
import requests
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# CONSTANTS
# ============================================================

WISCO_DOWNLOAD_URL = (
    "https://www.surveycodings.org/downloads/content_files/"
    "occupations_ISCO08_5dgt_55languages_4000titles_with_mapping_surveycodings_20230425.xlsx"
)

DEFAULT_WISCO_FILE = "occupations_ISCO08_5dgt_55languages_4000titles.xlsx"


# ============================================================
# HARD-CODED JOB REMAPPING (PRE-NORMALISATION)
# ============================================================

JOB_REMAP = {
    "dj": "disc jockey",
    "bounty hunter": "bailiff",
    "chef": "chef cook",
    "coach": "athletic coach",
    "customer support specialist": "call centre operator customer service",
    "florist": "florist, operating a shop",
    "model": "fashion model",
    "nurse": "certified nurse",
    "paramedic": "emergency paramedic",
    "pilot": "airline pilot, co-pilot",
    "umpire": "sports umpire",
}


# ============================================================
# UTILS
# ============================================================

def download_wisco_if_needed(filepath: Path) -> Path:
    if filepath.exists():
        return filepath

    print(f"WISCO file not found — downloading...")
    print(f"Source: {WISCO_DOWNLOAD_URL}")

    response = requests.get(WISCO_DOWNLOAD_URL, stream=True)
    response.raise_for_status()

    with open(filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"Downloaded WISCO file to: {filepath.resolve()}")
    return filepath


# ============================================================
# MAPPER CLASS
# ============================================================

class ISCOMapperWISCO:
    """Maps job titles to ISCO-08 codes using WageIndicator WISCO database"""

    def __init__(self, wisco_file: Path):
        self.wisco_file = wisco_file
        self.job_index = None
        self.loaded = False

    # --------------------------------------------------------

    def load_wisco_database(self):
        if not self.wisco_file.exists():
            raise FileNotFoundError(self.wisco_file)

        print(f"Loading WISCO database: {self.wisco_file}")
        df = pd.read_excel(self.wisco_file, sheet_name="CODESET")

        records = []
        for _, row in df.iterrows():
            code_13 = row["occupai3_API_13dgt"]
            label = row["MASTER LABEL 4000"]

            if pd.isna(code_13) or pd.isna(label) or label == "nr_empty":
                continue
            if code_13 < 0:
                continue

            code_str = f"{int(code_13)}"
            isco_4 = code_str[:4].zfill(4)

            # Armed forces correction
            if isco_4 in {"1100", "2100", "3100"}:
                isco_4 = "0" + isco_4[0] + isco_4[2:]

            records.append({
                "job_title": label,
                "job_title_lower": label.lower(),
                "isco_code": isco_4
            })

        self.job_index = pd.DataFrame(records).drop_duplicates("job_title")
        self.loaded = True

        print(f"Loaded {len(self.job_index):,} job titles")
        print(f"Unique ISCO-4 codes: {self.job_index['isco_code'].nunique()}")

    # --------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)
        return re.sub(r"\s+", " ", text)

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    def _apply_job_remap(self, job: str) -> str:
        job_norm = self._normalize(job)
        return JOB_REMAP.get(job_norm, job)

    # --------------------------------------------------------

    def map_job_title(self, job_title: str, threshold=0.7, max_results=3):
        if not self.loaded:
            raise RuntimeError("WISCO database not loaded")

        job_title = self._apply_job_remap(job_title)
        job_norm = self._normalize(job_title)

        matches = []

        exact = self.job_index[self.job_index["job_title_lower"] == job_norm]
        for _, r in exact.iterrows():
            matches.append({
                "matched_title": r["job_title"],
                "isco_code": r["isco_code"],
                "similarity": 1.0,
                "match_type": "exact"
            })

        if matches:
            return matches

        self.job_index["sim"] = self.job_index["job_title_lower"].apply(
            lambda x: self._similarity(job_norm, x)
        )

        fuzzy = (
            self.job_index[self.job_index["sim"] >= threshold]
            .sort_values("sim", ascending=False)
            .head(max_results)
        )

        for _, r in fuzzy.iterrows():
            matches.append({
                "matched_title": r["job_title"],
                "isco_code": r["isco_code"],
                "similarity": round(r["sim"], 3),
                "match_type": "fuzzy"
            })

        return matches

    # --------------------------------------------------------

    def map_job_list(self, job_list_file, threshold=0.7):
        if job_list_file.endswith(".csv"):
            jobs = pd.read_csv(job_list_file).iloc[:, 0].dropna().tolist()
        else:
            with open(job_list_file, "r", encoding="utf-8") as f:
                jobs = [l.strip() for l in f if l.strip()]

        print(f"🔄 Mapping {len(jobs)} job titles")

        rows = []
        for job in jobs:
            matches = self.map_job_title(job, threshold)

            if matches:
                best = matches[0]
                rows.append({
                    "job_title_original": job,
                    "job_title_remapped": self._apply_job_remap(job),
                    "matched_title": best["matched_title"],
                    "isco_code": best["isco_code"],
                    "similarity_score": best["similarity"],
                    "match_type": best["match_type"]
                })
            else:
                rows.append({
                    "job_title_original": job,
                    "job_title_remapped": self._apply_job_remap(job),
                    "matched_title": None,
                    "isco_code": None,
                    "similarity_score": 0.0,
                    "match_type": "none"
                })

        return pd.DataFrame(rows)


# ============================================================
# CLI ENTRYPOINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch Job Title → ISCO-08 Mapper (WISCO)"
    )

    parser.add_argument("--job-list", required=True, help="CSV or TXT file with job titles")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wisco-file", default=None)

    args = parser.parse_args()

    wisco_path = Path(args.wisco_file) if args.wisco_file else Path(DEFAULT_WISCO_FILE)
    wisco_path = download_wisco_if_needed(wisco_path)

    mapper = ISCOMapperWISCO(wisco_path)
    mapper.load_wisco_database()

    df = mapper.map_job_list(args.job_list, threshold=args.threshold)
    df.to_csv(args.output, index=False)

    matched = (df["match_type"] != "none").sum()
    print(f"Done — matched {matched}/{len(df)} ({matched/len(df)*100:.1f}%)")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()