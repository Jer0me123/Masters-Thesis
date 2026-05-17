import json
from pathlib import Path

# =========================
# CONFIG
# =========================
ROOT_DIR = Path(r"DebiasedImages")
OUTPUT_PATH = ROOT_DIR / "annotations.jsonl"

VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# =========================
# HELPERS
# =========================

def parse_folder_name(folder_name: str):
    """
    Example:
    CCv2_Gender_female_CCv2_MSTE_SkinTone_3
    """
    parts = folder_name.split("_")

    # Gender
    if "female" in parts:
        gender = "Female"
    elif "male" in parts:
        gender = "Male"
    else:
        raise ValueError(f"Gender not found in: {folder_name}")

    # MST label
    if "SkinTone" in parts:
        idx = parts.index("SkinTone")
        mst_label = int(parts[idx + 1])
    else:
        raise ValueError(f"MST label not found in: {folder_name}")

    return gender, mst_label


def mst_to_bin(mst_label: int):
    """
    Your binning:
    1-3  -> Light (0)
    4-7  -> Mid   (1)
    8-10 -> Dark  (2)
    """
    if 1 <= mst_label <= 3:
        return 0, "Light (1-3)"
    elif 4 <= mst_label <= 7:
        return 1, "Mid (4-7)"
    elif 8 <= mst_label <= 10:
        return 2, "Dark (8-10)"
    else:
        raise ValueError(f"Invalid MST label: {mst_label}")


# =========================
# MAIN
# =========================

def main():
    rows = []

    for folder in ROOT_DIR.iterdir():
        if not folder.is_dir():
            continue

        try:
            gender, mst_label = parse_folder_name(folder.name)
            bin_label, bin_name = mst_to_bin(mst_label)
        except Exception as e:
            print(f"Skipping folder: {folder.name} | {e}")
            continue

        for img_path in folder.rglob("*"):
            if img_path.suffix.lower() not in VALID_EXTS:
                continue

            rel_path = img_path.relative_to(ROOT_DIR).as_posix()

            row = {
                "image": rel_path,
                "gender": gender,
                "gender_confidence": None,
                "mst_label": mst_label,
                "bin_label": bin_label,
                "bin_name": bin_name,
                "skin_raw_output": None,
                "skin_confidence": None,
            }

            rows.append(row)

    # Write JSONL
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"Done. Wrote {len(rows)} entries → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()