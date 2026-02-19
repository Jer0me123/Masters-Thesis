import os
import json
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

CCV2_IMAGES_ROOT = Path(r"F:\Thesis\CasualConversationv2_Dataset\Images")
CCV2_JSON = Path(r"F:\Thesis\CasualConversationv2_Dataset\Annotations\CasualConversationsV2.json")

OUT_SKINTONE_ROOT = Path(r"ITI-GEN\data\CCv2_MSTE_SkinTone_benchmark\CCv2_MSTE_SkinTone")
OUT_GENDER_ROOT   = Path(r"ITI-GEN\data\CCv2_Gender_benchmark\CCv2_Gender")

CONFIDENCE_FILTER = None   # None | {"low"} | {"medium"} | {"high"}

OUT_SKINTONE_ROOT.mkdir(parents=True, exist_ok=True)
OUT_GENDER_ROOT.mkdir(parents=True, exist_ok=True)

# ============================================================
# PARSERS
# ============================================================

def parse_mst_label(mst_dict):
    scale = mst_dict.get("scale", "")
    digits = "".join(c for c in scale if c.isdigit())
    return int(digits) if digits.isdigit() else None


def parse_gender(raw):
    if not raw:
        return None
    g = raw.lower()
    if "female" in g or "woman" in g:
        return "female"
    if "male" in g or "man" in g:
        return "male"
    return None


# ============================================================
# LINK CREATION (Windows-safe)
# ============================================================

def safe_link(src: Path, dst: Path):
    """
    Prefer symlink; fall back to hard link if symlinks are unavailable.
    """
    try:
        os.symlink(src, dst)
    except OSError:
        os.link(src, dst)


# ============================================================
# MAIN PIPELINE
# ============================================================

def build_ccv2_joint_views():
    print("\n--- Loading Casual Conversations V2 annotations ---")

    with open(CCV2_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # --------------------------------------------------------
    # Subject-level joint filtering (STRICT)
    # --------------------------------------------------------

    eligible_subjects = {}

    for entry in data:
        sid = entry["subject_id"]
        if sid in eligible_subjects:
            continue

        mst = parse_mst_label(entry.get("monk_skin_tone", {}))
        gender = parse_gender(entry.get("gender", ""))

        # STRICT joint condition
        if mst is None or not (1 <= mst <= 10):
            continue
        if gender not in {"male", "female"}:
            continue

        conf = entry.get("monk_skin_tone", {}).get("confidence", "").lower()
        if CONFIDENCE_FILTER and conf not in CONFIDENCE_FILTER:
            continue

        eligible_subjects[sid] = {
            "mst": mst,
            "gender": gender,
        }

    print(f"Eligible subjects (MST ∧ gender): {len(eligible_subjects)}")

    # --------------------------------------------------------
    # Single image traversal → dual view creation
    # --------------------------------------------------------

    created_st = 0
    created_g  = 0
    skipped    = 0

    for sid, labels in tqdm(eligible_subjects.items(), desc="Processing subjects"):
        subj_dir = CCV2_IMAGES_ROOT / sid
        if not subj_dir.is_dir():
            continue

        mst = labels["mst"]
        gender = labels["gender"]

        st_dir = OUT_SKINTONE_ROOT / str(mst)
        g_dir  = OUT_GENDER_ROOT / gender

        st_dir.mkdir(parents=True, exist_ok=True)
        g_dir.mkdir(parents=True, exist_ok=True)

        for img_path in subj_dir.iterdir():
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            link_name = f"{sid}_{img_path.name}"

            st_dst = st_dir / link_name
            g_dst  = g_dir  / link_name

            # Create both or neither (atomic at image level)
            if st_dst.exists() or g_dst.exists():
                skipped += 1
                continue

            safe_link(img_path, st_dst)
            safe_link(img_path, g_dst)

            created_st += 1
            created_g  += 1

    print("\n--- Done ---")
    print(f"Skin-tone links created : {created_st}")
    print(f"Gender links created    : {created_g}")
    print(f"Skipped (existing)      : {skipped}")
    print(f"Skin-tone root          : {OUT_SKINTONE_ROOT}")
    print(f"Gender root             : {OUT_GENDER_ROOT}")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    build_ccv2_joint_views()
