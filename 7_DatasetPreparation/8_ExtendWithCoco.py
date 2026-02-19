import json
from pathlib import Path

# ==========================
# CONFIG
# ==========================
BASE_SPLITS_JSON = r"C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\7_DatasetPreparation\UniversalSplits\DatasetClassification\splits_face_combined_stratified_patchShufflePS16.json"
OUTPUT_SPLITS_JSON = r"C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\7_DatasetPreparation\UniversalSplits\DatasetClassificationCoco\coco_splits_face_combined_stratified_patchShufflePS16.json"

COCO_ROOT = Path(r"F:\ImageRetrieval\SpuriousFeatureImages\Coco\Shuffling&Colour\patch_shuffle_ps16")
COCO_SPLITS = {
    "train": COCO_ROOT / "train2017",
    "val":   COCO_ROOT / "val2017",
    "test":  COCO_ROOT / "test2017",
}

COCO_LABEL = 2
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# ==========================
# LOAD EXISTING SPLITS
# ==========================
with open(BASE_SPLITS_JSON, "r", encoding="utf-8") as f:
    splits = json.load(f)

# Ensure all splits exist
for split in ["train", "val", "test"]:
    splits.setdefault(split, [])

# ==========================
# ADD COCO IMAGES
# ==========================
for split_name, split_dir in COCO_SPLITS.items():
    if not split_dir.exists():
        print(f"⚠ Skipping missing split: {split_dir}")
        continue

    coco_images = [
        p for p in split_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS
    ]

    print(f"Adding {len(coco_images):,} COCO images to '{split_name}'")

    for img_path in coco_images:
        splits[split_name].append({
            "image": str(img_path.resolve()),
            "label": COCO_LABEL
        })

# ==========================
# SAVE NEW JSON
# ==========================
with open(OUTPUT_SPLITS_JSON, "w", encoding="utf-8") as f:
    json.dump(splits, f, indent=2)

print(f"\n✅ Saved merged splits to:\n{OUTPUT_SPLITS_JSON}")
