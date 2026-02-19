import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def load_splits(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def apply_suffix(path: Path, suffix: str | None) -> Path:
    if not suffix:
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")

def relabel_extend_and_suffix(samples, new_label, image_root, suffix):
    out = []
    for s in samples:
        # Copy ALL existing fields
        record = dict(s)

        # Update image path
        rel = Path(record["image"])
        full = apply_suffix(image_root / rel, suffix)
        record["image"] = str(full)

        # Overwrite label
        record["label"] = new_label

        out.append(record)
    return out

def extract_image_keys(split_json):
    return {
        split: {
            Path(s["image"]).as_posix()
            for s in split_json[split]
        }
        for split in ["train", "val", "test"]
    }

def filter_by_reference(samples, allowed_keys):
    out = []
    for s in samples:
        key = Path(s["image"]).as_posix()
        if key in allowed_keys:
            out.append(s)
    return out


# --------------------------------------------------
# Sampling utilities
# --------------------------------------------------

def balance_by_label(samples):
    buckets = defaultdict(list)
    for s in samples:
        buckets[s["label"]].append(s)

    min_n = min(len(v) for v in buckets.values())

    balanced = []
    for label in sorted(buckets):
        group = buckets[label]
        random.shuffle(group)
        balanced.extend(group[:min_n])

    random.shuffle(balanced)
    return balanced

def downsample(samples, max_n, stratified):
    if max_n is None or len(samples) <= max_n:
        return samples

    if not stratified:
        random.shuffle(samples)
        return samples[:max_n]

    buckets = defaultdict(list)
    for s in samples:
        buckets[s["label"]].append(s)

    total = len(samples)
    out = []

    for label, group in buckets.items():
        frac = len(group) / total
        k = max(1, int(frac * max_n))
        random.shuffle(group)
        out.extend(group[:k])

    random.shuffle(out)
    return out[:max_n]

def get_downsample_for_split(args, split):
    if split == "train" and args.downsample_train is not None:
        return args.downsample_train
    if split == "val" and args.downsample_val is not None:
        return args.downsample_val
    if split == "test" and args.downsample_test is not None:
        return args.downsample_test
    return args.downsample


def print_stats(name, split):
    total = len(split)
    counts = defaultdict(int)
    for s in split:
        counts[s["label"]] += 1

    print(f"\n[{name}] total={total}")
    for k in sorted(counts):
        pct = 100.0 * counts[k] / total if total else 0.0
        print(f"  label {k}: {counts[k]} ({pct:.2f}%)")

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Merge dataset splits with optional balancing and downsampling"
    )

    p.add_argument("--splits_a", required=True, type=Path)
    p.add_argument("--splits_b", required=True, type=Path)

    p.add_argument("--image_root_a", required=True, type=Path)
    p.add_argument("--image_root_b", required=True, type=Path)

    p.add_argument("--suffix_a", default=None)
    p.add_argument("--suffix_b", default=None)

    p.add_argument("--out_json", required=True, type=Path)

    p.add_argument("--label_a", default="DatasetA")
    p.add_argument("--label_b", default="DatasetB")

    p.add_argument("--balanced", action="store_true")
    p.add_argument("--downsample", type=int, default=None)
    p.add_argument("--downsample_train", type=int, default=None)
    p.add_argument("--downsample_val", type=int, default=None)
    p.add_argument("--downsample_test", type=int, default=None)

    p.add_argument(
        "--split_mode",
        choices=["random", "stratified"],
        default="stratified"
    )

    p.add_argument("--seed", type=int, default=0)

    p.add_argument(
        "--split_reference",
        type=Path,
        default=None,
        help="JSON file whose train/val/test image membership is authoritative"
    )


    args = p.parse_args()

    if args.split_reference is not None:
        if args.balanced or args.downsample is not None:
            raise ValueError(
                "When --split_reference is provided, --balanced and --downsample "
                "must NOT be used. Apply sampling when creating the reference splits."
            )

    set_seed(args.seed)

    A = load_splits(args.splits_a)
    B = load_splits(args.splits_b)

    reference_keys = None

    if args.split_reference is not None:
        ref = load_splits(args.split_reference)
        reference_keys = extract_image_keys(ref)

    merged = {}

    for split in ["train", "val", "test"]:

        A_split = A[split]
        B_split = B[split]

        if reference_keys is not None:
            A_split = filter_by_reference(A_split, reference_keys[split])
            B_split = filter_by_reference(B_split, reference_keys[split])

        combined = (
            relabel_extend_and_suffix(
                A_split, 0, args.image_root_a, args.suffix_a
            ) +
            relabel_extend_and_suffix(
                B_split, 1, args.image_root_b, args.suffix_b
            )
        )

        if reference_keys is None:
            if args.balanced:
                combined = balance_by_label(combined)

            max_n = get_downsample_for_split(args, split)

            combined = downsample(
                combined,
                max_n,
                stratified=(args.split_mode == "stratified")
            )

        merged[split] = combined

    merged["label_mapping"] = {
        0: args.label_a,
        1: args.label_b
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    print("Saved:", args.out_json)
    for split in ["train", "val", "test"]:
        print_stats(split, merged[split])

if __name__ == "__main__":
    main()

# python 6_MergeSplitsForDatasetClassification.py ^
# --splits_a "UniversalSplits\Professions_125k_ISCO_Aligned_1k_Subset\splits_gender_face_stratified.json" ^
# --splits_b "UniversalSplits\StableDiffusion\splits_gender_face_stratified.json" ^
# --image_root_a "F:\ImageRetrieval\Professions_125k_ISCO_Aligned_1k_Subset" ^
# --image_root_b "E:\ImageRetrieval\StableDiffusionGeneratedImages\valid" ^
# --suffix_a "" ^
# --suffix_b "" ^
# --label_a "Professions_125k_ISCO_Aligned_1k_Subset" ^
# --label_b "StableDiffusion" ^
# --out_json "UniversalSplits\DatasetClassification\splits_face_combined_stratified.json"

# python 6_MergeSplitsForDatasetClassification.py ^
# --splits_a "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\Shuffling&Colour\mean_rgb\splits_gender_face_stratified_meanrgb_normalized.json" ^
# --splits_b "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\Shuffling&Colour\mean_rgb\splits_gender_face_stratified_meanrgb_normalized.json" ^
# --image_root_a "F:\ImageRetrieval\Professions_125k_ISCO_Aligned_1k_Subset" ^
# --image_root_b "E:\ImageRetrieval\StableDiffusionGeneratedImages\valid" ^
# --suffix_a "" ^
# --suffix_b "" ^
# --label_a "Professions_125k_ISCO_Aligned_1k_Subset_MeanRGB" ^
# --label_b "StableDiffusion_MeanRGB" ^
# --out_json "UniversalSplits\DatasetClassification\splits_face_combined_stratified_meanrgb.json" ^
# --split_reference "UniversalSplits\DatasetClassification\splits_face_combined_stratified.json"


# python 6_MergeSplitsForDatasetClassification.py ^
# --splits_a "UniversalSplits\Professions_125k_ISCO_Aligned_1k_Subset\splits_gender_face_stratified.json" ^
# --splits_b "UniversalSplits\StableDiffusion\splits_gender_face_stratified.json" ^
# --image_root_a "F:\ImageRetrieval\Professions_125k_ISCO_Aligned_1k_Subset" ^
# --image_root_b "E:\ImageRetrieval\StableDiffusionGeneratedImages\valid" ^
# --suffix_a "" ^
# --suffix_b "" ^
# --label_a "Professions_125k_ISCO_Aligned_1k_Subset" ^
# --label_b "StableDiffusion" ^
# --out_json "UniversalSplits\DatasetClassification\splits_face_combined_stratified_10k.json" ^
# --downsample 10_000 ^
# --seed 42

# python 6_MergeSplitsForDatasetClassification.py ^
# --splits_a "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\Shuffling&Colour\mean_rgb\splits_gender_face_stratified_meanrgb_normalized.json" ^
# --splits_b "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\Shuffling&Colour\mean_rgb\splits_gender_face_stratified_meanrgb_normalized.json" ^
# --image_root_a "F:\ImageRetrieval\Professions_125k_ISCO_Aligned_1k_Subset" ^
# --image_root_b "E:\ImageRetrieval\StableDiffusionGeneratedImages\valid" ^
# --suffix_a "" ^
# --suffix_b "" ^
# --label_a "Professions_125k_ISCO_Aligned_1k_Subset_MeanRGB" ^
# --label_b "StableDiffusion_MeanRGB" ^
# --out_json "UniversalSplits\DatasetClassification\splits_face_combined_stratified_meanrgb_10k.json" ^
# --split_reference "UniversalSplits\DatasetClassification\splits_face_combined_stratified_10k.json"