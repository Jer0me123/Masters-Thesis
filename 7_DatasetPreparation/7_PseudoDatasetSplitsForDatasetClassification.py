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

def resolve_image_path(img_path: str, image_root: Path, suffix: str | None):
    p = Path(img_path)

    if not p.is_absolute():
        p = image_root / p

    p = apply_suffix(p, suffix)
    return str(p)

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
# Core logic
# --------------------------------------------------

def assign_pseudo_labels(samples, balanced=True):
    samples = list(samples)
    n = len(samples)

    if balanced:
        labels = [0] * (n // 2) + [1] * (n - n // 2)
        random.shuffle(labels)
    else:
        labels = [random.randint(0, 1) for _ in range(n)]

    out = []
    for s, lbl in zip(samples, labels):
        record = dict(s)
        record["label"] = lbl
        out.append(record)

    return out

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

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Create pseudo-dataset splits by random label assignment"
    )

    p.add_argument("--splits", required=True, type=Path)
    p.add_argument("--image_root", required=True, type=Path)
    p.add_argument("--suffix", default=None)

    p.add_argument("--out_json", required=True, type=Path)

    p.add_argument("--label_a", default="PseudoDatasetA")
    p.add_argument("--label_b", default="PseudoDatasetB")

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

    args = p.parse_args()
    set_seed(args.seed)

    splits = load_splits(args.splits)

    out = {}

    for split in ["train", "val", "test"]:
        samples = splits[split]

        # Assign arbitrary dataset labels
        samples = assign_pseudo_labels(samples, balanced=args.balanced)

        # Resolve image paths
        resolved = []
        for s in samples:
            record = dict(s)
            record["image"] = resolve_image_path(
                record["image"],
                args.image_root,
                args.suffix
            )
            resolved.append(record)

        # Optional downsampling
        max_n = get_downsample_for_split(args, split)
        resolved = downsample(
            resolved,
            max_n,
            stratified=(args.split_mode == "stratified")
        )

        out[split] = resolved

    out["label_mapping"] = {
        0: args.label_a,
        1: args.label_b
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Saved:", args.out_json)
    for split in ["train", "val", "test"]:
        print_stats(split, out[split])

if __name__ == "__main__":
    main()

# python 7_PseudoDatasetSplitsForDatasetClassification.py ^
# --splits "UniversalSplits\Professions_125k_ISCO_Aligned_1k_Subset\splits_gender_face_stratified.json" ^
# --out_json "UniversalSplits\PseudoDatasetClassification\ISCO_splits_face_combined_stratified.json" ^
# --image_root "F:\ImageRetrieval\Professions_125k_ISCO_Aligned_1k_Subset" ^
# --balanced ^
# --seed 42 ^
# --label_a "PseudoDatase1" ^
# --label_b "PseudoDatase2"


# python 7_PseudoDatasetSplitsForDatasetClassification.py ^
# --splits "UniversalSplits\StableDiffusion\splits_gender_face_stratified.json" ^
# --out_json "UniversalSplits\PseudoDatasetClassification\SD_splits_face_combined_stratified.json" ^
# --image_root "E:\ImageRetrieval\StableDiffusionGeneratedImages\valid" ^
# --balanced ^
# --seed 42 ^
# --label_a "PseudoDatase1" ^
# --label_b "PseudoDatase2"