import argparse
import json
import random
from pathlib import Path
from collections import defaultdict
import re

# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)

# --------------------------------------------------
# Helpers
# --------------------------------------------------

FACE_TOKEN_RE = re.compile(r"(?:_face)?\.[a-zA-Z0-9]+$")

def get_base_id(image_path: str) -> str:
    name = Path(image_path).name
    return FACE_TOKEN_RE.sub("", name)

def is_face_annotation(image_path: str, face_hint: str | None):
    if face_hint is None:
        return False
    return face_hint.lower() in image_path.lower()

def print_label_distribution(split_name, samples, id_to_label):
    total = len(samples)
    counts = defaultdict(int)

    for s in samples:
        counts[s["label"]] += 1

    print(f"\n[{split_name}]")
    for label_id, count in sorted(counts.items()):
        label_name = id_to_label[label_id]
        pct = 100.0 * count / total if total > 0 else 0.0
        print(f"  {label_name} ({label_id}): {count} ({pct:.2f}%)")

def balance_by_label(samples):
    buckets = defaultdict(list)
    for s in samples:
        buckets[s["label"]].append(s)

    min_count = min(len(v) for v in buckets.values())

    balanced = []
    for label, group in buckets.items():
        random.shuffle(group)
        balanced.extend(group[:min_count])

    random.shuffle(balanced)
    return balanced

# --------------------------------------------------
# Load JSONL
# --------------------------------------------------

def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    p = argparse.ArgumentParser()

    p.add_argument("--annotations", required=True, type=Path)
    p.add_argument("--poses", required=True, type=Path)
    p.add_argument("--out_json", required=True, type=Path)

    p.add_argument("--label_key", required=True)
    p.add_argument("--confidence_key", default=None)
    p.add_argument("--min_confidence", type=float, default=None)

    p.add_argument(
        "--annotation_source",
        choices=["face", "nonface"],
        required=True
    )

    p.add_argument("--face_path_hint", default=None)

    p.add_argument("--train_pct", type=float, default=0.7)
    p.add_argument("--val_pct", type=float, default=0.15)
    p.add_argument("--test_pct", type=float, default=0.15)

    p.add_argument("--balance", action="store_true")

    p.add_argument(
        "--split_mode",
        choices=["random", "stratified"],
        default="stratified"
    )

    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()
    assert abs(args.train_pct + args.val_pct + args.test_pct - 1.0) < 1e-6

    set_seed(args.seed)

    # --------------------------------------------------
    # Load inputs
    # --------------------------------------------------

    raw = load_jsonl(args.annotations)
    poses = load_jsonl(args.poses)

    # Build pose lookup WITHOUT any filtering
    pose_by_id = {
        get_base_id(p["image"]): p for p in poses
    }

    # --------------------------------------------------
    # Group annotations - IDENTICAL to base script
    # --------------------------------------------------

    grouped = defaultdict(list)
    for s in raw:
        base_id = get_base_id(s["image"])
        grouped[base_id].append(s)

    processed = []

    # Process in DETERMINISTIC order (sorted by base_id for reproducibility)
    for base_id in sorted(grouped.keys()):
        entries = grouped[base_id]
        chosen = None
        fallback = None

        for s in entries:
            is_face = is_face_annotation(s["image"], args.face_path_hint)

            if args.annotation_source == "face" and is_face:
                chosen = s
            elif args.annotation_source == "nonface" and not is_face:
                chosen = s

            if not is_face:
                fallback = s

        if chosen is None:
            continue

        # Confidence filtering - IDENTICAL to base script
        if args.confidence_key and args.min_confidence is not None:
            if chosen.get(args.confidence_key, 0.0) < args.min_confidence:
                continue

        # Always use non-face image path - IDENTICAL to base script
        image_path = fallback["image"] if fallback else chosen["image"]

        # Attach pose features (use zeros if pose not available)
        pose = pose_by_id.get(base_id)
        features = (
            pose["normalized_keypoints_with_visibility"]
            if pose is not None
            else [0.0] * 51
        )

        processed.append({
            "image": image_path,
            "label": chosen[args.label_key],
            "features": features
        })

    # --------------------------------------------------
    # Encode labels - IDENTICAL to base script
    # --------------------------------------------------

    label_set = sorted({s["label"] for s in processed})
    label_to_id = {k: i for i, k in enumerate(label_set)}
    id_to_label = {v: k for k, v in label_to_id.items()}

    for s in processed:
        s["label"] = label_to_id[s["label"]]

    # --------------------------------------------------
    # Balance - IDENTICAL to base script
    # --------------------------------------------------

    if args.balance:
        processed = balance_by_label(processed)

    # --------------------------------------------------
    # Split - IDENTICAL to base script
    # --------------------------------------------------

    if args.split_mode == "stratified":
        buckets = defaultdict(list)
        for s in processed:
            buckets[s["label"]].append(s)

        train, val, test = [], [], []
        # Process labels in sorted order for determinism
        for label in sorted(buckets.keys()):
            group = buckets[label]
            random.shuffle(group)
            n = len(group)
            n_train = int(n * args.train_pct)
            n_val = int(n * args.val_pct)

            train.extend(group[:n_train])
            val.extend(group[n_train:n_train + n_val])
            test.extend(group[n_train + n_val:])
    else:
        random.shuffle(processed)
        n = len(processed)
        n_train = int(n * args.train_pct)
        n_val = int(n * args.val_pct)

        train = processed[:n_train]
        val = processed[n_train:n_train + n_val]
        test = processed[n_train + n_val:]

    output = {"train": train, "val": val, "test": test}

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("Saved:", args.out_json)
    print("Label mapping:")
    for k, v in label_to_id.items():
        print(f"  {v}: {k}")

    print(f"\nTrain {len(train)} | Val {len(val)} | Test {len(test)}")
    print("\n=== Label distribution ===")
    print_label_distribution("train", train, id_to_label)
    print_label_distribution("val", val, id_to_label)
    print_label_distribution("test", test, id_to_label)

if __name__ == "__main__":
    main()

# python DefinePoseDatasetSplits.py ^
#   --annotations "E:\ImageRetrieval\StableDiffusionGeneratedImages_Annotations\annotations.jsonl" ^
#   --poses "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\PoseDetection\poses.jsonl" ^
#   --out_json pose_gender.json ^
#   --annotation_source face ^
#   --face_path_hint face ^
#   --label_key gender ^
#   --split_mode stratified ^
#   --balance ^
#   --seed 42


# python DefineDatasetSplits.py --annotations "E:\ImageRetrieval\StableDiffusionGeneratedImages_Annotations\annotations.jsonl" 
# --out_json "splits_gender_face_stratified_balanced.json" --label_key gender --annotation_source face --face_path_hint face --split_mode stratified --seed 42 --balance