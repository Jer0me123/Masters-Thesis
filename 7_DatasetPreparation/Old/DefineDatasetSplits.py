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
    """
    Extract a canonical image ID independent of crop type or extension.
    """
    name = Path(image_path).name
    return FACE_TOKEN_RE.sub("", name)

def is_face_annotation(image_path: str, face_hint: str | None):
    """
    Determines whether an annotation corresponds to a derived view
    (e.g. face crop). Uses substring matching, not hard-coded folder names.
    """
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
# Load annotations.jsonl
# --------------------------------------------------

def load_annotations(jsonl_path):
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    p = argparse.ArgumentParser()

    p.add_argument("--annotations", required=True, type=Path)
    p.add_argument("--out_json", required=True, type=Path)

    p.add_argument("--label_key", required=True)
    p.add_argument("--confidence_key", default=None)
    p.add_argument("--min_confidence", type=float, default=None)

    p.add_argument(
        "--annotation_source",
        choices=["face", "nonface"],
        required=True,
        help="Which annotations to trust for labels"
    )

    p.add_argument(
        "--face_path_hint",
        default=None,
        help="Substring that identifies derived views (e.g. 'face', 'crop')"
    )

    p.add_argument("--train_pct", type=float, default=0.7)
    p.add_argument("--val_pct", type=float, default=0.15)
    p.add_argument("--test_pct", type=float, default=0.15)

    p.add_argument(
        "--balance",
        action="store_true",
        help="Downsample classes to equal size before splitting"
    )

    p.add_argument("--split_mode",
                   choices=["random", "stratified"],
                   default="stratified")

    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()
    assert abs(args.train_pct + args.val_pct + args.test_pct - 1.0) < 1e-6

    set_seed(args.seed)

    raw = load_annotations(args.annotations)

    # --------------------------------------------------
    # Group annotations by base image ID
    # --------------------------------------------------

    grouped = defaultdict(list)
    for s in raw:
        base_id = get_base_id(s["image"])
        grouped[base_id].append(s)

    processed = []

    for base_id, entries in grouped.items():
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

        # Confidence filtering
        if args.confidence_key and args.min_confidence is not None:
            if chosen.get(args.confidence_key, 0.0) < args.min_confidence:
                continue

        # Always emit NON-FACE image path
        image_path = fallback["image"] if fallback else chosen["image"]

        processed.append({
            "image": image_path,
            "label": chosen[args.label_key]
        })

    # --------------------------------------------------
    # Encode labels
    # --------------------------------------------------

    label_set = sorted({s["label"] for s in processed})
    label_to_id = {k: i for i, k in enumerate(label_set)}
    id_to_label = {v: k for k, v in label_to_id.items()}

    for s in processed:
        s["label"] = label_to_id[s["label"]]

    # --------------------------------------------------
    # Balanced Distribution
    # --------------------------------------------------

    if args.balance:
        processed = balance_by_label(processed)

    # --------------------------------------------------
    # Split
    # --------------------------------------------------

    if args.split_mode == "stratified":
        buckets = defaultdict(list)
        for s in processed:
            buckets[s["label"]].append(s)

        train, val, test = [], [], []
        for label, group in buckets.items():
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


# python DefineDatasetSplits.py --annotations "E:\ImageRetrieval\StableDiffusionGeneratedImages_Annotations\annotations.jsonl" --out_json "splits_gender_face_stratified.json" --label_key gender --annotation_source face --face_path_hint face --split_mode stratified --seed 42

# python DefineDatasetSplits.py --annotations "E:\ImageRetrieval\StableDiffusionGeneratedImages_Annotations\annotations.jsonl" --out_json "splits_gender_face_stratified_balanced.json" --label_key gender --annotation_source face --face_path_hint face --split_mode stratified --seed 42 --balance


# python DefineDatasetSplits.py --annotations "E:\ImageRetrieval\StableDiffusionGeneratedImages_Annotations\annotations.jsonl" --out_json "splits_gender_face_stratified_balanced.json" --label_key gender --annotation_source face --face_path_hint face --split_mode stratified --seed 42 --balance