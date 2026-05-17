import argparse
import json
import re
from pathlib import Path

# --------------------------------------------------
# Helpers
# --------------------------------------------------

FACE_TOKEN_RE = re.compile(r"(?:_face)?\.[a-zA-Z0-9]+$")

def get_base_id(image_path: str) -> str:
    """Extract a canonical image ID independent of crop type or extension."""
    name = Path(image_path).name
    return FACE_TOKEN_RE.sub("", name)

def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def print_stats(split_name, samples, output_key="caption"):
    total = len(samples)
    with_caption = sum(1 for s in samples if s.get(output_key, "") != "")
    without_caption = total - with_caption

    print(f"[{split_name}]")
    print(f"  Total: {total}")
    print(f"  With caption: {with_caption} ({100.0 * with_caption / total:.2f}%)")
    print(f"  Without caption: {without_caption} ({100.0 * without_caption / total:.2f}%)")

    if with_caption > 0:
        avg_length = sum(len(s.get(output_key, "")) for s in samples if s.get(output_key, "")) / with_caption
        print(f"  Average caption length: {avg_length:.1f} characters")

    from collections import defaultdict
    by_label = defaultdict(lambda: {"total": 0, "with_caption": 0})
    for s in samples:
        lbl = str(s.get("label", "unknown"))
        by_label[lbl]["total"] += 1
        if s.get(output_key, "") != "":
            by_label[lbl]["with_caption"] += 1
    print(f"  Per-label breakdown:")
    for lbl in sorted(by_label.keys()):
        d = by_label[lbl]
        pct = 100.0 * d["with_caption"] / d["total"] if d["total"] else 0
        print(f"    Label {lbl}: {d['with_caption']}/{d['total']} ({pct:.1f}%) have captions")

# --------------------------------------------------
# Build caption lookup from a single JSONL file
# --------------------------------------------------

def build_caption_lookup(captions_path: Path, caption_type: str, sentence_index: int, default_caption: str) -> dict:
    """Returns {base_id: caption_text} for one captions JSONL file."""
    data = load_jsonl(captions_path)
    lookup = {}
    for record in data:
        filename = record.get("filename") or record.get("image", "")
        base_id = get_base_id(filename)

        sentences = record.get("sentences", [])
        if sentences and sentence_index < len(sentences):
            caption_text = sentences[sentence_index].get(caption_type, default_caption)
        else:
            caption_text = default_caption

        lookup[base_id] = caption_text

    print(f"  Loaded {len(lookup)} caption records from {captions_path.name}")
    return lookup

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=(
            "Enrich existing splits with captions from caption JSONL file(s).\n\n"
            "MODE A - single file (original behaviour):\n"
            "  --captions path/to/captions.jsonl\n\n"
            "MODE B - per-label files (multi-dataset), two sub-options:\n"
            "  B1: --captions_dir DIR --captions_map_file map.json   [recommended on Windows]\n"
            "  B2: --captions_dir DIR --captions_map '{\"0\": ...}'   [Linux/Mac inline JSON]\n\n"
            "map.json format:\n"
            "  {\n"
            "    \"0\": \"Professions_125k/ImageCaptioning/captions.jsonl\",\n"
            "    \"1\": \"StableDiffusionImages/ImageCaptioning/captions.jsonl\",\n"
            "    \"2\": \"Coco/ImageCaptioning/captions.jsonl\"\n"
            "  }"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--splits", required=True, type=Path,
                   help="JSON file with train/val/test splits")

    # Mode A (original)
    p.add_argument("--captions", type=Path, default=None,
                   help="[Mode A] Single JSONL file with caption data (original behaviour)")

    # Mode B (multi-dataset)
    p.add_argument("--captions_dir", type=Path, default=None,
                   help="[Mode B] Base directory for caption JSONL files")
    p.add_argument("--captions_map", type=str, default=None,
                   help=(
                       "[Mode B] Inline JSON string: label -> relative captions path. "
                       "Use --captions_map_file on Windows CMD to avoid quoting issues."
                   ))
    p.add_argument("--captions_map_file", type=Path, default=None,
                   help=(
                       "[Mode B] Path to a JSON file containing label -> relative captions path. "
                       "Recommended on Windows CMD. "
                       "Example: {\"0\": \"Prof/captions.jsonl\", \"1\": \"SD/captions.jsonl\"}"
                   ))

    # Shared
    p.add_argument("--out_json", required=True, type=Path,
                   help="Output JSON file with captions added")
    p.add_argument("--caption_type",
                   choices=["raw", "remapped"],
                   default="remapped",
                   help="Which caption field to use: 'raw' or 'remapped' (default: remapped)")
    p.add_argument("--sentence_index", type=int, default=0,
                   help="Which sentence to use if multiple exist (default: 0)")
    p.add_argument("--output_key", default="caption",
                   help="Key name for caption in output samples (default: 'caption')")
    p.add_argument("--default_caption", default="",
                   help="Default caption if image not found (default: empty string)")

    args = p.parse_args()

    # Validate mode selection
    use_single = args.captions is not None
    use_multi  = args.captions_dir is not None or args.captions_map is not None or args.captions_map_file is not None

    if use_single and use_multi:
        p.error("Specify either --captions (Mode A) OR Mode B args, not both.")
    if not use_single and not use_multi:
        p.error("Must specify either --captions (Mode A) or --captions_dir + --captions_map/--captions_map_file (Mode B).")
    if use_multi:
        if not args.captions_dir:
            p.error("Mode B requires --captions_dir.")
        if args.captions_map and args.captions_map_file:
            p.error("Provide either --captions_map or --captions_map_file, not both.")
        if not args.captions_map and not args.captions_map_file:
            p.error("Mode B requires either --captions_map or --captions_map_file.")

    # Load splits
    print(f"\nLoading splits from: {args.splits}")
    splits = load_json(args.splits)
    label_mapping = splits.get("label_mapping", {})
    print(f"Label mapping: {label_mapping}")

    # Build caption lookup(s)
    if use_single:
        print(f"\n[Mode A] Loading captions from: {args.captions}")
        shared_lookup = build_caption_lookup(
            args.captions, args.caption_type, args.sentence_index, args.default_caption
        )
        caption_lookups = None

    else:
        if args.captions_map_file:
            print(f"\n[Mode B] Reading captions map from file: {args.captions_map_file}")
            captions_map = load_json(args.captions_map_file)
        else:
            try:
                captions_map = json.loads(args.captions_map)
            except json.JSONDecodeError as e:
                p.error(
                    f"--captions_map is not valid JSON: {e}\n"
                    "Tip: on Windows CMD use --captions_map_file with a .json file instead."
                )

        print(f"\n[Mode B] Loading caption files from: {args.captions_dir}")
        caption_lookups: dict = {}
        for label_str, rel_path in captions_map.items():
            captions_path = args.captions_dir / rel_path
            if not captions_path.exists():
                print(f"  WARNING: Caption file not found for label {label_str}: {captions_path}")
                caption_lookups[label_str] = {}
            else:
                caption_lookups[label_str] = build_caption_lookup(
                    captions_path, args.caption_type, args.sentence_index, args.default_caption
                )
        shared_lookup = None

    # Enrich each split
    enriched = {}
    total_found = 0
    total_missing = 0

    for split_name in ["train", "val", "test"]:
        if split_name not in splits:
            continue

        enriched_samples = []
        split_found = 0
        split_missing = 0

        for sample in splits[split_name]:
            label_str = str(sample.get("label", ""))
            base_id = get_base_id(sample["image"])

            lookup = shared_lookup if use_single else caption_lookups.get(label_str, {})
            caption = lookup.get(base_id, args.default_caption)

            enriched_sample = sample.copy()
            enriched_sample[args.output_key] = caption

            if caption != args.default_caption:
                split_found += 1
            else:
                split_missing += 1

            enriched_samples.append(enriched_sample)

        enriched[split_name] = enriched_samples
        total_found += split_found
        total_missing += split_missing
        print(f"  [{split_name}] matched {split_found}, missing {split_missing}")

    # Preserve label_mapping
    if "label_mapping" in splits:
        enriched["label_mapping"] = splits["label_mapping"]

    # Save output
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2)

    print(f"\nSaved enriched splits to: {args.out_json}")
    print(f"Total matched: {total_found} | Total missing: {total_missing}")

    print(f"\n=== Caption coverage per split ===")
    for split_name in ["train", "val", "test"]:
        if split_name in enriched:
            print_stats(split_name, enriched[split_name], args.output_key)
            print()

    print("=== Sample outputs ===")
    if "train" in enriched and enriched["train"]:
        print("\nFirst 3 train samples:")
        for i, sample in enumerate(enriched["train"][:3]):
            print(f"\n  Sample {i+1}:")
            print(f"    Image:   {sample['image']}")
            print(f"    Label:   {sample.get('label', 'N/A')} "
                  f"({label_mapping.get(str(sample.get('label', '')), 'unknown')})")
            print(f"    Caption: {sample.get(args.output_key, '')[:120]}")


if __name__ == "__main__":
    main()

# python 4_CreateCaption_Splits.py ^
#   --splits "" ^
#   --captions_dir "" ^
#   --captions_map_file "captions_map.json" ^
#   --out_json ""