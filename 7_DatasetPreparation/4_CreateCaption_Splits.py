import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

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

def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def print_stats(split_name, samples):
    total = len(samples)
    with_caption = sum(1 for s in samples if s.get("caption", "") != "")
    without_caption = total - with_caption
    
    print(f"[{split_name}]")
    print(f"  Total: {total}")
    print(f"  With caption: {with_caption} ({100.0 * with_caption / total:.2f}%)")
    print(f"  Without caption: {without_caption} ({100.0 * without_caption / total:.2f}%)")
    
    if with_caption > 0:
        avg_length = sum(len(s.get("caption", "")) for s in samples if s.get("caption", "")) / with_caption
        print(f"  Average caption length: {avg_length:.1f} characters")

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Enrich existing splits with captions from captions JSONL"
    )

    p.add_argument("--splits", required=True, type=Path,
                   help="JSON file with train/val/test splits")
    p.add_argument("--captions", required=True, type=Path,
                   help="JSONL file with caption data")
    p.add_argument("--out_json", required=True, type=Path,
                   help="Output JSON file with captions added")
    p.add_argument("--caption_type", 
                   choices=["raw", "remapped"],
                   default="remapped",
                   help="Which caption to use: 'raw' or 'remapped'")
    p.add_argument("--sentence_index", type=int, default=0,
                   help="Which sentence to use if multiple (default: 0 for first)")
    p.add_argument("--output_key", default="caption",
                   help="Key name for caption in output (default: 'caption')")
    p.add_argument("--default_caption", default="",
                   help="Default caption if not found (default: empty string)")

    args = p.parse_args()

    # Load data
    print(f"Loading splits from {args.splits}")
    splits = load_json(args.splits)
    
    print(f"Loading captions from {args.captions}")
    captions_data = load_jsonl(args.captions)
    print(f"Loaded {len(captions_data)} caption records")

    # Build caption lookup by base image ID
    caption_by_id = {}
    
    for record in captions_data:
        # The captions file uses "filename" instead of "image"
        base_id = get_base_id(record["filename"])
        
        # Extract the requested caption
        sentences = record.get("sentences", [])
        if sentences and args.sentence_index < len(sentences):
            sentence = sentences[args.sentence_index]
            caption_text = sentence.get(args.caption_type, "")
        else:
            caption_text = args.default_caption
        
        caption_by_id[base_id] = caption_text

    print(f"Built caption lookup for {len(caption_by_id)} images")

    # Enrich each split
    enriched = {}
    
    for split_name in ["train", "val", "test"]:
        if split_name not in splits:
            continue
            
        enriched_samples = []
        
        for sample in splits[split_name]:
            base_id = get_base_id(sample["image"])
            caption = caption_by_id.get(base_id, args.default_caption)
            
            # Start with existing sample data
            enriched_sample = sample.copy()
            enriched_sample[args.output_key] = caption
            
            enriched_samples.append(enriched_sample)
        
        enriched[split_name] = enriched_samples

    # Preserve label mapping if it exists
    if "label_mapping" in splits:
        enriched["label_mapping"] = splits["label_mapping"]

    # Save output
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2)

    print(f"\nSaved: {args.out_json}")
    
    # Print statistics
    print(f"\n=== Caption coverage ===")
    for split_name in ["train", "val", "test"]:
        if split_name in enriched:
            print_stats(split_name, enriched[split_name])
            print()
    
    # Show some examples
    print("=== Sample outputs ===")
    if "train" in enriched and enriched["train"]:
        print("\nFirst 3 train samples:")
        for i, sample in enumerate(enriched["train"][:3]):
            print(f"\n  Sample {i+1}:")
            print(f"    Image: {sample['image']}")
            print(f"    Caption: {sample.get(args.output_key, '')}")
            if "label" in sample:
                print(f"    Label: {sample['label']}")


if __name__ == "__main__":
    main()

# Add remapped captions to your splits
# python 4_CreateCaption_Splits.py ^
#   --splits "UniversalSplits\StableDiffusion\splits_gender_face_stratified.json" ^
#   --captions "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\ImageCaptioning\captions.jsonl" ^
#   --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\ImageCaptioning\splits_gender_face_stratified_captions_remapped.json" ^
#   --caption_type remapped