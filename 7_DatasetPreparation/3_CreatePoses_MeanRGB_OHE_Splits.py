import argparse
import json
from pyexpat import features
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

def is_numeric_list(value):
    """Check if value is a list of numbers"""
    return isinstance(value, list) and all(isinstance(x, (int, float)) for x in value)

def get_default_value(sample_value):
    """Generate appropriate default value based on sample type"""
    if sample_value is None:
        return None
    elif isinstance(sample_value, list):
        if is_numeric_list(sample_value):
            return [0.0] * len(sample_value)
        else:
            return []
    elif isinstance(sample_value, dict):
        return {}
    elif isinstance(sample_value, (int, float)):
        return 0.0
    elif isinstance(sample_value, str):
        return ""
    else:
        return None

def print_stats(split_name, samples, feature_keys):
    total = len(samples)
    
    print(f"[{split_name}]")
    print(f"  Total: {total}")
    
    for key in feature_keys:
        if key not in samples[0]:
            continue
            
        sample_val = samples[0][key]
        default_val = get_default_value(sample_val)
        
        # Count samples with non-default values
        if is_numeric_list(sample_val):
            with_feature = sum(1 for s in samples if s.get(key) != default_val)
        else:
            with_feature = sum(1 for s in samples if s.get(key, default_val) != default_val)
        
        without_feature = total - with_feature
        
        print(f"  With '{key}': {with_feature} ({100.0 * with_feature / total:.2f}%)")
        print(f"  Without '{key}': {without_feature} ({100.0 * without_feature / total:.2f}%)")

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Enrich existing splits with features from JSONL file"
    )

    p.add_argument("--splits", required=True, type=Path,
                   help="JSON file with train/val/test splits")
    p.add_argument("--features", required=True, type=Path,
                   help="JSONL file with feature data")
    p.add_argument("--out_json", required=True, type=Path,
                   help="Output JSON file with features added")
    p.add_argument("--feature_keys", nargs="+", default=None,
                   help="Specific feature keys to extract (default: all except 'image')")
    p.add_argument("--output_key", default=None,
                   help="Single output key name (only valid with single feature_key)")

    args = p.parse_args()

    # Load data
    splits = load_json(args.splits)
    features = load_jsonl(args.features)

    # Build feature lookup by base image ID
    feature_by_id = {
        get_base_id(f["image"]): f for f in features
    }

    # Determine which keys to extract
    if features:
        sample_feature = features[0]
        available_keys = [k for k in sample_feature.keys() if k != "image"]
        
        if args.feature_keys:
            feature_keys = args.feature_keys
            # Validate requested keys exist
            invalid_keys = set(feature_keys) - set(available_keys)
            if invalid_keys:
                print(f"Warning: Keys not found in feature file: {invalid_keys}")
                feature_keys = [k for k in feature_keys if k in available_keys]
        else:
            feature_keys = available_keys
        
        if not feature_keys:
            raise ValueError("No valid feature keys to extract")
        
        print(f"Extracting features: {feature_keys}")
        
        # Validate output_key usage
        if args.output_key and len(feature_keys) != 1:
            raise ValueError("--output_key can only be used with a single --feature_keys")
    else:
        raise ValueError("Feature file is empty")

    # Determine output key mapping
    if args.output_key and len(feature_keys) == 1:
        key_mapping = {feature_keys[0]: args.output_key}
    else:
        key_mapping = {k: k for k in feature_keys}

    # Get sample values for default generation
    sample_feature_data = features[0] if features else {}
    default_values = {
        k: get_default_value(sample_feature_data.get(k))
        for k in feature_keys
    }

    # Enrich each split
    enriched = {}
    
    for split_name in ["train", "val", "test"]:
        if split_name not in splits:
            continue
            
        enriched_samples = []
        
        for sample in splits[split_name]:
            base_id = get_base_id(sample["image"])
            feature_data = feature_by_id.get(base_id)
            
            # Start with existing sample data
            enriched_sample = sample.copy()
            
            # Add requested features
            for src_key, dst_key in key_mapping.items():
                if feature_data is not None and src_key in feature_data:
                    enriched_sample[dst_key] = feature_data[src_key]
                else:
                    enriched_sample[dst_key] = default_values[src_key]
            
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
    print(f"\n=== Feature coverage ===")
    for split_name in ["train", "val", "test"]:
        if split_name in enriched:
            print_stats(split_name, enriched[split_name], list(key_mapping.values()))
            print()


if __name__ == "__main__":
    main()

# python 3_CreatePoses_MeanRGB_OHE_Splits.py ^
#   --splits "splits_gender_face_stratified.json" ^
#   --features "Fposes.jsonl" ^
#   --out_json "splits_gender_face_stratified_poses_normalized.json" ^
#   --feature_keys normalized_keypoints_with_visibility ^
#   --output_key features
