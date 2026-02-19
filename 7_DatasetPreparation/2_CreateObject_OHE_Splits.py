import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set

def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file into list of dictionaries"""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def extract_all_classes(detections: List[Dict]) -> List[str]:
    """Extract all unique object classes from detections, sorted alphabetically"""
    all_classes = set()
    for record in detections:
        all_classes.update(record["detections"].keys())
    return sorted(all_classes)

def load_class_list_from_json(path: Path) -> List[str]:
    """
    Load class list from JSON file with format {"0": "Class1", "1": "Class2", ...}
    Returns list ordered by integer keys.
    """
    with open(path, "r", encoding="utf-8") as f:
        class_dict = json.load(f)
    
    # Convert string keys to integers and sort
    sorted_items = sorted(class_dict.items(), key=lambda x: int(x[0]))
    return [name for _, name in sorted_items]

def create_one_hot(detections: Dict[str, List], class_list: List[str]) -> List[int]:
    """
    Create one-hot encoded vector for object presence.
    1 if class is present (at least one detection), 0 otherwise.
    """
    return [1 if cls in detections else 0 for cls in class_list]

def create_one_hot_normalized(detections: Dict[str, List], class_list: List[str]) -> List[float]:
    """
    Create normalized one-hot encoded vector (values as floats 0.0 or 1.0).
    """
    return [1.0 if cls in detections else 0.0 for cls in class_list]

def create_count_vector(detections: Dict[str, List], class_list: List[str]) -> List[int]:
    """
    Create count vector for number of detections per class.
    """
    return [len(detections.get(cls, [])) for cls in class_list]

def create_max_confidence_vector(detections: Dict[str, List], class_list: List[str]) -> List[float]:
    """
    Create vector with maximum confidence per class.
    0.0 if class not present, max confidence otherwise.
    """
    result = []
    for cls in class_list:
        if cls in detections:
            max_conf = max(det["confidence"] for det in detections[cls])
            result.append(max_conf)
        else:
            result.append(0.0)
    return result

def main():
    parser = argparse.ArgumentParser(
        description="Extract one-hot encoded object presence from detection JSONL"
    )
    
    parser.add_argument("--detections", required=True, type=Path,
                       help="Input JSONL file with object detections")
    parser.add_argument("--out_jsonl", required=True, type=Path,
                       help="Output JSONL file with one-hot encoded features")
    parser.add_argument("--encoding_type", 
                       choices=["binary", "normalized", "counts", "max_confidence"],
                       default="normalized",
                       help="Type of encoding: binary (0/1), normalized (0.0/1.0), counts, or max_confidence")
    parser.add_argument("--min_confidence", type=float, default=None,
                       help="Minimum confidence threshold for considering a detection")
    parser.add_argument("--class_list", type=Path, default=None,
                       help="Class list file: JSON with format {\"0\": \"Class1\", \"1\": \"Class2\"} or TXT with one class per line")
    parser.add_argument("--output_class_list", type=Path, default=None,
                       help="Save the class list to this file (JSON format)")
    parser.add_argument("--output_class_list_txt", type=Path, default=None,
                       help="Save the class list to this file (TXT format, one per line)")
    
    args = parser.parse_args()
    
    # Load detections
    print(f"Loading detections from {args.detections}")
    detections_data = load_jsonl(args.detections)
    print(f"Loaded {len(detections_data)} records")
    
    # Filter by confidence if specified
    if args.min_confidence is not None:
        print(f"Filtering detections with confidence >= {args.min_confidence}")
        for record in detections_data:
            filtered_detections = {}
            for cls, dets in record["detections"].items():
                filtered = [d for d in dets if d["confidence"] >= args.min_confidence]
                if filtered:
                    filtered_detections[cls] = filtered
            record["detections"] = filtered_detections
    
    # Determine class list
    if args.class_list:
        print(f"Loading class list from {args.class_list}")
        if args.class_list.suffix.lower() == '.json':
            class_list = load_class_list_from_json(args.class_list)
        else:
            # Assume TXT format
            with open(args.class_list, "r", encoding="utf-8") as f:
                class_list = [line.strip() for line in f if line.strip()]
        print(f"Using provided class list with {len(class_list)} classes")
    else:
        print("Extracting class list from detections")
        class_list = extract_all_classes(detections_data)
        print(f"Found {len(class_list)} unique classes in detections")
    
    print(f"Class list: {', '.join(class_list[:10])}" + 
          (f"... (+{len(class_list)-10} more)" if len(class_list) > 10 else ""))
    
    # Check which classes from class_list actually appear in detections
    detected_classes = extract_all_classes(detections_data)
    classes_in_list_not_detected = set(class_list) - set(detected_classes)
    classes_detected_not_in_list = set(detected_classes) - set(class_list)
    
    if classes_in_list_not_detected:
        print(f"\nNote: {len(classes_in_list_not_detected)} classes from class_list not found in detections (will be encoded as 0)")
        if len(classes_in_list_not_detected) <= 10:
            print(f"  Classes: {', '.join(sorted(classes_in_list_not_detected))}")
    
    if classes_detected_not_in_list:
        print(f"\nWarning: {len(classes_detected_not_in_list)} classes in detections not found in class_list (will be IGNORED)")
        if len(classes_detected_not_in_list) <= 10:
            print(f"  Classes: {', '.join(sorted(classes_detected_not_in_list))}")
        else:
            print(f"  First 10: {', '.join(sorted(list(classes_detected_not_in_list))[:10])}")
    
    # Save class list if requested
    if args.output_class_list:
        class_dict = {str(i): cls for i, cls in enumerate(class_list)}
        with open(args.output_class_list, "w", encoding="utf-8") as f:
            json.dump(class_dict, f, indent=2)
        print(f"\nSaved class list (JSON) to {args.output_class_list}")
    
    if args.output_class_list_txt:
        with open(args.output_class_list_txt, "w", encoding="utf-8") as f:
            for cls in class_list:
                f.write(f"{cls}\n")
        print(f"Saved class list (TXT) to {args.output_class_list_txt}")
    
    # Create output records
    output_records = []
    
    encoding_funcs = {
        "binary": create_one_hot,
        "normalized": create_one_hot_normalized,
        "counts": create_count_vector,
        "max_confidence": create_max_confidence_vector,
    }
    
    encode_func = encoding_funcs[args.encoding_type]
    
    for record in detections_data:
        encoding = encode_func(record["detections"], class_list)
        
        output_record = {
            "image": record["image"],
            "object_encoding": encoding,
            "num_classes_present": sum(1 for x in encoding if x > 0),
        }
        
        output_records.append(output_record)
    
    # Write output
    print(f"Writing to {args.out_jsonl}")
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for record in output_records:
            f.write(json.dumps(record) + "\n")
    
    # Print statistics
    print(f"\n=== Statistics ===")
    print(f"Total images: {len(output_records)}")
    print(f"Total classes in encoding: {len(class_list)}")
    
    avg_classes = sum(r["num_classes_present"] for r in output_records) / len(output_records)
    print(f"Average classes present per image: {avg_classes:.2f}")
    
    # Class frequency
    class_counts = defaultdict(int)
    for record in output_records:
        for i, cls in enumerate(class_list):
            if record["object_encoding"][i] > 0:
                class_counts[cls] += 1
    
    classes_with_detections = len([c for c in class_counts.values() if c > 0])
    classes_without_detections = len(class_list) - classes_with_detections
    
    print(f"Classes with at least 1 detection: {classes_with_detections}/{len(class_list)}")
    print(f"Classes with 0 detections: {classes_without_detections}/{len(class_list)}")
    
    if classes_with_detections > 0:
        print(f"\nTop 10 most frequent classes:")
        for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            pct = 100.0 * count / len(output_records)
            print(f"  {cls}: {count} ({pct:.1f}%)")
    
    # Show least frequent among those that have detections
    classes_with_counts = [(cls, cnt) for cls, cnt in class_counts.items() if cnt > 0]
    if len(classes_with_counts) > 10:
        print(f"\nBottom 10 least frequent classes (among those with detections):")
        for cls, count in sorted(classes_with_counts, key=lambda x: x[1])[:10]:
            pct = 100.0 * count / len(output_records)
            print(f"  {cls}: {count} ({pct:.1f}%)")

if __name__ == "__main__":
    main()

# # Extract one-hot encoded object presence (normalized)
# python 2_CreateObject_OHE_Splits.py ^
#   --detections "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\ObjectDetection\detections.jsonl" ^
#   --out_jsonl "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\ObjectDetection\binary_objects_detected.jsonl" ^
#   --encoding_type binary ^
#   --class_list "openimagesv7_classes_raw.json"

# # Then enrich your splits
# python 3_CreatePoses_MeanRGB_OHE_Splits.py ^
#   --splits "UniversalSplits\StableDiffusion\splits_gender_face_stratified.json" ^
#   --features "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\ObjectDetection\binary_objects_detected.jsonl" ^
#   --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\ObjectDetection\splits_gender_face_stratified_objects_OHE.json" ^
#   --feature_keys object_encoding ^
#   --output_key features





# # Extract one-hot encoded object presence (normalized)
# python 2_CreateObject_OHE_Splits.py ^
#   --detections "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\ObjectDetection_LabelRestricted\detections.jsonl" ^
#   --out_jsonl "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\ObjectDetection_LabelRestricted\binary_objects_detected.jsonl" ^
#   --encoding_type binary ^
#   --class_list "openimagesv7_classes_raw.json"

# # Then enrich your splits
# python 3_CreatePoses_MeanRGB_OHE_Splits.py ^
#   --splits "UniversalSplits\StableDiffusion\splits_gender_face_stratified.json" ^
#   --features "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\ObjectDetection_LabelRestricted\binary_objects_detected.jsonl" ^
#   --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\ObjectDetection_LabelRestricted\splits_gender_face_stratified_objects_OHE.json" ^
#   --feature_keys object_encoding ^
#   --output_key features