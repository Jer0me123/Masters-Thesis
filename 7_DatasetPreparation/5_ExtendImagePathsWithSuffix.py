import argparse
import json
from pathlib import Path

def add_suffix(path: str, suffix: str) -> Path:
    """
    Insert suffix before file extension.
    """
    p = Path(path)
    return p.with_name(p.stem + suffix + p.suffix)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_json", required=True, type=Path)
    parser.add_argument("--out_json", required=True, type=Path)
    parser.add_argument("--suffix", required=True,
                        help="Suffix to append before extension, e.g. _depth")
    parser.add_argument(
        "--base_path",
        required=True,
        type=Path,
        help="Base directory to prepend to image paths"
    )

    args = parser.parse_args()

    with open(args.in_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    for split in ["train", "val", "test"]:
        for record in data[split]:
            # 1) append suffix
            p = add_suffix(record["image"], args.suffix)

            # 2) prepend base path
            full_path = args.base_path / p

            # store as string (JSON-friendly)
            record["image"] = str(full_path)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("Saved:", args.out_json)

if __name__ == "__main__":
    main()