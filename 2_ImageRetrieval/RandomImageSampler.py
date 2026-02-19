"""
RandomImageSampler_Bitset.py
============================

True O(target_count) sampler for numeric Re-LAION shards.
Handles multi-GB JSONL exclusion files using bitsets.

Memory usage ≈ few MB.
"""
import os
import random
import shutil
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from bitarray import bitarray


# ------------------------------------------------------------
# Build exclusion bitsets (streaming JSONL)
# ------------------------------------------------------------

import os
import json
from tqdm import tqdm
from bitarray import bitarray


def build_exclusion_bitsets(jsonl_path, shard_dirs, max_index):

    # Normalize shard directory names once
    shard_map = {
        Path(d).name: bitarray(max_index + 1)
        for d in shard_dirs
    }

    for ba in shard_map.values():
        ba.setall(0)

    file_size = os.path.getsize(jsonl_path)

    print("[INFO] Streaming exclusion JSONL...")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        with tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            desc="Building exclusion index",
        ) as pbar:

            for line in f:
                try:
                    obj = json.loads(line)

                    for r in obj.get("results", []):
                        path_str = r["image_path"]

                        # Fast shard extraction
                        # Example: G:\Thesis\0002_images\1234567.jpg
                        parts = path_str.split("\\")
                        shard_name = parts[-2]  # 0002_images
                        filename = parts[-1]    # 1234567.jpg

                        if shard_name in shard_map:
                            idx = int(filename.split(".")[0])
                            if 0 <= idx <= max_index:
                                shard_map[shard_name][idx] = 1

                except:
                    pass

                pbar.update(len(line))

    print("[INFO] Exclusion bitsets built.")

    return shard_map




# ------------------------------------------------------------
# Sampling
# ------------------------------------------------------------

def run(args):

    random.seed(args.seed)

    shard_dirs = [Path(d).resolve() for d in args.image_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bitsets = build_exclusion_bitsets(
        args.jsonl_path,
        shard_dirs,
        args.max_index
    )

    saved = 0
    attempts = 0
    max_attempts = args.target_count * 500

    print(f"[INFO] Sampling {args.target_count:,} images...")

    with tqdm(total=args.target_count) as pbar:
        while saved < args.target_count and attempts < max_attempts:
            attempts += 1

            shard = random.choice(shard_dirs)
            idx = random.randint(0, args.max_index)

            # skip if excluded
            if bitsets[shard.name][idx]:
                continue

            img_path = shard / f"{idx}.jpg"

            if not img_path.exists():
                continue

            shard_prefix = shard.name.split("_")[0]   # "0000" from "0000_images"
            new_name = f"{shard_prefix}_{img_path.name}"
            dst = output_dir / new_name

            shutil.copy2(img_path, dst)

            bitsets[shard.name][idx] = 1  # prevent duplicates
            saved += 1
            pbar.update(1)

    print("\n" + "="*60)
    print(f"Finished.")
    print(f"Saved: {saved:,}")
    print(f"Attempts: {attempts:,}")
    print("="*60)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="True O(target_count) sampler using exclusion bitsets."
    )

    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--image_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--target_count", type=int, default=10000)
    parser.add_argument("--max_index", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    run(args)


# python RandomImageSampler.py ^
#     --jsonl_path "E:\ImageRetrieval\Professions_125k_ISCO_Aligned\ISCO_aligned_125k_retrieval_results_batchsize_10.jsonl" ^
#     --image_dirs "G:\Thesis\0000_images" "G:\Thesis\0001_images" "G:\Thesis\0002_images" "G:\Thesis\0003_images" ^
#     --output_dir "F:\ImageRetrieval\ReLaion5B_Random10k" ^
#     --target_count 10000 ^
#     --max_index 17_000_000 ^
#     --seed 42
