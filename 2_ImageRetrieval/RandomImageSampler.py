"""
RandomImageSampler_Bitset.py
============================

Supports two sampling modes:

1. SHARD mode (--mode shard):
   True O(target_count) sampler for numeric Re-LAION shards.
   Handles multi-GB JSONL exclusion files using bitsets.
   Memory usage ≈ few MB.

2. SIMPLE mode (--mode simple):
   Reproducible random sampler for arbitrary image directories.
   Supports any image filenames. No exclusion file required.
   Seed-controlled for full reproducibility.
"""

import os
import random
import shutil
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from bitarray import bitarray


# ============================================================
# SHARD MODE — exclusion bitset builder
# ============================================================

def build_exclusion_bitsets(jsonl_path, shard_dirs, max_index):
    """
    Streams a JSONL exclusion file and builds per-shard bitsets
    marking which image indices have already been used.

    Memory usage is O(max_index / 8) bytes per shard — typically a few MB.
    """
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
                        parts = path_str.split("\\")
                        shard_name = parts[-2]   # e.g. "0002_images"
                        filename   = parts[-1]   # e.g. "1234567.jpg"

                        if shard_name in shard_map:
                            idx = int(filename.split(".")[0])
                            if 0 <= idx <= max_index:
                                shard_map[shard_name][idx] = 1
                except Exception:
                    pass

                pbar.update(len(line))

    print("[INFO] Exclusion bitsets built.")
    return shard_map


def run_shard(args):
    """
    Shard-based sampling for numeric Re-LAION image directories.

    Images are identified by integer filename (e.g. 1234567.jpg).
    A JSONL exclusion file marks already-used indices via bitsets.
    After sampling, each selected index is marked to prevent duplicates.
    """
    random.seed(args.seed)

    shard_dirs = [Path(d).resolve() for d in args.image_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bitsets = build_exclusion_bitsets(
        args.jsonl_path,
        shard_dirs,
        args.max_index,
    )

    saved       = 0
    attempts    = 0
    max_attempts = args.target_count * 500

    print(f"[INFO] Sampling {args.target_count:,} images (shard mode)...")

    with tqdm(total=args.target_count) as pbar:
        while saved < args.target_count and attempts < max_attempts:
            attempts += 1

            shard = random.choice(shard_dirs)
            idx   = random.randint(0, args.max_index)

            if bitsets[shard.name][idx]:
                continue

            img_path = shard / f"{idx}.jpg"
            if not img_path.exists():
                continue

            shard_prefix = shard.name.split("_")[0]   # "0000" from "0000_images"
            new_name = f"{shard_prefix}_{img_path.name}"
            dst = output_dir / new_name

            shutil.copy2(img_path, dst)

            bitsets[shard.name][idx] = 1   # prevent duplicates
            saved   += 1
            pbar.update(1)

    print("\n" + "=" * 60)
    print("Finished (shard mode).")
    print(f"Saved:    {saved:,}")
    print(f"Attempts: {attempts:,}")
    print("=" * 60)


# ============================================================
# SIMPLE MODE — arbitrary directory sampler
# ============================================================

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def collect_images(src_dirs, recursive=False):
    """
    Collects all image file paths from one or more source directories.

    Args:
        src_dirs:  List of Path objects to search.
        recursive: If True, descend into subdirectories.

    Returns:
        Sorted list of Path objects (sorted for determinism before shuffling).
    """
    images = []
    for src in src_dirs:
        if recursive:
            candidates = src.rglob("*")
        else:
            candidates = src.iterdir()

        for p in candidates:
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                images.append(p)

    # Sort before shuffling so the starting pool is deterministic
    # regardless of filesystem ordering (which varies across OSes).
    images.sort()
    return images


def run_simple(args):
    """
    Simple reproducible random sampler for arbitrary image directories.

    Supports any image filenames and extensions. No exclusion file or
    numeric index required. Reproducibility is guaranteed by fixing the
    random seed before shuffling the full candidate list.

    The output filename is prefixed with the source directory name to
    avoid collisions when sampling from multiple directories.

    Args (from argparse):
        image_dirs:    One or more source directories to sample from.
        output_dir:    Destination directory for sampled images.
        target_count:  Number of images to sample.
        seed:          Random seed for reproducibility.
        recursive:     Whether to search subdirectories.
    """
    random.seed(args.seed)

    src_dirs   = [Path(d).resolve() for d in args.image_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Collecting images from {len(src_dirs)} director(y/ies)...")
    all_images = collect_images(src_dirs, recursive=getattr(args, "recursive", False))

    if not all_images:
        print("[ERROR] No images found in the specified directories.")
        return

    print(f"[INFO] Found {len(all_images):,} candidate images.")

    if args.target_count > len(all_images):
        print(
            f"[WARNING] target_count ({args.target_count:,}) exceeds available images "
            f"({len(all_images):,}). Sampling all available images."
        )

    # random.sample is O(target_count) and does not modify the source list.
    # The sorted + seeded combination ensures full reproducibility.
    sampled = random.sample(all_images, min(args.target_count, len(all_images)))

    print(f"[INFO] Copying {len(sampled):,} images to {output_dir} ...")

    for img_path in tqdm(sampled, desc="Copying"):
        # Prefix with parent dir name to avoid filename collisions across dirs.
        new_name = f"{img_path.parent.name}_{img_path.name}"
        dst = output_dir / new_name
        shutil.copy2(img_path, dst)

    print("\n" + "=" * 60)
    print("Finished (simple mode).")
    print(f"Sampled: {len(sampled):,} / {len(all_images):,} available")
    print(f"Seed:    {args.seed}  (re-run with the same seed to reproduce)")
    print(f"Output:  {output_dir}")
    print("=" * 60)


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Image sampler with two modes:\n"
            "  shard  — numeric Re-LAION shards with JSONL exclusion bitsets\n"
            "  simple — arbitrary directories, seed-reproducible random sample"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--mode",
        choices=["shard", "simple"],
        required=True,
        help="Sampling mode: 'shard' for Re-LAION shards, 'simple' for any directory.",
    )
    p.add_argument(
        "--image_dirs",
        type=str,
        nargs="+",
        required=True,
        help="One or more source image directories.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Destination directory for sampled images.",
    )
    p.add_argument(
        "--target_count",
        type=int,
        default=10000,
        help="Number of images to sample (default: 10000).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    # ── Shard-mode-only arguments ────────────────────────────
    shard_group = p.add_argument_group("shard mode options")
    shard_group.add_argument(
        "--jsonl_path",
        type=str,
        default=None,
        help="[shard] Path to JSONL exclusion file.",
    )
    shard_group.add_argument(
        "--max_index",
        type=int,
        default=None,
        help="[shard] Maximum numeric image index across all shards.",
    )

    # ── Simple-mode-only arguments ───────────────────────────
    simple_group = p.add_argument_group("simple mode options")
    simple_group.add_argument(
        "--recursive",
        action="store_true",
        help="[simple] Search subdirectories recursively.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "shard":
        if args.jsonl_path is None or args.max_index is None:
            raise ValueError(
                "Shard mode requires --jsonl_path and --max_index."
            )
        run_shard(args)

    elif args.mode == "simple":
        run_simple(args)


if __name__ == "__main__":
    main()


# python RandomImageSampler.py ^
#     --mode shard
#     --jsonl_path "E:\ImageRetrieval\Professions_125k_ISCO_Aligned\ISCO_aligned_125k_retrieval_results_batchsize_10.jsonl" ^
#     --image_dirs "G:\Thesis\0000_images" "G:\Thesis\0001_images" "G:\Thesis\0002_images" "G:\Thesis\0003_images" ^
#     --output_dir "F:\ImageRetrieval\TestingArbitrary10k\ReLaion5B_Random10k" ^
#     --target_count 10_000 ^
#     --max_index 17_000_000 ^
#     --seed 42

# python RandomImageSampler.py ^
#   --mode simple ^
#   --image_dirs "F:\ImageRetrieval\TestingArbitrary10k\CocoUnlabelled_Random10k\unlabeled2017" ^
#   --output_dir "F:\ImageRetrieval\TestingArbitrary10k\CocoUnlabelled_Random10k" ^
#   --target_count 10_000 ^
#   --seed 42