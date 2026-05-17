#!/usr/bin/env python3
"""
Resumable top-N image selection per subdirectory based on similarity score
encoded in filename. Copies images + facemesh and truncates annotations.jsonl.

Resume is handled via a JSONL manifest in the target directory.
NO deletions from source.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple


# -------------------------
# Config
# -------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SCORE_RE = re.compile(r"^([0-9]+\.[0-9]+)_")
MANIFEST_NAME = ".resume_manifest.jsonl"


# -------------------------
# Helpers
# -------------------------

def parse_score(filename: str) -> float | None:
    m = SCORE_RE.match(filename)
    return float(m.group(1)) if m else None


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


# -------------------------
# Manifest handling
# -------------------------

def load_completed_subdirs(manifest_path: Path) -> Set[str]:
    completed = set()
    if not manifest_path.exists():
        return completed

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            completed.add(obj["subdir"])
    return completed


def append_manifest(
    manifest_path: Path,
    subdir_name: str,
    selected_images: List[str],
) -> None:
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "subdir": subdir_name,
            "selected_images": selected_images
        }) + "\n")


def load_all_selected_images(manifest_path: Path) -> Set[str]:
    images = set()
    if not manifest_path.exists():
        return images

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            images.update(obj["selected_images"])
    return images


# -------------------------
# Core logic
# -------------------------

def select_topn_images(subdir: Path, top_n: int) -> List[Path]:
    scored: List[Tuple[float, Path]] = []

    for p in subdir.iterdir():
        if not p.is_file() or not is_image(p):
            continue
        score = parse_score(p.name)
        if score is not None:
            scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_n]]


def process_subdir(
    subdir: Path,
    src_root: Path,
    dst_root: Path,
    top_n: int,
) -> List[str]:
    """
    Process a single profession directory.
    Returns relative paths of copied images.
    """
    rel_subdir = subdir.relative_to(src_root)
    dst_subdir = dst_root / rel_subdir
    dst_subdir.mkdir(parents=True, exist_ok=True)

    src_facemesh = subdir / "facemesh"
    dst_facemesh = dst_subdir / "facemesh"
    if src_facemesh.exists():
        dst_facemesh.mkdir(exist_ok=True)

    selected = select_topn_images(subdir, top_n)
    copied_rel_paths: List[str] = []

    for img in selected:
        dst_img = dst_subdir / img.name
        if not dst_img.exists():
            shutil.copy2(img, dst_img)

        rel_img = str((rel_subdir / img.name).as_posix())
        copied_rel_paths.append(rel_img)

        if src_facemesh.exists():
            base = img.stem
            for ext in (".jpg", ".jpeg", ".png"):
                face_path = src_facemesh / f"{base}_face{ext}"
                if face_path.exists():
                    shutil.copy2(face_path, dst_facemesh / face_path.name)
                    copied = True
                    break

    return copied_rel_paths


def truncate_annotations(
    annotations_path: Path,
    dst_annotations_path: Path,
    kept_images: Set[str],
) -> None:
    kept = 0
    total = 0

    with annotations_path.open("r", encoding="utf-8") as fin, \
         dst_annotations_path.open("w", encoding="utf-8") as fout:

        for line in fin:
            total += 1
            obj = json.loads(line)
            if obj.get("image") in kept_images:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1

    print(f"[INFO] annotations.jsonl: kept {kept}/{total}")


# -------------------------
# CLI
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--face-dir", default="facemesh", type=Path)
    parser.add_argument("--target-root", required=True, type=Path)
    parser.add_argument("--top-n", required=True, type=int)
    args = parser.parse_args()

    src_root = args.source_root.resolve()
    dst_root = args.target_root.resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    manifest_path = dst_root / MANIFEST_NAME

    completed = load_completed_subdirs(manifest_path)
    print(f"[INFO] Resuming: {len(completed)} subdirectories already done")

    for subdir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        if subdir.name == args.face_dir:
            continue

        if subdir.name in completed:
            continue

        print(f"[INFO] Processing {subdir.name}")
        selected_rel_paths = process_subdir(
            subdir=subdir,
            src_root=src_root,
            dst_root=dst_root,
            top_n=args.top_n,
        )

        append_manifest(
            manifest_path=manifest_path,
            subdir_name=subdir.name,
            selected_images=selected_rel_paths,
        )

    print("[INFO] Rebuilding truncated annotations.jsonl")
    kept_images = load_all_selected_images(manifest_path)

    truncate_annotations(
        annotations_path=args.annotations.resolve(),
        dst_annotations_path=dst_root / "annotations.jsonl",
        kept_images=kept_images,
    )

    print("[DONE] Fully completed / safely resumable")


if __name__ == "__main__":
    main()


# python RELaion5B_Subset_Image_Retrieval.py --source-root "" --annotations "" --target-root "" --top-n 1000
