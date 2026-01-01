import os
import time
import argparse
from math import ceil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation
)

# ------------------------------------------------------------
# NumPy compatibility (same as Depth.py)
# ------------------------------------------------------------
np.float_ = np.float64
np.complex_ = np.complex128

# ------------------------------------------------------------
# Fast RGB Loader (unchanged semantics)
# ------------------------------------------------------------
class FastRGBLoader:
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size
        try:
            from turbojpeg import TurboJPEG
            self.jpeg = TurboJPEG()
            self.use_turbo = True
        except Exception:
            self.use_turbo = False

    def load(self, path: str):
        try:
            ext = os.path.splitext(path)[1].lower()
            if self.use_turbo and ext in (".jpg", ".jpeg"):
                with open(path, "rb") as f:
                    arr = self.jpeg.decode(f.read())
                img = Image.fromarray(arr, "RGB")
            else:
                img = Image.open(path).convert("RGB")

            if img.width <= 1 or img.height <= 1:
                raise ValueError

            orig_size = img.size
            img = img.resize(self.target_size, Image.BICUBIC)
            return img, orig_size

        except Exception:
            dummy = Image.new("RGB", self.target_size)
            return dummy, self.target_size


# ------------------------------------------------------------
# Input collection (unchanged semantics)
# ------------------------------------------------------------
def collect_images(image_dir, output_dir, exclude_dirs):
    image_dir = os.path.abspath(image_dir)
    output_dir = os.path.abspath(output_dir)
    exclude_dirs = {d.lower() for d in exclude_dirs}

    processed = set()
    if os.path.exists(output_dir):
        for root, _, files in os.walk(output_dir):
            for f in files:
                if f.endswith("_seg.png"):
                    rel = os.path.relpath(os.path.join(root, f), output_dir)
                    processed.add(rel.replace("_seg.png", ""))

    samples = []
    for root, dirs, files in os.walk(image_dir):
        dirs[:] = [d for d in dirs if d.lower() not in exclude_dirs]

        for f in files:
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            full = os.path.join(root, f)
            rel = os.path.relpath(full, image_dir)
            if os.path.splitext(rel)[0] in processed:
                continue

            samples.append((full, rel))

    return samples


# ------------------------------------------------------------
# Save segmentation map (unchanged output)
# ------------------------------------------------------------
def save_segmentation(seg_np, save_path):
    Image.fromarray(seg_np, mode="RGB").save(
        save_path, format="PNG", compress_level=1
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(args):
    start = time.time()
    device = torch.device(args.device)

    # ---- Output resize handling (unchanged semantics) ----
    if args.resize is None:
        output_resize = None
    elif len(args.resize) == 1:
        output_resize = (args.resize[0], args.resize[0])
    else:
        output_resize = tuple(args.resize)

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Model ----
    processor = AutoImageProcessor.from_pretrained(args.model_name, use_fast=True)

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        args.model_name
    ).to(device).eval()

    # Correct FP16 conversion
    model = model.half()

    # ADE20K color palette (150 classes) -> https://github.com/lllyasviel/ControlNet/issues/172
    ade_palette = np.array([
        (120,120,120),(180,120,120),(6,230,230),(80,50,50),(4,200,3),(120,120,80),
        (140,140,140),(204,5,255),(230,230,230),(4,250,7),(224,5,255),(235,255,7),
        (150,5,61),(120,120,70),(8,255,51),(255,6,82),(143,255,140),(204,255,4),
        (255,51,7),(204,70,3),(0,102,200),(61,230,250),(255,6,51),(11,102,255),
        (255,7,71),(255,9,224),(9,7,230),(220,220,220),(255,9,92),(112,9,255),
        (8,255,214),(7,255,224),(255,184,6),(10,255,71),(255,41,10),(7,255,255),
        (224,255,8),(102,8,255),(255,61,6),(255,194,7),(255,122,8),(0,255,20),
        (255,8,41),(255,5,153),(6,51,255),(235,12,255),(160,150,20),(0,163,255),
        (140,140,140),(250,10,15),(20,255,0),(31,255,0),(255,31,0),(255,224,0),
        (153,255,0),(0,0,255),(255,71,0),(0,235,255),(0,173,255),(31,0,255),
        (11,200,200),(255,82,0),(0,255,245),(0,61,255),(0,255,112),(0,255,133),
        (255,0,0),(255,163,0),(255,102,0),(194,255,0),(0,143,255),(51,255,0),
        (0,82,255),(0,255,41),(0,255,173),(10,0,255),(173,255,0),(0,255,153),
        (255,92,0),(255,0,255),(255,0,245),(255,0,102),(255,173,0),(255,0,20),
        (255,184,184),(0,31,255),(0,255,61),(0,71,255),(255,0,204),(0,255,194),
        (0,255,82),(0,10,255),(0,112,255),(51,0,255),(0,194,255),(0,122,255),
        (0,255,163),(255,153,0),(0,255,10),(255,112,0),(143,255,0),(82,0,255),
        (163,255,0),(255,235,0),(8,184,170),(133,0,255),(0,255,92),(184,0,255),
        (255,0,31),(0,184,255),(0,214,255),(255,0,112),(92,255,0),(0,224,255),
        (112,224,255),(70,184,160),(163,0,255),(153,0,255),(71,255,0),
        (255,0,163),(255,204,0),(255,0,143),(0,255,235),(133,255,0),
        (255,0,235),(245,0,255),(255,0,122),(255,245,0),(10,190,212),
        (214,255,0),(0,204,255),(20,0,255),(255,255,0),(0,153,255),
        (0,41,255),(0,255,204),(41,0,255),(41,255,0),(173,0,255),
        (0,245,255),(71,0,255),(122,0,255),(0,255,184),(0,92,255),
        (184,255,0),(0,133,255),(255,214,0),(25,194,194),(102,255,0),
        (92,0,255)
    ], dtype=np.uint8)

    samples = collect_images(
        args.image_dir, args.output_dir, args.exclude_dirs
    )

    if not samples:
        print("No remaining images to process.")
        return

    print(f"Images to process: {len(samples)}")

    loader = FastRGBLoader(target_size=tuple(args.fixed_size))
    total_batches = ceil(len(samples) / args.batch_size)

    with ThreadPoolExecutor(max_workers=args.num_workers) as pool, \
         tqdm(total=total_batches, desc="Semantic segmentation") as pbar:

        for i in range(0, len(samples), args.batch_size):
            batch = samples[i:i + args.batch_size]

            imgs, rels, orig_sizes = [], [], []
            for full, rel in batch:
                img, orig = loader.load(full)
                imgs.append(img)
                rels.append(rel)
                orig_sizes.append(orig)

            inputs = processor(images=imgs, return_tensors="pt")
            inputs = {
                k: v.pin_memory().to(device, non_blocking=True)
                for k, v in inputs.items()
            }

            with torch.no_grad(), torch.amp.autocast('cuda'):
                outputs = model(**inputs)

            seg_maps = processor.post_process_semantic_segmentation(
                outputs,
                target_sizes=[args.fixed_size] * len(imgs)
            )

            futures = []
            for seg, rel, orig_size in zip(seg_maps, rels, orig_sizes):
                seg = seg.cpu().numpy().astype(np.uint8)
                target_size = output_resize if output_resize is not None else orig_size
                seg = cv2.resize(seg, target_size, interpolation=cv2.INTER_NEAREST)
                seg_rgb = ade_palette[seg % len(ade_palette)]

                save_rel = os.path.splitext(rel)[0] + "_seg.png"
                save_path = os.path.join(args.output_dir, save_rel)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                futures.append(pool.submit(save_segmentation, seg_rgb, save_path))

            for f in futures:
                f.result()

            pbar.update(1)

    print(f"Done in {time.time() - start:.2f}s")


# ------------------------------------------------------------
# CLI (unchanged)
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Semantic segmentation (Depth-aligned pipeline)"
    )
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_name", type=str, default="facebook/mask2former-swin-large-ade-semantic")
    parser.add_argument("--fixed_size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--resize", type=int, nargs="+")
    parser.add_argument("--exclude_dirs", nargs="*", default=["facemesh"])

    main(parser.parse_args())


# ===========================================================
# EXAMPLE USAGE
# python SemanticSegmentation.py --image_dir "path/to/input" --resize 224 224 --batch_size 16 --num_workers 8 --output_dir "path/to/output" --exclude_dirs facemesh

# --resize 224 224 -> This is done as the classification model auto resizes images to 224 x 244 hence its better to resize them prior as this makes processing faster and storge requirements less.
# --exclude_dirs facemesh -> This is done to exclude any images in the facemesh directory from processing as these are not actual images but rather facemesh data.
# --fixed_size 512 512 -> This is done as the fixed size is used for the model input, and 512x512 is the expected size for semantic segmentation models.
# --model_name facebook/mask2former-swin-large-ade-semantic -> This is the model used for semantic segmentation in the paper, the difference is that this is derived from huggigface transformers library whilst the paper doesn't do that.

# "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\.venv-Copy-Copy\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\SemanticSegmentation\test.py" --image_dir "E:\ImageRetrieval\Professions_125k_Cleaned"  --output_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\SemanticSegmentation\test_2" --batch_size 8 --fixed_size 512 512