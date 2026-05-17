import os
import time
import argparse
from math import ceil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import cv2
import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# ------------------------------------------------------------
# NumPy compatibility
# ------------------------------------------------------------
# NOTE: Some third-party libraries (e.g., older Torch / OpenCV /
#       HF dependencies) still expect deprecated NumPy aliases.
#       These assignments ensure backward compatibility without
#       affecting numerical behavior.
np.float_ = np.float64
np.complex_ = np.complex128

# ------------------------------------------------------------
# Fast RGB Loader (same as RGB / Shuffle)
# ------------------------------------------------------------
class FastRGBLoader:
    """
    Fast image loader with optional TurboJPEG acceleration.

    NOTE:
    - This loader mirrors the RGB / Pixel-Shuffle pipelines to ensure
      that differences in downstream behavior are attributable solely
      to the *depth transformation*, not to I/O or resizing artifacts.
    - Images are always converted to RGB and resized to a fixed
      canonical size (384x384) before depth inference.
    """
        
    def __init__(self, target_size=(384, 384)):
        self.target_size = target_size

        try:
            from turbojpeg import TurboJPEG
            self.jpeg = TurboJPEG()
            self.use_turbo = True
        except Exception:
            self.jpeg = None
            self.use_turbo = False

    def load(self, path):
        ext = os.path.splitext(path)[1].lower()

        try:
            if self.use_turbo and ext in (".jpg", ".jpeg"):
                with open(path, "rb") as f:
                    arr = self.jpeg.decode(f.read())
                img = Image.fromarray(arr, "RGB")
            else:
                img = Image.open(path).convert("RGB")

            w, h = img.size
            if w <= 1 or h <= 1:
                raise ValueError

            original_size = img.size
            img = img.resize(self.target_size, Image.BICUBIC)
            return img, original_size

        except (UnidentifiedImageError, OSError, ValueError):
            return Image.new("RGB", self.target_size), self.target_size


# ------------------------------------------------------------
# Input collection
# ------------------------------------------------------------
def collect_images(image_dir, output_dir, exclude_dirs):
    """
    Collect unprocessed images while preserving directory structure.

    NOTE:
    - Images are skipped if a corresponding *_depth.png already exists.
    - This makes the pipeline resumable and safe for large-scale runs.
    - Directory exclusion (e.g., 'facemesh') prevents feedback loops
      across multiple transformation pipelines.
    """
    image_dir = os.path.abspath(image_dir)
    output_dir = os.path.abspath(output_dir)
    exclude_dirs = {d.lower() for d in exclude_dirs}

    samples = []
    processed = set()

    if os.path.exists(output_dir):
        for root, _, files in os.walk(output_dir):
            for f in files:
                if f.endswith("_depth.png"):
                    rel = os.path.relpath(os.path.join(root, f), output_dir)
                    processed.add(rel.replace("_depth.png", ""))

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
# Save depth map
# ------------------------------------------------------------
def save_depth_map(depth_np, save_path):
    """
    Save depth as a 3-channel PNG.

    NOTE:
    - Depth is replicated across RGB channels to match standard
      image-classification input expectations.
    - This mirrors the grayscale depth encoding used in dataset
      classification experiments.
    """
    depth_rgb = np.repeat(depth_np[..., None], 3, axis=-1)
    cv2.imwrite(save_path, depth_rgb)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(args):
    start_time = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)

    # ---- Output resize handling ----
    # NOTE:
    # - Optional *final* resize allows alignment with other pipelines
    #   (e.g., fixed 224×224 or 256×256 inputs for classifiers).
    # - Depth inference itself is always performed at 384×384.
    if args.resize is None:
        output_resize = None
    elif len(args.resize) == 1:
        output_resize = (args.resize[0], args.resize[0])
    else:
        output_resize = (args.resize[0], args.resize[1])

    # ---- Depth model ----
    # NOTE:
    # - Depth-Anything-V2 isolates *spatial geometry* while discarding
    #   texture and color.
    # - This corresponds exactly to the "depth" transformation used to
    #   measure structural dataset bias.
    model_name = f"depth-anything/Depth-Anything-V2-{args.model_size}-hf"
    processor = AutoImageProcessor.from_pretrained(
        model_name,
        use_fast=True,
        size={"height": 384, "width": 384},
    )
    model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device).eval()

    samples = collect_images(args.image_dir, args.output_dir, args.exclude_dirs)
    if not samples:
        print("No remaining images to process.")
        return

    print(f"Remaining images to process: {len(samples)}")

    loader = FastRGBLoader(target_size=(384, 384))
    total_batches = ceil(len(samples) / args.batch_size)

    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        pbar = tqdm(range(0, len(samples), args.batch_size),
                    total=total_batches,
                    desc="Processing batches")

        for i in pbar:
            batch = samples[i:i + args.batch_size]

            imgs, rels, orig_sizes = [], [], []
            for full, rel in batch:
                img, orig = loader.load(full)
                imgs.append(img)
                rels.append(rel)
                orig_sizes.append(orig)

            inputs = processor(images=imgs, return_tensors="pt").to(device)

            with torch.no_grad():
                predicted_depth = model(**inputs).predicted_depth

            futures = []
            for depth_map, rel, orig_size in zip(predicted_depth, rels, orig_sizes):

                # ---- Upsample to original resolution ----
                # NOTE:
                # - Bicubic interpolation preserves smooth geometry.
                # - align_corners=False avoids edge distortions.
                depth = torch.nn.functional.interpolate(
                    depth_map[None, None],
                    size=orig_size[::-1],
                    mode="bicubic",
                    align_corners=False,
                )[0, 0].cpu().numpy()

                if depth.max() != depth.min():
                    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                else:
                    depth = np.zeros_like(depth)

                depth = depth.astype(np.uint8)

                # ---- FINAL RESIZE (optional) ----
                # NOTE:
                # - Applied *after* depth computation to avoid altering
                #   spatial relationships learned by the depth model.
                if output_resize is not None:
                    depth = cv2.resize(depth, output_resize, interpolation=cv2.INTER_CUBIC)

                save_rel = os.path.splitext(rel)[0] + "_depth.png"
                save_path = os.path.join(args.output_dir, save_rel)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                futures.append(pool.submit(save_depth_map, depth, save_path))

            for f in futures:
                f.result()

    print(f"\nDepth maps saved to {args.output_dir}")
    print(f"Total time: {time.time() - start_time:.2f}s")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Depth pipeline aligned with RGB / Shuffle (+ output resize)"
    )
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model_size",
        default="Small",
        choices=["Small", "Base", "Large", "Giant"],
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs="+",
        help="Final output size: one value (square) or two values (W H)",
    )
    parser.add_argument(
        "--exclude_dirs",
        nargs="*",
        default=["facemesh"],
    )

    args = parser.parse_args()
    main(args)

# ===========================================================
# EXAMPLE USAGE
# python Depth.py --image_dir "path/to/input" --resize 224 224 --batch_size 16 --num_workers 8 --output_dir "path/to/output" --exclude_dirs facemesh --device cuda --model_size Small