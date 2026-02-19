"""
Pixel / Patch Shuffling and Mean-RGB Pipeline
=============================================

This script implements controlled image transformations used to
isolate and study spurious correlations and dataset bias in large-scale
image datasets (e.g. LAION derivatives).

The implementation closely follows the experimental methodology of:

1. "Understanding Bias in Large-Scale Visual Datasets"
   (Zeng et al., NeurIPS 2024)

2. "Gender Artifacts in Visual Datasets"
   (Meister et al., ICCV 2023)

Core idea:
----------
Apply destructive-but-controlled image transformations that remove
semantic or spatial information while preserving low-level statistics
(e.g. color distributions), then measure what predictive signal remains.

The transformations implemented here are:
- Pixel shuffling
- Patch shuffling (multiple patch sizes)
- Mean RGB images + RGB statistics (for linear probes)

The pipeline is designed to scale to hundreds of thousands or millions
of images, with resumability and minimal startup overhead.
"""

import os
import glob
import json
import argparse
from tqdm import tqdm
from threading import Lock, Thread
from typing import Tuple
from queue import Queue
import numpy as np
import cv2

# ============================================================
# FAST RGB DECODER
# ============================================================
class FastRGBDecoder:
    """
    High-performance image decoder that returns RGB NumPy arrays.

    Both referenced papers operate at very large scale. Image decoding
    quickly becomes a dominant bottleneck if PIL or naive OpenCV loading
    is used.

    This class:
    - Uses libjpeg-turbo (via PyTurboJPEG) when available
    - Falls back to OpenCV otherwise
    - Guarantees RGB channel ordering
    """
        
    def __init__(self):
        try:
            from turbojpeg import TurboJPEG
            self.jpeg = TurboJPEG(
                r"C:\libjpeg-turbo-gcc64\bin\libturbojpeg.dll"
            )
            self.use_turbo = True
            print("[Decoder] TurboJPEG enabled")
        except Exception:
            self.use_turbo = False
            print("[Decoder] TurboJPEG unavailable, using OpenCV")

    def load(self, path: str) -> np.ndarray | None:
        try:
            ext = os.path.splitext(path)[1].lower()
            if self.use_turbo and ext in {".jpg", ".jpeg"}:
                with open(path, "rb") as f:
                    return self.jpeg.decode(f.read())
            arr = cv2.imread(path)
            if arr is None:
                return None
            return arr[..., ::-1]
        except Exception:
            return None

# ============================================================
# ASYNC IMAGE WRITER (KEY PERFORMANCE FIX)
# ============================================================
class AsyncImageWriter:
    def __init__(self, num_workers=4, max_queue=8192):
        self.queue = Queue(maxsize=max_queue)
        self.stop_token = object()
        self.workers = []

        for _ in range(num_workers):
            t = Thread(target=self._worker, daemon=True)
            t.start()
            self.workers.append(t)

    def _worker(self):
        while True:
            item = self.queue.get()
            if item is self.stop_token:
                break
            path, arr = item
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, arr)
            self.queue.task_done()

    def submit(self, path: str, arr: np.ndarray):
        self.queue.put((path, arr))

    def close(self):
        self.queue.join()
        for _ in self.workers:
            self.queue.put(self.stop_token)

# ============================================================
# MANIFEST (BUFFERED)
# ============================================================
class TransformManifest:
    """
    Persistent, append-only manifest tracking completed transforms.

    Why this exists:
    ----------------
    At large scale, scanning output directories to determine what has
    already been processed is prohibitively slow.

    Instead, we maintain a JSONL manifest of the form:

        {"image": "relative/path.jpg", "transform": "pixel_shuffle"}

    An image is considered *complete* only if ALL required transforms
    are present. Partial completion is correctly handled.

    This directly supports resumable, fault-tolerant execution.
    """
        
    def __init__(self, path: str, flush_every: int = 512):
        self.path = path
        self.flush_every = flush_every
        self.lock = Lock()
        self.data = {}
        self.buffer = []

        os.makedirs(os.path.dirname(path), exist_ok=True)

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    self.data.setdefault(r["image"], set()).add(r["transform"])

    def record(self, image: str, transform: str):
        with self.lock:
            self.data.setdefault(image, set()).add(transform)
            self.buffer.append({"image": image, "transform": transform})
            if len(self.buffer) >= self.flush_every:
                self.flush()

    def flush(self):
        if not self.buffer:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            for r in self.buffer:
                f.write(json.dumps(r) + "\n")
        self.buffer.clear()

# ============================================================
# DATASET
# ============================================================
class ImageDataset:
    """
    Enumerates images that still require processing.

    - Recursively scan an image directory
    - Ignore specified subdirectories (e.g. 'facemesh')
    - Exclude images that are already fully processed
    - Decode images once per batch

    This mirrors dataset sampling used in both papers.
    """

    def __init__(self, image_dir, completed, exclude_dirs, resize_width = None, resize_height = None):
        self.decoder = FastRGBDecoder()
        self.image_dir = image_dir
        self.resize_width = resize_width
        self.resize_height = resize_height
        exclude_dirs = {d.lower() for d in exclude_dirs}

        self.paths = []
        for p in glob.glob(os.path.join(image_dir, "**", "*.*"), recursive=True):
            if not p.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            if set(os.path.normpath(p).lower().split(os.sep)) & exclude_dirs:
                continue
            rel = os.path.relpath(p, image_dir).replace("\\", "/")
            if os.path.splitext(rel)[0] in completed:
                continue
            self.paths.append(p)

    def __len__(self):
        return len(self.paths)

    def get_batch(self, i, bs):
        paths = self.paths[i:i + bs]
        arrays = []
        for p in paths:
            arr = self.decoder.load(p)
            if arr is None or arr.shape[0] <= 1 or arr.shape[1] <= 1:
                arr = np.zeros((224, 224, 3), dtype=np.uint8)
            
            elif self.resize_width and self.resize_height:
                arr = cv2.resize(arr, (self.resize_width, self.resize_height), interpolation=cv2.INTER_LINEAR)
            arrays.append(arr)
        return arrays, paths

# ============================================================
# TRANSFORMS (ARRAY-BASED)
# ============================================================
def pixel_shuffle(arr: np.ndarray) -> np.ndarray:
    """
    Pixel shuffling transformation.

    Paper alignment:
    "Understanding Bias in Large-Scale Visual Datasets" (Zeng et al., NeurIPS 2024)
    """
        
    h, w, c = arr.shape
    flat = arr.reshape(-1, c)
    rng = np.random.default_rng(int(arr[0, 0].sum()))
    return flat[rng.permutation(len(flat))].reshape(h, w, c)

def mean_rgb(arr: np.ndarray) -> Tuple[np.ndarray, list, list]:
    """
    Mean RGB transformation.

    Outputs:
    - Mean-RGB image (for CNN evaluation)
    - Raw RGB vector
    - Normalized RGB vector (for logistic regression)

    Paper alignment:
    1. "Understanding Bias in Large-Scale Visual Datasets" (Zeng et al., NeurIPS 2024)

    2. "Gender Artifacts in Visual Datasets" (Meister et al., ICCV 2023)
    """

    mean = arr.mean(axis=(0, 1))
    img = np.ones_like(arr, dtype=np.uint8) * mean.astype(np.uint8)
    return img, mean.tolist(), (mean / 255.0).tolist()

def patch_shuffle(arr: np.ndarray, ps: int) -> np.ndarray:
    """
    Patch shuffling transformation.

    Paper alignment:
    1. "Understanding Bias in Large-Scale Visual Datasets" (Zeng et al., NeurIPS 2024)
    """
        
    h, w, c = arr.shape

    gh = h // ps
    gw = w // ps

    # If no full patches fit, do nothing
    if gh == 0 or gw == 0:
        return arr.copy()

    # Region that will be shuffled
    H = gh * ps
    W = gw * ps

    core = arr[:H, :W]

    # Extract patches
    patches = (
        core
        .reshape(gh, ps, gw, ps, c)
        .transpose(0, 2, 1, 3, 4)
        .reshape(-1, ps, ps, c)
    )

    rng = np.random.default_rng(int(patches[0].sum()))
    rng.shuffle(patches)

    # Reassemble shuffled core
    shuffled_core = (
        patches
        .reshape(gh, gw, ps, ps, c)
        .transpose(0, 2, 1, 3, 4)
        .reshape(H, W, c)
    )

    # Insert back into original image
    out = arr.copy()
    out[:H, :W] = shuffled_core

    return out

# ============================================================
# MAIN
# ============================================================
def main(args):
    """
    Main execution loop.

    High-level logic:
    -----------------
    1. Load transform manifest
    2. Determine required transformations
    3. Identify completed images
    4. Build dataset of incomplete images
    5. Decode each image once
    6. Apply all requested transforms
    7. Save outputs and update manifest
    """
        
    manifest = TransformManifest(os.path.join(args.output_dir, "transform_manifest.jsonl"))

    required = set()
    if args.do_pixel_shuffle:
        required.add("pixel_shuffle")
    if args.do_mean_rgb:
        required.add("mean_rgb")
    if args.do_patch_shuffle:
        for ps in args.patch_sizes:
            required.add(f"patch_shuffle_ps{ps}")

    completed = {
        os.path.splitext(img)[0]
        for img, done in manifest.data.items()
        if required.issubset(done)
    }
    print(f"Skipping {len(completed)} completed images.")    

    if args.resize is not None:
        resize_width, resize_height = args.resize
    else:
        resize_width = resize_height = None

    dataset = ImageDataset(args.image_dir, completed, args.exclude_dirs, resize_width=resize_width, resize_height=resize_height)
    print(f"Images to process: {len(dataset)}")

    mean_json_path = os.path.join(args.output_dir, "mean_rgb", "mean_rgb_stats.jsonl")
    os.makedirs(os.path.dirname(mean_json_path), exist_ok=True)
    mean_lock = Lock()

    mean_lock = Lock()

    image_writer = AsyncImageWriter(
        num_workers=min(4, args.num_workers),
        max_queue=8192,
    )

    for i in tqdm(range(0, len(dataset), args.batch_size)):
        arrays, paths = dataset.get_batch(i, args.batch_size)

        for arr, path in zip(arrays, paths):
            rel = os.path.relpath(path, args.image_dir).replace("\\", "/")
            stem = os.path.splitext(rel)[0]

            if args.do_pixel_shuffle:
                out = pixel_shuffle(arr)
                image_writer.submit(
                    os.path.join(args.output_dir, "pixel_shuffle", stem + "_pixel.png"),
                    out,
                )
                manifest.record(rel, "pixel_shuffle")

            if args.do_mean_rgb:
                img, rgb, norm = mean_rgb(arr)
                image_writer.submit(
                    os.path.join(args.output_dir, "mean_rgb", stem + "_mean.png"),
                    img,
                )
                with mean_lock:
                    with open(mean_json_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "image": rel,
                            "rgb": rgb,
                            "normalized_rgb": norm
                        }) + "\n")
                manifest.record(rel, "mean_rgb")

            if args.do_patch_shuffle:
                for ps in args.patch_sizes:
                    out = patch_shuffle(arr, ps)
                    image_writer.submit(
                        os.path.join(
                            args.output_dir,
                            f"patch_shuffle_ps{ps}",
                            stem + f"_ps{ps}.png",
                        ),
                        out,
                    )
                    manifest.record(rel, f"patch_shuffle_ps{ps}")

    image_writer.close()
    manifest.flush()
    print("Done.")

# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--resize", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=None, help="Optional fixed resize as: --resize WIDTH HEIGHT (e.g. --resize 224 224)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--exclude_dirs", nargs="+", default=["facemesh"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--patch_sizes", type=int, nargs="+", default=[16])
    parser.add_argument("--do_pixel_shuffle", action="store_true")
    parser.add_argument("--do_patch_shuffle", action="store_true")
    parser.add_argument("--do_mean_rgb", action="store_true")
    main(parser.parse_args())

# ===========================================================
# EXAMPLE USAGE
# python PixelPatchShufflingMeanRGB.py --image_dir "path/to/input" --resize 224 224 --batch_size 16 --num_workers 8 --output_dir "path/to/output" --exclude_dirs facemesh --patch_sizes 1 2 4 8 16 --do_pixel_shuffle --do_patch_shuffle --do_mean_rgb

# --resize 224 224 -> This is done as the classification model auto resizes images to 224 x 244 hence its better to resize them prior as this makes processing faster and storge requirements less.
# --patch_sizes 1 2 4 8 16 -> This is done as in the paper "Understanding Bias in Large-Scale Visual Datasets" in Fig 5 they denote that these patch sizes where used for patch shuffling.  
# --exclude_dirs facemesh -> This is done to exclude any images in the facemesh directory from processing as these are not actual images but rather facemesh data.


# NOTE: Pixel and Patch Shuffling are randomized independently per image.
# The random seed is deterministically derived from the image content
# (specifically, the RGB sum of a reference pixel), ensuring that each
# image is shuffled in a reproducible way across runs, provided the
# preprocessing pipeline (decoding, resizing, color ordering) is unchanged.


# python PixelPatchShufflingMeanRGB.py --image_dir "E:\ImageRetrieval\StableDiffusionGeneratedImages\valid" --resize 224 224 --batch_size 16 --num_workers 8 --output_dir "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\Shuffling&Colour" --exclude_dirs face_crops --patch_sizes 2 4 8 16 --do_pixel_shuffle --do_patch_shuffle --do_mean_rgb

# python PixelPatchShufflingMeanRGB.py --image_dir "F:\ImageRetrieval\Professions_125k_ISCO_Aligned_1k_Subset" --resize 224 224 --batch_size 16 --num_workers 8 --output_dir "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\Shuffling&Colour" --exclude_dirs facemesh --patch_sizes 2 4 8 16 --do_pixel_shuffle --do_patch_shuffle --do_mean_rgb

# python PixelPatchShufflingMeanRGB.py --image_dir "F:\ImageRetrieval\Coco" --resize 224 224 --batch_size 16 --num_workers 8 --output_dir "F:\ImageRetrieval\SpuriousFeatureImages\Coco\Shuffling&Colour" --exclude_dirs facemesh --patch_sizes 2 4 8 16 --do_pixel_shuffle --do_patch_shuffle --do_mean_rgb





# python PixelPatchShufflingMeanRGB.py --image_dir "F:\ImageRetrieval\Professions_125k_ISCO_Aligned_1k_Subset\A_photo_of_a_florist" --resize 224 224 --batch_size 16 --num_workers 8 --output_dir "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\Shuffling&Colour\pixel_shuffle\A_photo_of_a_florist" --exclude_dirs facemesh --do_pixel_shuffle