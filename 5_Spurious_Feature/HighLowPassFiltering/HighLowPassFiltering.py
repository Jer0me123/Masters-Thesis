"""
High / Low-Pass Frequency Filtering Pipeline
===========================================

This script implements frequency-domain image transformations used to
analyze reliance on low- vs high-frequency visual cues in large-scale
vision datasets (e.g., LAION-5B derivatives).

Methodological alignment:
-------------------------
This implementation follows the experimental protocol described in:

"Understanding Bias in Large-Scale Visual Datasets"
(Zeng et al., NeurIPS 2024)

Specifically:
- Images are converted to grayscale
- A 2D FFT is applied
- Either low-frequency or high-frequency components are isolated
- The inverse FFT reconstructs filtered images
- Filtered images are used downstream to measure performance degradation
  and identify spurious correlations

Design goals:
-------------
- Scales to hundreds of thousands or millions of images
- Fully resumable (no directory scanning on restart)
- Deterministic and reproducible
- Matches the architecture and conventions of PixelPatchShufflingMeanRGB.py
"""

import os
import glob
import json
import argparse
from threading import Lock, Thread
from queue import Queue
from typing import Tuple, List

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm


# ============================================================
# FAST RGB DECODER
# ============================================================
class FastRGBDecoder:
    """
    High-performance RGB image decoder.

    Why this exists:
    ---------------
    At LAION scale, JPEG decoding dominates runtime if done via PIL.
    Both referenced papers operate at massive scale, so decoding must
    be optimized.

    Implementation:
    ---------------
    - Uses libjpeg-turbo (via PyTurboJPEG) when available
    - Falls back to OpenCV otherwise
    - Guarantees RGB output ordering
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
        """
        Load an image as an RGB NumPy array.

        Returns:
        --------
        - np.ndarray (H, W, 3) in RGB order
        - None if decoding fails
        """
        try:
            ext = os.path.splitext(path)[1].lower()
            if self.use_turbo and ext in {".jpg", ".jpeg"}:
                with open(path, "rb") as f:
                    return self.jpeg.decode(f.read())
            arr = cv2.imread(path)
            if arr is None:
                return None
            return arr[..., ::-1]  # BGR → RGB
        except Exception:
            return None


# ============================================================
# ASYNC IMAGE WRITER
# ============================================================
class AsyncImageWriter:
    """
    Asynchronous disk writer.

    Why this exists:
    ---------------
    Writing PNGs synchronously becomes the dominant bottleneck once
    FFT and filtering are optimized.

    This class:
    - Decouples compute from I/O
    - Allows FFT/filtering to proceed without blocking
    """

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
# TRANSFORM MANIFEST (RESUME LOGIC)
# ============================================================
class TransformManifest:
    """
    Persistent, append-only manifest tracking completed transforms.

    Why this is critical:
    ---------------------
    Scanning output directories at startup is infeasible at scale.
    Instead, we track completion explicitly.

    Manifest format (JSONL):
        {"image": "relative/path.jpg", "transform": "ideal_lowpass_r40"}

    An image is considered *complete* only if ALL required transforms
    are present.
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
    Enumerates images that still require frequency filtering.

    Responsibilities:
    -----------------
    - Recursive directory traversal
    - Exclude non-image or auxiliary directories (e.g., facemesh)
    - Preserve directory structure
    - Apply optional resize before FFT
    """

    def __init__(
        self,
        image_dir: str,
        completed: set,
        exclude_dirs: List[str],
        resize: Tuple[int, int] | None,
    ):
        self.decoder = FastRGBDecoder()
        self.image_dir = image_dir
        self.resize = resize
        exclude_dirs = {d.lower() for d in exclude_dirs}

        self.paths = []
        for p in glob.glob(os.path.join(image_dir, "**", "*.*"), recursive=True):
            if not p.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            if set(os.path.normpath(p).lower().split(os.sep)) & exclude_dirs:
                continue

            rel = os.path.relpath(p, image_dir).replace("\\", "/")
            if rel in completed:
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
            elif self.resize is not None:
                arr = cv2.resize(arr, self.resize, interpolation=cv2.INTER_LINEAR)

            arrays.append(arr)

        return arrays, paths


# ============================================================
# FREQUENCY FILTERS
# ============================================================
def build_ideal_masks(shape, radius):
    """
    Ideal (hard cutoff) frequency masks.

    Used in the main paper experiments.
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    d = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
    low = (d <= radius).astype(np.float32)
    high = 1.0 - low
    return low, high


def build_butterworth_masks(shape, radius, order):
    """
    Butterworth frequency masks.

    Used as a smooth alternative to ideal filters.
    The paper specifies order = 2 when Butterworth is used.
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    d = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
    low = 1.0 / (1.0 + (d / radius) ** (2 * order))
    high = 1.0 - low
    return low.astype(np.float32), high.astype(np.float32)


def apply_frequency_filter(arr_rgb, radius, filter_type, butterworth_order):
    """
    Apply frequency-domain low/high-pass filtering to a single image.

    Steps:
    ------
    1. Convert to grayscale
    2. Apply FFT
    3. Apply selected frequency mask
    4. Inverse FFT
    5. Normalize to uint8
    """
    gray = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    if filter_type == "butterworth":
        low_mask, high_mask = build_butterworth_masks(
            gray.shape, radius, butterworth_order
        )
    else:
        low_mask, high_mask = build_ideal_masks(gray.shape, radius)

    low = np.fft.ifft2(np.fft.ifftshift(fshift * low_mask)).real
    high = np.fft.ifft2(np.fft.ifftshift(fshift * high_mask)).real

    low = cv2.normalize(low, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    high = cv2.normalize(high, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return low, high


# ============================================================
# MAIN
# ============================================================
def main(args):
    """
    Main execution loop.

    Mirrors PixelPatchShufflingMeanRGB.py:
    - Manifest-based resume
    - Decode once per image
    - Apply all requested transforms
    - Async I/O
    """

    filter_root = (
        "ideal"
        if args.filter_type == "ideal"
        else f"butterworth_o{args.butterworth_order}"
    )

    manifest = TransformManifest(
        os.path.join(args.output_dir, "transform_manifest.jsonl")
    )

    required = set()
    for r in args.radius:
        if args.do_low:
            required.add(f"{filter_root}_lowpass_r{r}")
        if args.do_high:
            required.add(f"{filter_root}_highpass_r{r}")

    completed = {
        img for img, done in manifest.data.items()
        if required.issubset(done)
    }
    print(f"Skipping {len(completed)} completed images.")   

    resize = tuple(args.resize) if args.resize else None

    dataset = ImageDataset(
        args.image_dir,
        completed,
        args.exclude_dirs,
        resize,
    )

    print(f"Images to process: {len(dataset)}")

    writer = AsyncImageWriter(num_workers=min(4, args.num_workers))

    for i in tqdm(range(0, len(dataset), args.batch_size)):
        arrays, paths = dataset.get_batch(i, args.batch_size)

        for arr, path in zip(arrays, paths):
            rel = os.path.relpath(path, args.image_dir).replace("\\", "/")
            stem = os.path.splitext(rel)[0]

            for r in args.radius:
                low, high = apply_frequency_filter(
                    arr, r, args.filter_type, args.butterworth_order
                )

                if args.do_low:
                    writer.submit(
                        os.path.join(
                            args.output_dir,
                            filter_root,
                            f"radius_{r}",
                            "low_pass",
                            stem + "_low.png",
                        ),
                        low,
                    )
                    manifest.record(rel, f"{filter_root}_lowpass_r{r}")

                if args.do_high:
                    writer.submit(
                        os.path.join(
                            args.output_dir,
                            filter_root,
                            f"radius_{r}",
                            "high_pass",
                            stem + "_high.png",
                        ),
                        high,
                    )
                    manifest.record(rel, f"{filter_root}_highpass_r{r}")

    writer.close()
    manifest.flush()
    print("Done.")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="High / Low-pass filtering aligned with PixelPatchShufflingMeanRGB"
    )
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--exclude_dirs", nargs="+", default=["facemesh"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--radius", type=int, nargs="+", default=[40])
    parser.add_argument("--filter_type", choices=["ideal", "butterworth"], default="ideal")
    parser.add_argument("--butterworth_order", type=int, default=2)
    parser.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"), default=None)
    parser.add_argument("--do_low", action="store_true")
    parser.add_argument("--do_high", action="store_true")
    main(parser.parse_args())

# ===========================================================
# EXAMPLE USAGE
# python HighLowPassFiltering.py --image_dir "path/to/input" --resize 224 224 --batch_size 16 --num_workers 8 --output_dir "path/to/output" --exclude_dirs facemesh --radius 40 --do_low --do_high --filter_type ideal

# --resize 224 224 -> This is done as the classification model auto resizes images to 224 x 244 hence its better to resize them prior as this makes processing faster and storge requirements less.
# --exclude_dirs facemesh -> This is done to exclude any images in the facemesh directory from processing as these are not actual images but rather facemesh data.
# --filter_type ideal -> This is done as the ideal filter type was the primary filter used in the paper, however the butterworth filter was mentioned & tested in the Appendix, hence its inclusion
#  --radius 40 -> This is the primary radius value used in the paper for both high and low pass filtering with ideal filter, however other radius values were tested in the Appendix.

# NOTE: Radius 40 was used in the paper: "We then apply an ideal filter [22] with a hard threshold radius of 40 in the frequency domain, so as to only keep either high (i.e., high-pass filter) or low (i.e., low-pass filter) frequencies"

# "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\.venv-Copy-Copy\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\HighLowPassFiltering\test2.py" --image_dir "E:\ImageRetrieval\Professions_125k_Cleaned" --batch_size 8 --num_workers 8 --output_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\HighLowPassFiltering\test" --radius 40 --do_low --do_high --filter_type ideal

# python HighLowPassFiltering.py --image_dir "E:\ImageRetrieval\StableDiffusionGeneratedImages\valid" --resize 224 224 --batch_size 16 --num_workers 8 --output_dir "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\High&LowPassFilter" --exclude_dirs face_crops --radius 40 --do_low --do_high --filter_type ideal

# python HighLowPassFiltering.py --image_dir "F:\ImageRetrieval\Professions_125k_ISCO_Aligned_1k_Subset" --resize 224 224 --batch_size 16 --num_workers 8 --output_dir "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\High&LowPassFilter" --exclude_dirs facemesh --radius 40 --do_low --do_high --filter_type ideal

# python HighLowPassFiltering.py --image_dir "F:\ImageRetrieval\Coco" --resize 224 224 --batch_size 16 --num_workers 8 --output_dir "F:\ImageRetrieval\SpuriousFeatureImages\Coco\High&LowPassFilter" --exclude_dirs facemesh --radius 40 --do_low --do_high --filter_type ideal