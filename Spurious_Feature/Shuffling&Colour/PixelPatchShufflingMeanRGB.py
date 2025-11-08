import os
import glob
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

from PIL import Image, UnidentifiedImageError
import numpy as np
import cv2

# ---------------- Dataset ----------------
class PILImageDataset:
    """Dataset wrapper that can exclude already processed images."""
    def __init__(self, image_dir, exclude_basenames=None, extensions=(".jpg", ".jpeg", ".png")):
        exclude_basenames = exclude_basenames or set()
        self.image_paths = [
            p for p in glob.glob(os.path.join(image_dir, "**", "*.*"), recursive=True)
            if p.lower().endswith(extensions)
            and os.path.splitext(os.path.basename(p))[0] not in exclude_basenames
        ]

    def __len__(self):
        return len(self.image_paths)

    def get_batch(self, start_idx: int, batch_size: int):
        batch_paths = self.image_paths[start_idx:start_idx + batch_size]
        batch_imgs = []
        for path in batch_paths:
            try:
                img = Image.open(path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                w, h = img.size
                if w <= 1 or h <= 1:
                    raise ValueError("Invalid image dimensions")
            except (UnidentifiedImageError, OSError, ValueError) as e:
                img = Image.new("RGB", (224, 224), color=(0, 0, 0))
                print(f"Replaced corrupted image with black placeholder: {path} ({e})")
            batch_imgs.append(img)
        return batch_imgs, batch_paths


# ---------------- Save Utility ----------------
def save_image_cv2(image: Image.Image, save_path: str):
    arr = np.array(image)
    if len(arr.shape) == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    cv2.imwrite(save_path, arr)


# ---------------- Transform Functions ----------------
def pixel_shuffle_single_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    h, w, c = arr.shape
    rng = np.random.default_rng(seed=int(np.sum(arr[0, 0, :])))
    flat = arr.reshape(-1, c)
    perm = rng.permutation(flat.shape[0])
    shuffled = flat[perm].reshape(h, w, c)
    return Image.fromarray(shuffled)


def patch_shuffle_single_image(path: str, patch_size: int = 16) -> Image.Image:
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    h, w, c = arr.shape

    # Pad to multiple of patch_size
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    if pad_h > 0 or pad_w > 0:
        padded_arr = np.zeros((h + pad_h, w + pad_w, c), dtype=arr.dtype)
        padded_arr[:h, :w, :] = arr
        arr = padded_arr
        h, w = arr.shape[:2]

    ph, pw = patch_size, patch_size
    patches = [arr[i:i+ph, j:j+pw, :] for i in range(0, h, ph) for j in range(0, w, pw)]

    rng = np.random.default_rng(seed=int(np.sum(patches[0])))
    perm = rng.permutation(len(patches))

    shuffled_arr = np.zeros_like(arr)
    idx = 0
    for i in range(0, h, ph):
        for j in range(0, w, pw):
            shuffled_arr[i:i+ph, j:j+pw, :] = patches[perm[idx]]
            idx += 1

    shuffled_arr = shuffled_arr[:h - pad_h or None, :w - pad_w or None, :]
    return Image.fromarray(shuffled_arr)


def mean_rgb_single_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    mean_val = arr.mean(axis=(0, 1)).astype(np.uint8)
    mean_img = np.ones_like(arr) * mean_val
    return Image.fromarray(mean_img)


# ---------------- Batch Processing ----------------
def process_transform_batch(batch_paths: List[str], output_dir: str, transform_fn, max_workers: int = 8, patch_size: int = 16):
    os.makedirs(output_dir, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for path in batch_paths:
            if transform_fn.__name__ == "patch_shuffle_single_image":
                futures.append(executor.submit(transform_fn, path, patch_size))
            else:
                futures.append(executor.submit(transform_fn, path))

        for path, future in zip(batch_paths, as_completed(futures)):
            try:
                img = future.result()
                save_name = os.path.splitext(os.path.basename(path))[0] + f"_{transform_fn.__name__}.png"
                save_path = os.path.join(output_dir, save_name)
                save_image_cv2(img, save_path)
            except Exception as e:
                print(f"Error processing {path}: {e}")


# ---------------- Helpers ----------------
def get_already_processed_basenames(output_dir: str) -> set:
    """Collect base filenames that are already processed in any output folder."""
    processed = set()
    if not os.path.exists(output_dir):
        return processed

    for subdir in ["pixel_shuffle", "patch_shuffle", "mean_rgb"]:
        subpath = os.path.join(output_dir, subdir)
        if not os.path.isdir(subpath):
            continue
        for f in os.listdir(subpath):
            if f.lower().endswith(".png"):
                base = os.path.basename(f)
                base = base.split("_")[0]  # remove suffix
                processed.add(base)
    return processed


# ---------------- Main ----------------
def main(args):
    # Determine already processed images
    already_done = get_already_processed_basenames(args.output_dir)
    print(f"Found {len(already_done)} already processed images, skipping them.")

    # Create dataset excluding them
    dataset = PILImageDataset(args.image_dir, exclude_basenames=already_done)
    total_images = len(dataset)
    print(f"Remaining images to process: {total_images}")

    batch_size = args.batch_size

    # Output directories
    pixel_dir = os.path.join(args.output_dir, "pixel_shuffle")
    patch_dir = os.path.join(args.output_dir, "patch_shuffle")
    mean_dir = os.path.join(args.output_dir, "mean_rgb")

    for start_idx in tqdm(range(0, total_images, batch_size), desc="Processing batches"):
        _, batch_paths = dataset.get_batch(start_idx, batch_size)

        if args.do_pixel_shuffle:
            process_transform_batch(batch_paths, pixel_dir, pixel_shuffle_single_image, max_workers=args.num_workers)

        if args.do_patch_shuffle:
            process_transform_batch(batch_paths, patch_dir, patch_shuffle_single_image, max_workers=args.num_workers, patch_size=args.patch_size)

        if args.do_mean_rgb:
            process_transform_batch(batch_paths, mean_dir, mean_rgb_single_image, max_workers=args.num_workers)

    print(f"\nAll transformations complete. Output in: {args.output_dir}")


# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batched Pixel / Patch / Mean RGB transformations with resume (skipping preprocessed files)")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--do_pixel_shuffle", action="store_true", help="Enable pixel shuffle")
    parser.add_argument("--do_patch_shuffle", action="store_true", help="Enable patch shuffle")
    parser.add_argument("--do_mean_rgb", action="store_true", help="Enable mean RGB")
    args = parser.parse_args()
    main(args)

# "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\.venv-Copy-Copy\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\Shuffling&Colour\PixelPatchShufflingMeanRGB.py" --image_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ImageCaptioningEvaluationDatasets\LAION-5B-10k\LAION-5B-10k-images" --batch_size 8 --num_workers 8 --output_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\Shuffling&Colour\LAION-5B-10k-pixelColour" --patch_size 16 --do_pixel_shuffle --do_patch_shuffle --do_mean_rgb

# NOTE: Both Pixel and Patch Shuffling are randomized per image, but the random seed is derived from the image's first pixel value. This makes the shuffling deterministic (reproducible) for each image across runs.