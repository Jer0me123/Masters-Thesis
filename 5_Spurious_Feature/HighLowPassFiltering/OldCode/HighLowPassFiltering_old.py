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
        return self.image_paths[start_idx:start_idx + batch_size]


# ---------------- Save Utility ----------------
def save_image_cv2(image: np.ndarray, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, image)


# ---------------- Transform Function ----------------
def high_low_pass_filter(image_path: str, radius: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    """Apply ideal low-pass and high-pass filters on a single image."""
    img = Image.open(image_path).convert("L")  # grayscale
    img_np = np.array(img, dtype=np.float32)

    # 2D FFT
    f = np.fft.fft2(img_np)
    fshift = np.fft.fftshift(f)

    rows, cols = img_np.shape
    crow, ccol = rows // 2, cols // 2

    # Create ideal circular masks
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)

    mask_low = (distance <= radius).astype(np.float32)
    mask_high = 1 - mask_low

    # Apply masks
    fshift_low = fshift * mask_low
    fshift_high = fshift * mask_high

    # Inverse FFT
    img_low = np.fft.ifft2(np.fft.ifftshift(fshift_low)).real
    img_high = np.fft.ifft2(np.fft.ifftshift(fshift_high)).real

    # Normalize 0–255 for visualization
    img_low = cv2.normalize(img_low, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img_high = cv2.normalize(img_high, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return img_low, img_high


# ---------------- Batch Processing ----------------
def process_filter_batch(
    batch_paths: List[str],
    output_dir: str,
    radius: int,
    max_workers: int = 8,
    do_low: bool = True,
    do_high: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(high_low_pass_filter, path, radius): path for path in batch_paths}

        for future in as_completed(futures):
            path = futures[future]
            try:
                img_low, img_high = future.result()
                base = os.path.splitext(os.path.basename(path))[0]
                if do_low:
                    save_path = os.path.join(output_dir, "low_pass", f"{base}_lowpass.png")
                    save_image_cv2(img_low, save_path)
                if do_high:
                    save_path = os.path.join(output_dir, "high_pass", f"{base}_highpass.png")
                    save_image_cv2(img_high, save_path)
            except Exception as e:
                print(f"⚠️ Error processing {path}: {e}")


# ---------------- Helpers ----------------
def get_already_processed_basenames(output_dir: str) -> set:
    """Collect base filenames that are already processed."""
    processed = set()
    if not os.path.exists(output_dir):
        return processed

    for subdir in ["low_pass", "high_pass"]:
        subpath = os.path.join(output_dir, subdir)
        if not os.path.isdir(subpath):
            continue
        for f in os.listdir(subpath):
            if f.lower().endswith(".png"):
                base = os.path.basename(f).split("_")[0]
                processed.add(base)
    return processed


# ---------------- Main ----------------
def main(args):
    # Skip already processed images
    already_done = get_already_processed_basenames(args.output_dir)
    print(f"Found {len(already_done)} already processed images, skipping them.")

    dataset = PILImageDataset(args.image_dir, exclude_basenames=already_done)
    total_images = len(dataset)
    print(f"Remaining images to process: {total_images}")

    for start_idx in tqdm(range(0, total_images, args.batch_size), desc="Processing batches"):
        batch_paths = dataset.get_batch(start_idx, args.batch_size)
        process_filter_batch(
            batch_paths=batch_paths,
            output_dir=args.output_dir,
            radius=args.radius,
            max_workers=args.num_workers,
            do_low=args.do_low,
            do_high=args.do_high,
        )

    print(f"\nAll filtering complete. Results saved to: {args.output_dir}")


# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply batched low-pass and high-pass filtering with resume support")
    parser.add_argument("--image_dir", type=str, required=True, help="Input image directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for parallel processing")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of threads")
    parser.add_argument("--radius", type=int, default=40, help="Cutoff radius for frequency filtering")
    parser.add_argument("--do_low", action="store_true", help="Enable low-pass filter")
    parser.add_argument("--do_high", action="store_true", help="Enable high-pass filter")
    args = parser.parse_args()
    main(args)


# "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\.venv-Copy-Copy\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\HighLowPassFiltering\HighLowPassFiltering.py" --image_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ImageCaptioningEvaluationDatasets\LAION-5B-10k\LAION-5B-10k-images" --batch_size 8 --num_workers 8 --output_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\HighLowPassFiltering\LAION-5B-10k-highLowFilter" --radius 40 --do_low --do_high

# NOTE: Radius 40 was used in the paper: "We then apply an ideal filter [22] with a hard threshold radius of 40 in the frequency domain, so as to only keep either high (i.e., high-pass filter) or low (i.e., low-pass filter) frequencies"