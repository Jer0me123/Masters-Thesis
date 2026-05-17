import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================

INPUT_DIR = Path(r"FACET_Dataset\Segmented_FACET_0.2_fixed_label")
OUTPUT_DIR = Path(r"FACET_Dataset\Segmented_FACET_0.2_fixed_label_BGFixed")

BLACK_THRESHOLD = 10  # pixels with RGB <= this are considered background
NUM_WORKERS = 8
JPEG_QUALITY = 95

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================
# BACKGROUND CORRECTION LOGIC
# ==============================

def process_image(input_path: Path):
    relative_path = input_path.relative_to(INPUT_DIR)
    output_path = OUTPUT_DIR / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        return

    try:
        img = Image.open(input_path).convert("RGB")
        arr = np.array(img)

        # Detect black pixels
        black_mask = np.all(arr <= BLACK_THRESHOLD, axis=2)

        if np.sum(~black_mask) == 0:
            # Edge case: entire image black
            img.save(output_path, quality=JPEG_QUALITY)
            return

        # Compute mean face colour
        face_pixels = arr[~black_mask]
        mean_color = face_pixels.mean(axis=0).astype(np.uint8)

        # Replace black background
        arr[black_mask] = mean_color

        corrected_img = Image.fromarray(arr)
        corrected_img.save(output_path, quality=JPEG_QUALITY)

    except Exception as e:
        print(f"Error processing {input_path}: {e}")


# ==============================
# MAIN
# ==============================

def main():
    image_paths = list(INPUT_DIR.rglob("*.jpg"))

    print(f"Found {len(image_paths)} images")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        list(tqdm(executor.map(process_image, image_paths), total=len(image_paths)))

    print("Background-corrected dataset created.")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


