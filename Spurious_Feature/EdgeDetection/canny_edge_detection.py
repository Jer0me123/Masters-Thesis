# import os
# import time
# import json
# from math import ceil
# from tqdm import tqdm
# import argparse
# from concurrent.futures import ThreadPoolExecutor, as_completed

# import torch
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image, UnidentifiedImageError
# import numpy as np
# import cv2

# np.float_ = np.float64
# np.complex_ = np.complex128

# # ---------------- Dataset ----------------
# class PILImageDataset(Dataset):
#     """Loads images as PIL.Image objects, no conversion to tensors."""
#     def __init__(self, image_dir, extensions=(".jpg", ".jpeg", ".png")):
#         self.image_paths = [
#             os.path.join(image_dir, f)
#             for f in os.listdir(image_dir)
#             if f.lower().endswith(extensions)
#         ]

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         path = self.image_paths[idx]
#         try:
#             img = Image.open(path)

#             # Handle grayscale or other modes
#             if img.mode != "RGB":
#                 img = img.convert("RGB")

#             # Check for invalid dimensions
#             w, h = img.size
#             if w <= 1 or h <= 1:
#                 raise ValueError("Invalid image dimensions")

#         except (UnidentifiedImageError, OSError, ValueError) as e:
#             # Return black placeholder if corrupted
#             img = Image.new("RGB", (224, 224), color=(0, 0, 0))
#             print(f"⚠️ Replaced corrupted image with black placeholder: {path} ({e})")

#         return img, path

# # ---------------- Collate Function ----------------
# def collate_images(batch):
#     imgs, paths = zip(*batch)
#     return list(imgs), list(paths)

# # ---------------- DataLoader ----------------
# def get_pil_image_loader(image_dir, batch_size=32, num_workers=4):
#     dataset = PILImageDataset(image_dir)
#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         shuffle=False,
#         collate_fn=collate_images,
#         pin_memory=True
#     )
#     return loader, len(dataset)

# # ---------------- Canny Edge Detection Logic ----------------
# class EdgeDetector:
#     def __init__(self):
#         pass

#     def detect_edge(self, image):
#         image = np.array(image)
#         if len(image.shape) == 3:
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#         image = cv2.GaussianBlur(image, (3, 3), 0)
#         edges = cv2.Canny(image, threshold1=255/3, threshold2=255)
#         return edges

# # ---------------- Parallel Processing Function ----------------
# def process_image(detector, img_path_tuple, output_dir, imgid):
#     img, path = img_path_tuple
#     try:
#         edges = detector.detect_edge(img)
#         edge_filename = os.path.splitext(os.path.basename(path))[0] + "_edges.png"
#         edge_path = os.path.join(output_dir, edge_filename)
#         cv2.imwrite(edge_path, edges)
#         return {
#             "filename": os.path.basename(path),
#             "edge_map": edge_filename,
#             "imgid": imgid
#         }
#     except Exception as e:
#         print(f"⚠️ Error processing {path}: {e}")
#         return None

# # ---------------- Main Script ----------------
# def main(args):
#     total_start_time = time.time()

#     image_dir = args.image_dir
#     batch_size = args.batch_size
#     num_workers = args.num_workers
#     output_dir = args.output_dir
#     os.makedirs(output_dir, exist_ok=True)

#     detector = EdgeDetector()
#     loader, dataset_len = get_pil_image_loader(image_dir, batch_size=batch_size, num_workers=num_workers)
#     startup_time = time.time() - total_start_time

#     edge_data = {"dataset": os.path.basename(image_dir), "images": []}
#     imgid_counter = 0

#     detection_start_time = time.time()

#     for imgs, paths in tqdm(loader, desc="Processing batches", total=ceil(dataset_len / batch_size)):
#         # Build tuples for processing
#         batch_tuples = list(zip(imgs, paths))
#         results = []

#         # Parallel processing using ThreadPoolExecutor
#         with ThreadPoolExecutor(max_workers=num_workers) as executor:
#             futures = [
#                 executor.submit(process_image, detector, t, output_dir, imgid_counter + i)
#                 for i, t in enumerate(batch_tuples)
#             ]
#             for future in as_completed(futures):
#                 result = future.result()
#                 if result is not None:
#                     results.append(result)

#         # Update imgid_counter
#         imgid_counter += len(results)
#         edge_data["images"].extend(results)

#         # Stream JSON after each batch
#         json_output_path = os.path.join(output_dir, "edge_detection_results.json")
#         with open(json_output_path, "w", encoding="utf-8") as f:
#             json.dump(edge_data, f, indent=2)

#     detection_time = time.time() - detection_start_time
#     print(f"\nEdge maps saved to {output_dir}")
#     print(f"\nTiming Summary:")
#     print(f" - Startup time (setup): {startup_time:.2f}s")
#     print(f" - Detection time (processing only): {detection_time:.2f}s")

# # ---------------- CLI ----------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Batch Parallel Canny Edge Detection")
#     parser.add_argument("--image_dir", type=str, required=True)
#     parser.add_argument("--batch_size", type=int, default=25)
#     parser.add_argument("--num_workers", type=int, default=4)
#     parser.add_argument("--output_dir", type=str, required=True)
#     args = parser.parse_args()
#     main(args)

import os
import time
from math import ceil
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
import numpy as np
import cv2

np.float_ = np.float64
np.complex_ = np.complex128

# ---------------- Dataset ----------------
class PILImageDataset(Dataset):
    """Loads images as PIL.Image objects, skips already processed images if provided."""
    def __init__(self, image_dir, skip_basenames=None, extensions=(".jpg", ".jpeg", ".png")):
        skip_basenames = skip_basenames or set()
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(extensions) and os.path.splitext(f)[0] not in skip_basenames
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path)

            if img.mode != "RGB":
                img = img.convert("RGB")

            w, h = img.size
            if w <= 1 or h <= 1:
                raise ValueError("Invalid image dimensions")

        except (UnidentifiedImageError, OSError, ValueError) as e:
            img = Image.new("RGB", (224, 224), color=(0, 0, 0))
            print(f"⚠️ Replaced corrupted image with black placeholder: {path} ({e})")

        return img, path

# ---------------- Collate Function ----------------
def collate_images(batch):
    imgs, paths = zip(*batch)
    return list(imgs), list(paths)

# ---------------- DataLoader ----------------
def get_pil_image_loader(image_dir, skip_basenames=None, batch_size=32, num_workers=4):
    dataset = PILImageDataset(image_dir, skip_basenames=skip_basenames)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_images,
        pin_memory=True
    )
    return loader, len(dataset)

# ---------------- Canny Edge Detection Logic ----------------
class EdgeDetector:
    def detect_edge(self, image):
        image = np.array(image)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.GaussianBlur(image, (3, 3), 0)
        edges = cv2.Canny(image, threshold1=255/3, threshold2=255)
        return edges

# ---------------- Parallel Processing Function ----------------
def process_image(detector, img_path_tuple, output_dir):
    img, path = img_path_tuple
    try:
        edges = detector.detect_edge(img)
        edge_filename = os.path.splitext(os.path.basename(path))[0] + "_edges.png"
        edge_path = os.path.join(output_dir, edge_filename)
        cv2.imwrite(edge_path, edges)
    except Exception as e:
        print(f"⚠️ Error processing {path}: {e}")

# ---------------- Main Script ----------------
def main(args):
    total_start_time = time.time()

    image_dir = args.image_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # --- Resume Logic: skip already processed images ---
    processed_basenames = set()
    for f in os.listdir(output_dir):
        if f.lower().endswith("_edges.png"):
            processed_basenames.add(os.path.splitext(f)[0].replace("_edges", ""))
    if processed_basenames:
        print(f"Resuming: skipping {len(processed_basenames)} already processed images.")

    detector = EdgeDetector()
    loader, dataset_len = get_pil_image_loader(
        image_dir, skip_basenames=processed_basenames,
        batch_size=batch_size, num_workers=num_workers
    )
    startup_time = time.time() - total_start_time

    if dataset_len == 0:
        print("No remaining images to process. Exiting.")
        return

    detection_start_time = time.time()

    for imgs, paths in tqdm(loader, desc="Processing batches", total=ceil(dataset_len / batch_size)):
        batch_tuples = list(zip(imgs, paths))
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_image, detector, t, output_dir) for t in batch_tuples]
            for f in futures:
                f.result()

    detection_time = time.time() - detection_start_time
    print(f"\nEdge maps saved to {output_dir}")
    print(f"\nTiming Summary:")
    print(f" - Startup time (setup): {startup_time:.2f}s")
    print(f" - Detection time (processing only): {detection_time:.2f}s")

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Parallel Canny Edge Detection with Resume")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)

# "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\.venv-Copy-Copy\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\EdgeDetection\canny_edge_detection.py" --image_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ImageCaptioningEvaluationDatasets\LAION-5B-10k\LAION-5B-10k-images" --batch_size 8 --num_workers 8 --output_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\EdgeDetection\LAION-5B-10k-canny-edge-detected" 