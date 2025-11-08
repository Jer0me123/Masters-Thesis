import os
import time
from math import ceil
from tqdm.auto import tqdm
import argparse
import cv2
from concurrent.futures import ThreadPoolExecutor
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Ensure NumPy types are correctly defined
np.float_ = np.float64
np.complex_ = np.complex128

# ---------------- Dataset ----------------
class DepthImageDataset(Dataset):
    """
    Loads images as PIL.Image objects, resizing them to a fixed size for batching,
    returning the image, path, and original size.
    Skips images that are already processed (resume logic).
    """
    def __init__(self, image_dir, output_dir=None, target_size=(384, 384), extensions=(".jpg", ".jpeg", ".png")):
        self.target_size = target_size
        all_images = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(extensions)
        ]

        # Determine already processed images
        processed_basenames = set()
        if output_dir and os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                if f.lower().endswith("_depth.png"):
                    processed_basenames.add(os.path.splitext(f)[0].replace("_depth", ""))

        # Only keep unprocessed images
        self.image_paths = [p for p in all_images if os.path.splitext(os.path.basename(p))[0] not in processed_basenames]
        if processed_basenames:
            print(f"Resuming: skipping {len(processed_basenames)} already processed images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        original_size = None
        img = None
        
        try:
            img = Image.open(path).convert("RGB")
            original_size = img.size  # store original (W, H)

            w, h = original_size
            if w <= 1 or h <= 1:
                raise ValueError("Invalid image dimensions")
            
            # Resize to fixed target size for batching
            img = img.resize(self.target_size, Image.BICUBIC)

        except (UnidentifiedImageError, OSError, ValueError) as e:
            # Robust error handling: Return black placeholder
            img = Image.new("RGB", self.target_size, color=(0, 0, 0))
            original_size = self.target_size
            print(f"⚠️ Replaced corrupted image with black placeholder: {path} ({e})")
        
        return img, path, original_size

# ---------------- Collate Function ----------------
def collate_depth_batch(batch):
    """Simple collate to group items from the dataset."""
    imgs, paths, sizes = zip(*batch)
    return list(imgs), list(paths), list(sizes)

# ---------------- DataLoader ----------------
def get_depth_image_loader(image_dir, output_dir=None, batch_size=32, num_workers=4):
    dataset = DepthImageDataset(image_dir, output_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_depth_batch,
        pin_memory=True
    )
    return loader, len(dataset)

# ---------------- Save Depth Map ----------------
def save_depth_map_cv2(depth_np, save_path):
    depth_rgb = np.repeat(depth_np[..., np.newaxis], 3, axis=-1)
    cv2.imwrite(save_path, depth_rgb)

# ---------------- Main Script ----------------
def main(args):
    start_time = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- 1. Setup Model and Processor ---
    device = torch.device(args.device)
    
    model_name = f"depth-anything/Depth-Anything-V2-{args.model_size}-hf"
    image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True, size={"height": 384, "width": 384})
    model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device).eval()

    # --- 2. Load DataLoader (with resume) ---
    loader, dataset_len = get_depth_image_loader(
        args.image_dir, 
        output_dir=args.output_dir,
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    
    if dataset_len == 0:
        print("No remaining images to process. Exiting.")
        return

    print(f"Remaining images to process: {dataset_len}")
    imgid_counter = 0
    total_batches = ceil(dataset_len / args.batch_size)

    # --- 3. Process Batches ---
    pbar = tqdm(loader, desc="Processing Batches", total=total_batches)
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        for batch_imgs, batch_paths, batch_sizes in pbar:
            inputs = image_processor(images=batch_imgs, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth

            futures = []
            for depth_map, path, original_size in zip(predicted_depth, batch_paths, batch_sizes):
                prediction = torch.nn.functional.interpolate(
                    depth_map.unsqueeze(0).unsqueeze(0),
                    size=original_size[::-1],
                    mode="bicubic",
                    align_corners=False
                ).squeeze()

                depth_np = prediction.cpu().numpy()
                if depth_np.max() != depth_np.min():
                    depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min()) * 255.0
                else:
                    depth_np = np.zeros_like(depth_np)
                depth_np = depth_np.astype(np.uint8)

                save_filename = os.path.splitext(os.path.basename(path))[0] + "_depth.png"
                save_path = os.path.join(args.output_dir, save_filename)
                futures.append(executor.submit(save_depth_map_cv2, depth_np, save_path))
                imgid_counter += 1

            for f in futures:
                f.result()

    total_time = time.time() - start_time
    print(f"\nDepth maps saved to {args.output_dir}")
    print(f"Total images processed: {imgid_counter}")
    print(f"Total processing time: {total_time:.2f}s")

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Depth Estimation with resume (skip already processed images)")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--model_size", type=str, default="Large", choices=["Small", "Base", "Large", "Giant"], 
                        help="Depth-Anything V2 model size to use (Small is fastest).")
    args = parser.parse_args()
    main(args)

# Small & Base should be sufficiently fast Large (was used in the paper not fast enough)
# "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\.venv-Copy-Copy\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\Depth\depth.py" --image_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ImageCaptioningEvaluationDatasets\LAION-5B-10k\LAION-5B-10k-images" --batch_size 8 --num_workers 8 --output_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\Depth\LAION-5B-10k-depth" --model_size "Small"