import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, Future
from typing import List, Tuple
import argparse
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms.functional import resize
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

# ---------------- Worker Function ----------------
def process_seg_map_resize_and_save(
    seg_map_tensor: torch.Tensor,
    original_size: Tuple[int, int],
    save_path: str,
    palette: np.ndarray
) -> str:
    """
    Post-processing function executed in a separate CPU process.
    1. Convert tensor to NumPy array.
    2. Resize segmentation map back to original image size.
    3. Apply ADE20K color palette to visualize segmentation.
    4. Save colored image as PNG with minimal compression for speed.
    
    Args:
        seg_map_tensor: Tensor with predicted class indices.
        original_size: Tuple (H, W) to resize back to original dimensions.
        save_path: Full path where colored segmentation image is saved.
        palette: ADE20K color palette array for visualization.
    
    Returns:
        The path of the saved image.
    """
    seg_map_np = seg_map_tensor.numpy().astype(np.uint8)
    
    # Resize segmentation map to original image size
    seg_map_resized = cv2.resize(seg_map_np, dsize=(original_size[1], original_size[0]),
                                 interpolation=cv2.INTER_NEAREST)
    
    # Map class indices to RGB colors
    colored = palette[seg_map_resized % len(palette)]
    
    # Save the final colored segmentation map
    Image.fromarray(colored, mode="RGB").save(save_path, format="PNG", compress_level=1)
    return save_path

# ---------------- Dataset & Collate ----------------
class ImageDataset(Dataset):
    """
    Custom PyTorch Dataset to handle image loading and optional resizing.
    Allows efficient batched loading for GPU inference.
    """
    def __init__(self, image_dir, fixed_size=(512, 512), extensions=(".jpg", ".jpeg", ".png")):
        self.fixed_size = fixed_size
        image_dir = Path(image_dir)
        # Collect all image paths in the directory with specified extensions
        self.image_paths = [str(p) for p in sorted(image_dir.iterdir()) if p.suffix.lower() in extensions]
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = read_image(path).float() / 255.0  # Convert to float [0,1]
        original_size = img.shape[1:]  # Save original HxW for later resizing
        if self.fixed_size is not None:
            img = resize(img, self.fixed_size)  # Resize for model input
        return img, path, original_size

def collate_fixed(batch):
    """
    Custom collate function to combine individual dataset items into a batch.
    Returns:
        imgs_tensor: Stacked image tensor (B,C,H,W)
        paths: List of image paths
        sizes: List of original image sizes
    """
    imgs, paths, sizes = zip(*batch)
    imgs_tensor = torch.stack(imgs, dim=0)
    return imgs_tensor, list(paths), list(sizes)

# ---------------- Main Segmentation Function ----------------
def segment_images(image_dir: str, output_dir: str, batch_size: int = 8,
                   fixed_size=(512, 512), save_results: bool = True, 
                   checkpoint_batches: int = 100, resume: bool = True):
    """
    Core function performing batch semantic segmentation:
    1. Loads Mask2Former model and processor.
    2. Prepares dataset and dataloader.
    3. Runs GPU inference in batches.
    4. Offloads CPU post-processing (resizing + coloring + saving) to multiprocessing workers.
    5. Supports checkpointing and resuming to handle large datasets.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: Running on CPU. Execution time will be very high.")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_dir = Path(output_dir)

    # Load pretrained Mask2Former model and processor for semantic segmentation
    model_name = "facebook/mask2former-swin-large-ade-semantic"
    print("Loading model and processor...")
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_name#, dtype=torch.float16
    ).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)

    # Pre-calculate normalization tensors and move to device
    mean = torch.tensor(processor.image_mean, device=device, dtype=torch.float16).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, device=device, dtype=torch.float16).view(1, 3, 1, 1)

    # ADE20K color palette (150 classes) -> https://github.com/lllyasviel/ControlNet/issues/172
    ade_palette = np.array([
        (120, 120, 120), (180, 120, 120), (6, 230, 230), (80, 50, 50),
        (4, 200, 3), (120, 120, 80), (140, 140, 140), (204, 5, 255),
        (230, 230, 230), (4, 250, 7), (224, 5, 255), (235, 255, 7),
        (150, 5, 61), (120, 120, 70), (8, 255, 51), (255, 6, 82),
        (143, 255, 140), (204, 255, 4), (255, 51, 7), (204, 70, 3),
        (0, 102, 200), (61, 230, 250), (255, 6, 51), (11, 102, 255),
        (255, 7, 71), (255, 9, 224), (9, 7, 230), (220, 220, 220),
        (255, 9, 92), (112, 9, 255), (8, 255, 214), (7, 255, 224),
        (255, 184, 6), (10, 255, 71), (255, 41, 10), (7, 255, 255),
        (224, 255, 8), (102, 8, 255), (255, 61, 6), (255, 194, 7),
        (255, 122, 8), (0, 255, 20), (255, 8, 41), (255, 5, 153),
        (6, 51, 255), (235, 12, 255), (160, 150, 20), (0, 163, 255),
        (140, 140, 140), (250, 10, 15), (20, 255, 0), (31, 255, 0),
        (255, 31, 0), (255, 224, 0), (153, 255, 0), (0, 0, 255),
        (255, 71, 0), (0, 235, 255), (0, 173, 255), (31, 0, 255),
        (11, 200, 200), (255, 82, 0), (0, 255, 245), (0, 61, 255),
        (0, 255, 112), (0, 255, 133), (255, 0, 0), (255, 163, 0),
        (255, 102, 0), (194, 255, 0), (0, 143, 255), (51, 255, 0),
        (0, 82, 255), (0, 255, 41), (0, 255, 173), (10, 0, 255),
        (173, 255, 0), (0, 255, 153), (255, 92, 0), (255, 0, 255),
        (255, 0, 245), (255, 0, 102), (255, 173, 0), (255, 0, 20),
        (255, 184, 184), (0, 31, 255), (0, 255, 61), (0, 71, 255),
        (255, 0, 204), (0, 255, 194), (0, 255, 82), (0, 10, 255),
        (0, 112, 255), (51, 0, 255), (0, 194, 255), (0, 122, 255),
        (0, 255, 163), (255, 153, 0), (0, 255, 10), (255, 112, 0),
        (143, 255, 0), (82, 0, 255), (163, 255, 0), (255, 235, 0),
        (8, 184, 170), (133, 0, 255), (0, 255, 92), (184, 0, 255),
        (255, 0, 31), (0, 184, 255), (0, 214, 255), (255, 0, 112),
        (92, 255, 0), (0, 224, 255), (112, 224, 255), (70, 184, 160),
        (163, 0, 255), (153, 0, 255), (71, 255, 0), (255, 0, 163),
        (255, 204, 0), (255, 0, 143), (0, 255, 235), (133, 255, 0),
        (255, 0, 235), (245, 0, 255), (255, 0, 122), (255, 245, 0),
        (10, 190, 212), (214, 255, 0), (0, 204, 255), (20, 0, 255),
        (255, 255, 0), (0, 153, 255), (0, 41, 255), (0, 255, 204),
        (41, 0, 255), (41, 255, 0), (173, 0, 255), (0, 245, 255),
        (71, 0, 255), (122, 0, 255), (0, 255, 184), (0, 92, 255),
        (184, 255, 0), (0, 133, 255), (255, 214, 0), (25, 194, 194),
        (102, 255, 0), (92, 0, 255)
    ], dtype=np.uint8)

    # Initialize dataset
    dataset = ImageDataset(image_dir, fixed_size=fixed_size)
    
    # --- Filter already processed images if resuming ---
    if resume:
        processed = {p.stem.replace("_seg","") for p in output_dir.glob("*_seg.png")}
        dataset.image_paths = [p for p in dataset.image_paths if Path(p).stem not in processed]
        print(f"Resuming: {len(dataset.image_paths)} images left to process")

    if not dataset.image_paths:
        print("All images already processed. Exiting.")
        return

    # DataLoader handles batching, parallel loading, and collating
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=max(1, os.cpu_count() // 2), 
                        pin_memory=True, collate_fn=collate_fixed)

    total_batches = len(loader)
    print(f"Processing {len(dataset)} images on {device.upper()} in {total_batches} batches...")

    all_futures: List[Future] = []

    # Multiprocessing for CPU-bound post-processing
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor, \
         tqdm(total=total_batches, desc="Inference (GPU) & Post-Proc (CPU)") as pbar:

        for batch_idx, (imgs_tensor, paths, original_sizes) in enumerate(loader):
            # GPU inference preparation
            imgs_tensor = imgs_tensor.to(device, non_blocking=True, dtype=torch.float16).clamp(0.0, 1.0)
            pixel_values = (imgs_tensor - mean) / std

            # Run inference with mixed precision
            with torch.no_grad(), torch.amp.autocast(device_type=device):
                outputs = model(pixel_values=pixel_values)

            # Post-process outputs (class indices)
            target_sizes_for_postproc = [fixed_size for _ in imgs_tensor]
            seg_maps = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes_for_postproc)

            # Submit CPU-heavy post-processing tasks
            if save_results:
                for seg_map, orig_size, p in zip(seg_maps, original_sizes, paths):
                    save_path = output_dir / (Path(p).stem + "_seg.png")
                    if save_path.exists() and resume:
                        continue
                    # Move tensor to CPU before sending to worker
                    future = executor.submit(process_seg_map_resize_and_save,
                                             seg_map.cpu(), orig_size, str(save_path), ade_palette)
                    all_futures.append(future)

            pbar.update(1)

            # Periodic checkpoint: wait for all pending saves to finish
            if save_results and (batch_idx + 1) % checkpoint_batches == 0:
                print(f"\nCheckpoint: Waiting for {len(all_futures)} save tasks to complete...")
                for future in tqdm(all_futures, desc=f"Saving checkpoint (Batch {batch_idx + 1})"):
                    future.result()
                all_futures = []
                print("Checkpoint complete. Resuming GPU inference...")

        # Wait for any remaining saves
        if save_results and all_futures:
            print(f"\nFinal wait: Waiting for {len(all_futures)} remaining save tasks to complete...")
            for future in tqdm(all_futures, desc="Saving final results"):
                future.result()

    print("Segmentation completed successfully!")

# ---------------- Command-Line Interface ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Semantic Segmentation using Mask2Former")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save segmented images")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for GPU inference")
    parser.add_argument("--fixed_size", type=int, nargs=2, default=[512, 512], help="Fixed size (H W) for model input")
    parser.add_argument("--save_results", type=bool, default=True, help="Whether to save segmentation results")
    parser.add_argument("--checkpoint_batches", type=int, default=100, help="Checkpoint frequency in batches")

    args = parser.parse_args()

    segment_images(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        fixed_size=tuple(args.fixed_size),
        save_results=args.save_results,
        checkpoint_batches=args.checkpoint_batches
    )

# ---------------- Example Usage ----------------
# "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\.venv-Copy-Copy\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\SemanticSegmentation\semantic_seg.py"  --image_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ImageCaptioningEvaluationDatasets\LAION-5B-10k\LAION-5B-10k-images"  --output_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\SemanticSegmentation\LAION-5B-10k-segmented" --batch_size 8 --fixed_size 512 512 --save_results True --checkpoint_batches 100

# "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\.venv-Copy-Copy\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\SemanticSegmentation\semantic_seg.py" --image_dir "E:\ImageRetrieval\Professions_125k_Cleaned\Female_Accountant"  --output_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\SemanticSegmentation\test" --batch_size 8 --fixed_size 512 512 --save_results True --checkpoint_batches 100
