import os
import time
import json
from math import ceil
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
import numpy as np

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ---------------- Dataset ----------------
class PILImageDataset(Dataset):
    """Loads images as PIL.Image objects, no conversion to tensors."""
    def __init__(self, image_dir, extensions=(".jpg", ".jpeg", ".png"), preprocess_size=500):
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(extensions)
        ]
        self.preprocess_size = preprocess_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            # Resize if min dimension > preprocess_size
            if min(img.size) > self.preprocess_size:
                img = img.resize((int(img.width * self.preprocess_size / min(img.size)),
                                  int(img.height * self.preprocess_size / min(img.size))),
                                 Image.BICUBIC)
        except (UnidentifiedImageError, OSError, ValueError) as e:
            img = Image.new("RGB", (224, 224), color=(0, 0, 0))
            print(f"⚠️ Replaced corrupted image with black placeholder: {path} ({e})")
        return img, path

# ---------------- Collate Function ----------------
def collate_images(batch):
    imgs, paths = zip(*batch)
    return list(imgs), list(paths)

# ---------------- DataLoader ----------------
def get_pil_image_loader(image_dir, batch_size=32, num_workers=4):
    dataset = PILImageDataset(image_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_images,
        pin_memory=True
    )
    return loader, len(dataset)

# ---------------- SAM Edge Detection Logic ----------------
class SAMEdgeDetector:
    def __init__(self, model_type, model_path, device="cuda", points_per_side=32):
        self.sam = sam_model_registry[model_type](checkpoint=model_path).to(device)
        self.sam.eval()
        self.mask_generator = SamAutomaticMaskGenerator(
            self.sam,
            points_per_side=points_per_side
        )
        self.device = device

    @staticmethod
    def merge_anns(anns, h, w):
        """Merge overlapping SAM masks into single object map."""
        sorted_anns = sorted(anns, key=(lambda x: x['predicted_iou']), reverse=True)
        img = np.zeros((h, w))
        for idx, ann in enumerate(sorted_anns):
            img[ann['segmentation']] = idx + 1
        return img

    @staticmethod
    def contour(input_map):
        """Generate boundary map from per-pixel object IDs."""
        H, W = input_map.shape
        binary_map = np.zeros((H, W), dtype=int)
        original = input_map[1:-1, 1:-1]
        up = input_map[:-2, 1:-1]
        down = input_map[2:, 1:-1]
        left = input_map[1:-1, :-2]
        right = input_map[1:-1, 2:]
        binary_map[1:-1, 1:-1] = ((original != up) |
                                  (original != down) |
                                  (original != left) |
                                  (original != right)).astype(int)
        return binary_map

    def detect_edge(self, image):
        """Generate SAM-based edge map for a single image."""
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        masks = self.mask_generator.generate(img_np)
        total_mask = self.merge_anns(masks, h, w)
        edge_map = np.uint8(self.contour(total_mask) * 255)
        return edge_map

# ---------------- Parallel Processing Function ----------------
def process_image(detector, img_path_tuple, output_dir, imgid):
    img, path = img_path_tuple
    try:
        edges = detector.detect_edge(img)
        edge_filename = os.path.splitext(os.path.basename(path))[0] + "_sam_edges.png"
        edge_path = os.path.join(output_dir, edge_filename)
        Image.fromarray(edges).save(edge_path)
        return {
            "filename": os.path.basename(path),
            "edge_map": edge_filename,
            "imgid": imgid
        }
    except Exception as e:
        print(f"⚠️ Error processing {path}: {e}")
        return None

# ---------------- Main Script ----------------
def main(args):
    total_start_time = time.time()

    image_dir = args.image_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nLoading Detector")
    detector = SAMEdgeDetector(args.model_type, args.model_path, device=args.device)
    print(f"\nLoaded Detector")

    print(f"\nSetting Up Image Loader")
    loader, dataset_len = get_pil_image_loader(image_dir, batch_size=batch_size, num_workers=num_workers)
    print(f"\nImage Loader Setup Complete")
    startup_time = time.time() - total_start_time

    edge_data = {"dataset": os.path.basename(image_dir), "images": []}
    imgid_counter = 0

    detection_start_time = time.time()

    for imgs, paths in tqdm(loader, desc="Processing batches", total=ceil(dataset_len / batch_size)):
        # Sequential processing per batch
        for i, (img, path) in enumerate(zip(imgs, paths)):
            try:
                edges = detector.detect_edge(img)
                edge_filename = os.path.splitext(os.path.basename(path))[0] + "_sam_edges.png"
                edge_path = os.path.join(output_dir, edge_filename)
                Image.fromarray(edges).save(edge_path)

                edge_data["images"].append({
                    "filename": os.path.basename(path),
                    "edge_map": edge_filename,
                    "imgid": imgid_counter
                })
                imgid_counter += 1

            except Exception as e:
                print(f"⚠️ Error processing {path}: {e}")

        # Write JSON after each batch
        json_output_path = os.path.join(output_dir, "sam_edge_detection_results.json")
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(edge_data, f, indent=2)

    detection_time = time.time() - detection_start_time
    print(f"\nSAM-based edge maps saved to {output_dir}")
    print(f"\nTiming Summary:")
    print(f" - Startup time (setup): {startup_time:.2f}s")
    print(f" - Detection time (processing only): {detection_time:.2f}s")

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Parallel SAM Edge Detection")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True, help="Path to SAM checkpoint (e.g., sam_vit_l_0b3195.pth)")
    parser.add_argument("--model_type", type=str, required=True, help="Type of SAM Model (e.g., vit_l)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda", help="Device to run SAM model on (cuda or cpu)")
    args = parser.parse_args()

    main(args)

# "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\.venv-Copy-Copy\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\EdgeDetection\sam_edge_detection.py" --image_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ImageCaptioningEvaluationDatasets\LAION-5B-10k\LAION-5B-10k-images" --batch_size 8 --num_workers 8 --output_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\EdgeDetection\LAION-5B-10k-sam-edge-detected" --model_path "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\EdgeDetection\sam_vit_l_0b3195.pth" --model_type "vit_l"

# NOTE: Does not have resume functionality
# NOTE: This model is quite costly time wise and additionally in the paper it was noted that The classification accuracy on SAM contours (73.2%) is slightly higher than that on Canny edge (71.0%), thus its better to use Canny Edge as opposed to this.


# "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\.venv-Copy-Copy\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\EdgeDetection\sam_edge_detection.py" --image_dir "E:\ImageRetrieval\Professions_125k_Cleaned\Female_Accountant" --batch_size 8 --num_workers 8 --output_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\EdgeDetection\sam_test" --model_path "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\EdgeDetection\sam_vit_l_0b3195.pth" --model_type "vit_l"