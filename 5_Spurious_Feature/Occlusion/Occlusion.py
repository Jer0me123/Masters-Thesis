import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path
from typing import Tuple, Optional, Set, Dict
from abc import ABC, abstractmethod
import argparse
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms.functional import resize
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import contextlib
import sys
import os
import json
from threading import Lock
import time
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

write_lock = Lock()

from occlusionModelsHelper import (
    get_model, ImageDataset,
    MIN_MASK_AREA_RATIO, MAX_MASK_AREA_RATIO,
    MIN_BBOX_AREA_RATIO, MAX_BBOX_AREA_RATIO
)

# ================================
# FAST RGB DECODER
# ================================
class FastRGBDecoder:
    """
    High-performance image decoder that returns RGB NumPy arrays.
    Uses libjpeg-turbo when available, falls back to OpenCV otherwise.
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


# ================================
# MANIFEST (BUFFERED)
# ================================
class TransformManifest:
    """
    Persistent, append-only manifest tracking completed transforms.
    An image is considered complete only if ALL required operations are present.
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
                    self.data.setdefault(r["image"], set()).add(r["operation"])

    def record(self, image: str, operation: str):
        with self.lock:
            self.data.setdefault(image, set()).add(operation)
            self.buffer.append({"image": image, "operation": operation})
            if len(self.buffer) >= self.flush_every:
                self.flush()

    def flush(self):
        if not self.buffer:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            for r in self.buffer:
                f.write(json.dumps(r) + "\n")
        self.buffer.clear()


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# ================================
# CPU POST-PROCESS (UPDATED WITH RESIZE)
# ================================
def process_occlusions_and_save(person_mask, original_size, image_path, output_root, 
                                operations, resize_output=None, decoder=None):
    """
    Takes a person mask and generates requested occlusion variants
    person_mask: binary mask at fixed_size resolution
    operations: set of operations to perform
    resize_output: optional tuple (W, H) to resize final output images
    """
    H, W = original_size
    
    # Resize mask to original size
    person_mask_resized = cv2.resize(person_mask, (W, H), cv2.INTER_NEAREST)
    
    # Load original image using TurboJPEG if available
    if decoder is not None:
        image = decoder.load(image_path)
        if image is None:
            return None
    else:
        image = cv2.imread(image_path)
        if image is None:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Quality checks
    img_area = H * W
    mask_area = int(person_mask_resized.sum())
    mask_ratio = mask_area / img_area
    
    # if mask_ratio < MIN_MASK_AREA_RATIO or mask_ratio > MAX_MASK_AREA_RATIO:
    #     return None
    
    ys, xs = np.where(person_mask_resized == 1)
    if len(xs) == 0:
        return None
    
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    bbox_area = (x2 - x1) * (y2 - y1)
    bbox_ratio = bbox_area / img_area
    
    # if bbox_ratio < MIN_BBOX_AREA_RATIO or bbox_ratio > MAX_BBOX_AREA_RATIO:
    #     return None
    
    # Get relative path structure
    # rel_path = os.path.relpath(image_path, Path(image_path).parent.parent)
    # stem = Path(rel_path).stem
    # subdir = Path(rel_path).parent

    rel_path = os.path.relpath(image_path, args.image_dir).replace("\\", "/")
    stem = Path(rel_path).stem
    subdir = Path(rel_path).parent
    image_id = os.path.splitext(rel_path)[0]

    
    # Generate requested occlusion variants
    outputs = {}
    
    if "Full" in operations:
        outputs["Full"] = image.copy()
    
    if "Full_NoBg" in operations:
        outputs["Full_NoBg"] = image * person_mask_resized[..., None]
    
    if "MaskSegm" in operations:
        masksegm = image.copy()
        masksegm[person_mask_resized == 1] = 255
        outputs["MaskSegm"] = masksegm
    
    if "MaskSegm_NoBg" in operations:
        masksegm_nobg = np.zeros_like(image)
        masksegm_nobg[person_mask_resized == 1] = 255
        outputs["MaskSegm_NoBg"] = masksegm_nobg
    
    if "MaskRect" in operations:
        maskrect = image.copy()
        maskrect[y1:y2, x1:x2] = 255
        outputs["MaskRect"] = maskrect
    
    if "MaskRect_NoBg" in operations:
        maskrect_nobg = np.zeros_like(image)
        maskrect_nobg[y1:y2, x1:x2] = 255
        outputs["MaskRect_NoBg"] = maskrect_nobg
    
    # Save all variants
    for k, img in outputs.items():
        # Apply resize if specified
        if resize_output is not None:
            img = cv2.resize(img, resize_output, interpolation=cv2.INTER_LINEAR)
        
        # --- Ensure RGB uint8 ---
        if img.ndim == 2:
            # Grayscale -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 3:
            # Assume OpenCV-style BGR -> RGB
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pass  # already RGB
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")

        # Maintain directory structure
        out_dir = Path(output_root) / k / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}.png"
        Image.fromarray(img).save(out_path, compress_level=1)
    
    # return {
    #     "image": str(Path(rel_path).with_suffix("")),
    #     "mask_ratio": round(mask_ratio, 4),
    #     "bbox_ratio": round(bbox_ratio, 4)
    # }

    completed_ops = list(outputs.keys())

    return {
        "image": image_id,
        "completed_ops": completed_ops,
        "mask_ratio": round(mask_ratio, 4),
        "bbox_ratio": round(bbox_ratio, 4),
    }




# ================================
# DATASET
# ================================
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
        tensors = []
        orig_sizes = []

        for p in paths:
            arr = self.decoder.load(p)

            if arr is None or arr.shape[0] <= 1 or arr.shape[1] <= 1:
                arr = np.zeros((224, 224, 3), dtype=np.uint8)
                orig_h, orig_w = 224, 224
            else:
                orig_h, orig_w = arr.shape[:2]

            if self.resize_width and self.resize_height:
                arr = cv2.resize(
                    arr,
                    (self.resize_width, self.resize_height),
                    interpolation=cv2.INTER_LINEAR
                )

            # HWC → CHW
            arr = arr.transpose(2, 0, 1)

            # uint8 → float32, normalized
            tensor = torch.from_numpy(arr).float().div_(255.0)

            tensors.append(tensor)
            orig_sizes.append((orig_h, orig_w))

        batch = torch.stack(tensors, dim=0)

        return batch, paths, orig_sizes

def collate_fixed(batch):
    imgs, paths, sizes = zip(*batch)
    return torch.stack(imgs, dim=0), list(paths), list(sizes)


# ================================
# MAIN (UPDATED)
# ================================
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize TurboJPEG decoder
    decoder = FastRGBDecoder()
    
    # Load specified model
    model = get_model(args.model, device)
    
    # Use model's preferred size if not specified
    fixed_size = model.get_fixed_size() if args.fixed_size is None else tuple(args.fixed_size)
    
    # Resume logic using manifest
    manifest_path = Path(args.output_dir) / "occlusion_manifest.jsonl"
    manifest = TransformManifest(str(manifest_path))
    
    # An image is complete if ALL requested operations are present
    completed = set()
    operations = set(args.operations)
    for img, ops in manifest.data.items():
        if operations.issubset(ops):
            completed.add(img)
    
    print(f"Skipping {len(completed)} completed images.")    

    # Startup message
    all_images_count = sum(1 for root, dirs, files in os.walk(args.image_dir) 
                          for f in files if f.lower().endswith((".jpg", ".jpeg", ".png")))
    
    print(f"- Remaining images to process: {all_images_count - len(completed)}")
    print(f"- Using model: {args.model}")
    print(f"- Fixed size: {fixed_size}")
    print(f"- Operations: {', '.join(sorted(operations))}")
    if args.resize:
        print(f"- Output resize: {args.resize}")
    if args.exclude_dirs:
        print(f"- Excluding directories: {', '.join(args.exclude_dirs)}")
    
    # Dataset
    dataset = ImageDataset(args.image_dir, completed, args.exclude_dirs, resize_width=fixed_size[0], resize_height=fixed_size[1])

    # CSV for QC stats
    csv_path = Path(args.output_dir) / "occlusion_qc.csv"
    
    with open(csv_path, "a+", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "mask_ratio", "bbox_ratio"])

        for i in tqdm(range(0, len(dataset), args.batch_size)):
            imgs_tensor, paths, sizes = dataset.get_batch(i, args.batch_size)

            imgs_tensor = imgs_tensor.to(device, dtype=torch.float16)

            person_masks = model.predict(imgs_tensor)

            num_workers = os.cpu_count()/2  

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        process_occlusions_and_save,
                        person_mask,
                        size,
                        p,
                        args.output_dir,
                        operations,
                        args.resize,
                        decoder
                    )
                    for person_mask, size, p in zip(person_masks, sizes, paths)
                ]

                for future in as_completed(futures):
                    res = future.result()
                    if res:
                        with write_lock:
                            # writer.writerow(res)
                            # for op in operations:
                            #     manifest.record(res["image"], op)

                            csv_row = {
                                "image": res["image"],
                                "mask_ratio": res["mask_ratio"],
                                "bbox_ratio": res["bbox_ratio"],
                            }
                            writer.writerow(csv_row)

                            for op in res["completed_ops"]:
                                manifest.record(res["image"], op)



    manifest.flush()
    print("Completed successfully.")


# ================================
# CLI
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate occlusions using various segmentation models")
    parser.add_argument("--image_dir", required=True, help="Input image directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for occlusions")
    parser.add_argument("--model", choices=["mask2former_ade20k", "mask2former_coco", "maskrcnn", "yolact", "sam", "lang_sam"],
                        default="mask2former_ade20k", help="Segmentation model to use")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--fixed_size", type=int, nargs=2, default=None, help="Fixed input size (H W)")
    parser.add_argument("--resize", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=None, 
                        help="Optional fixed resize for output images as: --resize WIDTH HEIGHT (e.g. --resize 224 224)")
    parser.add_argument("--operations", nargs="+", 
                        choices=["Full", "Full_NoBg", "MaskSegm", "MaskSegm_NoBg", "MaskRect", "MaskRect_NoBg"],
                        default=["Full", "Full_NoBg", "MaskSegm", "MaskSegm_NoBg", "MaskRect", "MaskRect_NoBg"],
                        help="Operations to perform (default: all)")
    parser.add_argument("--exclude_dirs", nargs="+", default=["facemesh"], 
                        help="Subdirectories to exclude from processing")
    
    args = parser.parse_args()
    
    main(args)

# Example usage:
# python Occlusion.py --image_dir "G:\Thesis\ImageRetrieval\Professions_125k_Cleaned" --output_dir test --model mask2former_coco --batch_size 8 --resize 224 224 --operations Full MaskSegm --exclude_dirs facemesh

# python Occlusion.py --image_dir "G:\Thesis\ImageRetrieval\Professions_125k_Cleaned" --output_dir test_ade20k --model mask2former_ade20k --batch_size 8 --resize 224 224 --exclude_dirs facemesh

# python Occlusion.py --image_dir "G:\Thesis\ImageRetrieval\Professions_125k_Cleaned" --output_dir test_mrcnn --model maskrcnn --batch_size 8 --resize 224 224 --exclude_dirs facemesh

# python Occlusion.py --image_dir "G:\Thesis\ImageRetrieval\Professions_125k_Cleaned" --output_dir test_y --model yolact --batch_size 8 --resize 224 224 --exclude_dirs facemesh

# python Occlusion.py --image_dir "G:\Thesis\ImageRetrieval\Professions_125k_Cleaned" --output_dir test_s --model sam --batch_size 8 --resize 224 224 --exclude_dirs facemesh

# python Occlusion.py --image_dir "G:\Thesis\ImageRetrieval\Professions_125k_Cleaned" --output_dir test_ls --model lang_sam --batch_size 8 --resize 224 224 --exclude_dirs facemesh

# ===========================================================
# EXAMPLE USAGE
# python Occlusion.py --model mask2former_coco --image_dir  "path/to/input" --output_dir  "path/to/output"  --batch_size 8 --fixed_size 224 224 --resize 224 224 --operations Full_NoBg MaskSegm MaskSegm_NoBg MaskRect MaskRect_NoBg --exclude_dirs facemesh

# --model mask2former_coco -> This is done as X was the best performing model overall in terms or speed, accuracy and output quality.
# --fixed_size 224 224 -> This can be ommited or set to 224 224, if ommitted the model preferred size will be used however it will slow down exectuion. With model mask2former_coco using 224 224 didn't appear to affect output quality.
# --resize 224 224 -> This is done as the classification model auto resizes images to 224 x 244 hence its better to resize them prior as this makes processing faster and storge requirements less.
# --operations Full_NoBg MaskSegm MaskSegm_NoBg MaskRect MaskRect_NoBg -> Full is ommited from the list as this simply copies the image over creating and unneeded duplicate.
# --exclude_dirs facemesh -> This is done to exclude any images in the facemesh directory from processing as these are not actual images but rather facemesh data.

#NOTE: Mask2Former-COCO is used as the primary segmentation model because,
# despite being slightly slower per image than Mask R-CNN, it yields
# higher detection rates, more consistent IoU, and fewer low-quality
# masks that fail QC. This leads to more usable outputs and comparable
# end-to-end throughput when processed in batches.


# "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\5_Spurious_Feature\Occlusion\.venv\Scripts\python.exe" Occlusion.py --model mask2former_coco --image_dir  "E:\ImageRetrieval\StableDiffusionGeneratedImages\valid" --output_dir  "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\Occlusion"  --batch_size 4 --fixed_size 224 224 --resize 224 224 --operations Full_NoBg MaskSegm MaskSegm_NoBg MaskRect MaskRect_NoBg --exclude_dirs face_crops

# "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\5_Spurious_Feature\Occlusion\.venv\Scripts\python.exe" Occlusion.py --model mask2former_coco --image_dir  "F:\ImageRetrieval\Professions_125k_ISCO_Aligned_1k_Subset" --output_dir  "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\Occlusion"  --batch_size 4 --fixed_size 224 224 --resize 224 224 --operations Full_NoBg MaskSegm MaskSegm_NoBg MaskRect MaskRect_NoBg --exclude_dirs facemesh