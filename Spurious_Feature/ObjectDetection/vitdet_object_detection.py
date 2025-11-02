import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, Future
from typing import List, Tuple
import torch
import cv2
from PIL import Image
import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.transforms import ResizeShortestEdge
from detectron2.utils.visualizer import Visualizer, ColorMode

# ---------------- Worker Function ----------------
def save_detectron2_image(image: np.ndarray, instances_info: List[Tuple[np.ndarray, str, Tuple[int,int,int]]], save_path: str) -> str:
    v = Visualizer(image, instance_mode=ColorMode.IMAGE)
    for bbox, cls_name, color in instances_info:
        norm_color = [c / 255 for c in color]
        v.draw_box(bbox, alpha=1, edge_color=norm_color)
        x_center = (bbox[0] + bbox[2]) / 2
        y = min(bbox[1], bbox[3])
        v.draw_text(cls_name, (x_center, y), color=norm_color)
    Image.fromarray(v.get_output().get_image()).save(save_path)
    return save_path

# ---------------- Dataset ----------------
class ImageDataset(Dataset):
    def __init__(self, image_dir, min_resize=500, extensions=(".jpg", ".jpeg", ".png")):
        self.min_resize = min_resize
        image_dir = Path(image_dir)
        self.image_paths = [str(p) for p in sorted(image_dir.iterdir()) if p.suffix.lower() in extensions]
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        orig_w, orig_h = img.size
        if min(orig_w, orig_h) > self.min_resize:
            scale = self.min_resize / min(orig_w, orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            img = img.resize((new_w, new_h), Image.BICUBIC)
        img_np = np.array(img, dtype=np.uint8)
        return img_np, path, (orig_h, orig_w)

def collate_images(batch):
    imgs, paths, orig_sizes = zip(*batch)
    return list(imgs), list(paths), list(orig_sizes)

# ---------------- Main Object Detection Function ----------------
def detect_objects(
    image_dir: str,
    output_dir: str,
    model_path: str,
    checkpoint_path: str,
    color_json: str,
    batch_size: int = 4,
    num_workers: int = 4,
    conf_thresh: float = 0.5,
    resume: bool = True
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: Running on CPU. This will be slow.")

    os.makedirs(output_dir, exist_ok=True)
    output_dir = Path(output_dir)

    # Load model
    cfg = LazyConfig.load(model_path)
    cfg.train.init_checkpoint = checkpoint_path
    np.float_ = np.float = np.float64
    np.complex_ = np.complex = np.complex128
    model = instantiate(cfg.model)
    DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    model.eval().to(device)

    # Load class-color mapping
    with open(color_json, "r") as f:
        color_dict = json.load(f)
    class_names = list(color_dict.keys())
    class_colors = list(color_dict.values())

    # Prepare dataset
    dataset = ImageDataset(image_dir)
    if resume:
        processed = {p.stem for p in output_dir.glob("*.png")}
        dataset.image_paths = [p for p in dataset.image_paths if Path(p).stem not in processed]
        print(f"Resuming: {len(dataset.image_paths)} images left to process")
    if not dataset.image_paths:
        print("All images already processed. Exiting.")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, collate_fn=collate_images)

    transform = ResizeShortestEdge(short_edge_length=1024, max_size=1024)
    all_futures: List[Future] = []

    # Executor for async saving
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor, \
         tqdm(total=len(loader), desc="Object Detection (GPU) & Save (CPU)") as pbar:

        for imgs, paths, orig_sizes in loader:
            # Apply Detectron2 transform on batch
            imgs_resized = []
            inputs = []
            for img_np, orig_size in zip(imgs, orig_sizes):
                tf = transform.get_transform(img_np)
                resized = tf.apply_image(img_np)
                imgs_resized.append(resized)
                img_tensor = torch.tensor(resized).permute(2, 0, 1).float().to(device, non_blocking=True)
                inputs.append({"image": img_tensor, "height": img_np.shape[0], "width": img_np.shape[1]})

            # Batch inference with FP16
            with torch.amp.autocast(device_type=device), torch.no_grad():
                outputs = model(inputs)

            # Submit saving tasks asynchronously
            for resized, output, path, orig_size in zip(imgs_resized, outputs, paths, orig_sizes):
                pred_instances = output["instances"]
                keep = pred_instances.scores > conf_thresh
                pred_instances = pred_instances[keep]

                instances_info = [
                    (bbox.cpu().numpy(), class_names[i], tuple(class_colors[i]))
                    for bbox, i in zip(pred_instances.pred_boxes.tensor, pred_instances.pred_classes.cpu().numpy())
                ]

                img_final = cv2.resize(resized, (orig_size[1], orig_size[0]))
                save_path = output_dir / (Path(path).stem + ".png")
                if save_path.exists() and resume:
                    continue

                future = executor.submit(save_detectron2_image, img_final, instances_info, str(save_path))
                all_futures.append(future)

            pbar.update(1)

        # Wait for remaining saves
        if all_futures:
            for future in tqdm(all_futures, desc="Saving final results"):
                future.result()

    print("Object detection completed successfully!")

# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch Object Detection using Detectron2")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--color_json", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    args = parser.parse_args()

    detect_objects(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        checkpoint_path=args.checkpoint_path,
        color_json=args.color_json,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        conf_thresh=args.conf_thresh
    )

# ---------------- Example Usage ----------------
# VitDet huge model command:
# "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\.venv-Copy-Copy\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\vitdet_object_detection.py" --image_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ImageCaptioningEvaluationDatasets\LAION-5B-10k\LAION-5B-10k-images" --output_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\LAION-5B-10k-vitdet-huge-detected" --model_path "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\detectron2\projects\ViTDet\configs\LVIS\cascade_mask_rcnn_vitdet_h_100ep.py" --checkpoint_path "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\VitDet_huge_LVIS.pkl" --color_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\LVIS_color_map.json" --batch_size 4 --num_workers 8

# VitDet base model command:
# "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\.venv-Copy-Copy\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\vitdet_object_detection.py" --image_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ImageCaptioningEvaluationDatasets\LAION-5B-10k\LAION-5B-10k-images" --output_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\LAION-5B-10k-vitdet-base-detected" --model_path "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\detectron2\projects\ViTDet\configs\LVIS\cascade_mask_rcnn_vitdet_b_100ep.py" --checkpoint_path "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\VitDet_base_LVIS.pkl" --color_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\LVIS_color_map.json" --batch_size 4 --num_workers 8

# Around 7hrs for 10k images using VitDet base model on RTX 3060ti, VitDet huge model takes much longer due to memory constraints.