# import os
# from pathlib import Path
# from concurrent.futures import ProcessPoolExecutor, Future
# from typing import List, Tuple
# import torch
# import cv2
# from PIL import Image
# import numpy as np
# import json
# from tqdm import tqdm
# from torch.utils.data import Dataset, DataLoader
# from torchvision.io import read_image
# from torchvision.transforms.functional import resize
# from detectron2.config import LazyConfig, instantiate
# from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.data.transforms import ResizeShortestEdge
# from detectron2.utils.visualizer import Visualizer, ColorMode

# # ---------------- Worker Function ----------------
# def save_detectron2_image(image: np.ndarray, instances_info: List[Tuple[np.ndarray, str, Tuple[int,int,int]]], save_path: str) -> str:
#     """
#     Draw bounding boxes and labels on image and save to disk.
#     """
#     v = Visualizer(image, instance_mode=ColorMode.IMAGE)
#     for bbox, cls_name, color in instances_info:
#         norm_color = [c / 255 for c in color]
#         v.draw_box(bbox, alpha=1, edge_color=norm_color)
#         x_center = (bbox[0] + bbox[2]) / 2
#         y = min(bbox[1], bbox[3])
#         v.draw_text(cls_name, (x_center, y), color=norm_color)
#     Image.fromarray(v.get_output().get_image()).save(save_path)
#     return save_path

# # ---------------- Dataset & Collate ----------------
# class ImageDataset(Dataset):
#     """
#     Dataset to load images for object detection with optional fixed-size resizing.
#     """
#     def __init__(self, image_dir, fixed_size=(1024, 1024), extensions=(".jpg", ".jpeg", ".png")):
#         self.fixed_size = fixed_size
#         image_dir = Path(image_dir)
#         self.image_paths = [str(p) for p in sorted(image_dir.iterdir()) if p.suffix.lower() in extensions]
#         if not self.image_paths:
#             raise ValueError(f"No images found in {image_dir}")

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         path = self.image_paths[idx]
#         img = read_image(path).float() / 255.0  # float tensor [0,1]
#         original_size = img.shape[1:]  # HxW
#         if self.fixed_size:
#             img = resize(img, self.fixed_size)
#         return img, path, original_size

# def collate_images(batch):
#     imgs, paths, orig_sizes = zip(*batch)
#     imgs_tensor = torch.stack(imgs, dim=0)
#     return imgs_tensor, list(paths), list(orig_sizes)

# # ---------------- Main Object Detection Function ----------------
# def detect_objects(
#     image_dir: str,
#     output_dir: str,
#     model_path: str,
#     checkpoint_path: str,
#     color_json: str,
#     batch_size: int = 4,
#     num_workers: int = 2,
#     conf_thresh: float = 0.5,
#     fixed_size: Tuple[int,int]=(1024,1024),
#     resume: bool = True
# ):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     if device == "cpu":
#         print("WARNING: Running on CPU. This will be slow.")

#     os.makedirs(output_dir, exist_ok=True)
#     output_dir = Path(output_dir)

#     # --- Load Detectron2 model ---
#     cfg = LazyConfig.load(model_path)
#     cfg.train.init_checkpoint = checkpoint_path
#     np.float_ = np.float = np.float64
#     np.complex_ = np.complex = np.complex128
#     model = instantiate(cfg.model)
#     DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
#     model.eval().to(device)

#     # --- Load class-color mapping ---
#     with open(color_json, "r") as f:
#         color_dict = json.load(f)
#     class_names = list(color_dict.keys())
#     class_colors = list(color_dict.values())

#     # --- Prepare dataset and dataloader ---
#     dataset = ImageDataset(image_dir, fixed_size=fixed_size)
#     if resume:
#         processed = {p.stem for p in output_dir.glob("*_det.png")}
#         dataset.image_paths = [p for p in dataset.image_paths if Path(p).stem not in processed]
#         print(f"Resuming: {len(dataset.image_paths)} images left to process")

#     if not dataset.image_paths:
#         print("All images already processed. Exiting.")
#         return

#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
#                         num_workers=num_workers, pin_memory=True, collate_fn=collate_images)

#     transform = ResizeShortestEdge(short_edge_length=1024, max_size=1024)
#     all_futures: List[Future] = []

#     with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor, \
#          tqdm(total=len(loader), desc="Object Detection (GPU) & Save (CPU)") as pbar:

#         for imgs_tensor, paths, original_sizes in loader:
#             imgs_tensor = imgs_tensor.to(device)

#             for img_tensor, path, orig_size in zip(imgs_tensor, paths, original_sizes):
#                 # Convert tensor -> HWC uint8
#                 img = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

#                 # Apply Detectron2 transform
#                 tf = transform.get_transform(img)
#                 resized = tf.apply_image(img)
#                 img_tensor_resized = torch.tensor(resized).permute(2, 0, 1).float().to(device)

#                 # Model inference
#                 with torch.no_grad():
#                     inputs = [{"image": img_tensor_resized, "height": img.shape[0], "width": img.shape[1]}]
#                     outputs = model(inputs)

#                 pred_instances = outputs[0]["instances"]
#                 keep = pred_instances.scores > conf_thresh
#                 pred_instances = pred_instances[keep]

#                 # Prepare info for saving
#                 instances_info = [
#                     (bbox.cpu().numpy(), class_names[i], tuple(class_colors[i]))
#                     for bbox, i in zip(pred_instances.pred_boxes.tensor, pred_instances.pred_classes.cpu().numpy())
#                 ]

#                 # Resize image back to original size before saving
#                 img_final = cv2.resize(resized, (orig_size[1], orig_size[0]))
#                 save_path = output_dir / (Path(path).stem + "_det.png")
#                 if save_path.exists() and resume:
#                     continue

#                 future = executor.submit(save_detectron2_image, img_final, instances_info, str(save_path))
#                 all_futures.append(future)

#             pbar.update(1)

#         # Wait for remaining saves
#         if all_futures:
#             for future in tqdm(all_futures, desc="Saving final results"):
#                 future.result()

#     print("Object detection completed successfully!")

# # ---------------- CLI ----------------
# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Batch Object Detection using Detectron2")
#     parser.add_argument("--image_dir", type=str, required=True)
#     parser.add_argument("--output_dir", type=str, required=True)
#     parser.add_argument("--model_path", type=str, required=True)
#     parser.add_argument("--checkpoint_path", type=str, required=True)
#     parser.add_argument("--color_json", type=str, required=True)
#     parser.add_argument("--batch_size", type=int, default=4)
#     parser.add_argument("--num_workers", type=int, default=2)
#     parser.add_argument("--conf_thresh", type=float, default=0.5)
#     parser.add_argument("--fixed_size", type=int, nargs=2, default=[256,256])
#     args = parser.parse_args()

#     detect_objects(
#         image_dir=args.image_dir,
#         output_dir=args.output_dir,
#         model_path=args.model_path,
#         checkpoint_path=args.checkpoint_path,
#         color_json=args.color_json,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         conf_thresh=args.conf_thresh,
#         fixed_size=tuple(args.fixed_size)
#     )






# import os
# from pathlib import Path
# from concurrent.futures import ProcessPoolExecutor, Future
# from typing import List, Tuple
# import torch
# import cv2
# from PIL import Image
# import numpy as np
# import json
# from tqdm import tqdm
# from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms.functional import resize as tv_resize
# from detectron2.config import LazyConfig, instantiate
# from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.data.transforms import ResizeShortestEdge
# from detectron2.utils.visualizer import Visualizer, ColorMode

# # ---------------- Worker Function ----------------
# def save_detectron2_image(image: np.ndarray, instances_info: List[Tuple[np.ndarray, str, Tuple[int,int,int]]], save_path: str) -> str:
#     """
#     Draw bounding boxes and labels on image and save to disk.
#     """
#     v = Visualizer(image, instance_mode=ColorMode.IMAGE)
#     for bbox, cls_name, color in instances_info:
#         norm_color = [c / 255 for c in color]
#         v.draw_box(bbox, alpha=1, edge_color=norm_color)
#         x_center = (bbox[0] + bbox[2]) / 2
#         y = min(bbox[1], bbox[3])
#         v.draw_text(cls_name, (x_center, y), color=norm_color)
#     Image.fromarray(v.get_output().get_image()).save(save_path)
#     return save_path

# # ---------------- Dataset & Collate ----------------
# class ImageDataset(Dataset):
#     """
#     Dataset to load images for object detection with aspect-ratio preserving resizing.
#     """
#     def __init__(self, image_dir, min_resize=500, extensions=(".jpg", ".jpeg", ".png")):
#         self.min_resize = min_resize
#         image_dir = Path(image_dir)
#         self.image_paths = [str(p) for p in sorted(image_dir.iterdir()) if p.suffix.lower() in extensions]
#         if not self.image_paths:
#             raise ValueError(f"No images found in {image_dir}")

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         path = self.image_paths[idx]
#         img = Image.open(path).convert("RGB")
#         orig_w, orig_h = img.size

#         # Resize only if min side > min_resize, preserve aspect ratio
#         if min(orig_w, orig_h) > self.min_resize:
#             scale = self.min_resize / min(orig_w, orig_h)
#             new_w, new_h = int(orig_w * scale), int(orig_h * scale)
#             img = img.resize((new_w, new_h), Image.BICUBIC)

#         img_np = np.array(img, dtype=np.uint8)
#         return img_np, path, (orig_h, orig_w)  # return original HxW

# def collate_images(batch):
#     imgs, paths, orig_sizes = zip(*batch)
#     return list(imgs), list(paths), list(orig_sizes)

# # ---------------- Main Object Detection Function ----------------
# def detect_objects(
#     image_dir: str,
#     output_dir: str,
#     model_path: str,
#     checkpoint_path: str,
#     color_json: str,
#     batch_size: int = 4,
#     num_workers: int = 2,
#     conf_thresh: float = 0.5,
#     resume: bool = True
# ):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     if device == "cpu":
#         print("WARNING: Running on CPU. This will be slow.")

#     os.makedirs(output_dir, exist_ok=True)
#     output_dir = Path(output_dir)

#     # --- Load Detectron2 model ---
#     cfg = LazyConfig.load(model_path)
#     cfg.train.init_checkpoint = checkpoint_path
#     np.float_ = np.float = np.float64
#     np.complex_ = np.complex = np.complex128
#     model = instantiate(cfg.model)
#     DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
#     model.eval().to(device)

#     # --- Load class-color mapping ---
#     with open(color_json, "r") as f:
#         color_dict = json.load(f)
#     class_names = list(color_dict.keys())
#     class_colors = list(color_dict.values())

#     # --- Prepare dataset and dataloader ---
#     dataset = ImageDataset(image_dir)
#     if resume:
#         processed = {p.stem for p in output_dir.glob("*_det.png")}
#         dataset.image_paths = [p for p in dataset.image_paths if Path(p).stem not in processed]
#         print(f"Resuming: {len(dataset.image_paths)} images left to process")

#     if not dataset.image_paths:
#         print("All images already processed. Exiting.")
#         return

#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
#                         num_workers=num_workers, pin_memory=True, collate_fn=collate_images)

#     transform = ResizeShortestEdge(short_edge_length=1024, max_size=1024)
#     all_futures: List[Future] = []

#     with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor, \
#          tqdm(total=len(loader), desc="Object Detection (GPU) & Save (CPU)") as pbar:

#         for imgs, paths, original_sizes in loader:
#             # Model expects batch items as individual images
#             for img_np, path, orig_size in zip(imgs, paths, original_sizes):
#                 # Apply Detectron2 transform
#                 tf = transform.get_transform(img_np)
#                 resized = tf.apply_image(img_np)
#                 img_tensor_resized = torch.tensor(resized).permute(2, 0, 1).float().to(device)

#                 # Model inference
#                 with torch.no_grad():
#                     inputs = [{"image": img_tensor_resized, "height": img_np.shape[0], "width": img_np.shape[1]}]
#                     outputs = model(inputs)

#                 pred_instances = outputs[0]["instances"]
#                 keep = pred_instances.scores > conf_thresh
#                 pred_instances = pred_instances[keep]

#                 # Prepare info for saving
#                 instances_info = [
#                     (bbox.cpu().numpy(), class_names[i], tuple(class_colors[i]))
#                     for bbox, i in zip(pred_instances.pred_boxes.tensor, pred_instances.pred_classes.cpu().numpy())
#                 ]

#                 # Resize image back to original size before saving
#                 img_final = cv2.resize(resized, (orig_size[1], orig_size[0]))
#                 save_path = output_dir / (Path(path).stem + "_det.png")
#                 if save_path.exists() and resume:
#                     continue

#                 future = executor.submit(save_detectron2_image, img_final, instances_info, str(save_path))
#                 all_futures.append(future)

#             pbar.update(1)

#         if all_futures:
#             for future in tqdm(all_futures, desc="Saving final results"):
#                 future.result()

#     print("Object detection completed successfully!")

# # ---------------- CLI ----------------
# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Batch Object Detection using Detectron2")
#     parser.add_argument("--image_dir", type=str, required=True)
#     parser.add_argument("--output_dir", type=str, required=True)
#     parser.add_argument("--model_path", type=str, required=True)
#     parser.add_argument("--checkpoint_path", type=str, required=True)
#     parser.add_argument("--color_json", type=str, required=True)
#     parser.add_argument("--batch_size", type=int, default=4)
#     parser.add_argument("--num_workers", type=int, default=2)
#     parser.add_argument("--conf_thresh", type=float, default=0.5)
#     args = parser.parse_args()

#     detect_objects(
#         image_dir=args.image_dir,
#         output_dir=args.output_dir,
#         model_path=args.model_path,
#         checkpoint_path=args.checkpoint_path,
#         color_json=args.color_json,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         conf_thresh=args.conf_thresh
#     )












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
        processed = {p.stem for p in output_dir.glob("*_det.png")}
        dataset.image_paths = [p for p in dataset.image_paths if Path(p).stem not in processed]
        print(f"Resuming: {len(dataset.image_paths)} images left to process")
    if not dataset.image_paths:
        print("All images already processed. Exiting.")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, collate_fn=collate_images)

    # transform = ResizeShortestEdge(short_edge_length=1024, max_size=1024)
    transform = ResizeShortestEdge(short_edge_length=256, max_size=256)
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
                save_path = output_dir / (Path(path).stem + "_det.png")
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






# import os
# from pathlib import Path
# from concurrent.futures import ProcessPoolExecutor, Future, as_completed
# from typing import List, Tuple
# import torch
# import cv2
# import numpy as np
# import json
# from tqdm import tqdm
# from torch.utils.data import Dataset, DataLoader
# from detectron2.config import LazyConfig, instantiate
# from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.data.transforms import ResizeShortestEdge
# from detectron2.utils.visualizer import Visualizer, ColorMode
# import time # <-- Imported time

# # Define the model input transformation once
# MODEL_TRANSFORM = ResizeShortestEdge(short_edge_length=1024, max_size=1024)

# # Global dictionary to track cumulative times
# total_timing = {
#     "batches_run": 0,
#     "cpu_preprocess_sec": 0.0,
#     "gpu_inference_sec": 0.0,
#     "postprocess_submit_sec": 0.0,
# }

# # ---------------- Worker Function ----------------
# def save_detectron2_image(
#     image_resized: np.ndarray, 
#     instances_info: List[Tuple[np.ndarray, str, Tuple[int,int,int]]], 
#     save_path: str,
#     orig_size: Tuple[int, int] # (H, W) tuple is now passed
# ) -> str:
#     """
#     Worker function to perform final resizing, visualization, and saving.
#     """
#     # 1. Final resizing (CPU parallelized)
#     image_final = cv2.resize(image_resized, (orig_size[1], orig_size[0]))

#     # 2. Visualization
#     v = Visualizer(image_final, instance_mode=ColorMode.IMAGE) 
    
#     for bbox, cls_name, color in instances_info:
#         norm_color = [c / 255 for c in color]
#         v.draw_box(bbox, alpha=1, edge_color=norm_color)
#         x_center = (bbox[0] + bbox[2]) / 2
#         y = min(bbox[1], bbox[3])
#         v.draw_text(cls_name, (x_center, y), color=norm_color)
        
#     # 3. Optimized I/O
#     final_output_bgr = v.get_output().get_image()[:, :, ::-1]
#     cv2.imwrite(save_path, final_output_bgr)
    
#     return save_path

# # ---------------- Dataset ----------------
# class ImageDataset(Dataset):
#     def __init__(self, image_dir, extensions=(".jpg", ".jpeg", ".png")):
#         image_dir = Path(image_dir)
#         self.image_paths = [str(p) for p in sorted(image_dir.iterdir()) if p.suffix.lower() in extensions]
#         if not self.image_paths:
#             raise ValueError(f"No images found in {image_dir}")
#         self.transform = MODEL_TRANSFORM 

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         path = self.image_paths[idx]
        
#         # Load image with OpenCV in BGR, then convert to RGB
#         img_np_bgr = cv2.imread(path)
#         if img_np_bgr is None:
#              raise IOError(f"Failed to load image at {path}")
        
#         img_np_rgb = img_np_bgr[:, :, ::-1] 
#         orig_h, orig_w = img_np_rgb.shape[:2]

#         # Parallelize Model Input Resizing
#         tf = self.transform.get_transform(img_np_rgb)
#         resized_img_np_rgb = tf.apply_image(img_np_rgb)
        
#         return resized_img_np_rgb, path, (orig_h, orig_w)

# def collate_images(batch):
#     imgs, paths, orig_sizes = zip(*batch)
#     return list(imgs), list(paths), list(orig_sizes)

# # ---------------- Main Object Detection Function ----------------
# def detect_objects(
#     image_dir: str,
#     output_dir: str,
#     model_path: str,
#     checkpoint_path: str,
#     color_json: str,
#     batch_size: int = 4,
#     num_workers: int = 4,
#     conf_thresh: float = 0.5,
#     resume: bool = True
# ):
#     global total_timing
    
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     if device == "cpu":
#         print("WARNING: Running on CPU. This will be slow.")

#     os.makedirs(output_dir, exist_ok=True)
#     output_dir = Path(output_dir)

#     # Load model
#     cfg = LazyConfig.load(model_path)
#     cfg.train.init_checkpoint = checkpoint_path
#     np.float_ = np.float = np.float64
#     np.complex_ = np.complex = np.complex128
#     model = instantiate(cfg.model)
#     DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
#     model.eval().to(device)

#     # Load class-color mapping
#     with open(color_json, "r") as f:
#         color_dict = json.load(f)
#     class_names = list(color_dict.keys())
#     class_colors = list(color_dict.values())

#     # Prepare dataset
#     dataset = ImageDataset(image_dir)
#     if resume:
#         processed = {p.stem for p in output_dir.glob("*_det.png")}
#         dataset.image_paths = [p for p in dataset.image_paths if Path(p).stem not in processed]
#         print(f"Resuming: {len(dataset.image_paths)} images left to process")
    
#     total_images_to_process = len(dataset.image_paths)
#     if not dataset.image_paths:
#         print("All images already processed. Exiting.")
#         return

#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
#                          num_workers=num_workers, pin_memory=True, collate_fn=collate_images)

#     active_futures: List[Future] = []
#     MAX_CONCURRENT_SAVES = os.cpu_count() * 2 
#     total_batches = len(loader)

#     # Executor for async saving
#     with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor, \
#           tqdm(total=total_images_to_process, desc="Object Detection (GPU) & Save (CPU)") as pbar:

#         for i, (imgs_resized_rgb, paths, orig_sizes) in enumerate(loader):
#             total_timing["batches_run"] += 1
            
#             # --- START CPU PRE-PROCESSING TIMER ---
#             start_cpu_preprocess = time.time()
            
#             inputs = []
#             for resized_img_np_rgb, orig_size in zip(imgs_resized_rgb, orig_sizes):
#                 # Convert RGB to BGR and force memory copy
#                 resized_img_np_bgr = resized_img_np_rgb[:, :, ::-1].copy() 
                
#                 # C, H, W BGR tensor for model input
#                 img_tensor = torch.tensor(resized_img_np_bgr).permute(2, 0, 1).float().to(device, non_blocking=True)
                
#                 # Get resized dimensions for input dict
#                 H, W = resized_img_np_bgr.shape[:2]
#                 inputs.append({"image": img_tensor, "height": H, "width": W})
            
#             # --- END CPU PRE-PROCESSING TIMER ---
#             end_cpu_preprocess = time.time()
#             cpu_preprocess_time = end_cpu_preprocess - start_cpu_preprocess
#             total_timing["cpu_preprocess_sec"] += cpu_preprocess_time


#             # --- START GPU INFERENCE TIMER ---
#             start_gpu_inference = time.time()

#             # Batch inference with FP16
#             with torch.amp.autocast(device_type=device), torch.no_grad():
#                 outputs = model(inputs)
            
#             # Wait for GPU operations to ensure accurate timing
#             if device == "cuda":
#                 torch.cuda.synchronize()

#             # --- END GPU INFERENCE TIMER ---
#             end_gpu_inference = time.time()
#             gpu_inference_time = end_gpu_inference - start_gpu_inference
#             total_timing["gpu_inference_sec"] += gpu_inference_time
            
            
#             # --- START POST-PROCESSING/SUBMISSION TIMER ---
#             start_postprocess_submit = time.time()

#             # Submit saving tasks asynchronously
#             images_submitted_in_batch = 0
#             for resized_img_np_rgb, output, path, orig_size in zip(imgs_resized_rgb, outputs, paths, orig_sizes):
#                 pred_instances = output["instances"]
#                 keep = pred_instances.scores > conf_thresh
#                 pred_instances = pred_instances[keep]

#                 # Extract info and move tensors to CPU *once*
#                 instances_info = [
#                     (bbox.cpu().numpy(), class_names[i], tuple(class_colors[i]))
#                     for bbox, i in zip(pred_instances.pred_boxes.tensor, pred_instances.pred_classes.cpu().numpy())
#                 ]
                
#                 save_path = output_dir / (Path(path).stem + "_det.png")
#                 if save_path.exists() and resume:
#                     continue

#                 # Submit save task
#                 future = executor.submit(save_detectron2_image, resized_img_np_rgb, instances_info, str(save_path), orig_size)
#                 active_futures.append(future)
#                 images_submitted_in_batch += 1

#             # Bounded Queue Logic: Wait for some tasks to finish if the queue is too large
#             if len(active_futures) > MAX_CONCURRENT_SAVES:
#                 # Wait for the next one to complete
#                 done = next(as_completed(active_futures, timeout=None))

#                 # Remove the completed future from the list
#                 active_futures = [f for f in active_futures if not f.done()]
                
#                 # Check result of the completed task to catch exceptions early
#                 done.result()
                
#             # --- END POST-PROCESSING/SUBMISSION TIMER ---
#             end_postprocess_submit = time.time()
#             postprocess_submit_time = end_postprocess_submit - start_postprocess_submit
#             total_timing["postprocess_submit_sec"] += postprocess_submit_time
            
#             pbar.update(images_submitted_in_batch) # Update based on how many images were submitted for saving

#             # --- PRINT PER-BATCH TIMING ---
#             print(f"\nBatch {i+1}/{total_batches} ({images_submitted_in_batch} images) completed:")
#             print(f"  CPU Pre-process: {cpu_preprocess_time:.4f}s")
#             print(f"  GPU Inference:   {gpu_inference_time:.4f}s")
#             print(f"  Post-process/Submit: {postprocess_submit_time:.4f}s")
#             print(f"  Total Batch Time:  {cpu_preprocess_time + gpu_inference_time + postprocess_submit_time:.4f}s")
#             print("-" * 30)


#         # Wait for all remaining saves
#         for future in tqdm(as_completed(active_futures), total=len(active_futures), desc="Saving final results"):
#             future.result()

#     print("Object detection completed successfully!")
#     print("\n--- AGGREGATE PERFORMANCE BREAKDOWN ---")
#     if total_timing["batches_run"] > 0:
#         avg_batch = total_timing["batches_run"]
#         print(f"Total Batches Processed: {avg_batch}")
#         print(f"Average CPU Pre-processing (per batch): {total_timing['cpu_preprocess_sec'] / avg_batch:.4f} seconds")
#         print(f"Average GPU Inference (per batch): {total_timing['gpu_inference_sec'] / avg_batch:.4f} seconds")
#         print(f"Average Post-processing/Submission (per batch): {total_timing['postprocess_submit_sec'] / avg_batch:.4f} seconds")
    
# # ---------------- CLI ----------------
# if __name__ == "__main__":
#     import argparse

#     # ... (CLI arguments remain unchanged)
#     parser = argparse.ArgumentParser(description="Batch Object Detection using Detectron2")
#     parser.add_argument("--image_dir", type=str, required=True)
#     parser.add_argument("--output_dir", type=str, required=True)
#     parser.add_argument("--model_path", type=str, required=True)
#     parser.add_argument("--checkpoint_path", type=str, required=True)
#     parser.add_argument("--color_json", type=str, required=True)
#     parser.add_argument("--batch_size", type=int, default=4)
#     parser.add_argument("--num_workers", type=int, default=4)
#     parser.add_argument("--conf_thresh", type=float, default=0.5)
#     args = parser.parse_args()

#     detect_objects(
#         image_dir=args.image_dir,
#         output_dir=args.output_dir,
#         model_path=args.model_path,
#         checkpoint_path=args.checkpoint_path,
#         color_json=args.color_json,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         conf_thresh=args.conf_thresh
#     )



# ---------------- Example Usage ----------------
# "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\.venv-Copy-Copy\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\object_detection.py" --image_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ImageCaptioningEvaluationDatasets\LAION-5B-10k\LAION-5B-10k-images" --output_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\LAION-5B-10k-detected" --model_path "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\detectron2\projects\ViTDet\configs\LVIS\cascade_mask_rcnn_vitdet_h_100ep.py" --checkpoint_path "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\VitDet_huge_LVIS.pkl" --color_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\LVIS_color_map.json" --batch_size 4 --num_workers 8


# "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\.venv-Copy-Copy\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\object_detection.py" --image_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ImageCaptioningEvaluationDatasets\LAION-5B-10k\LAION-5B-10k-images" --output_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\LAION-5B-10k-detected" --model_path "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\detectron2\projects\ViTDet\configs\LVIS\cascade_mask_rcnn_vitdet_b_100ep.py" --checkpoint_path "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\VitDet_base_LVIS.pkl" --color_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\LVIS_color_map.json" --batch_size 4 --num_workers 8

# --checkpoint_batches 100