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

# import sys
# from pathlib import Path

# YOLACT_PATH = Path(__file__).resolve().parent / "yolact"
# sys.path.insert(0, str(YOLACT_PATH))

# # ================================
# # CONFIG
# # ================================
# ADE20K_PERSON_CLASS = 12
# COCO_PERSON_CLASS = 0

# MIN_MASK_AREA_RATIO = 0.02
# MAX_MASK_AREA_RATIO = 0.60
# MIN_BBOX_AREA_RATIO = 0.03
# MAX_BBOX_AREA_RATIO = 0.80


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


# # ================================
# # BASE MODEL INTERFACE
# # ================================
# class BaseSegmentationModel(ABC):
#     """Abstract base class for segmentation models"""
    
#     def __init__(self, device="cuda"):
#         self.device = device
        
#     @abstractmethod
#     def load_model(self):
#         """Load the model"""
#         pass
    
#     @abstractmethod
#     def predict(self, images_tensor: torch.Tensor) -> list:
#         """
#         Run inference on batch of images
#         Returns list of person masks (one per image)
#         Each mask should be a binary numpy array (H, W)
#         """
#         pass
    
#     @abstractmethod
#     def get_fixed_size(self) -> Tuple[int, int]:
#         """Return the fixed input size for the model"""
#         pass


# # ================================
# # MASK2FORMER (ADE20K) - YOUR ORIGINAL
# # ================================
# class Mask2FormerADE20K(BaseSegmentationModel):
#     def __init__(self, device="cuda"):
#         super().__init__(device)
#         self.model = None
#         self.processor = None
#         self.mean = None
#         self.std = None
        
#     def load_model(self):
#         print("Loading Mask2Former (ADE20K)...")
#         model_name = "facebook/mask2former-swin-large-ade-semantic"
#         self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
#             model_name, torch_dtype=torch.float16
#         ).to(self.device).eval()
        
#         self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
#         self.mean = torch.tensor(self.processor.image_mean, device=self.device).view(1, 3, 1, 1)
#         self.std = torch.tensor(self.processor.image_std, device=self.device).view(1, 3, 1, 1)
        
#     def predict(self, images_tensor: torch.Tensor) -> list:
#         pixel_values = (images_tensor - self.mean) / self.std
        
#         with torch.no_grad(), torch.amp.autocast("cuda"):
#             outputs = self.model(pixel_values=pixel_values)
        
#         fixed_size = self.get_fixed_size()
#         seg_maps = self.processor.post_process_semantic_segmentation(
#             outputs,
#             target_sizes=[fixed_size for _ in images_tensor]
#         )
        
#         # Extract person masks
#         person_masks = []
#         for seg_map in seg_maps:
#             seg_np = seg_map.cpu().numpy().astype(np.uint8)
#             person_mask = (seg_np == ADE20K_PERSON_CLASS).astype(np.uint8)
#             person_masks.append(person_mask)
        
#         return person_masks
    
#     def get_fixed_size(self) -> Tuple[int, int]:
#         return (512, 512)

# # ================================
# # MASK2FORMER (COCO INSTANCE)
# # ================================
# class Mask2FormerCOCO(BaseSegmentationModel):
#     def __init__(self, device="cuda"):
#         super().__init__(device)
#         self.model = None
#         self.processor = None
#         self.mean = None
#         self.std = None
        
#     def load_model(self):
#         print("Loading Mask2Former (COCO Instance)...")
#         model_name = "facebook/mask2former-swin-large-coco-instance"
#         self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
#             model_name, torch_dtype=torch.float16
#         ).to(self.device).eval()
        
#         self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
#         self.mean = torch.tensor(self.processor.image_mean, device=self.device).view(1, 3, 1, 1)
#         self.std = torch.tensor(self.processor.image_std, device=self.device).view(1, 3, 1, 1)
        
#     def predict(self, images_tensor: torch.Tensor) -> list:
#         pixel_values = (images_tensor - self.mean) / self.std
        
#         with torch.no_grad(), torch.amp.autocast("cuda"):
#             outputs = self.model(pixel_values=pixel_values)
        
#         fixed_size = self.get_fixed_size()
#         results = self.processor.post_process_instance_segmentation(
#             outputs,
#             target_sizes=[fixed_size for _ in images_tensor],
#             threshold=0.5
#         )
        
#         # Extract person masks (combine all person instances)
#         person_masks = []
#         for result in results:
#             h, w = fixed_size
#             combined_mask = np.zeros((h, w), dtype=np.uint8)
            
#             # Result contains 'segmentation' (H, W) tensor and 'segments_info' list
#             if 'segmentation' in result and 'segments_info' in result:
#                 seg_map = result['segmentation'].cpu().numpy()
                
#                 # Find all person segments (label_id == 0 for person in COCO)
#                 for segment_info in result['segments_info']:
#                     if segment_info['label_id'] == COCO_PERSON_CLASS:
#                         # Extract this instance's mask
#                         instance_mask = (seg_map == segment_info['id'])
#                         combined_mask = np.logical_or(combined_mask, instance_mask).astype(np.uint8)
            
#             person_masks.append(combined_mask)
        
#         return person_masks
    
#     def get_fixed_size(self) -> Tuple[int, int]:
#         return (512, 512)


# # ================================
# # DETECTRON2 MASK R-CNN
# # ================================
# class MaskRCNNDetectron2(BaseSegmentationModel):
#     def __init__(self, device="cuda", confidence_threshold=0.5):
#         super().__init__(device)
#         self.predictor = None
#         self.confidence_threshold = confidence_threshold
        
#     def load_model(self):
#         print("Loading Mask R-CNN (Detectron2)...")
#         try:
#             from detectron2 import model_zoo
#             from detectron2.engine import DefaultPredictor
#             from detectron2.config import get_cfg
#         except ImportError:
#             raise ImportError("Please install detectron2: pip install detectron2")
        
#         cfg = get_cfg()
#         cfg.merge_from_file(model_zoo.get_config_file(
#             "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
#         ))
#         cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
#             "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
#         )
#         cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
#         cfg.MODEL.DEVICE = self.device
        
#         self.predictor = DefaultPredictor(cfg)
        
#     def predict(self, images_tensor: torch.Tensor) -> list:
#         # Detectron2 expects BGR numpy arrays
#         person_masks = []
        
#         for img_tensor in images_tensor:
#             # Convert to numpy BGR
#             img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
#             img_np = (img_np * 255).astype(np.uint8)
#             img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
#             # Run inference
#             outputs = self.predictor(img_bgr)
#             instances = outputs["instances"]
            
#             # Filter for person class (0 in COCO)
#             person_instances = instances[instances.pred_classes == COCO_PERSON_CLASS]
            
#             # Combine all person masks
#             h, w = img_bgr.shape[:2]
#             combined_mask = np.zeros((h, w), dtype=np.uint8)
            
#             if len(person_instances) > 0:
#                 masks = person_instances.pred_masks.cpu().numpy()
#                 combined_mask = np.any(masks, axis=0).astype(np.uint8)
            
#             person_masks.append(combined_mask)
        
#         return person_masks
    
#     def get_fixed_size(self) -> Tuple[int, int]:
#         return (512, 512)


# # ================================
# # YOLACT (REAL INTEGRATION)
# # ================================
# class YOLACTModel(BaseSegmentationModel):
#     def __init__(self, device="cuda", confidence_threshold=0.3):
#         super().__init__(device)
#         self.model = None
#         self.confidence_threshold = confidence_threshold
#         self.transform = None

#     def load_model(self):
#         print("Loading YOLACT...")

#         from data import cfg, set_cfg

#         # ✅ Match your downloaded checkpoint
#         set_cfg("yolact_resnet50_config")

#         cfg.use_fast_nms = True
#         cfg.use_cross_class_nms = True
        
#         from yolact import Yolact
#         from utils.augmentations import FastBaseTransform

#         # ✅ Build absolute path to weights
#         yolact_root = Path(__file__).resolve().parent / "yolact"
#         weights_path = yolact_root / "weights" / "yolact_resnet50_54_800000.pth"

#         if not weights_path.exists():
#             raise FileNotFoundError(
#                 f"YOLACT weights not found at:\n{weights_path}\n\n"
#                 "Make sure the file exists in yolact/weights/"
#             )

#         self.model = Yolact()
#         self.model.load_weights(str(weights_path))
#         self.model.eval()
#         self.model = self.model.to(self.device)

#         self.transform = FastBaseTransform()

#         print("✅ YOLACT loaded successfully from:")
#         print(weights_path)

#     def predict(self, images_tensor: torch.Tensor) -> list:
#         from layers.output_utils import postprocess

#         person_masks = []

#         with torch.no_grad():
#             for img_tensor in images_tensor:
#                 # --- Convert CHW -> HWC uint8 ---
#                 img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
#                 img_np = (img_np * 255).astype(np.uint8)
#                 h, w = img_np.shape[:2]

#                 # ✅ Convert to tensor and ADD BATCH DIMENSION [1, H, W, 3]
#                 frame = torch.from_numpy(img_np).float().to(self.device)
#                 frame = frame.unsqueeze(0)  # [1, H, W, 3]

#                 # ✅ Apply YOLACT transform (expects 4D input)
#                 frame = self.transform(frame)

#                 # Forward through YOLACT
#                 preds = self.model(frame)

#                 # ✅ Correct call to postprocess:
#                 #    - keep default batch_idx=0
#                 #    - pass score_threshold as a keyword
#                 classes, scores, boxes, masks = postprocess(
#                     preds,
#                     w,
#                     h,
#                     score_threshold=self.confidence_threshold,
#                 )

#                 combined_mask = np.zeros((h, w), dtype=np.uint8)

#                 # classes is a tensor of class IDs (COCO class 0 == person)
#                 if classes is not None and len(classes) > 0:
#                     person_indices = (classes == 0)

#                     if person_indices.any():
#                         person_masks_np = masks[person_indices].cpu().numpy()
#                         combined_mask = np.any(person_masks_np, axis=0).astype(np.uint8)

#                 person_masks.append(combined_mask)

#         return person_masks


#     def get_fixed_size(self) -> Tuple[int, int]:
#         # YOLACT resizes internally
#         return (550, 550)

# # ================================
# # SAM (Segment Anything Model)
# # ================================
# class SAMModel(BaseSegmentationModel):
#     def __init__(self, device="cuda", model_size="base"):
#         super().__init__(device)
#         self.model = None
#         self.processor = None
#         self.model_size = model_size

#     def load_model(self):
#         print(f"Loading SAM ({self.model_size})...")
#         try:
#             from transformers import SamModel, SamProcessor
#         except ImportError:
#             raise ImportError("Please install: pip install transformers>=4.35.0")

#         model_names = {
#             "base": "facebook/sam-vit-base",
#             "large": "facebook/sam-vit-large",
#             "huge": "facebook/sam-vit-huge",
#         }
#         model_name = model_names[self.model_size]

#         self.model = SamModel.from_pretrained(model_name).to(self.device).eval()
#         self.processor = SamProcessor.from_pretrained(model_name)
#         print("⚠️  SAM is designed for interactive use, not batch processing.")
#         print("   This will be VERY SLOW. Consider using Mask2Former or MaskRCNN instead.")

#     def predict(self, images_tensor: torch.Tensor) -> list:
#         """
#         Heuristic: pick the largest reasonably central mask as 'person'.
#         """
#         person_masks = []

#         for img_tensor in images_tensor:
#             # Convert to uint8 H×W×C
#             img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
#             img_np = (img_np * 255).astype(np.uint8)

#             from PIL import Image as PILImage
#             pil_img = PILImage.fromarray(img_np)

#             # Prepare SAM inputs
#             inputs = self.processor(pil_img, return_tensors="pt").to(self.device)

#             with torch.no_grad():
#                 outputs = self.model(**inputs, multimask_output=True)

#             # post_process_masks returns a list (batch) of tensors
#             masks = self.processor.post_process_masks(
#                 outputs.pred_masks.cpu(),
#                 inputs["original_sizes"].cpu(),
#                 inputs["reshaped_input_sizes"].cpu(),
#             )[0]  # first (and only) image

#             # Ensure shape is (num_masks, H, W)
#             if isinstance(masks, torch.Tensor):
#                 masks = masks.cpu()
#             if masks.ndim == 4:
#                 # (num_masks, 1, H, W) -> (num_masks, H, W)
#                 masks = masks.squeeze(1)

#             masks_np = masks.numpy()
#             num_masks = masks_np.shape[0]

#             # Default: no person mask
#             h_mask = w_mask = None
#             combined_mask = None

#             if num_masks > 0:
#                 # Use mask resolution for center, not original image
#                 h_mask, w_mask = masks_np.shape[1], masks_np.shape[2]
#                 center_y, center_x = h_mask // 2, w_mask // 2

#                 best_score = -1.0
#                 best_mask = None

#                 # Check up to top 3 masks
#                 for i in range(min(num_masks, 3)):
#                     mask = masks_np[i]

#                     # ✅ FORCE ANY POSSIBLE SHAPE INTO A SINGLE 2D MASK
#                     if mask.ndim == 2:
#                         pass  # already HxW
#                     elif mask.ndim == 3:
#                         # Could be (1, H, W) or (H, W, 1) or (K, H, W)
#                         if mask.shape[0] == 1:          # (1, H, W)
#                             mask = mask[0]
#                         elif mask.shape[-1] == 1:       # (H, W, 1)
#                             mask = mask[:, :, 0]
#                         else:                           # (K, H, W) → collapse by OR
#                             mask = np.max(mask, axis=0)
#                     elif mask.ndim == 4:
#                         # (1, 1, H, W) or similar
#                         mask = np.max(mask, axis=(0, 1))
#                     else:
#                         # Absolute fallback
#                         mask = np.max(mask)

#                     mask = (mask > 0).astype(np.uint8)

#                     area = float(mask.sum())
#                     if area == 0:
#                         continue

#                     # ✅ SAFE CENTER CHECK (ALWAYS SCALAR NOW)
#                     center_val = mask[center_y, center_x]
#                     contains_center = int(center_val) > 0

#                     # Simple heuristic: area * centrality bonus
#                     score = area * (2.0 if contains_center else 1.0)

#                     if score > best_score:
#                         best_score = score
#                         best_mask = mask

#                 if best_mask is not None:
#                     combined_mask = best_mask

#             if combined_mask is None:
#                 # Fallback: empty mask (no valid person-like region)
#                 if h_mask is None or w_mask is None:
#                     h_mask, w_mask = img_np.shape[:2]
#                 combined_mask = np.zeros((h_mask, w_mask), dtype=np.uint8)

#             person_masks.append(combined_mask)

#         return person_masks

#     def get_fixed_size(self) -> Tuple[int, int]:
#         # SAM internally works at 1024×1024, but we can keep your external fixed_size independent.
#         return (1024, 1024)

# # ================================
# # LANG-SAM (TEXT PROMPTED PERSON SEGMENTATION)
# # ================================
# class LangSAMPersonModel(BaseSegmentationModel):
#     def __init__(self, device="cuda", prompt="person"):
#         super().__init__(device)
#         self.prompt = prompt
#         self.model = None

#     def load_model(self):
#         print("Loading Lang-Segment-Anything (Text-Prompted SAM)...")
#         try:
#             from lang_sam import LangSAM
#         except ImportError:
#             raise ImportError(
#                 "Please install Lang-SAM:\n"
#                 "pip install git+https://github.com/luca-medeiros/lang-segment-anything.git"
#             )

#         # ✅ Do NOT pass device here – repo handles it internally via DEVICE
#         self.model = LangSAM()
#         print(f"✅ Lang-SAM loaded with prompt: '{self.prompt}'")

#     def predict(self, images_tensor: torch.Tensor) -> list:
#         """
#         Returns one binary person mask per input image.
#         """
#         from PIL import Image as PILImage

#         person_masks = []

#         for img_tensor in images_tensor:
#             # ✅ Convert CHW tensor → HWC uint8
#             img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
#             img_np = (img_np * 255).astype(np.uint8)
#             pil_img = PILImage.fromarray(img_np)

#             # ✅ Lang-SAM expects LISTS
#             with suppress_stdout():
#                 results = self.model.predict(
#                     images_pil=[pil_img],
#                     texts_prompt=[self.prompt],
#                     box_threshold=0.5,
#                     text_threshold=0.5,
#                 )

#             result = results[0]  # one image → one result

#             masks = result.get("masks", None)

#             if masks is None or len(masks) == 0:
#                 # ❌ No person detected → return empty mask
#                 h, w = img_np.shape[:2]
#                 empty = np.zeros((h, w), dtype=np.uint8)
#                 person_masks.append(empty)
#                 continue

#             # ✅ masks shape: (N, H, W)
#             # Combine all detected "person" regions
#             combined_mask = np.any(masks, axis=0).astype(np.uint8)

#             person_masks.append(combined_mask)

#         return person_masks

#     def get_fixed_size(self) -> Tuple[int, int]:
#         # SAM operates internally near 1024
#         return (1024, 1024)


# # ================================
# # MODEL FACTORY
# # ================================
# def get_model(model_name: str, device: str = "cuda") -> BaseSegmentationModel:
#     """Factory function to get the specified model"""
#     models = {
#         "mask2former_ade20k": Mask2FormerADE20K,
#         "mask2former_coco": Mask2FormerCOCO,
#         "maskrcnn": MaskRCNNDetectron2,
#         "yolact": YOLACTModel,
#         "sam": SAMModel,
#         "lang_sam": LangSAMPersonModel
#     }
    
#     if model_name not in models:
#         raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
#     model = models[model_name](device)
#     model.load_model()
#     return model


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
    
    if mask_ratio < MIN_MASK_AREA_RATIO or mask_ratio > MAX_MASK_AREA_RATIO:
        return None
    
    ys, xs = np.where(person_mask_resized == 1)
    if len(xs) == 0:
        return None
    
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    bbox_area = (x2 - x1) * (y2 - y1)
    bbox_ratio = bbox_area / img_area
    
    if bbox_ratio < MIN_BBOX_AREA_RATIO or bbox_ratio > MAX_BBOX_AREA_RATIO:
        return None
    
    # Get relative path structure
    rel_path = os.path.relpath(image_path, Path(image_path).parent.parent)
    stem = Path(rel_path).stem
    subdir = Path(rel_path).parent
    
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
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")

        # Maintain directory structure
        out_dir = Path(output_root) / k / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}.png"
        Image.fromarray(img).save(out_path, compress_level=1)
    
    return {
        "image": str(Path(rel_path).with_suffix("")),
        "mask_ratio": round(mask_ratio, 4),
        "bbox_ratio": round(bbox_ratio, 4)
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
                            writer.writerow(res)
                            for op in operations:
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
