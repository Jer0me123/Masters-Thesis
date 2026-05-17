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

import sys
from pathlib import Path

YOLACT_PATH = Path(__file__).resolve().parent / "yolact"
sys.path.insert(0, str(YOLACT_PATH))

# ================================
# CONFIG
# ================================
ADE20K_PERSON_CLASS = 12
COCO_PERSON_CLASS = 0

MIN_MASK_AREA_RATIO = 0.02
MAX_MASK_AREA_RATIO = 0.60
MIN_BBOX_AREA_RATIO = 0.03
MAX_BBOX_AREA_RATIO = 0.80

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
# BASE MODEL INTERFACE
# ================================
class BaseSegmentationModel(ABC):
    """Abstract base class for segmentation models"""
    
    def __init__(self, device="cuda"):
        self.device = device
        
    @abstractmethod
    def load_model(self):
        """Load the model"""
        pass
    
    @abstractmethod
    def predict(self, images_tensor: torch.Tensor) -> list:
        """
        Run inference on batch of images
        Returns list of person masks (one per image)
        Each mask should be a binary numpy array (H, W)
        """
        pass
    
    @abstractmethod
    def get_fixed_size(self) -> Tuple[int, int]:
        """Return the fixed input size for the model"""
        pass


# ================================
# MASK2FORMER (ADE20K) - YOUR ORIGINAL
# ================================
class Mask2FormerADE20K(BaseSegmentationModel):
    def __init__(self, device="cuda"):
        super().__init__(device)
        self.model = None
        self.processor = None
        self.mean = None
        self.std = None
        
    def load_model(self):
        print("Loading Mask2Former (ADE20K)...")
        model_name = "facebook/mask2former-swin-large-ade-semantic"
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(self.device).eval()
        
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.mean = torch.tensor(self.processor.image_mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.processor.image_std, device=self.device).view(1, 3, 1, 1)
        
    def predict(self, images_tensor: torch.Tensor) -> list:
        pixel_values = (images_tensor - self.mean) / self.std
        
        with torch.no_grad(), torch.amp.autocast("cuda"):
            outputs = self.model(pixel_values=pixel_values)
        
        fixed_size = self.get_fixed_size()
        seg_maps = self.processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[fixed_size for _ in images_tensor]
        )
        
        # Extract person masks
        person_masks = []
        for seg_map in seg_maps:
            seg_np = seg_map.cpu().numpy().astype(np.uint8)
            person_mask = (seg_np == ADE20K_PERSON_CLASS).astype(np.uint8)
            person_masks.append(person_mask)
        
        return person_masks
    
    def get_fixed_size(self) -> Tuple[int, int]:
        return (512, 512)

# ================================
# MASK2FORMER (COCO INSTANCE)
# ================================
class Mask2FormerCOCO(BaseSegmentationModel):
    def __init__(self, device="cuda"):
        super().__init__(device)
        self.model = None
        self.processor = None
        self.mean = None
        self.std = None
        
    def load_model(self):
        print("Loading Mask2Former (COCO Instance)...")
        model_name = "facebook/mask2former-swin-large-coco-instance"
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(self.device).eval()
        
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.mean = torch.tensor(self.processor.image_mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.processor.image_std, device=self.device).view(1, 3, 1, 1)
        
    def predict(self, images_tensor: torch.Tensor) -> list:
        pixel_values = (images_tensor - self.mean) / self.std
        
        with torch.no_grad(), torch.amp.autocast("cuda"):
            outputs = self.model(pixel_values=pixel_values)
        
        fixed_size = self.get_fixed_size()
        results = self.processor.post_process_instance_segmentation(
            outputs,
            target_sizes=[fixed_size for _ in images_tensor],
            threshold=0.5
        )
        
        # Extract person masks (combine all person instances)
        person_masks = []
        for result in results:
            h, w = fixed_size
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Result contains 'segmentation' (H, W) tensor and 'segments_info' list
            if 'segmentation' in result and 'segments_info' in result:
                seg_map = result['segmentation'].cpu().numpy()
                
                # Find all person segments (label_id == 0 for person in COCO)
                for segment_info in result['segments_info']:
                    if segment_info['label_id'] == COCO_PERSON_CLASS:
                        # Extract this instance's mask
                        instance_mask = (seg_map == segment_info['id'])
                        combined_mask = np.logical_or(combined_mask, instance_mask).astype(np.uint8)
            
            person_masks.append(combined_mask)
        
        return person_masks
    
    def get_fixed_size(self) -> Tuple[int, int]:
        return (512, 512)


# ================================
# DETECTRON2 MASK R-CNN
# ================================
class MaskRCNNDetectron2(BaseSegmentationModel):
    def __init__(self, device="cuda", confidence_threshold=0.5):
        super().__init__(device)
        self.predictor = None
        self.confidence_threshold = confidence_threshold
        
    def load_model(self):
        print("Loading Mask R-CNN (Detectron2)...")
        try:
            from detectron2 import model_zoo
            from detectron2.engine import DefaultPredictor
            from detectron2.config import get_cfg
        except ImportError:
            raise ImportError("Please install detectron2: pip install detectron2")
        
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        ))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
        cfg.MODEL.DEVICE = self.device
        
        self.predictor = DefaultPredictor(cfg)
        
    def predict(self, images_tensor: torch.Tensor) -> list:
        # Detectron2 expects BGR numpy arrays
        person_masks = []
        
        for img_tensor in images_tensor:
            # Convert to numpy BGR
            img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # Run inference
            outputs = self.predictor(img_bgr)
            instances = outputs["instances"]
            
            # Filter for person class (0 in COCO)
            person_instances = instances[instances.pred_classes == COCO_PERSON_CLASS]
            
            # Combine all person masks
            h, w = img_bgr.shape[:2]
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            
            if len(person_instances) > 0:
                masks = person_instances.pred_masks.cpu().numpy()
                combined_mask = np.any(masks, axis=0).astype(np.uint8)
            
            person_masks.append(combined_mask)
        
        return person_masks
    
    def get_fixed_size(self) -> Tuple[int, int]:
        return (512, 512)


# ================================
# YOLACT (REAL INTEGRATION)
# ================================
class YOLACTModel(BaseSegmentationModel):
    def __init__(self, device="cuda", confidence_threshold=0.3):
        super().__init__(device)
        self.model = None
        self.confidence_threshold = confidence_threshold
        self.transform = None

    def load_model(self):
        print("Loading YOLACT...")

        from data import cfg, set_cfg

        #  Match your downloaded checkpoint
        set_cfg("yolact_resnet50_config")

        cfg.use_fast_nms = True
        cfg.use_cross_class_nms = True
        
        from yolact import Yolact
        from utils.augmentations import FastBaseTransform

        #  Build absolute path to weights
        yolact_root = Path(__file__).resolve().parent / "yolact"
        weights_path = yolact_root / "weights" / "yolact_resnet50_54_800000.pth"

        if not weights_path.exists():
            raise FileNotFoundError(
                f"YOLACT weights not found at:\n{weights_path}\n\n"
                "Make sure the file exists in yolact/weights/"
            )

        self.model = Yolact()
        self.model.load_weights(str(weights_path))
        self.model.eval()
        self.model = self.model.to(self.device)

        self.transform = FastBaseTransform()

        print(" YOLACT loaded successfully from:")
        print(weights_path)

    def predict(self, images_tensor: torch.Tensor) -> list:
        from layers.output_utils import postprocess

        person_masks = []

        with torch.no_grad():
            for img_tensor in images_tensor:
                # --- Convert CHW -> HWC uint8 ---
                img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np * 255).astype(np.uint8)
                h, w = img_np.shape[:2]

                #  Convert to tensor and ADD BATCH DIMENSION [1, H, W, 3]
                frame = torch.from_numpy(img_np).float().to(self.device)
                frame = frame.unsqueeze(0)  # [1, H, W, 3]

                #  Apply YOLACT transform (expects 4D input)
                frame = self.transform(frame)

                # Forward through YOLACT
                preds = self.model(frame)

                #  Correct call to postprocess:
                #    - keep default batch_idx=0
                #    - pass score_threshold as a keyword
                classes, scores, boxes, masks = postprocess(
                    preds,
                    w,
                    h,
                    score_threshold=self.confidence_threshold,
                )

                combined_mask = np.zeros((h, w), dtype=np.uint8)

                # classes is a tensor of class IDs (COCO class 0 == person)
                if classes is not None and len(classes) > 0:
                    person_indices = (classes == 0)

                    if person_indices.any():
                        person_masks_np = masks[person_indices].cpu().numpy()
                        combined_mask = np.any(person_masks_np, axis=0).astype(np.uint8)

                person_masks.append(combined_mask)

        return person_masks


    def get_fixed_size(self) -> Tuple[int, int]:
        # YOLACT resizes internally
        return (550, 550)

# ================================
# SAM (Segment Anything Model)
# ================================
class SAMModel(BaseSegmentationModel):
    def __init__(self, device="cuda", model_size="base"):
        super().__init__(device)
        self.model = None
        self.processor = None
        self.model_size = model_size

    def load_model(self):
        print(f"Loading SAM ({self.model_size})...")
        try:
            from transformers import SamModel, SamProcessor
        except ImportError:
            raise ImportError("Please install: pip install transformers>=4.35.0")

        model_names = {
            "base": "facebook/sam-vit-base",
            "large": "facebook/sam-vit-large",
            "huge": "facebook/sam-vit-huge",
        }
        model_name = model_names[self.model_size]

        self.model = SamModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = SamProcessor.from_pretrained(model_name)
        print("  SAM is designed for interactive use, not batch processing.")
        print("   This will be VERY SLOW. Consider using Mask2Former or MaskRCNN instead.")

    def predict(self, images_tensor: torch.Tensor) -> list:
        """
        Heuristic: pick the largest reasonably central mask as 'person'.
        """
        person_masks = []

        for img_tensor in images_tensor:
            # Convert to uint8 H×W×C
            img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * 255).astype(np.uint8)

            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(img_np)

            # Prepare SAM inputs
            inputs = self.processor(pil_img, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, multimask_output=True)

            # post_process_masks returns a list (batch) of tensors
            masks = self.processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )[0]  # first (and only) image

            # Ensure shape is (num_masks, H, W)
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu()
            if masks.ndim == 4:
                # (num_masks, 1, H, W) -> (num_masks, H, W)
                masks = masks.squeeze(1)

            masks_np = masks.numpy()
            num_masks = masks_np.shape[0]

            # Default: no person mask
            h_mask = w_mask = None
            combined_mask = None

            if num_masks > 0:
                # Use mask resolution for center, not original image
                h_mask, w_mask = masks_np.shape[1], masks_np.shape[2]
                center_y, center_x = h_mask // 2, w_mask // 2

                best_score = -1.0
                best_mask = None

                # Check up to top 3 masks
                for i in range(min(num_masks, 3)):

                    mask = masks_np[i]

                    if mask.ndim == 2:
                        pass  # already HxW
                    elif mask.ndim == 3:
                        # Could be (1, H, W) or (H, W, 1) or (K, H, W)
                        if mask.shape[0] == 1:          # (1, H, W)
                            mask = mask[0]
                        elif mask.shape[-1] == 1:       # (H, W, 1)
                            mask = mask[:, :, 0]
                        else:                           # (K, H, W) → collapse by OR
                            mask = np.max(mask, axis=0)
                    elif mask.ndim == 4:
                        # (1, 1, H, W) or similar
                        mask = np.max(mask, axis=(0, 1))
                    else:
                        # Absolute fallback
                        mask = np.max(mask)

                    mask = (mask > 0).astype(np.uint8)

                    area = float(mask.sum())
                    if area == 0:
                        continue

                    #  SAFE CENTER CHECK (ALWAYS SCALAR NOW)
                    center_val = mask[center_y, center_x]
                    contains_center = int(center_val) > 0

                    # Simple heuristic: area * centrality bonus
                    score = area * (2.0 if contains_center else 1.0)

                    if score > best_score:
                        best_score = score
                        best_mask = mask

                if best_mask is not None:
                    combined_mask = best_mask

            if combined_mask is None:
                # Fallback: empty mask (no valid person-like region)
                if h_mask is None or w_mask is None:
                    h_mask, w_mask = img_np.shape[:2]
                combined_mask = np.zeros((h_mask, w_mask), dtype=np.uint8)

            person_masks.append(combined_mask)

        return person_masks

    def get_fixed_size(self) -> Tuple[int, int]:
        # SAM internally works at 1024×1024, but we can keep your external fixed_size independent.
        return (1024, 1024)

# ================================
# LANG-SAM (TEXT PROMPTED PERSON SEGMENTATION)
# ================================
class LangSAMPersonModel(BaseSegmentationModel):
    def __init__(self, device="cuda", prompt="person"):
        super().__init__(device)
        self.prompt = prompt
        self.model = None

    def load_model(self):
        print("Loading Lang-Segment-Anything (Text-Prompted SAM)...")
        try:
            from lang_sam import LangSAM
        except ImportError:
            raise ImportError(
                "Please install Lang-SAM:\n"
                "pip install git+https://github.com/luca-medeiros/lang-segment-anything.git"
            )

        #  Do NOT pass device here — repo handles it internally via DEVICE
        self.model = LangSAM()
        print(f" Lang-SAM loaded with prompt: '{self.prompt}'")

    def predict(self, images_tensor: torch.Tensor) -> list:
        """
        Returns one binary person mask per input image.
        """
        from PIL import Image as PILImage

        person_masks = []

        for img_tensor in images_tensor:
            #  Convert CHW tensor → HWC uint8
            img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * 255).astype(np.uint8)
            pil_img = PILImage.fromarray(img_np)

            #  Lang-SAM expects LISTS
            with suppress_stdout():
                results = self.model.predict(
                    images_pil=[pil_img],
                    texts_prompt=[self.prompt],
                    box_threshold=0.5,
                    text_threshold=0.5,
                )

            result = results[0]  # one image → one result

            masks = result.get("masks", None)

            if masks is None or len(masks) == 0:
                #  No person detected → return empty mask
                h, w = img_np.shape[:2]
                empty = np.zeros((h, w), dtype=np.uint8)
                person_masks.append(empty)
                continue

            #  masks shape: (N, H, W)
            # Combine all detected "person" regions
            combined_mask = np.any(masks, axis=0).astype(np.uint8)

            person_masks.append(combined_mask)

        return person_masks

    def get_fixed_size(self) -> Tuple[int, int]:
        # SAM operates internally near 1024
        return (1024, 1024)
    
# ================================
# MODEL FACTORY
# ================================
def get_model(model_name: str, device: str = "cuda") -> BaseSegmentationModel:
    """Factory function to get the specified model"""
    models = {
        "mask2former_ade20k": Mask2FormerADE20K,
        "mask2former_coco": Mask2FormerCOCO,
        "maskrcnn": MaskRCNNDetectron2,
        "yolact": YOLACTModel,
        "sam": SAMModel,
        "lang_sam": LangSAMPersonModel
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    model = models[model_name](device)
    model.load_model()
    return model

# ================================
# DATASET (UNCHANGED)
# ================================
class ImageDataset(Dataset):
    def __init__(self, image_dir, fixed_size=(512, 512), exclude_set: Optional[Set[str]] = None):
        self.fixed_size = fixed_size
        all_paths = sorted(
            [str(p) for p in Path(image_dir).iterdir()
             if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        )
        
        if exclude_set:
            self.image_paths = [
                p for p in all_paths if Path(p).stem not in exclude_set
            ]
        else:
            self.image_paths = all_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = read_image(path).float() / 255.0

        if img.shape[0] == 1:  # Grayscale image
            img = img.repeat(3, 1, 1)  # Repeat channel 3 times to make RGB

        original_size = img.shape[1:]
        img = resize(img, self.fixed_size)
        return img, path, original_size


def collate_fixed(batch):
    imgs, paths, sizes = zip(*batch)
    return torch.stack(imgs, dim=0), list(paths), list(sizes)