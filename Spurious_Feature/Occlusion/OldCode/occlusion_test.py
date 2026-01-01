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

        # ✅ Match your downloaded checkpoint
        set_cfg("yolact_resnet50_config")

        cfg.use_fast_nms = True
        cfg.use_cross_class_nms = True
        
        from yolact import Yolact
        from utils.augmentations import FastBaseTransform

        # ✅ Build absolute path to weights
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

        print("✅ YOLACT loaded successfully from:")
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

                # ✅ Convert to tensor and ADD BATCH DIMENSION [1, H, W, 3]
                frame = torch.from_numpy(img_np).float().to(self.device)
                frame = frame.unsqueeze(0)  # [1, H, W, 3]

                # ✅ Apply YOLACT transform (expects 4D input)
                frame = self.transform(frame)

                # Forward through YOLACT
                preds = self.model(frame)

                # ✅ Correct call to postprocess:
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
        print("⚠️  SAM is designed for interactive use, not batch processing.")
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
                    # mask = masks_np[i].astype(np.uint8)  # (H, W)

                    # area = float(mask.sum())
                    # if area == 0:
                    #     continue

                    # # Safe center access in mask coords
                    # contains_center = bool(mask[center_y, center_x] > 0)

                    mask = masks_np[i]

                    # ✅ FORCE ANY POSSIBLE SHAPE INTO A SINGLE 2D MASK
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

                    # ✅ SAFE CENTER CHECK (ALWAYS SCALAR NOW)
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

        # ✅ Do NOT pass device here — repo handles it internally via DEVICE
        self.model = LangSAM()
        print(f"✅ Lang-SAM loaded with prompt: '{self.prompt}'")

    def predict(self, images_tensor: torch.Tensor) -> list:
        """
        Returns one binary person mask per input image.
        """
        from PIL import Image as PILImage

        person_masks = []

        for img_tensor in images_tensor:
            # ✅ Convert CHW tensor → HWC uint8
            img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * 255).astype(np.uint8)
            pil_img = PILImage.fromarray(img_np)

            # ✅ Lang-SAM expects LISTS
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
                # ❌ No person detected → return empty mask
                h, w = img_np.shape[:2]
                empty = np.zeros((h, w), dtype=np.uint8)
                person_masks.append(empty)
                continue

            # ✅ masks shape: (N, H, W)
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
# CPU POST-PROCESS (UNCHANGED)
# ================================
def process_occlusions_and_save(person_mask, original_size, image_path, output_root):
    """
    Takes a person mask and generates all occlusion variants
    person_mask: binary mask at fixed_size resolution
    """
    H, W = original_size
    
    # Resize mask to original size
    person_mask_resized = cv2.resize(person_mask, (W, H), cv2.INTER_NEAREST)
    
    # Load original image
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
    
    stem = Path(image_path).stem
    
    # Generate all occlusion variants
    full = image.copy()
    full_nobg = image * person_mask_resized[..., None]
    
    masksegm = image.copy()
    masksegm[person_mask_resized == 1] = 255
    
    masksegm_nobg = np.zeros_like(image)
    masksegm_nobg[person_mask_resized == 1] = 255
    
    maskrect = image.copy()
    maskrect[y1:y2, x1:x2] = 255
    
    maskrect_nobg = np.zeros_like(image)
    maskrect_nobg[y1:y2, x1:x2] = 255
    
    outputs = {
        "Full": full,
        "Full_NoBg": full_nobg,
        "MaskSegm": masksegm,
        "MaskSegm_NoBg": masksegm_nobg,
        "MaskRect": maskrect,
        "MaskRect_NoBg": maskrect_nobg,
    }
    
    # Save all variants
    for k, img in outputs.items():
        out_dir = Path(output_root) / k
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}.png"
        Image.fromarray(img).save(out_path, compress_level=1)
    
    return {
        "image": stem,
        "mask_ratio": round(mask_ratio, 4),
        "bbox_ratio": round(bbox_ratio, 4)
    }


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


# ================================
# MAIN (UPDATED TO USE MODEL FACTORY)
# ================================
def main(image_dir, output_dir, model_name="mask2former_ade20k", 
         batch_size=4, fixed_size=None, resume=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load specified model
    model = get_model(model_name, device)
    
    # Use model's preferred size if not specified
    if fixed_size is None:
        fixed_size = model.get_fixed_size()
    
    # Resume logic
    csv_path = Path(output_dir) / "occlusion_qc.csv"
    processed = set()
    
    if resume and csv_path.exists():
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed.add(row["image"])
    
    # Startup message
    all_images = [
        p.stem for p in Path(image_dir).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]
    
    if resume:
        print(f"🔄 TRUE RESUME ENABLED")
        print(f"✅ Already completed: {len(processed)}")
        print(f"⏳ Remaining to process: {len(all_images) - len(processed)}")
    else:
        print(f"🆕 Fresh run (no resume)")
        print(f"⏳ Total images to process: {len(all_images)}")
    
    print(f"🤖 Using model: {model_name}")
    print(f"📐 Fixed size: {fixed_size}")
    
    # Dataset
    dataset = ImageDataset(image_dir, fixed_size, exclude_set=processed if resume else None)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fixed
    )
    
    # CSV write mode
    csv_mode = "a" if resume and csv_path.exists() else "w"
    
    with open(csv_path, csv_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "mask_ratio", "bbox_ratio"])
        if csv_mode == "w":
            writer.writeheader()
        
        for imgs_tensor, paths, sizes in tqdm(loader, total=len(loader), desc=f"Processing with {model_name}"):
            imgs_tensor = imgs_tensor.to(device, dtype=torch.float16)
            
            # Get person masks from model
            person_masks = model.predict(imgs_tensor)
            
            # Process each image
            for person_mask, size, p in zip(person_masks, sizes, paths):
                res = process_occlusions_and_save(person_mask, size, p, output_dir)
                if res:
                    writer.writerow(res)
    
    print("✅ Completed successfully.")


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
    parser.add_argument("--resume", action="store_true", help="Resume from previous run")
    
    args = parser.parse_args()
    
    fixed_size = tuple(args.fixed_size) if args.fixed_size else None
    
    main(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        fixed_size=fixed_size,
        resume=args.resume
    )

# python occlusion_test.py --image_dir "G:\Thesis\ImageRetrieval\Professions_125k_Cleaned" --output_dir test --model mask2former_coco --batch_size 8 --resume

# # Segmentation Models: Papers & Architecture Summary

# ## 🥇 Mask2Former COCO

# **Paper:** [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527)
# - **Authors:** Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, Rohit Girdhar
# - **Published:** NeurIPS 2022
# - **Meta AI Research (FAIR)**

# **Architecture Summary:**
# Mask2Former uses a **Transformer-based architecture** with masked attention for universal segmentation. The model consists of three main components: (1) a **pixel decoder** (using Swin Transformer backbone) that extracts multi-scale features, (2) a **Transformer decoder** with masked attention that attends only to predicted mask regions rather than all pixels (making it more efficient), and (3) dynamic **mask embeddings** that are learned per-instance. Unlike the original MaskFormer, it uses **masked attention** where each query only attends to the foreground regions predicted by the previous layer, significantly improving efficiency and quality. The architecture is "universal" because it can handle panoptic, instance, and semantic segmentation with the same model design. For COCO instance segmentation, it's trained to predict individual object instances with associated class labels and precise binary masks.

# **Key Innovation:** Masked attention mechanism that restricts attention to predicted mask regions, improving both speed and accuracy.

# ---

# ## 🥈 Mask R-CNN (via Detectron2)

# **Paper:** [Mask R-CNN](https://arxiv.org/abs/1703.06870)
# - **Authors:** Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick
# - **Published:** ICCV 2017
# - **Facebook AI Research (FAIR)**

# **Architecture Summary:**
# Mask R-CNN extends **Faster R-CNN** by adding a parallel branch for predicting segmentation masks alongside the existing branches for classification and bounding box regression. The architecture consists of: (1) a **backbone CNN** (typically ResNet-50 or ResNet-101 with FPN - Feature Pyramid Network) for feature extraction, (2) a **Region Proposal Network (RPN)** that generates candidate object bounding boxes, (3) **RoIAlign** (instead of RoIPool) which precisely aligns extracted features with the input without quantization, preserving spatial correspondence, and (4) a **mask branch** - a small FCN that predicts a binary mask for each RoI in parallel with the class and box outputs. The key insight is that the mask prediction is decoupled from class prediction, allowing the network to generate masks for each class independently. This simple yet effective design became the foundation for most modern instance segmentation systems.

# **Key Innovation:** Adding a mask prediction branch to Faster R-CNN with RoIAlign for pixel-accurate segmentation.

# ---

# ## 🥉 Lang-SAM (Language Segment-Anything)

# **Combined from two papers:**

# ### 1. Segment Anything Model (SAM)
# **Paper:** [Segment Anything](https://arxiv.org/abs/2304.02643)
# - **Authors:** Alexander Kirillov, Eric Mintun, Nikhila Ravi, et al.
# - **Published:** ICCV 2023
# - **Meta AI Research**

# ### 2. Grounding DINO
# **Paper:** [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499)
# - **Authors:** Shilong Liu, Zhaoyang Zeng, Tianhe Ren, et al.
# - **Published:** ECCV 2024
# - **IDEA Research**

# **Architecture Summary:**
# Lang-SAM is a **combination architecture** that merges two powerful models. First, **Grounding DINO** acts as the "detector" - it uses a **Transformer-based architecture** with text-image fusion to detect objects based on natural language queries (e.g., "person"). It combines DINO (self-distillation with no labels) with grounded pre-training, using a language backbone to encode text and a vision backbone for images, then fusing them through cross-modality attention. The detected bounding boxes are then passed to **SAM** as prompts. SAM itself consists of: (1) a heavy **image encoder** (ViT-H, ViT-L, or ViT-B) that processes the entire image once to create embeddings, (2) a lightweight **prompt encoder** that handles points, boxes, or text, and (3) a **mask decoder** that combines image embeddings with prompt embeddings to generate high-quality segmentation masks. SAM was trained on 11M images with 1.1B masks, making it extremely robust. The combination allows text-based ("person") zero-shot instance segmentation with exceptional mask quality.

# **Key Innovation:** Zero-shot segmentation via language prompts with state-of-the-art mask quality from massive-scale pretraining.

# ---

# ## 📉 Mask2Former ADE20K

# **Same architecture as Mask2Former COCO above**, but trained differently:

# **Paper:** [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527)
# - Same paper as Mask2Former COCO

# **Architecture Summary:**
# Identical transformer-based architecture as Mask2Former COCO (Swin backbone + masked attention decoder), but trained on the **ADE20K dataset** for **semantic segmentation** rather than instance segmentation. The key difference is the training objective: instead of predicting individual object instances, it predicts semantic classes for each pixel (150 classes in ADE20K, including "person" as class 12). This means multiple people in an image are treated as one semantic region rather than separate instances. The architecture handles this by having the queries predict class-specific segments rather than instance-specific segments. While technically the model can distinguish instances through different query outputs, the semantic training makes it less precise at instance separation compared to the COCO instance-trained version.

# **Key Difference from COCO version:** Semantic segmentation (one "person" class) vs. instance segmentation (separate person instances).

# ---

# ## 🚀 YOLACT

# **Paper:** [YOLACT: Real-time Instance Segmentation](https://arxiv.org/abs/1904.02689)
# - **Authors:** Daniel Bolya, Chong Zhou, Fanyi Xiao, Yong Jae Lee
# - **Published:** ICCV 2019
# - **UC Davis**

# **Follow-up:** [YOLACT++: Better Real-time Instance Segmentation](https://arxiv.org/abs/1912.06218) (2019)

# **Architecture Summary:**
# YOLACT (You Only Look At CoefficienTs) achieves real-time instance segmentation through a clever **decoupled approach**. Unlike Mask R-CNN which predicts masks for each RoI, YOLACT splits mask generation into two parallel tasks: (1) a **ProtoNet** branch generates a set of "prototype masks" (k=32 typically) - these are full-image-sized masks that represent basic shapes and patterns, and (2) a **prediction head** (built on top of RetinaNet/FPN) predicts class, box, and k **mask coefficients** for each detected object. The final instance mask is created by taking a **linear combination** of the prototype masks weighted by the predicted coefficients, followed by cropping with the bounding box. This assembly is performed by a simple matrix multiplication, making it extremely fast. The architecture uses **ResNet-101 with FPN** as the backbone. YOLACT++ adds deformable convolutions and better backbone features for improved accuracy while maintaining speed.

# **Key Innovation:** Decoupling mask prediction into prototype generation + coefficient prediction for real-time performance (30+ FPS).

# ---

# ## 📊 Architecture Comparison Table

# | Model | Architecture Type | Backbone | Key Component | Parameters | Year |
# |-------|------------------|----------|---------------|------------|------|
# | **Mask2Former COCO** | Transformer | Swin-L | Masked Attention Decoder | ~200M | 2022 |
# | **MaskRCNN** | CNN + RPN | ResNet-50/101 + FPN | RoIAlign + Mask Head | ~50M | 2017 |
# | **Lang-SAM** | Hybrid (DINO + SAM) | ViT-H/L/B | Text-Image Fusion + Prompt Decoder | ~640M-1.25B | 2023 |
# | **Mask2Former ADE20K** | Transformer | Swin-L | Masked Attention Decoder | ~200M | 2022 |
# | **YOLACT** | CNN + RetinaNet | ResNet-101 + FPN | ProtoNet + Coefficients | ~36M | 2019 |

# ---

# ## 🔍 Detailed Architecture Components

# ### Backbone Networks Used:
# - **ResNet-50/101:** Traditional CNN with residual connections (He et al., 2016)
# - **ResNet-101 + FPN:** ResNet with Feature Pyramid Network for multi-scale features (Lin et al., 2017)
# - **Swin Transformer:** Hierarchical vision transformer with shifted windows (Liu et al., 2021)
# - **ViT (Vision Transformer):** Pure transformer for vision (Dosovitskiy et al., 2021)

# ### Key Technical Components:

# **RoIAlign (Mask R-CNN):**
# - Precise spatial alignment of features without quantization
# - Paper: [Mask R-CNN](https://arxiv.org/abs/1703.06870)

# **Feature Pyramid Network (FPN):**
# - Multi-scale feature representation
# - Paper: [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)

# **Masked Attention:**
# - Attention restricted to predicted foreground regions
# - Paper: [Masked-attention Mask Transformer](https://arxiv.org/abs/2112.01527)

# **Swin Transformer:**
# - Hierarchical vision transformer with shifted windows
# - Paper: [Swin Transformer](https://arxiv.org/abs/2103.14030)

# ---

# ## 📚 Additional Reading

# ### Foundational Papers:

# **Faster R-CNN** (basis for Mask R-CNN):
# - Paper: [Faster R-CNN: Towards Real-Time Object Detection](https://arxiv.org/abs/1506.01497)
# - Authors: Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun
# - NeurIPS 2015

# **RetinaNet** (basis for YOLACT):
# - Paper: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
# - Authors: Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár
# - ICCV 2017

# **Vision Transformer (ViT)** (basis for SAM):
# - Paper: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
# - Authors: Alexey Dosovitskiy, et al.
# - ICLR 2021

# ### Datasets:

# **COCO Dataset:**
# - Paper: [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312)
# - Authors: Tsung-Yi Lin, et al.
# - ECCV 2014

# **ADE20K Dataset:**
# - Paper: [Scene Parsing through ADE20K Dataset](https://arxiv.org/abs/1608.05442)
# - Authors: Bolei Zhou, et al.
# - CVPR 2017

# ---

# ## 🔗 Official Implementation Links

# - **Mask2Former:** https://github.com/facebookresearch/Mask2Former
# - **Mask R-CNN (Detectron2):** https://github.com/facebookresearch/detectron2
# - **SAM:** https://github.com/facebookresearch/segment-anything
# - **Grounding DINO:** https://github.com/IDEA-Research/GroundingDINO
# - **Lang-SAM:** https://github.com/luca-medeiros/lang-segment-anything
# - **YOLACT:** https://github.com/dbolya/yolact

# ---

# ## 💡 Evolution Timeline
# ```
# 2015: Faster R-CNN (object detection)
#   ↓
# 2017: Mask R-CNN (+ segmentation masks)
#   ↓
# 2017: RetinaNet (focal loss for dense detection)
#   ↓
# 2019: YOLACT (real-time instance segmentation)
#   ↓
# 2021: Vision Transformer (transformer for vision)
#   ↓
# 2021: Swin Transformer (hierarchical ViT)
#   ↓
# 2022: Mask2Former (masked attention for universal segmentation)
#   ↓
# 2023: SAM (foundation model for segmentation)
#   ↓
# 2023: Grounding DINO (language-grounded detection)
#   ↓
# 2023: Lang-SAM (combining DINO + SAM)
# ```

# Each model builds on previous innovations, with modern transformers (Mask2Former, SAM) representing the current state-of-the-art, while classical CNNs (Mask R-CNN, YOLACT) remain practical choices for many applications.