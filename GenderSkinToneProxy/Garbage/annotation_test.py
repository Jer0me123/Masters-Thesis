"""
Gender & Skin Tone Annotation Pipeline
======================================

This script processes images to extract:
- Gender classification (using RealisticGenderClassifier)
- Skin tone prediction (using VGG16 MST models)

Results are saved in a JSONL manifest for downstream analysis.

The implementation follows the style and resumability patterns of
PixelPatchShufflingMeanRGB.py for large-scale processing.
"""

import os
import glob
import json
import argparse
from tqdm import tqdm
from threading import Lock
from typing import Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from skimage.color import rgb2lab


# ============================================================
# SKIN TONE BINNING
# ============================================================
def mst_to_bin(mst: int) -> int:
    """3-bin grouping: Light/Mid/Dark (1-3 / 4-7 / 8-10)"""
    if mst <= 3:
        return 0  # Light
    elif mst <= 7:
        return 1  # Mid
    else:
        return 2  # Dark


def bin_to_name(bin_id: int) -> str:
    """Convert bin ID to human-readable name"""
    bin_names = {0: "Light (1-3)", 1: "Mid (4-7)", 2: "Dark (8-10)"}
    return bin_names.get(bin_id, "Unknown")


# ============================================================
# VGG16 MST Classifier & Regressor
# ============================================================

class VGG16MSTClassifier(nn.Module):
    def __init__(self, input_mode="rgb", use_bn=True, num_classes=10, dropout_p=0.5):
        super().__init__()

        self.input_mode = input_mode

        base = (
            models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
            if use_bn
            else models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        )

        features = base.features

        # Adjust first conv for hybrid mode (6 channels)
        if input_mode == "hybrid":
            old_conv = features[0]
            assert isinstance(old_conv, nn.Conv2d)
            new_conv = nn.Conv2d(
                in_channels=6,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None),
            )

            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = old_conv.weight.clone()
                new_conv.weight[:, 3:, :, :] = old_conv.weight.clone()
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)

            features[0] = new_conv

        self.features = features

        # Freeze policy
        if input_mode == "rgb":
            for p in self.features.parameters():
                p.requires_grad = False
            for p in list(self.features.parameters())[-10:]:
                p.requires_grad = True
        elif input_mode == "lab":
            nn.init.kaiming_normal_(features[0].weight, mode='fan_out', nonlinearity='relu')
            if features[0].bias is not None:
                nn.init.zeros_(features[0].bias)

            for p in self.features.parameters():
                p.requires_grad = False

            for p in features[0].parameters():
                p.requires_grad = True

            for p in list(self.features.parameters())[-10:]:
                p.requires_grad = True
        else:
            for p in self.features.parameters():
                p.requires_grad = True

        # Infer flattened feature dimension
        with torch.no_grad():
            if input_mode == "hybrid":
                dummy = torch.zeros(1, 6, 224, 224)
            else:
                dummy = torch.zeros(1, 3, 224, 224)
            feat = self.features(dummy)
            flat_dim = feat.view(1, -1).shape[1]
            print(f"[INFO] VGG feature dimension: {flat_dim}")

        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(512, num_classes),
        )

        self._init_head_weights()

    def _init_head_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits

class VGG16MSTRegressor(nn.Module):
    def __init__(self, input_mode="rgb", use_bn=True, dropout_p=0.5):
        super().__init__()
        self.input_mode = input_mode

        base = (
            models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
            if use_bn
            else models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        )

        features = base.features

        if input_mode == "hybrid":
            old_conv = features[0]
            assert isinstance(old_conv, nn.Conv2d)
            new_conv = nn.Conv2d(
                in_channels=6,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None),
            )

            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = old_conv.weight.clone()
                new_conv.weight[:, 3:, :, :] = old_conv.weight.clone()
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)

            features[0] = new_conv

        self.features = features

        if input_mode == "rgb":
            for p in self.features.parameters():
                p.requires_grad = False
            for p in list(self.features.parameters())[-10:]:
                p.requires_grad = True
        elif input_mode == "lab":
            nn.init.kaiming_normal_(features[0].weight, mode='fan_out', nonlinearity='relu')
            if features[0].bias is not None:
                nn.init.zeros_(features[0].bias)
            
            for p in self.features.parameters():
                p.requires_grad = False
            for p in features[0].parameters():
                p.requires_grad = True
            for p in list(self.features.parameters())[-10:]:
                p.requires_grad = True
        else:
            for p in self.features.parameters():
                p.requires_grad = True

        with torch.no_grad():
            if input_mode == "hybrid":
                dummy = torch.zeros(1, 6, 224, 224)
            else:
                dummy = torch.zeros(1, 3, 224, 224)
            feat = self.features(dummy)
            flat_dim = feat.view(1, -1).shape[1]
            print(f"[INFO] VGG feature dimension: {flat_dim}")

        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

        self._init_head_weights()

    def _init_head_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x).squeeze(-1)


# ============================================================
# IMAGE TRANSFORMS
# ============================================================
class RGBTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __call__(self, img_pil):
        return self.transform(img_pil)


class LABTransform:
    def __init__(self, lab_mean, lab_std):
        self.resize = transforms.Resize((224, 224))
        self.lab_mean = lab_mean.astype(np.float32)
        self.lab_std = lab_std.astype(np.float32)
    
    def __call__(self, img_pil):
        img = self.resize(img_pil)
        rgb = np.asarray(img).astype(np.float32) / 255.0
        lab = rgb2lab(rgb).astype(np.float32)
        lab_norm = (lab - self.lab_mean) / self.lab_std
        return torch.from_numpy(lab_norm.transpose(2, 0, 1)).float()


class HybridTransform:
    def __init__(self, lab_mean, lab_std):
        self.resize = transforms.Resize((224, 224))
        self.lab_mean = lab_mean.astype(np.float32)
        self.lab_std = lab_std.astype(np.float32)
        self.rgb_norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __call__(self, img_pil):
        img = self.resize(img_pil)
        rgb_arr = np.asarray(img).astype(np.float32) / 255.0
        
        # RGB branch
        rgb_tensor = torch.from_numpy(rgb_arr.transpose(2, 0, 1)).float()
        rgb_tensor = self.rgb_norm(rgb_tensor)
        
        # LAB branch
        lab_arr = rgb2lab(rgb_arr).astype(np.float32)
        lab_norm = (lab_arr - self.lab_mean) / self.lab_std
        lab_tensor = torch.from_numpy(lab_norm.transpose(2, 0, 1)).float()
        
        return torch.cat([rgb_tensor, lab_tensor], dim=0)


# ============================================================
# SKIN TONE MODEL
# ============================================================
class SkinTonePredictor:
    """
    Wrapper for VGG16-based skin tone models.
    Automatically adapts to classifier head size from checkpoint.
    """

    def __init__(
        self,
        model_path: str,
        model_type: str = "classifier",
        output_mode: str = "unbinned",
        input_mode: str = "rgb",
        lab_mean: Optional[np.ndarray] = None,
        lab_std: Optional[np.ndarray] = None,
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.output_mode = output_mode
        self.input_mode = input_mode

        print(f"[SkinTone] Loading checkpoint: {model_path}")

        # --------------------------------------------------
        # Load checkpoint FIRST
        # --------------------------------------------------
        ckpt = torch.load(model_path, map_location="cpu")

        # --------------------------------------------------
        # Infer classifier output size
        # --------------------------------------------------
        if model_type == "classifier":
            head_weight = ckpt["classifier.6.weight"]
            num_classes = head_weight.shape[0]
            print(f"[SkinTone] Detected classifier with {num_classes} output classes")
        else:
            num_classes = None
            print(f"[SkinTone] Detected regressor model")

        # --------------------------------------------------
        # Build model with correct head size
        # --------------------------------------------------
        if model_type == "classifier":
            self.model = VGG16MSTClassifier(
                input_mode=input_mode,
                num_classes=num_classes
            )
        elif model_type == "regressor":
            self.model = VGG16MSTRegressor(input_mode=input_mode)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # --------------------------------------------------
        # Load weights (now compatible)
        # --------------------------------------------------
        self.model.load_state_dict(ckpt, strict=True)
        self.model.to(self.device)
        self.model.eval()

        # --------------------------------------------------
        # Setup transform
        # --------------------------------------------------
        if input_mode == "rgb":
            self.transform = RGBTransform()
        elif input_mode == "lab":
            if lab_mean is None or lab_std is None:
                raise ValueError("LAB mode requires lab_mean and lab_std")
            self.transform = LABTransform(lab_mean, lab_std)
        elif input_mode == "hybrid":
            if lab_mean is None or lab_std is None:
                raise ValueError("Hybrid mode requires lab_mean and lab_std")
            self.transform = HybridTransform(lab_mean, lab_std)
        else:
            raise ValueError(f"Unknown input_mode: {input_mode}")

        self.num_classes = num_classes
        print(f"[SkinTone] Model ready on {self.device}")

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------
    def predict(self, img_pil: Image.Image):
        try:
            x = self.transform(img_pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                if self.model_type == "classifier":
                    logits = self.model(x)
                    pred = logits.argmax(dim=1).item()

                    # -----------------------------
                    # Interpret output
                    # -----------------------------
                    if self.num_classes == 10:
                        mst_label = pred + 1
                        bin_label = mst_to_bin(mst_label)

                    elif self.num_classes == 3:
                        bin_label = pred
                        mst_label = pred  # symbolic

                    elif self.num_classes == 2:
                        # binary → map to Light / Dark
                        bin_label = 0 if pred == 0 else 2
                        mst_label = bin_label

                    else:
                        mst_label = pred
                        bin_label = None

                else:  # regressor
                    val = self.model(x).item()
                    mst_label = int(np.clip(round(val * 9 + 1), 1, 10))
                    bin_label = mst_to_bin(mst_label)

                return mst_label, bin_label

        except Exception as e:
            print(f"[WARN] Skin tone prediction failed: {e}")
            return None, None

    # --------------------------------------------------
    # Batch prediction
    # --------------------------------------------------
    def predict_batch(self, img_pils: list):
        if not img_pils:
            return []

        try:
            xs = torch.stack([self.transform(img) for img in img_pils]).to(self.device)

            with torch.no_grad():
                if self.model_type == "classifier":
                    preds = self.model(xs).argmax(dim=1).cpu().numpy()

                    results = []
                    for p in preds:
                        if self.num_classes == 10:
                            mst = p + 1
                            bin_ = mst_to_bin(mst)
                        elif self.num_classes == 3:
                            mst = p
                            bin_ = p
                        elif self.num_classes == 2:
                            mst = p
                            bin_ = 0 if p == 0 else 2
                        else:
                            mst = p
                            bin_ = None

                        results.append((mst, bin_))
                    return results

                else:
                    vals = self.model(xs).cpu().numpy()
                    results = []
                    for v in vals:
                        mst = int(np.clip(round(v * 9 + 1), 1, 10))
                        bin_ = mst_to_bin(mst)
                        results.append((mst, bin_))
                    return results

        except Exception as e:
            print(f"[WARN] Batch skin tone prediction failed: {e}")
            return [(None, None)] * len(img_pils)



# ============================================================
# GENDER MODEL
# ============================================================
class GenderPredictor:
    """
    Wrapper for RealisticGenderClassifier
    """
    
    def __init__(self, device: str = "cuda"):
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        
        model_name = "prithivMLmods/Realistic-Gender-Classification"
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        print(f"[Gender] Loading {model_name} on {self.device}...")
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"[Gender] Model loaded successfully")
    
    def predict(self, img_pil: Image.Image) -> Tuple[Optional[str], Optional[float]]:
        """
        Returns: (gender, confidence)
        """
        try:
            inputs = self.processor(images=img_pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                pred_class = probs.argmax(dim=1).item()
                confidence = probs[0, pred_class].item()
            
            # Extract gender from label (e.g., "male portrait" → "Male")
            label = self.model.config.id2label[pred_class]
            gender = label.split()[0].capitalize()
            
            return gender, confidence
        
        except Exception as e:
            print(f"[WARN] Gender prediction failed: {e}")
            return None, None
    
    def predict_batch(self, img_pils: list) -> list:
        """
        Batch prediction for multiple images
        Returns: list of (gender, confidence) tuples
        """
        if not img_pils:
            return []
        
        try:
            inputs = self.processor(images=img_pils, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                pred_classes = probs.argmax(dim=1).cpu().numpy()
                confidences = probs.max(dim=1).values.cpu().numpy()
            
            results = []
            for pred_class, conf in zip(pred_classes, confidences):
                label = self.model.config.id2label[int(pred_class)]
                gender = label.split()[0].capitalize()
                results.append((gender, float(conf)))
            
            return results
        
        except Exception as e:
            print(f"[WARN] Batch gender prediction failed: {e}")
            return [(None, None)] * len(img_pils)


# ============================================================
# ANNOTATION MANIFEST
# ============================================================
class AnnotationManifest:
    """
    JSONL manifest tracking completed annotations
    """
    
    def __init__(self, path: str, flush_every: int = 128):
        self.path = path
        self.flush_every = flush_every
        self.lock = Lock()
        self.processed = set()
        self.buffer = []
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    self.processed.add(record["image"])
            print(f"[Manifest] Loaded {len(self.processed)} existing annotations")
    
    def is_processed(self, image: str) -> bool:
        return image in self.processed
    
    # def record(self, image: str, gender: str, gender_conf: float,
    #            mst_label: int, bin_label: int, bin_name: str):
    #     with self.lock:
    #         record = {
    #             "image": image,
    #             "gender": gender,
    #             "gender_confidence": gender_conf,
    #             "mst_label": mst_label,
    #             "bin_label": bin_label,
    #             "bin_name": bin_name
    #         }
    #         self.buffer.append(record)
    #         self.processed.add(image)
            
    #         if len(self.buffer) >= self.flush_every:
    #             self.flush()

    def record(self, image: str, gender: str, gender_conf,
           mst_label, bin_label, bin_name):
        with self.lock:
            record = {
                "image": str(image),
                "gender": str(gender) if gender is not None else None,
                "gender_confidence": float(gender_conf) if gender_conf is not None else None,
                "mst_label": int(mst_label) if mst_label is not None else None,
                "bin_label": int(bin_label) if bin_label is not None else None,
                "bin_name": str(bin_name) if bin_name is not None else None,
            }

            self.buffer.append(record)
            self.processed.add(image)

            if len(self.buffer) >= self.flush_every:
                self.flush()

    
    def flush(self):
        if not self.buffer:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            for record in self.buffer:
                f.write(json.dumps(record) + "\n")
        self.buffer.clear()


# ============================================================
# IMAGE DATASET
# ============================================================
class ImageDataset:
    """
    Enumerates images that need annotation
    """
    
    def __init__(self, image_dir: str, manifest: AnnotationManifest, exclude_dirs: list, target_subdir: str = None):
        self.image_dir = image_dir
        exclude_dirs = {d.lower() for d in exclude_dirs}
        
        self.paths = []
        
        if target_subdir:
            # Process only images in specified subdirectory within each folder
            print(f"[Dataset] Looking for images in '{target_subdir}' subdirectories...")
            
            # Find all directories that match the target subdirectory pattern
            for root, dirs, files in os.walk(image_dir):
                # Check if current directory name matches target_subdir
                if os.path.basename(root).lower() == target_subdir.lower():
                    # Check if this directory should be excluded
                    path_parts = set(os.path.normpath(root).lower().split(os.sep))
                    if path_parts & exclude_dirs:
                        continue
                    
                    # Process all images in this directory (non-recursive)
                    for f in files:
                        if f.lower().endswith((".jpg", ".jpeg", ".png")):
                            abs_path = os.path.join(root, f)
                            rel = os.path.relpath(abs_path, image_dir).replace("\\", "/")
                            
                            if manifest.is_processed(rel):
                                continue
                            
                            self.paths.append((abs_path, rel))
        else:
            # Original behavior: process all images recursively
            for p in glob.glob(os.path.join(image_dir, "**", "*.*"), recursive=True):
                if not p.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                if set(os.path.normpath(p).lower().split(os.sep)) & exclude_dirs:
                    continue
                
                rel = os.path.relpath(p, image_dir).replace("\\", "/")
                if manifest.is_processed(rel):
                    continue
                
                self.paths.append((p, rel))
        
        print(f"[Dataset] Found {len(self.paths)} images to process")
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        return self.paths[idx]
    
    def get_batch(self, start_idx: int, batch_size: int):
        """
        Get a batch of images
        Returns: (images, paths) where images is list of PIL Images or None
        """
        end_idx = min(start_idx + batch_size, len(self.paths))
        batch_paths = self.paths[start_idx:end_idx]
        
        images = []
        paths = []
        
        for abs_path, rel_path in batch_paths:
            try:
                img_pil = Image.open(abs_path).convert("RGB")
                images.append(img_pil)
                paths.append((abs_path, rel_path))
            except Exception as e:
                print(f"[WARN] Failed to load {abs_path}: {e}")
                images.append(None)
                paths.append((abs_path, rel_path))
        
        return images, paths


# ============================================================
# MAIN
# ============================================================
def main(args):
    """
    Main annotation pipeline
    """
    
    # Initialize manifest
    manifest_path = os.path.join(args.output_dir, "annotations.jsonl")
    manifest = AnnotationManifest(manifest_path)
    
    # Build dataset
    dataset = ImageDataset(args.image_dir, manifest, args.exclude_dirs, args.target_subdir)
    
    if len(dataset) == 0:
        print("No images to process. Exiting.")
        return
    
    # Initialize models
    gender_model = GenderPredictor(device=args.device)
    
    # Load LAB statistics if needed
    lab_mean = None
    lab_std = None
    if args.skin_input_mode in ["lab", "hybrid"]:
        if args.lab_stats_path is None:
            raise ValueError("LAB/Hybrid mode requires --lab_stats_path")
        
        stats = np.load(args.lab_stats_path)
        lab_mean = stats["mean"]
        lab_std = stats["std"]
        print(f"[LAB] Loaded statistics from {args.lab_stats_path}")
    
    skin_model = SkinTonePredictor(
        model_path=args.skin_model_path,
        model_type=args.skin_model_type,
        output_mode=args.skin_output_mode,
        input_mode=args.skin_input_mode,
        lab_mean=lab_mean,
        lab_std=lab_std,
        device=args.device
    )
    
    # Process images
    print(f"\n[Processing] Starting annotation of {len(dataset)} images...")
    
    for i in tqdm(range(0, len(dataset), args.batch_size), desc="Annotating"):
        # Get batch of images
        images, paths = dataset.get_batch(i, args.batch_size)
        
        # Filter out failed loads
        valid_images = []
        valid_paths = []
        valid_indices = []
        
        for idx, (img, path) in enumerate(zip(images, paths)):
            if img is not None:
                valid_images.append(img)
                valid_paths.append(path)
                valid_indices.append(idx)
        
        if not valid_images:
            continue
        
        # Batch predict gender
        gender_results = gender_model.predict_batch(valid_images)
        
        # Batch predict skin tone
        skin_results = skin_model.predict_batch(valid_images)
        
        # Record results
        for (abs_path, rel_path), (gender, gender_conf), (mst_label, bin_label) in zip(
            valid_paths, gender_results, skin_results
        ):
            # Skip if predictions failed
            if gender is None or mst_label is None:
                print(f"[WARN] Skipping {rel_path} due to prediction failure")
                continue
            
            bin_name = bin_to_name(bin_label)
            
            # Record annotation
            manifest.record(
                image=rel_path,
                gender=gender,
                gender_conf=gender_conf,
                mst_label=mst_label,
                bin_label=bin_label,
                bin_name=bin_name
            )
    
    # Final flush
    manifest.flush()
    print(f"\n[Complete] Annotations saved to {manifest_path}")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Annotate images with gender and skin tone predictions"
    )
    
    # Input/Output
    parser.add_argument("--image_dir", required=True, help="Directory containing images")
    parser.add_argument("--output_dir", required=True, help="Output directory for annotations")
    parser.add_argument("--exclude_dirs", nargs="+", default=[], 
                       help="Subdirectories to exclude")
    parser.add_argument("--target_subdir", type=str, 
                       help="Subdirectories to process")
    
    # Skin tone model
    parser.add_argument("--skin_model_path", required=True, help="Path to skin tone model checkpoint")
    parser.add_argument("--skin_model_type", choices=["classifier", "regressor"], 
                       default="classifier", help="Type of skin tone model")
    parser.add_argument("--skin_output_mode", choices=["binned", "unbinned"], 
                       default="unbinned", 
                       help="binned: model outputs 3 bins directly | unbinned: model outputs MST 1-10")
    parser.add_argument("--skin_input_mode", choices=["rgb", "lab", "hybrid"], 
                       default="rgb", help="Input color space for skin tone model")
    parser.add_argument("--lab_stats_path", help="Path to LAB statistics (.npz file) if using lab/hybrid mode")
    
    # Hardware
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    
    args = parser.parse_args()
    main(args)


# ===========================================================
# EXAMPLE USAGE
# 
# For a classifier model outputting MST 1-10 (unbinned):
# python GenderSkinToneAnnotation.py \
#     --image_dir "path/to/images" \
#     --output_dir "path/to/output" \
#     --skin_model_path "models/vgg16_mst10_rgb.pth" \
#     --skin_model_type classifier \
#     --skin_output_mode unbinned \
#     --skin_input_mode rgb \
#     --batch_size 32
#
# For a classifier model directly outputting 3 bins:
# python GenderSkinToneAnnotation.py \
#     --image_dir "path/to/images" \
#     --output_dir "path/to/output" \
#     --skin_model_path "models/vgg16_mst3_hybrid.pth" \
#     --skin_model_type classifier \
#     --skin_output_mode binned \
#     --skin_input_mode hybrid \
#     --lab_stats_path "models/lab_stats.npz" \
#     --batch_size 16
#
# For a regressor model (always unbinned, always sigmoid output):
# python GenderSkinToneAnnotation.py \
#     --image_dir "path/to/images" \
#     --output_dir "path/to/output" \
#     --skin_model_path "models/vgg16_regressor.pth" \
#     --skin_model_type regressor \
#     --skin_output_mode unbinned \
#     --skin_input_mode lab \
#     --lab_stats_path "models/lab_stats.npz" \
#     --batch_size 64

# python GenderSkinToneAnnotation.py --image_dir "G:\Thesis\ImageRetrieval\Professions_125k_Cleaned" --output_dir "G:\Thesis\ImageRetrieval\test" --skin_model_path "G:\Thesis\CasualConversationv2_Dataset\Models\rgb\Model 1\vgg16_mst_best.pth" --skin_model_type classifier --skin_output_mode unbinned --skin_input_mode rgb


# python GenderSkinToneAnnotation.py --image_dir "E:\ImageRetrieval\Professions_125k_ISCO_Aligned" --output_dir "E:\ImageRetrieval\Professions_125k_ISCO_Aligned_Annotations" --skin_model_path "G:\Thesis\CasualConversationv2_Dataset\Models\rgb\Model 1\vgg16_mst_best.pth" --skin_model_type classifier --skin_output_mode unbinned --skin_input_mode rgb

# python GenderSkinToneAnnotation.py --image_dir "E:\ImageRetrieval\Professions_125k_ISCO_Aligned" --output_dir "E:\ImageRetrieval\Professions_125k_ISCO_Aligned_Annotations" --skin_model_path "G:\Thesis\CasualConversationv2_Dataset\Models\rgb\Model 1\vgg16_mst_best.pth" --skin_model_type classifier --skin_output_mode unbinned --skin_input_mode rgb

# python GenderSkinToneAnnotation.py --image_dir "E:\ImageRetrieval\StableDiffusionGeneratedImages\valid" --output_dir "E:\ImageRetrieval\StableDiffusionGeneratedImages_Annotations" --skin_model_path "G:\Thesis\CasualConversationv2_Dataset\Models\rgb\Model 1\vgg16_mst_best.pth" --skin_model_type classifier --skin_output_mode unbinned --skin_input_mode rgb