"""
Gender & Skin Tone Annotation Pipeline
======================================

This script processes images to extract:
- Gender classification (using RealisticGenderClassifier, always on RGB)
- Skin tone prediction (using VGG16 or ResNet18 MST models)

Model architectures match vgg16_mst_classification_regression_rgb_lab.py
exactly so that trained checkpoints load without modification.

Supported skin tone model combinations:
  --skin_arch  : vgg16 | resnet18
  --skin_mode  : classification | regression | coral
  --skin_input : rgb | lab
  --num_classes: output classes matching the label_mapping used at training
                 (e.g. 3, 4, 10 for classification; same value used for coral;
                  ignored for regression since output is always 1 scalar)

Color space note:
  When --skin_input lab is used, the skin model receives a LAB-normalised
  tensor while the gender model still receives the original RGB PIL image.
  Pass --lab_stats_path pointing to either a .npz (keys: "mean", "std") or
  a .json (keys: "mean", "std") file containing per-channel LAB statistics
  computed over the training dataset.

Results are saved in a JSONL manifest for downstream analysis.
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
import torch.nn.functional as F
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
# MODEL ARCHITECTURES
# These mirror vgg16_mst_classification_regression_rgb_lab.py
# exactly so that saved checkpoints load without any key mismatches.
# ============================================================

class VGG16MSTModel(nn.Module):
    """
    VGG16 with contrastive projection head.
    Supports classification, regression, and CORAL ordinal modes.
    Matches the training-script architecture exactly.
    """
    def __init__(self, num_outputs: int, mode: str = "classification",
                 dropout: float = 0.5, use_bn: bool = False,
                 pretrained: bool = False):
        super().__init__()
        self.mode = mode
        self.num_outputs = num_outputs

        if mode == "classification":
            out_dim = num_outputs
        elif mode == "coral":
            out_dim = num_outputs - 1  # K-1 thresholds
        else:  # regression
            out_dim = 1

        vgg16 = (
            models.vgg16_bn(weights=None)
            if use_bn
            else models.vgg16(weights=None)
        )
        self.features = vgg16.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # Projection head (contrastive learning; not used at inference)
        self.projection = nn.Sequential(
            nn.Linear(512 * 7 * 7, 2048),
            nn.ReLU(),
            nn.Linear(2048, 128),
        )

        if mode in ["classification", "coral"]:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(4096, out_dim),
            )
        else:  # regression
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(4096, 1024),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(1024, out_dim),
            )

    def forward(self, x, return_features: bool = False):
        x = self.features(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self.classifier(features)
        if self.mode == "regression":
            logits = torch.clamp(logits, 0.0, 10.0)
        if return_features:
            return logits, self.projection(features)
        return logits


class ResNet18MSTModel(nn.Module):
    """
    ResNet18 with optional contrastive projection head.
    Supports classification, regression, and CORAL ordinal modes.
    Matches the training-script architecture exactly.
    """
    def __init__(self, num_outputs: int, mode: str = "classification",
                 dropout: float = 0.5, pretrained: bool = False):
        super().__init__()
        self.mode = mode

        resnet = models.resnet18(weights=None)
        in_features = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.backbone = resnet

        self.projection = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )

        self.dropout = nn.Dropout(dropout)

        if mode == "classification":
            self.head = nn.Linear(in_features, num_outputs)
        elif mode == "coral":
            self.head = nn.Linear(in_features, num_outputs - 1)
        else:  # regression
            self.head = nn.Linear(in_features, 1)

    def forward(self, x, return_features: bool = False):
        features = self.backbone(x)
        features = self.dropout(features)
        logits = self.head(features)
        if self.mode == "regression":
            logits = torch.clamp(logits, 0.0, 10.0)
        if return_features:
            return logits, self.projection(features)
        return logits


def build_skintone_model(arch: str, mode: str, num_classes: int) -> nn.Module:
    """
    Factory: build the correct model architecture for inference.

    Args:
        arch       : "vgg16" or "resnet18"
        mode       : "classification", "regression", or "coral"
        num_classes: number of output classes used at training
                     (e.g. 3, 4, 10 for classification/coral;
                      ignored for regression — always 1 output)

    Note: dropout is not passed because model.eval() disables it entirely;
    the rate has no effect on inference and PyTorch loads the saved scalar
    from the checkpoint regardless of the initialised value.
    """
    if arch == "vgg16":
        return VGG16MSTModel(num_outputs=num_classes, mode=mode, pretrained=False)
    elif arch == "resnet18":
        return ResNet18MSTModel(num_outputs=num_classes, mode=mode, pretrained=False)
    else:
        raise ValueError(f"Unknown arch: {arch!r}. Choose 'vgg16' or 'resnet18'.")


# ============================================================
# IMAGE TRANSFORMS
# ============================================================
class RGBTransform:
    """Standard ImageNet preprocessing for RGB models."""
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __call__(self, img_pil: Image.Image) -> torch.Tensor:
        return self.transform(img_pil)


class LABTransform:
    """
    Converts an RGB PIL image to a normalised LAB tensor.

    lab_mean / lab_std should be numpy arrays of shape (3,) or (1,1,3)
    computed over the training dataset (matching the values used during
    training in vgg16_mst_classification_regression_rgb_lab.py).
    """
    def __init__(self, lab_mean: np.ndarray, lab_std: np.ndarray):
        self.resize = transforms.Resize((224, 224))
        self.lab_mean = np.array(lab_mean, dtype=np.float32).reshape(1, 1, 3)
        self.lab_std  = np.array(lab_std,  dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, img_pil: Image.Image) -> torch.Tensor:
        img = self.resize(img_pil)
        rgb = np.asarray(img).astype(np.float32) / 255.0
        lab = rgb2lab(rgb).astype(np.float32)           # H×W×3, L∈[0,100]
        lab_norm = (lab - self.lab_mean) / (self.lab_std + 1e-6)
        return torch.from_numpy(lab_norm.transpose(2, 0, 1)).float()


# ============================================================
# CHECKPOINT LOADING
# ============================================================
def _load_state_dict(path: str, device: torch.device) -> dict:
    """
    Load model weights from a checkpoint, handling two formats:
      1. Raw state dict  – saved as: torch.save(model.state_dict(), path)
      2. Full checkpoint – saved as: torch.save({'model_state_dict': ...,
                                                  'epoch': ..., ...}, path)
    Returns the raw OrderedDict of parameter tensors.
    """
    raw = torch.load(path, map_location=device, weights_only=False)

    if isinstance(raw, dict) and "model_state_dict" in raw:
        print(f"[Checkpoint] Detected full checkpoint format (epoch {raw.get('epoch', '?')})")
        return raw["model_state_dict"]

    # Assume it is already a plain state dict (OrderedDict)
    return raw



# ============================================================
# LABEL MAPPING
# ============================================================
def load_label_mapping(path: str):
    """
    Load a label_mapping_*.json produced by the training script.

    Expected format:
        {
            "label_mapping": {"1": 0, "2": 0, "3": 1, ...},
            "num_classes": 4
        }

    Returns
    -------
    class_to_msts : dict  {int class_idx → sorted list of MST ints}
    class_to_rep  : dict  {int class_idx → representative MST int (median)}
    mst_to_class  : dict  {int MST → int class_idx}
    num_classes   : int
    """
    with open(path, "r") as f:
        cfg = json.load(f)

    raw         = cfg["label_mapping"]            # {"1": 0, "2": 0, ...}
    num_classes = cfg.get("num_classes", max(int(v) for v in raw.values()) + 1)
    mst_to_class = {int(k): int(v) for k, v in raw.items()}

    missing = [m for m in range(1, 11) if m not in mst_to_class]
    if missing:
        raise ValueError(f"label_mapping missing MST values: {missing}")

    # Invert: class_idx → sorted list of MST values
    class_to_msts: dict = {}
    for mst, cls in mst_to_class.items():
        class_to_msts.setdefault(cls, []).append(mst)
    for cls in class_to_msts:
        class_to_msts[cls].sort()

    # Representative MST = median of the group (middle element)
    class_to_rep = {
        cls: msts[len(msts) // 2]
        for cls, msts in class_to_msts.items()
    }

    print(f"[LabelMapping] {num_classes} classes loaded from {path}")
    for cls in sorted(class_to_rep):
        msts = class_to_msts[cls]
        label = (f"MST {msts[0]}" if len(msts) == 1
                 else f"MST {msts[0]}-{msts[-1]}" if msts == list(range(msts[0], msts[-1]+1))
                 else f"MST {','.join(map(str, msts))}")
        print(f"  class {cls} → {label}  (representative MST {class_to_rep[cls]})")

    return class_to_msts, class_to_rep, mst_to_class, num_classes


# ============================================================
# SKIN TONE PREDICTOR
# ============================================================
class SkinTonePredictor:
    """
    Unified wrapper for all skin-tone model variants produced by
    vgg16_mst_classification_regression_rgb_lab.py.

    Supported combinations
    ──────────────────────
    arch   : "vgg16"  | "resnet18"
    mode   : "classification" | "regression" | "coral"
    input  : "rgb"    | "lab"

    Output
    ──────
    predict / predict_batch → (mst_label, bin_label)
      mst_label : predicted MST value (1-10) or bin index when output_mode="binned"
      bin_label : 3-bin index (0=Light, 1=Mid, 2=Dark), always computed
    """

    def __init__(
        self,
        model_path: str,
        arch: str = "vgg16",
        mode: str = "classification",
        num_classes: int = 10,
        output_mode: str = "unbinned",
        input_mode: str = "rgb",
        label_mapping_path: Optional[str] = None,
        lab_mean: Optional[np.ndarray] = None,
        lab_std: Optional[np.ndarray] = None,
        device: str = "cuda",
    ):
        """
        Args:
            model_path         : Path to .pth / .pt checkpoint file.
            arch               : "vgg16" or "resnet18".
            mode               : "classification", "regression", or "coral".
            num_classes        : Number of output classes used at training.
                                 For regression this is ignored (output is 1 scalar).
                                 For coral this is K (model has K-1 output logits).
            output_mode        : "binned"   → model output classes ARE the 3 bins (0-2)
                                 "unbinned" → model output is MST-aligned; converted to bins
            input_mode         : "rgb" or "lab".
            label_mapping_path : Optional path to label_mapping_*.json used at training.
                                 When provided, model class indices are mapped back to MST
                                 values using the same mapping, ensuring regression clamping
                                 and classification decoding are consistent with training.
                                 When omitted, falls back to simple +1 offset (assumes
                                 MST 1-10 identity mapping).
            lab_mean           : Per-channel LAB mean (shape 3).  Required for lab mode.
            lab_std            : Per-channel LAB std  (shape 3).  Required for lab mode.
            device             : "cuda" or "cpu".
        """
        self.device      = torch.device(device if torch.cuda.is_available() else "cpu")
        self.arch        = arch
        self.mode        = mode
        self.num_classes = num_classes
        self.output_mode = output_mode
        self.input_mode  = input_mode

        print(f"[SkinTone] arch={arch} | mode={mode} | classes={num_classes} "
              f"| input={input_mode} | output={output_mode}")
        print(f"[SkinTone] Loading checkpoint: {model_path}")

        # ── Build model ────────────────────────────────────────────────
        self.model = build_skintone_model(arch, mode, num_classes)
        state_dict = _load_state_dict(model_path, self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print(f"[SkinTone] Model ready on {self.device}")

        # ── Label mapping (optional) ───────────────────────────────────
        # When provided, overrides the fallback +1 offset decoding.
        if label_mapping_path is not None:
            (self._class_to_msts,
             self._class_to_rep,
             self._mst_to_class,
             self._map_num_classes) = load_label_mapping(label_mapping_path)
        else:
            self._class_to_msts  = None
            self._class_to_rep   = None
            self._mst_to_class   = None
            self._map_num_classes = num_classes

        # ── Image transform ────────────────────────────────────────────
        if input_mode == "rgb":
            self.transform = RGBTransform()
        elif input_mode == "lab":
            if lab_mean is None or lab_std is None:
                raise ValueError("--lab_stats_path is required when --skin_input lab")
            self.transform = LABTransform(lab_mean, lab_std)
        else:
            raise ValueError(f"Unknown input_mode: {input_mode!r}. Choose 'rgb' or 'lab'.")

    # ── Internal helpers ───────────────────────────────────────────────

    def _class_idx_to_mst(self, class_idx: int) -> int:
        """
        Convert a model class index to a representative MST value.

        With a label mapping loaded: looks up the median MST of the class group.
        Without a mapping (fallback): assumes identity mapping, class 0 → MST 1.
        In both cases the index is clamped to the valid class range first.
        """
        max_cls = self._map_num_classes - 1
        class_idx = int(np.clip(class_idx, 0, max_cls))
        if self._class_to_rep is not None:
            return self._class_to_rep[class_idx]
        # Fallback: 0-indexed class → MST (class 0 → MST 1)
        return class_idx + 1

    def _decode_logits(self, logits: torch.Tensor) -> list:
        """
        Convert raw model output to (mst_label, bin_label, pred_raw, pred_conf) per sample.

        classification : argmax → class index → MST via label mapping (or +1 fallback)
                         pred_raw  = raw class index from model (int, before mapping)
                         pred_conf = softmax probability of the predicted class
        regression     : float output rounded to nearest class index → MST via mapping
                         pred_raw  = raw float from model (before rounding/mapping)
                         pred_conf = None (regression has no probability estimate)
        coral          : thresholds exceeded → rank → MST via mapping
                         pred_raw  = predicted rank (int, before mapping)
                         pred_conf = None (no single probability for ordinal rank)

        When a label_mapping_path was provided, clamping uses the mapping's
        num_classes and the representative MST for each class is the median MST
        of all MST values that were assigned to that class at training time.
        """
        results = []

        if self.mode == "classification":
            probs        = torch.softmax(logits, dim=1).cpu().numpy()
            pred_classes = probs.argmax(axis=1)
            for pred, prob_row in zip(pred_classes, probs):
                pred_conf = float(prob_row[int(np.clip(pred, 0, self._map_num_classes - 1))])
                if self.output_mode == "binned":
                    # Model outputs bin indices directly; no MST mapping needed
                    bin_label = int(np.clip(pred, 0, self._map_num_classes - 1))
                    mst_label = bin_label
                else:
                    mst_label = self._class_idx_to_mst(pred)
                    bin_label = mst_to_bin(mst_label)
                results.append((mst_label, bin_label, int(pred), pred_conf))

        elif self.mode == "regression":
            # Model outputs a continuous value in [0, num_classes-1] (0-indexed).
            # Round to nearest class index, clamp to valid range, then map to MST.
            preds = logits.squeeze(-1).cpu().numpy()
            if preds.ndim == 0:          # single-sample edge case
                preds = preds.reshape(1)
            for val in preds:
                pred_raw  = float(val)
                class_idx = int(round(pred_raw))   # nearest class index (0-based)
                mst_label = self._class_idx_to_mst(class_idx)
                bin_label = mst_to_bin(mst_label)
                results.append((mst_label, bin_label, pred_raw, None))

        elif self.mode == "coral":
            # CORAL: predicted rank = number of thresholds exceeded (logit > 0)
            pred_ranks = (logits > 0).sum(dim=1).cpu().numpy()
            for rank in pred_ranks:
                if self.output_mode == "binned":
                    bin_label = int(np.clip(rank, 0, self._map_num_classes - 1))
                    mst_label = bin_label
                else:
                    mst_label = self._class_idx_to_mst(rank)
                    bin_label = mst_to_bin(mst_label)
                results.append((mst_label, bin_label, int(rank), None))

        else:
            raise ValueError(f"Unknown mode: {self.mode!r}")

        return results

    # ── Public API ─────────────────────────────────────────────────────

    def predict(self, img_pil: Image.Image) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[float]]:
        """Single-image prediction. Returns (mst_label, bin_label, pred_raw, pred_conf)."""
        try:
            tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
            return self._decode_logits(logits)[0]
        except Exception as e:
            print(f"[WARN] Skin tone prediction failed: {e}")
            return None, None, None, None

    def predict_batch(self, img_pils: list) -> list:
        """
        Batch prediction. Returns list of (mst_label, bin_label, pred_raw, pred_conf) tuples.
        """
        if not img_pils:
            return []
        try:
            tensors = [self.transform(img) for img in img_pils]
            batch   = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                logits = self.model(batch)
            return self._decode_logits(logits)
        except Exception as e:
            print(f"[WARN] Batch skin tone prediction failed: {e}")
            return [(None, None, None, None)] * len(img_pils)


# ============================================================
# GENDER MODEL  (always uses RGB regardless of skin input mode)
# ============================================================
class GenderPredictor:
    """
    Wrapper for RealisticGenderClassifier (HuggingFace).
    Always operates on the original RGB PIL image.
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
        print("[Gender] Model loaded successfully")

    def predict(self, img_pil: Image.Image) -> Tuple[Optional[str], Optional[float]]:
        """Returns (gender, confidence)."""
        try:
            inputs = self.processor(images=img_pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                pred_class = probs.argmax(dim=1).item()
                confidence = probs[0, pred_class].item()
            label  = self.model.config.id2label[pred_class]
            gender = label.split()[0].capitalize()
            return gender, confidence
        except Exception as e:
            print(f"[WARN] Gender prediction failed: {e}")
            return None, None

    def predict_batch(self, img_pils: list) -> list:
        """Batch prediction. Returns list of (gender, confidence) tuples."""
        if not img_pils:
            return []
        try:
            inputs = self.processor(images=img_pils, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs       = F.softmax(outputs.logits, dim=-1)
                pred_classes = probs.argmax(dim=1).cpu().numpy()
                confidences  = probs.max(dim=1).values.cpu().numpy()
            results = []
            for pred_class, conf in zip(pred_classes, confidences):
                label  = self.model.config.id2label[int(pred_class)]
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
    """JSONL manifest tracking completed annotations with resume support."""

    def __init__(self, path: str, flush_every: int = 128):
        self.path        = path
        self.flush_every = flush_every
        self.lock        = Lock()
        self.processed   = set()
        self.buffer      = []

        os.makedirs(os.path.dirname(path), exist_ok=True)

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    self.processed.add(record["image"])
            print(f"[Manifest] Loaded {len(self.processed)} existing annotations")

    def is_processed(self, image: str) -> bool:
        return image in self.processed

    def record(self, image: str, gender: str, gender_conf: float,
               mst_label: int, bin_label: int, bin_name: str,
               skin_raw: Optional[float], skin_conf: Optional[float]):
        with self.lock:
            entry = {
                "image":             image,
                "gender":            gender,
                "gender_confidence": gender_conf,
                "mst_label":         mst_label,
                "bin_label":         bin_label,
                "bin_name":          bin_name,
                "skin_raw_output":   round(skin_raw, 4) if skin_raw is not None else None,
                "skin_confidence":   round(skin_conf, 4) if skin_conf is not None else None,
            }
            self.buffer.append(entry)
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
    """Enumerates images that still need annotation (respects manifest)."""

    def __init__(self, image_dir: str, manifest: AnnotationManifest,
                 exclude_dirs: list, target_subdir: str = None):
        self.image_dir  = image_dir
        exclude_dirs    = {d.lower() for d in exclude_dirs}
        self.paths      = []

        if target_subdir:
            print(f"[Dataset] Looking for images in '{target_subdir}' subdirectories...")
            for root, dirs, files in os.walk(image_dir):
                if os.path.basename(root).lower() != target_subdir.lower():
                    continue
                path_parts = set(os.path.normpath(root).lower().split(os.sep))
                if path_parts & exclude_dirs:
                    continue
                for f in files:
                    if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    abs_path = os.path.join(root, f)
                    rel = os.path.relpath(abs_path, image_dir).replace("\\", "/")
                    if not manifest.is_processed(rel):
                        self.paths.append((abs_path, rel))
        else:
            for p in glob.glob(os.path.join(image_dir, "**", "*.*"), recursive=True):
                if not p.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                if set(os.path.normpath(p).lower().split(os.sep)) & exclude_dirs:
                    continue
                rel = os.path.relpath(p, image_dir).replace("\\", "/")
                if not manifest.is_processed(rel):
                    self.paths.append((p, rel))

        print(f"[Dataset] Found {len(self.paths)} images to process")

    def __len__(self):
        return len(self.paths)

    def get_batch(self, start_idx: int, batch_size: int):
        end_idx    = min(start_idx + batch_size, len(self.paths))
        batch_paths = self.paths[start_idx:end_idx]
        images, paths = [], []
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
# LAB STATS LOADING
# ============================================================
def load_lab_stats(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load per-channel LAB mean/std from either:
      .npz  – np.load(path) with keys "mean" and "std"
      .json – json.load(path) with keys "mean" and "std"
    Returns (mean, std) as float32 numpy arrays of shape (3,).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        stats = np.load(path)
        return stats["mean"].astype(np.float32), stats["std"].astype(np.float32)
    elif ext == ".json":
        with open(path) as f:
            stats = json.load(f)
        return (np.array(stats["mean"], dtype=np.float32),
                np.array(stats["std"],  dtype=np.float32))
    else:
        raise ValueError(f"Unsupported lab_stats format: {ext!r}. Use .npz or .json.")


# ============================================================
# MAIN
# ============================================================
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    manifest_path = os.path.join(args.output_dir, "annotations.jsonl")
    manifest      = AnnotationManifest(manifest_path)

    dataset = ImageDataset(
        args.image_dir, manifest, args.exclude_dirs, args.target_subdir
    )

    if len(dataset) == 0:
        print("No images to process. Exiting.")
        return

    # ── Gender model (always RGB) ──────────────────────────────────────
    gender_model = GenderPredictor(device=args.device)

    # ── LAB statistics (only needed for LAB input mode) ───────────────
    lab_mean = lab_std = None
    if args.skin_input == "lab":
        if args.lab_stats_path is None:
            raise ValueError("--lab_stats_path is required when --skin_input lab")
        lab_mean, lab_std = load_lab_stats(args.lab_stats_path)
        print(f"[LAB] mean={lab_mean.tolist()}  std={lab_std.tolist()}")

    # ── Skin tone model ────────────────────────────────────────────────
    skin_model = SkinTonePredictor(
        model_path          = args.skin_model_path,
        arch                = args.skin_arch,
        mode                = args.skin_mode,
        num_classes         = args.num_classes,
        output_mode         = args.skin_output_mode,
        input_mode          = args.skin_input,
        label_mapping_path  = args.label_mapping_path,
        lab_mean            = lab_mean,
        lab_std             = lab_std,
        device              = args.device,
    )

    # ── Processing loop ────────────────────────────────────────────────
    print(f"\n[Processing] Annotating {len(dataset)} images "
          f"(batch_size={args.batch_size})...")

    for i in tqdm(range(0, len(dataset), args.batch_size), desc="Annotating"):
        images, paths = dataset.get_batch(i, args.batch_size)

        # Filter failed loads
        valid_images  = []
        valid_paths   = []
        for img, path in zip(images, paths):
            if img is not None:
                valid_images.append(img)
                valid_paths.append(path)

        if not valid_images:
            continue

        # Gender always on original RGB PIL images
        gender_results = gender_model.predict_batch(valid_images)

        # Skin tone on original RGB PIL images; transform handles color space
        skin_results   = skin_model.predict_batch(valid_images)

        for (abs_path, rel_path), (gender, gender_conf), (mst_label, bin_label, skin_raw, skin_conf) in zip(
            valid_paths, gender_results, skin_results
        ):
            if gender is None or mst_label is None:
                print(f"[WARN] Skipping {rel_path} due to prediction failure")
                continue

            manifest.record(
                image       = rel_path,
                gender      = gender,
                gender_conf = gender_conf,
                mst_label   = mst_label,
                bin_label   = bin_label,
                bin_name    = bin_to_name(bin_label),
                skin_raw    = skin_raw,
                skin_conf   = skin_conf,
            )

    manifest.flush()
    print(f"\n[Complete] Annotations saved to {manifest_path}")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Annotate images with gender and skin tone predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Input / Output ─────────────────────────────────────────────────
    parser.add_argument("--image_dir",    required=True,
                        help="Root directory containing images")
    parser.add_argument("--output_dir",   required=True,
                        help="Output directory for annotations.jsonl")
    parser.add_argument("--exclude_dirs", nargs="+", default=[],
                        help="Subdirectory names to exclude from processing")
    parser.add_argument("--target_subdir", type=str, default=None,
                        help="Only process images inside subdirs with this name")

    # ── Skin tone model ────────────────────────────────────────────────
    parser.add_argument("--skin_model_path", required=True,
                        help="Path to the trained skin tone model checkpoint (.pt/.pth)")

    parser.add_argument("--skin_arch",
                        choices=["vgg16", "resnet18"], default="vgg16",
                        help="Backbone architecture used at training")

    parser.add_argument("--skin_mode",
                        choices=["classification", "regression", "coral"],
                        default="classification",
                        help="Training mode of the skin tone model")

    parser.add_argument("--num_classes", type=int, default=10,
                        help="Number of output classes used at training "
                             "(e.g. 3, 4, 10 for classification/coral; "
                             "ignored for regression)")

    parser.add_argument("--skin_output_mode",
                        choices=["binned", "unbinned"], default="unbinned",
                        help="binned: model outputs 3-bin indices directly | "
                             "unbinned: model outputs MST-aligned classes → mapped to bins")

    parser.add_argument("--skin_input",
                        choices=["rgb", "lab"], default="rgb",
                        help="Input colour space for the skin tone model. "
                             "Gender model always uses RGB regardless of this setting.")

    parser.add_argument("--lab_stats_path", default=None,
                        help="Path to LAB statistics file (.npz or .json with "
                             "'mean' and 'std' keys). Required when --skin_input lab.")

    parser.add_argument("--label_mapping_path", default=None,
                        help="Path to label_mapping_*.json used at training. "
                             "Maps model class indices back to MST values so that "
                             "decoding and clamping match the training configuration exactly. "
                             "When omitted, falls back to a simple +1 offset (assumes "
                             "class 0 = MST 1, class 1 = MST 2, etc.).")

    # ── Hardware ───────────────────────────────────────────────────────
    parser.add_argument("--device",     default="cuda",
                        help="Compute device ('cuda' or 'cpu')")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")

    args = parser.parse_args()
    main(args)


# ============================================================
# EXAMPLE USAGE
# ============================================================
#
# ── VGG16 · Classification · RGB (10-class MST) ──────────────────────
# python GenderSkinToneAnnotation.py \
#     --image_dir   "path/to/images" \
#     --output_dir  "path/to/output" \
#     --skin_model_path "models/vgg16_mst10_rgb_best.pt" \
#     --skin_arch   vgg16 \
#     --skin_mode   classification \
#     --num_classes 10 \
#     --skin_output_mode unbinned \
#     --skin_input  rgb \
#     --batch_size  32 \
#     --label_mapping_path "models/label_mapping_10class.json"
