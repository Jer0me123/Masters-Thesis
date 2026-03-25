"""
Gender & Skin Tone Annotation Pipeline (Multi-Task MST + 3-bin)
==============================================================

This script processes images to extract:
- Gender classification (using prithivMLmods/Realistic-Gender-Classification)
- Skin tone prediction using your trained VGG16 multi-task model:
    - MST 10-class head (MST 1..10)
    - 3-bin head (Light/Mid/Dark)

Results are saved in a JSONL manifest for downstream analysis.

Key fix vs previous version:
- Loads the checkpoint into the SAME architecture used in training:
  VGG16MSTMultiTask(feature_extractor + mst_classifier + bin_classifier [+ projection])

Notes:
- Your training saved model.state_dict() to vgg16_mst_best.pth (best_path).
- Some runs might save a dict with "model_state_dict". We support both.
- We ALWAYS return mst_label in 1..10 and bin_label in 0..2.
"""

import os
import glob
import json
import argparse
from tqdm import tqdm
from threading import Lock
from typing import Optional, Tuple, List
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
    return bin_names.get(int(bin_id), "Unknown")


# ============================================================
# VGG16 MULTI-TASK MODEL (MATCHES TRAINING SCRIPT)
# ============================================================
class VGG16MSTMultiTask(nn.Module):
    """
    Matches training architecture:
    - base.features from VGG16 (BN if use_bn=True)
    - feature_extractor MLP: flat_dim -> 1024 -> 512
    - mst_classifier: 512 -> 10
    - bin_classifier: 512 -> 3
    - projection head exists in training; not required for inference, but we include it
      so the checkpoint loads with strict=True
    """
    def __init__(self, input_mode: str = "rgb", use_bn: bool = True, dropout_p: float = 0.5):
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

        # Infer flattened feature dimension
        with torch.no_grad():
            c = 6 if input_mode == "hybrid" else 3
            dummy = torch.zeros(1, c, 224, 224)
            feat = self.features(dummy)
            flat_dim = feat.view(1, -1).shape[1]
            print(f"[INFO] VGG feature dimension: {flat_dim}")

        self.feature_extractor = nn.Sequential(
            nn.Linear(flat_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
        )

        self.mst_classifier = nn.Linear(512, 10)
        self.bin_classifier = nn.Linear(512, 3)

        # Present in training checkpoint; include to load strictly
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        feats = self.feature_extractor(x)
        mst_logits = self.mst_classifier(feats)
        bin_logits = self.bin_classifier(feats)
        return mst_logits, bin_logits


# ============================================================
# IMAGE TRANSFORMS (INFERENCE)
# ============================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RGBTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __call__(self, img_pil: Image.Image) -> torch.Tensor:
        return self.transform(img_pil)


class LABTransform:
    def __init__(self, lab_mean: np.ndarray, lab_std: np.ndarray):
        self.resize = transforms.Resize((224, 224))
        self.lab_mean = np.asarray(lab_mean, dtype=np.float32)
        self.lab_std = np.asarray(lab_std, dtype=np.float32)

    def __call__(self, img_pil: Image.Image) -> torch.Tensor:
        img = self.resize(img_pil)
        rgb = np.asarray(img).astype(np.float32) / 255.0
        lab = rgb2lab(rgb).astype(np.float32)
        lab_norm = (lab - self.lab_mean) / self.lab_std
        return torch.from_numpy(lab_norm.transpose(2, 0, 1)).float()


class HybridTransform:
    """
    Concatenate RGB (ImageNet norm) and LAB (dataset norm) -> [6, H, W]
    """
    def __init__(self, lab_mean: np.ndarray, lab_std: np.ndarray):
        self.resize = transforms.Resize((224, 224))
        self.lab_mean = np.asarray(lab_mean, dtype=np.float32)
        self.lab_std = np.asarray(lab_std, dtype=np.float32)
        self.to_tensor = transforms.ToTensor()
        self.rgb_norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, img_pil: Image.Image) -> torch.Tensor:
        img = self.resize(img_pil)

        # RGB branch
        rgb_tensor = self.rgb_norm(self.to_tensor(img))  # [3,H,W]

        # LAB branch
        rgb_np = np.asarray(img).astype(np.float32) / 255.0
        lab = rgb2lab(rgb_np).astype(np.float32)
        lab_norm = (lab - self.lab_mean) / self.lab_std
        lab_tensor = torch.from_numpy(lab_norm.transpose(2, 0, 1)).float()

        return torch.cat([rgb_tensor, lab_tensor], dim=0)


# ============================================================
# SKIN TONE PREDICTOR (MULTI-TASK)
# ============================================================
class SkinTonePredictor:
    """
    Wrapper for your trained multi-task MST model.

    Outputs:
      - mst_label: 1..10 (from mst head argmax + 1)
      - bin_label: 0..2  (from bin head argmax)

    If you trained mst3-only models, this script is not for that checkpoint.
    This matches your posted training code (mst_classifier=10, bin_classifier=3).
    """

    def __init__(
        self,
        model_path: str,
        input_mode: str = "rgb",
        lab_mean: Optional[np.ndarray] = None,
        lab_std: Optional[np.ndarray] = None,
        device: str = "cuda",
        use_bn: bool = True,
        dropout_p: float = 0.5,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.input_mode = input_mode
        self.use_bn = use_bn

        print(f"[SkinTone] Loading multi-task model from {model_path}")
        print(f"[SkinTone] Input mode: {input_mode} | use_bn: {use_bn} | device: {self.device}")

        # # Build model architecture exactly like training
        # self.model = VGG16MSTMultiTask(
        #     input_mode=input_mode,
        #     use_bn=use_bn,
        #     dropout_p=dropout_p,
        # )

        # --------------------------------------------------
        # Infer BN vs non-BN from checkpoint
        # --------------------------------------------------
        def _checkpoint_uses_bn(state_dict):
            # BN layers have running_mean / running_var
            return any(
                ("running_mean" in k or "running_var" in k)
                and k.startswith("features.")
                for k in state_dict.keys()
            )

        ckpt = torch.load(model_path, map_location=self.device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        uses_bn = _checkpoint_uses_bn(state_dict)

        print(f"[SkinTone] Detected backbone: {'VGG16_BN' if uses_bn else 'VGG16 (no BN)'}")

        # --------------------------------------------------
        # Build EXACT matching architecture
        # --------------------------------------------------
        self.model = VGG16MSTMultiTask(
            input_mode=input_mode,
            use_bn=uses_bn,          # 🔥 THIS IS THE FIX
            dropout_p=dropout_p,
        )

        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()


        ckpt = torch.load(model_path, map_location=self.device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()
        print("[SkinTone] Model loaded successfully")

        # Transform
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

    def predict(self, img_pil: Image.Image) -> Tuple[Optional[int], Optional[int]]:
        try:
            x = self.transform(img_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                mst_logits, bin_logits = self.model(x)

            mst_idx = int(mst_logits.argmax(dim=1).item())  # 0..9
            bin_idx = int(bin_logits.argmax(dim=1).item())  # 0..2

            mst_label = mst_idx + 1  # 1..10
            bin_label = bin_idx       # 0..2

            return mst_label, bin_label
        except Exception as e:
            print(f"[WARN] Skin tone prediction failed: {e}")
            return None, None

    def predict_batch(self, img_pils: List[Image.Image]) -> List[Tuple[Optional[int], Optional[int]]]:
        if not img_pils:
            return []

        try:
            tensors = [self.transform(img) for img in img_pils]
            batch = torch.stack(tensors).to(self.device)

            with torch.no_grad():
                mst_logits, bin_logits = self.model(batch)

            mst_idxs = mst_logits.argmax(dim=1).cpu().numpy()  # 0..9
            bin_idxs = bin_logits.argmax(dim=1).cpu().numpy()  # 0..2

            out = []
            for mi, bi in zip(mst_idxs, bin_idxs):
                out.append((int(mi) + 1, int(bi)))
            return out
        except Exception as e:
            print(f"[WARN] Batch skin tone prediction failed: {e}")
            return [(None, None)] * len(img_pils)


# ============================================================
# GENDER MODEL
# ============================================================
class GenderPredictor:
    """
    Wrapper for prithivMLmods/Realistic-Gender-Classification
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
        try:
            inputs = self.processor(images=img_pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                pred_class = probs.argmax(dim=1).item()
                confidence = probs[0, pred_class].item()

            label = self.model.config.id2label[pred_class]
            gender = label.split()[0].capitalize()
            return gender, float(confidence)

        except Exception as e:
            print(f"[WARN] Gender prediction failed: {e}")
            return None, None

    def predict_batch(self, img_pils: List[Image.Image]) -> List[Tuple[Optional[str], Optional[float]]]:
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

            out = []
            for c, conf in zip(pred_classes, confidences):
                label = self.model.config.id2label[int(c)]
                gender = label.split()[0].capitalize()
                out.append((gender, float(conf)))
            return out

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

    def record(
        self,
        image: str,
        gender: str,
        gender_conf: float,
        mst_label: int,
        bin_label: int,
        bin_name: str,
    ):
        with self.lock:
            record = {
                "image": image,
                "gender": gender,
                "gender_confidence": gender_conf,
                "mst_label": int(mst_label),
                "bin_label": int(bin_label),
                "bin_name": bin_name,
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

    def __init__(
        self,
        image_dir: str,
        manifest: AnnotationManifest,
        exclude_dirs: List[str],
        target_subdir: Optional[str] = None,
    ):
        self.image_dir = image_dir
        exclude_dirs = {d.lower() for d in exclude_dirs}
        self.paths = []

        if target_subdir:
            print(f"[Dataset] Looking for images in '{target_subdir}' subdirectories...")
            for root, _, files in os.walk(image_dir):
                if os.path.basename(root).lower() == target_subdir.lower():
                    path_parts = set(os.path.normpath(root).lower().split(os.sep))
                    if path_parts & exclude_dirs:
                        continue

                    for fname in files:
                        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                            continue
                        abs_path = os.path.join(root, fname)
                        rel = os.path.relpath(abs_path, image_dir).replace("\\", "/")
                        if manifest.is_processed(rel):
                            continue
                        self.paths.append((abs_path, rel))
        else:
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
    manifest_path = os.path.join(args.output_dir, "annotations.jsonl")
    manifest = AnnotationManifest(manifest_path)

    dataset = ImageDataset(args.image_dir, manifest, args.exclude_dirs, args.target_subdir)
    if len(dataset) == 0:
        print("No images to process. Exiting.")
        return

    gender_model = GenderPredictor(device=args.device)

    # Load LAB stats if needed
    lab_mean = None
    lab_std = None
    if args.skin_input_mode in ["lab", "hybrid"]:
        if args.lab_stats_path is None:
            raise ValueError("LAB/Hybrid mode requires --lab_stats_path (.npz with keys 'mean' and 'std').")
        stats = np.load(args.lab_stats_path)
        if "mean" not in stats or "std" not in stats:
            raise ValueError("LAB stats file must contain 'mean' and 'std' arrays.")
        lab_mean = stats["mean"]
        lab_std = stats["std"]
        print(f"[LAB] Loaded statistics from {args.lab_stats_path}")

    skin_model = SkinTonePredictor(
        model_path=args.skin_model_path,
        input_mode=args.skin_input_mode,
        lab_mean=lab_mean,
        lab_std=lab_std,
        device=args.device,
        use_bn=not args.no_bn,       # default True (matches training --use-bn)
        dropout_p=args.skin_dropout, # should match training if you want strict parity
    )

    print(f"\n[Processing] Starting annotation of {len(dataset)} images...")

    for i in tqdm(range(0, len(dataset), args.batch_size), desc="Annotating"):
        images, paths = dataset.get_batch(i, args.batch_size)

        valid_images = []
        valid_paths = []
        for img, path in zip(images, paths):
            if img is not None:
                valid_images.append(img)
                valid_paths.append(path)

        if not valid_images:
            continue

        gender_results = gender_model.predict_batch(valid_images)
        skin_results = skin_model.predict_batch(valid_images)

        for (abs_path, rel_path), (gender, gender_conf), (mst_label, bin_label) in zip(
            valid_paths, gender_results, skin_results
        ):
            if gender is None or mst_label is None or bin_label is None:
                print(f"[WARN] Skipping {rel_path} due to prediction failure")
                continue

            # Optional consistency check: warn if bin head disagrees with mst->bin
            derived_bin = mst_to_bin(int(mst_label))
            if int(bin_label) != derived_bin:
                # Not fatal (bin head can disagree), but useful diagnostic
                print(f"[WARN] Bin mismatch for {rel_path}: bin_head={bin_label} vs mst_to_bin={derived_bin}")

            manifest.record(
                image=rel_path,
                gender=gender,
                gender_conf=gender_conf,
                mst_label=int(mst_label),
                bin_label=int(bin_label),
                bin_name=bin_to_name(bin_label),
            )

    manifest.flush()
    print(f"\n[Complete] Annotations saved to {manifest_path}")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate images with gender and skin tone predictions")

    # Input/Output
    parser.add_argument("--image_dir", required=True, help="Directory containing images")
    parser.add_argument("--output_dir", required=True, help="Output directory for annotations")
    parser.add_argument("--exclude_dirs", nargs="+", default=[], help="Subdirectories to exclude")
    parser.add_argument("--target_subdir", type=str, help="Only process images under directories named this")

    # Skin tone model (multi-task)
    parser.add_argument("--skin_model_path", required=True, help="Path to multi-task checkpoint (vgg16_mst_best.pth)")
    parser.add_argument("--skin_input_mode", choices=["rgb", "lab", "hybrid"], default="rgb",
                        help="Input color space for skin tone model")
    parser.add_argument("--lab_stats_path", help="Path to LAB statistics (.npz) if using lab/hybrid mode")

    # Inference architecture knobs (should match training)
    parser.add_argument("--no_bn", action="store_true",
                        help="Set if your checkpoint was trained WITHOUT batch norm (most likely you used BN)")
    parser.add_argument("--skin_dropout", type=float, default=0.5,
                        help="Dropout used in feature_extractor during training (default 0.5)")

    # Hardware
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)


# python "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\GenderSkinToneProxy\annotation_test_2.py" --image_dir "F:\Thesis\Segmented_FairFace" --output_dir "F:\Thesis\Segmented_FairFace_test2" --skin_model_path "G:\Thesis\CasualConversationv2_Dataset\Models\ImprovedTest_Balanced\vgg16_mst_best.pth" --skin_input_mode rgb

# python "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\GenderSkinToneProxy\annotation_test_2.py" --image_dir "F:\Thesis\Segmented_FairFace" --output_dir "F:\Thesis\Segmented_FairFace_test3" --skin_model_path "G:\Thesis\CasualConversationv2_Dataset\Models\BalancedV3_500\vgg16_mst_best.pth" --skin_input_mode rgb

# ===========================================================
# EXAMPLE USAGE
#
# RGB + BN checkpoint (your posted training run used --use-bn and input_mode rgb):
# python GenderSkinToneAnnotation.py ^
#   --image_dir "E:\ImageRetrieval\Professions_125k_ISCO_Aligned" ^
#   --output_dir "E:\ImageRetrieval\Professions_125k_ISCO_Aligned_Annotations" ^
#   --skin_model_path "G:\Thesis\CasualConversationv2_Dataset\Models\ImprovedTest_Balanced\vgg16_mst_best.pth" ^
#   --skin_input_mode rgb ^
#   --batch_size 32
#
# LAB mode:
# python GenderSkinToneAnnotation.py ^
#   --image_dir "..." ^
#   --output_dir "..." ^
#   --skin_model_path "...\vgg16_mst_best.pth" ^
#   --skin_input_mode lab ^
#   --lab_stats_path "...\lab_stats.npz"
#
# HYBRID mode:
# python GenderSkinToneAnnotation.py ^
#   --image_dir "..." ^
#   --output_dir "..." ^
#   --skin_model_path "...\vgg16_mst_best.pth" ^
#   --skin_input_mode hybrid ^
#   --lab_stats_path "...\lab_stats.npz"
