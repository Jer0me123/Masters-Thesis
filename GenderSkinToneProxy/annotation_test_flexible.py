# # """
# # Gender & Skin Tone Annotation Pipeline (Auto-Detecting Architecture)
# # =====================================================================

# # This script auto-detects the model architecture from the checkpoint:
# # - Single-task vs multi-task (checks for bin_classifier)
# # - Number of MST classes (5, 10, or other)
# # - BatchNorm vs no BatchNorm

# # Results are saved in a JSONL manifest for downstream analysis.
# # """

# # import os
# # import glob
# # import json
# # import argparse
# # from tqdm import tqdm
# # from threading import Lock
# # from typing import Optional, Tuple, List, Dict, Any
# # import numpy as np

# # import torch
# # import torch.nn as nn
# # from PIL import Image
# # from torchvision import models, transforms
# # from skimage.color import rgb2lab


# # # ============================================================
# # # SKIN TONE BINNING
# # # ============================================================
# # def mst_to_bin(mst: int, num_mst_classes: int = 10) -> int:
# #     """
# #     Convert MST label to 3-bin grouping.
    
# #     For 10-class MST: Light/Mid/Dark (1-3 / 4-7 / 8-10)
# #     For 5-class MST: Light/Mid/Dark (1 / 2-3 / 4-5)
# #     """
# #     if num_mst_classes == 10:
# #         if mst <= 3:
# #             return 0  # Light
# #         elif mst <= 7:
# #             return 1  # Mid
# #         else:
# #             return 2  # Dark
# #     elif num_mst_classes == 5:
# #         if mst <= 1:
# #             return 0  # Light
# #         elif mst <= 3:
# #             return 1  # Mid
# #         else:
# #             return 2  # Dark
# #     else:
# #         # Generic fallback: split into thirds
# #         third = num_mst_classes // 3
# #         if mst <= third:
# #             return 0
# #         elif mst <= 2 * third:
# #             return 1
# #         else:
# #             return 2


# # def bin_to_name(bin_id: int, num_mst_classes: int = 10) -> str:
# #     """Convert bin ID to human-readable name"""
# #     if num_mst_classes == 10:
# #         bin_names = {0: "Light (1-3)", 1: "Mid (4-7)", 2: "Dark (8-10)"}
# #     elif num_mst_classes == 5:
# #         bin_names = {0: "Light (1)", 1: "Mid (2-3)", 2: "Dark (4-5)"}
# #     else:
# #         bin_names = {0: "Light", 1: "Mid", 2: "Dark"}
# #     return bin_names.get(int(bin_id), "Unknown")


# # # ============================================================
# # # ARCHITECTURE DETECTION
# # # ============================================================
# # def inspect_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
# #     """
# #     Inspect checkpoint to determine architecture parameters.
    
# #     Returns:
# #         dict with keys:
# #         - uses_bn: bool
# #         - has_bin_classifier: bool
# #         - num_mst_classes: int
# #         - num_bin_classes: int (if applicable)
# #     """
# #     print(f"[Inspector] Loading checkpoint from {checkpoint_path}")
# #     ckpt = torch.load(checkpoint_path, map_location="cpu")
    
# #     # Handle dict wrapper
# #     if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
# #         state_dict = ckpt["model_state_dict"]
# #     else:
# #         state_dict = ckpt
    
# #     # 1. Check for BatchNorm
# #     uses_bn = any(
# #         ("running_mean" in k or "running_var" in k)
# #         and k.startswith("features.")
# #         for k in state_dict.keys()
# #     )
    
# #     # 2. Check for bin_classifier
# #     has_bin_classifier = any(k.startswith("bin_classifier.") for k in state_dict.keys())
    
# #     # 3. Determine number of MST classes
# #     mst_weight_key = "mst_classifier.weight"
# #     if mst_weight_key not in state_dict:
# #         raise ValueError(f"Missing '{mst_weight_key}' in checkpoint!")
    
# #     num_mst_classes = state_dict[mst_weight_key].shape[0]
    
# #     # 4. Determine number of bin classes (if applicable)
# #     num_bin_classes = None
# #     if has_bin_classifier:
# #         bin_weight_key = "bin_classifier.weight"
# #         num_bin_classes = state_dict[bin_weight_key].shape[0]
    
# #     info = {
# #         "uses_bn": uses_bn,
# #         "has_bin_classifier": has_bin_classifier,
# #         "num_mst_classes": num_mst_classes,
# #         "num_bin_classes": num_bin_classes,
# #     }
    
# #     print(f"[Inspector] Architecture detected:")
# #     print(f"  - Backbone: {'VGG16_BN' if uses_bn else 'VGG16 (no BN)'}")
# #     print(f"  - MST classes: {num_mst_classes}")
# #     print(f"  - Multi-task: {'Yes' if has_bin_classifier else 'No (single-task)'}")
# #     if has_bin_classifier:
# #         print(f"  - Bin classes: {num_bin_classes}")
    
# #     return info


# # # ============================================================
# # # FLEXIBLE VGG16 MODEL
# # # ============================================================
# # class VGG16MSTFlexible(nn.Module):
# #     """
# #     Flexible VGG16-based MST model that adapts to checkpoint architecture.
# #     """
# #     def __init__(
# #         self,
# #         input_mode: str = "rgb",
# #         use_bn: bool = True,
# #         dropout_p: float = 0.5,
# #         num_mst_classes: int = 10,
# #         num_bin_classes: Optional[int] = None,
# #     ):
# #         super().__init__()
# #         self.input_mode = input_mode
# #         self.num_mst_classes = num_mst_classes
# #         self.num_bin_classes = num_bin_classes
# #         self.is_multitask = num_bin_classes is not None

# #         # Load base VGG16
# #         base = (
# #             models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
# #             if use_bn
# #             else models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
# #         )

# #         features = base.features

# #         # Adjust first conv for hybrid mode (6 channels)
# #         if input_mode == "hybrid":
# #             old_conv = features[0]
# #             assert isinstance(old_conv, nn.Conv2d)
# #             new_conv = nn.Conv2d(
# #                 in_channels=6,
# #                 out_channels=old_conv.out_channels,
# #                 kernel_size=old_conv.kernel_size,
# #                 stride=old_conv.stride,
# #                 padding=old_conv.padding,
# #                 bias=(old_conv.bias is not None),
# #             )
# #             with torch.no_grad():
# #                 new_conv.weight[:, :3, :, :] = old_conv.weight.clone()
# #                 new_conv.weight[:, 3:, :, :] = old_conv.weight.clone()
# #                 if old_conv.bias is not None:
# #                     new_conv.bias.copy_(old_conv.bias)
# #             features[0] = new_conv

# #         self.features = features

# #         # Infer flattened feature dimension
# #         with torch.no_grad():
# #             c = 6 if input_mode == "hybrid" else 3
# #             dummy = torch.zeros(1, c, 224, 224)
# #             feat = self.features(dummy)
# #             flat_dim = feat.view(1, -1).shape[1]

# #         self.feature_extractor = nn.Sequential(
# #             nn.Linear(flat_dim, 1024),
# #             nn.ReLU(inplace=True),
# #             nn.Dropout(dropout_p),
# #             nn.Linear(1024, 512),
# #             nn.ReLU(inplace=True),
# #             nn.Dropout(dropout_p),
# #         )

# #         self.mst_classifier = nn.Linear(512, num_mst_classes)
        
# #         if self.is_multitask:
# #             self.bin_classifier = nn.Linear(512, num_bin_classes)
# #             # Projection head (may exist in some checkpoints)
# #             self.projection = nn.Sequential(
# #                 nn.Linear(512, 256),
# #                 nn.ReLU(inplace=True),
# #                 nn.Linear(256, 128),
# #             )

# #     def forward(self, x):
# #         x = self.features(x)
# #         x = x.view(x.size(0), -1)
# #         feats = self.feature_extractor(x)
# #         mst_logits = self.mst_classifier(feats)
        
# #         if self.is_multitask:
# #             bin_logits = self.bin_classifier(feats)
# #             return mst_logits, bin_logits
# #         else:
# #             return mst_logits


# # # ============================================================
# # # IMAGE TRANSFORMS (INFERENCE)
# # # ============================================================
# # IMAGENET_MEAN = [0.485, 0.456, 0.406]
# # IMAGENET_STD = [0.229, 0.224, 0.225]


# # class RGBTransform:
# #     def __init__(self):
# #         self.transform = transforms.Compose([
# #             transforms.Resize((224, 224)),
# #             transforms.ToTensor(),
# #             transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
# #         ])

# #     def __call__(self, img_pil: Image.Image) -> torch.Tensor:
# #         return self.transform(img_pil)


# # class LABTransform:
# #     def __init__(self, lab_mean: np.ndarray, lab_std: np.ndarray):
# #         self.resize = transforms.Resize((224, 224))
# #         self.lab_mean = np.asarray(lab_mean, dtype=np.float32)
# #         self.lab_std = np.asarray(lab_std, dtype=np.float32)

# #     def __call__(self, img_pil: Image.Image) -> torch.Tensor:
# #         img = self.resize(img_pil)
# #         rgb = np.asarray(img).astype(np.float32) / 255.0
# #         lab = rgb2lab(rgb).astype(np.float32)
# #         lab_norm = (lab - self.lab_mean) / self.lab_std
# #         return torch.from_numpy(lab_norm.transpose(2, 0, 1)).float()


# # class HybridTransform:
# #     """Concatenate RGB (ImageNet norm) and LAB (dataset norm) -> [6, H, W]"""
# #     def __init__(self, lab_mean: np.ndarray, lab_std: np.ndarray):
# #         self.resize = transforms.Resize((224, 224))
# #         self.lab_mean = np.asarray(lab_mean, dtype=np.float32)
# #         self.lab_std = np.asarray(lab_std, dtype=np.float32)
# #         self.to_tensor = transforms.ToTensor()
# #         self.rgb_norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

# #     def __call__(self, img_pil: Image.Image) -> torch.Tensor:
# #         img = self.resize(img_pil)
# #         rgb_tensor = self.rgb_norm(self.to_tensor(img))
# #         rgb_np = np.asarray(img).astype(np.float32) / 255.0
# #         lab = rgb2lab(rgb_np).astype(np.float32)
# #         lab_norm = (lab - self.lab_mean) / self.lab_std
# #         lab_tensor = torch.from_numpy(lab_norm.transpose(2, 0, 1)).float()
# #         return torch.cat([rgb_tensor, lab_tensor], dim=0)


# # # ============================================================
# # # SKIN TONE PREDICTOR (FLEXIBLE)
# # # ============================================================
# # class SkinTonePredictor:
# #     """
# #     Auto-detecting skin tone predictor.
    
# #     Outputs:
# #       - mst_label: 1..N (where N is num_mst_classes)
# #       - bin_label: 0..2 (derived from MST or from bin_classifier if available)
# #     """

# #     def __init__(
# #         self,
# #         model_path: str,
# #         input_mode: str = "rgb",
# #         lab_mean: Optional[np.ndarray] = None,
# #         lab_std: Optional[np.ndarray] = None,
# #         device: str = "cuda",
# #         dropout_p: float = 0.5,
# #     ):
# #         self.device = torch.device(device if torch.cuda.is_available() else "cpu")
# #         self.input_mode = input_mode

# #         # Inspect checkpoint
# #         arch_info = inspect_checkpoint(model_path)
# #         self.uses_bn = arch_info["uses_bn"]
# #         self.has_bin_classifier = arch_info["has_bin_classifier"]
# #         self.num_mst_classes = arch_info["num_mst_classes"]
# #         self.num_bin_classes = arch_info["num_bin_classes"]

# #         print(f"\n[SkinTone] Building model architecture...")
# #         print(f"  - Input mode: {input_mode}")
# #         print(f"  - Device: {self.device}")

# #         # Build matching architecture
# #         self.model = VGG16MSTFlexible(
# #             input_mode=input_mode,
# #             use_bn=self.uses_bn,
# #             dropout_p=dropout_p,
# #             num_mst_classes=self.num_mst_classes,
# #             num_bin_classes=self.num_bin_classes,
# #         )

# #         # Load checkpoint
# #         ckpt = torch.load(model_path, map_location=self.device)
# #         if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
# #             state_dict = ckpt["model_state_dict"]
# #         else:
# #             state_dict = ckpt

# #         # Load with strict=False to handle optional projection head
# #         missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        
# #         # Only warn about truly problematic missing keys (not projection head)
# #         critical_missing = [k for k in missing if not k.startswith("projection.")]
# #         if critical_missing:
# #             print(f"[WARN] Missing critical keys: {critical_missing}")
# #         if unexpected:
# #             print(f"[WARN] Unexpected keys: {unexpected}")

# #         self.model.to(self.device)
# #         self.model.eval()
# #         print("[SkinTone] Model loaded successfully\n")

# #         # Transform
# #         if input_mode == "rgb":
# #             self.transform = RGBTransform()
# #         elif input_mode == "lab":
# #             if lab_mean is None or lab_std is None:
# #                 raise ValueError("LAB mode requires lab_mean and lab_std")
# #             self.transform = LABTransform(lab_mean, lab_std)
# #         elif input_mode == "hybrid":
# #             if lab_mean is None or lab_std is None:
# #                 raise ValueError("Hybrid mode requires lab_mean and lab_std")
# #             self.transform = HybridTransform(lab_mean, lab_std)
# #         else:
# #             raise ValueError(f"Unknown input_mode: {input_mode}")

# #     def predict(self, img_pil: Image.Image) -> Tuple[Optional[int], Optional[int]]:
# #         try:
# #             x = self.transform(img_pil).unsqueeze(0).to(self.device)
# #             with torch.no_grad():
# #                 output = self.model(x)
                
# #                 if self.has_bin_classifier:
# #                     mst_logits, bin_logits = output
# #                     bin_idx = int(bin_logits.argmax(dim=1).item())
# #                 else:
# #                     mst_logits = output
# #                     bin_idx = None

# #             mst_idx = int(mst_logits.argmax(dim=1).item())
# #             mst_label = mst_idx + 1  # Convert to 1-indexed

# #             # Derive bin if not from classifier
# #             if bin_idx is None:
# #                 bin_label = mst_to_bin(mst_label, self.num_mst_classes)
# #             else:
# #                 bin_label = bin_idx

# #             return mst_label, bin_label
            
# #         except Exception as e:
# #             print(f"[WARN] Skin tone prediction failed: {e}")
# #             return None, None

# #     def predict_batch(self, img_pils: List[Image.Image]) -> List[Tuple[Optional[int], Optional[int]]]:
# #         if not img_pils:
# #             return []

# #         try:
# #             tensors = [self.transform(img) for img in img_pils]
# #             batch = torch.stack(tensors).to(self.device)

# #             with torch.no_grad():
# #                 output = self.model(batch)
                
# #                 if self.has_bin_classifier:
# #                     mst_logits, bin_logits = output
# #                     bin_idxs = bin_logits.argmax(dim=1).cpu().numpy()
# #                 else:
# #                     mst_logits = output
# #                     bin_idxs = None

# #             mst_idxs = mst_logits.argmax(dim=1).cpu().numpy()

# #             out = []
# #             for i, mi in enumerate(mst_idxs):
# #                 mst_label = int(mi) + 1
                
# #                 if bin_idxs is None:
# #                     bin_label = mst_to_bin(mst_label, self.num_mst_classes)
# #                 else:
# #                     bin_label = int(bin_idxs[i])
                
# #                 out.append((mst_label, bin_label))
# #             return out
            
# #         except Exception as e:
# #             print(f"[WARN] Batch skin tone prediction failed: {e}")
# #             return [(None, None)] * len(img_pils)


# # # ============================================================
# # # GENDER MODEL
# # # ============================================================
# # class GenderPredictor:
# #     """Wrapper for prithivMLmods/Realistic-Gender-Classification"""

# #     def __init__(self, device: str = "cuda"):
# #         from transformers import AutoImageProcessor, AutoModelForImageClassification

# #         model_name = "prithivMLmods/Realistic-Gender-Classification"
# #         self.device = torch.device(device if torch.cuda.is_available() else "cpu")

# #         print(f"[Gender] Loading {model_name} on {self.device}...")
# #         self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
# #         self.model = AutoModelForImageClassification.from_pretrained(model_name)
# #         self.model = self.model.to(self.device)
# #         self.model.eval()
# #         print("[Gender] Model loaded successfully\n")

# #     def predict(self, img_pil: Image.Image) -> Tuple[Optional[str], Optional[float]]:
# #         try:
# #             inputs = self.processor(images=img_pil, return_tensors="pt")
# #             inputs = {k: v.to(self.device) for k, v in inputs.items()}

# #             with torch.no_grad():
# #                 outputs = self.model(**inputs)
# #                 probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
# #                 pred_class = probs.argmax(dim=1).item()
# #                 confidence = probs[0, pred_class].item()

# #             label = self.model.config.id2label[pred_class]
# #             gender = label.split()[0].capitalize()
# #             return gender, float(confidence)

# #         except Exception as e:
# #             print(f"[WARN] Gender prediction failed: {e}")
# #             return None, None

# #     def predict_batch(self, img_pils: List[Image.Image]) -> List[Tuple[Optional[str], Optional[float]]]:
# #         if not img_pils:
# #             return []

# #         try:
# #             inputs = self.processor(images=img_pils, return_tensors="pt")
# #             inputs = {k: v.to(self.device) for k, v in inputs.items()}

# #             with torch.no_grad():
# #                 outputs = self.model(**inputs)
# #                 probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
# #                 pred_classes = probs.argmax(dim=1).cpu().numpy()
# #                 confidences = probs.max(dim=1).values.cpu().numpy()

# #             out = []
# #             for c, conf in zip(pred_classes, confidences):
# #                 label = self.model.config.id2label[int(c)]
# #                 gender = label.split()[0].capitalize()
# #                 out.append((gender, float(conf)))
# #             return out

# #         except Exception as e:
# #             print(f"[WARN] Batch gender prediction failed: {e}")
# #             return [(None, None)] * len(img_pils)


# # # ============================================================
# # # ANNOTATION MANIFEST
# # # ============================================================
# # class AnnotationManifest:
# #     """JSONL manifest tracking completed annotations"""
    
# #     def __init__(self, path: str, flush_every: int = 128):
# #         self.path = path
# #         self.flush_every = flush_every
# #         self.lock = Lock()
# #         self.processed = set()
# #         self.buffer = []

# #         os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

# #         if os.path.exists(path):
# #             with open(path, "r", encoding="utf-8") as f:
# #                 for line in f:
# #                     record = json.loads(line)
# #                     self.processed.add(record["image"])
# #             print(f"[Manifest] Loaded {len(self.processed)} existing annotations")

# #     def is_processed(self, image: str) -> bool:
# #         return image in self.processed

# #     def record(
# #         self,
# #         image: str,
# #         gender: str,
# #         gender_conf: float,
# #         mst_label: int,
# #         bin_label: int,
# #         bin_name: str,
# #         num_mst_classes: int,
# #     ):
# #         with self.lock:
# #             record = {
# #                 "image": image,
# #                 "gender": gender,
# #                 "gender_confidence": gender_conf,
# #                 "mst_label": int(mst_label),
# #                 "mst_max": num_mst_classes,
# #                 "bin_label": int(bin_label),
# #                 "bin_name": bin_name,
# #             }
# #             self.buffer.append(record)
# #             self.processed.add(image)

# #             if len(self.buffer) >= self.flush_every:
# #                 self.flush()

# #     def flush(self):
# #         if not self.buffer:
# #             return
# #         with open(self.path, "a", encoding="utf-8") as f:
# #             for record in self.buffer:
# #                 f.write(json.dumps(record) + "\n")
# #         self.buffer.clear()


# # # ============================================================
# # # IMAGE DATASET
# # # ============================================================
# # class ImageDataset:
# #     """Enumerates images that need annotation"""

# #     def __init__(
# #         self,
# #         image_dir: str,
# #         manifest: AnnotationManifest,
# #         exclude_dirs: List[str],
# #         target_subdir: Optional[str] = None,
# #     ):
# #         self.image_dir = image_dir
# #         exclude_dirs = {d.lower() for d in exclude_dirs}
# #         self.paths = []

# #         if target_subdir:
# #             print(f"[Dataset] Looking for images in '{target_subdir}' subdirectories...")
# #             for root, _, files in os.walk(image_dir):
# #                 if os.path.basename(root).lower() == target_subdir.lower():
# #                     path_parts = set(os.path.normpath(root).lower().split(os.sep))
# #                     if path_parts & exclude_dirs:
# #                         continue

# #                     for fname in files:
# #                         if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
# #                             continue
# #                         abs_path = os.path.join(root, fname)
# #                         rel = os.path.relpath(abs_path, image_dir).replace("\\", "/")
# #                         if manifest.is_processed(rel):
# #                             continue
# #                         self.paths.append((abs_path, rel))
# #         else:
# #             for p in glob.glob(os.path.join(image_dir, "**", "*.*"), recursive=True):
# #                 if not p.lower().endswith((".jpg", ".jpeg", ".png")):
# #                     continue
# #                 if set(os.path.normpath(p).lower().split(os.sep)) & exclude_dirs:
# #                     continue

# #                 rel = os.path.relpath(p, image_dir).replace("\\", "/")
# #                 if manifest.is_processed(rel):
# #                     continue

# #                 self.paths.append((p, rel))

# #         print(f"[Dataset] Found {len(self.paths)} images to process\n")

# #     def __len__(self):
# #         return len(self.paths)

# #     def __getitem__(self, idx):
# #         return self.paths[idx]

# #     def get_batch(self, start_idx: int, batch_size: int):
# #         end_idx = min(start_idx + batch_size, len(self.paths))
# #         batch_paths = self.paths[start_idx:end_idx]

# #         images = []
# #         paths = []
# #         for abs_path, rel_path in batch_paths:
# #             try:
# #                 img_pil = Image.open(abs_path).convert("RGB")
# #                 images.append(img_pil)
# #                 paths.append((abs_path, rel_path))
# #             except Exception as e:
# #                 print(f"[WARN] Failed to load {abs_path}: {e}")
# #                 images.append(None)
# #                 paths.append((abs_path, rel_path))

# #         return images, paths


# # # ============================================================
# # # MAIN
# # # ============================================================
# # def main(args):
# #     manifest_path = os.path.join(args.output_dir, "annotations.jsonl")
# #     manifest = AnnotationManifest(manifest_path)

# #     dataset = ImageDataset(args.image_dir, manifest, args.exclude_dirs, args.target_subdir)
# #     if len(dataset) == 0:
# #         print("No images to process. Exiting.")
# #         return

# #     gender_model = GenderPredictor(device=args.device)

# #     # Load LAB stats if needed
# #     lab_mean = None
# #     lab_std = None
# #     if args.skin_input_mode in ["lab", "hybrid"]:
# #         if args.lab_stats_path is None:
# #             raise ValueError("LAB/Hybrid mode requires --lab_stats_path (.npz with keys 'mean' and 'std').")
# #         stats = np.load(args.lab_stats_path)
# #         if "mean" not in stats or "std" not in stats:
# #             raise ValueError("LAB stats file must contain 'mean' and 'std' arrays.")
# #         lab_mean = stats["mean"]
# #         lab_std = stats["std"]
# #         print(f"[LAB] Loaded statistics from {args.lab_stats_path}\n")

# #     skin_model = SkinTonePredictor(
# #         model_path=args.skin_model_path,
# #         input_mode=args.skin_input_mode,
# #         lab_mean=lab_mean,
# #         lab_std=lab_std,
# #         device=args.device,
# #         dropout_p=args.skin_dropout,
# #     )

# #     print(f"[Processing] Starting annotation of {len(dataset)} images...\n")

# #     for i in tqdm(range(0, len(dataset), args.batch_size), desc="Annotating"):
# #         images, paths = dataset.get_batch(i, args.batch_size)

# #         valid_images = []
# #         valid_paths = []
# #         for img, path in zip(images, paths):
# #             if img is not None:
# #                 valid_images.append(img)
# #                 valid_paths.append(path)

# #         if not valid_images:
# #             continue

# #         gender_results = gender_model.predict_batch(valid_images)
# #         skin_results = skin_model.predict_batch(valid_images)

# #         for (abs_path, rel_path), (gender, gender_conf), (mst_label, bin_label) in zip(
# #             valid_paths, gender_results, skin_results
# #         ):
# #             if gender is None or mst_label is None or bin_label is None:
# #                 print(f"[WARN] Skipping {rel_path} due to prediction failure")
# #                 continue

# #             manifest.record(
# #                 image=rel_path,
# #                 gender=gender,
# #                 gender_conf=gender_conf,
# #                 mst_label=int(mst_label),
# #                 bin_label=int(bin_label),
# #                 bin_name=bin_to_name(bin_label, skin_model.num_mst_classes),
# #                 num_mst_classes=skin_model.num_mst_classes,
# #             )

# #     manifest.flush()
# #     print(f"\n[Complete] Annotations saved to {manifest_path}")


# # # ============================================================
# # # CLI
# # # ============================================================
# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser(
# #         description="Annotate images with gender and skin tone predictions (auto-detecting model architecture)"
# #     )

# #     # Input/Output
# #     parser.add_argument("--image_dir", required=True, help="Directory containing images")
# #     parser.add_argument("--output_dir", required=True, help="Output directory for annotations")
# #     parser.add_argument("--exclude_dirs", nargs="+", default=[], help="Subdirectories to exclude")
# #     parser.add_argument("--target_subdir", type=str, help="Only process images under directories named this")

# #     # Skin tone model (auto-detecting)
# #     parser.add_argument("--skin_model_path", required=True, help="Path to model checkpoint (.pth)")
# #     parser.add_argument(
# #         "--skin_input_mode",
# #         choices=["rgb", "lab", "hybrid"],
# #         default="rgb",
# #         help="Input color space for skin tone model",
# #     )
# #     parser.add_argument("--lab_stats_path", help="Path to LAB statistics (.npz) if using lab/hybrid mode")

# #     # Inference parameters
# #     parser.add_argument(
# #         "--skin_dropout",
# #         type=float,
# #         default=0.5,
# #         help="Dropout used in feature_extractor (should match training)",
# #     )

# #     # Hardware
# #     parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
# #     parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")

# #     args = parser.parse_args()
# #     os.makedirs(args.output_dir, exist_ok=True)
# #     main(args)

# """
# Gender & Skin Tone Annotation Pipeline (RGB & LAB Support)
# ============================================================

# FIXED VERSION with full LAB color space support:
# - Auto-detects model architecture (single-task vs multi-task, num classes, BatchNorm)
# - Supports RGB, LAB, and Hybrid input modes
# - Loads LAB stats from JSON (training format) or NPZ (legacy format)
# - Properly initializes LAB first conv layer (adapts from ImageNet)

# Results are saved in a JSONL manifest for downstream analysis.
# """

# import os
# import glob
# import json
# import argparse
# from tqdm import tqdm
# from threading import Lock
# from typing import Optional, Tuple, List, Dict, Any
# from pathlib import Path
# import numpy as np

# import torch
# import torch.nn as nn
# from PIL import Image
# from torchvision import models, transforms
# from skimage.color import rgb2lab


# # ============================================================
# # SKIN TONE BINNING
# # ============================================================
# def mst_to_bin(mst: int, num_mst_classes: int = 10) -> int:
#     """
#     Convert MST label to 3-bin grouping.
    
#     For 10-class MST: Light/Mid/Dark (1-3 / 4-7 / 8-10)
#     For 5-class MST: Light/Mid/Dark (1 / 2-3 / 4-5)
#     For binary: Light/Dark (1-5 / 6-10 for 10-class, 0 / 1 for 2-class)
#     """
#     if num_mst_classes == 10:
#         if mst <= 3:
#             return 0  # Light
#         elif mst <= 7:
#             return 1  # Mid
#         else:
#             return 2  # Dark
#     elif num_mst_classes == 5:
#         if mst <= 1:
#             return 0  # Light
#         elif mst <= 3:
#             return 1  # Mid
#         else:
#             return 2  # Dark
#     elif num_mst_classes == 2:
#         # Binary classification
#         return mst - 1  # Assumes MST labels are 1 or 2
#     else:
#         # Generic fallback: split into thirds
#         third = num_mst_classes // 3
#         if mst <= third:
#             return 0
#         elif mst <= 2 * third:
#             return 1
#         else:
#             return 2


# def bin_to_name(bin_id: int, num_mst_classes: int = 10) -> str:
#     """Convert bin ID to human-readable name"""
#     if num_mst_classes == 10:
#         bin_names = {0: "Light (1-3)", 1: "Mid (4-7)", 2: "Dark (8-10)"}
#     elif num_mst_classes == 5:
#         bin_names = {0: "Light (1)", 1: "Mid (2-3)", 2: "Dark (4-5)"}
#     elif num_mst_classes == 2:
#         bin_names = {0: "Light", 1: "Dark"}
#     else:
#         bin_names = {0: "Light", 1: "Mid", 2: "Dark"}
#     return bin_names.get(int(bin_id), "Unknown")


# # ============================================================
# # LAB STATISTICS LOADING
# # ============================================================
# def load_lab_stats(stats_path: str) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Load LAB statistics from JSON or NPZ format.
    
#     JSON format (from training):
#         {
#             "lab_mean": [L, a, b],
#             "lab_std": [L, a, b],
#             ...
#         }
    
#     NPZ format (legacy):
#         {
#             "mean": np.array([L, a, b]),
#             "std": np.array([L, a, b])
#         }
    
#     Returns:
#         (lab_mean, lab_std) as numpy arrays
#     """
#     stats_path = Path(stats_path)
    
#     if not stats_path.exists():
#         raise FileNotFoundError(f"LAB stats file not found: {stats_path}")
    
#     # Try JSON first (training format)
#     if stats_path.suffix == '.json':
#         with open(stats_path, 'r') as f:
#             stats = json.load(f)
        
#         if "lab_mean" not in stats or "lab_std" not in stats:
#             raise ValueError("JSON file must contain 'lab_mean' and 'lab_std' keys")
        
#         lab_mean = np.array(stats["lab_mean"], dtype=np.float32)
#         lab_std = np.array(stats["lab_std"], dtype=np.float32)
        
#         print(f"[LAB Stats] Loaded from JSON: {stats_path}")
#         if "num_images" in stats:
#             print(f"[LAB Stats] Computed on {stats['num_images']} images")
#         if "include_augmentation" in stats:
#             print(f"[LAB Stats] With augmentation: {stats['include_augmentation']}")
    
#     # Try NPZ (legacy format)
#     elif stats_path.suffix == '.npz':
#         stats = np.load(stats_path)
        
#         if "mean" not in stats or "std" not in stats:
#             raise ValueError("NPZ file must contain 'mean' and 'std' arrays")
        
#         lab_mean = stats["mean"].astype(np.float32)
#         lab_std = stats["std"].astype(np.float32)
        
#         print(f"[LAB Stats] Loaded from NPZ: {stats_path}")
    
#     else:
#         raise ValueError(f"Unsupported LAB stats format: {stats_path.suffix} (use .json or .npz)")
    
#     print(f"[LAB Stats] LAB_MEAN = {lab_mean.tolist()}")
#     print(f"[LAB Stats] LAB_STD  = {lab_std.tolist()}\n")
    
#     return lab_mean, lab_std


# # ============================================================
# # ARCHITECTURE DETECTION
# # ============================================================
# def inspect_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
#     """
#     Inspect checkpoint to determine architecture parameters.
    
#     Returns:
#         dict with keys:
#         - uses_bn: bool
#         - has_bin_classifier: bool
#         - num_mst_classes: int
#         - num_bin_classes: int (if applicable)
#         - input_channels: int (3 for RGB/LAB, 6 for hybrid)
#     """
#     print(f"[Inspector] Loading checkpoint from {checkpoint_path}")
#     ckpt = torch.load(checkpoint_path, map_location="cpu")
    
#     # Handle dict wrapper
#     if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
#         state_dict = ckpt["model_state_dict"]
#     else:
#         state_dict = ckpt
    
#     # 1. Check for BatchNorm
#     uses_bn = any(
#         ("running_mean" in k or "running_var" in k)
#         and k.startswith("features.")
#         for k in state_dict.keys()
#     )
    
#     # 2. Check for bin_classifier
#     has_bin_classifier = any(k.startswith("bin_classifier.") for k in state_dict.keys())
    
#     # 3. Determine number of MST classes
#     mst_weight_key = "mst_classifier.weight"
#     if mst_weight_key not in state_dict:
#         raise ValueError(f"Missing '{mst_weight_key}' in checkpoint!")
    
#     num_mst_classes = state_dict[mst_weight_key].shape[0]
    
#     # 4. Determine number of bin classes (if applicable)
#     num_bin_classes = None
#     if has_bin_classifier:
#         bin_weight_key = "bin_classifier.weight"
#         num_bin_classes = state_dict[bin_weight_key].shape[0]
    
#     # 5. Determine input channels from first conv layer
#     first_conv_key = "features.0.weight"
#     if first_conv_key not in state_dict:
#         raise ValueError(f"Missing '{first_conv_key}' in checkpoint!")
    
#     input_channels = state_dict[first_conv_key].shape[1]  # (out, in, h, w)
    
#     info = {
#         "uses_bn": uses_bn,
#         "has_bin_classifier": has_bin_classifier,
#         "num_mst_classes": num_mst_classes,
#         "num_bin_classes": num_bin_classes,
#         "input_channels": input_channels,
#     }
    
#     print(f"[Inspector] Architecture detected:")
#     print(f"  - Backbone: {'VGG16_BN' if uses_bn else 'VGG16 (no BN)'}")
#     print(f"  - Input channels: {input_channels} ({'Hybrid' if input_channels == 6 else 'RGB/LAB'})")
#     print(f"  - MST classes: {num_mst_classes}")
#     print(f"  - Multi-task: {'Yes' if has_bin_classifier else 'No (single-task)'}")
#     if has_bin_classifier:
#         print(f"  - Bin classes: {num_bin_classes}")
    
#     return info


# # ============================================================
# # FLEXIBLE VGG16 MODEL (WITH LAB SUPPORT)
# # ============================================================
# class VGG16MSTFlexible(nn.Module):
#     """
#     Flexible VGG16-based MST model with full LAB support.
    
#     FIXED: Properly initializes LAB first conv layer by adapting ImageNet weights
#     """
#     def __init__(
#         self,
#         input_mode: str = "rgb",
#         use_bn: bool = True,
#         dropout_p: float = 0.5,
#         num_mst_classes: int = 10,
#         num_bin_classes: Optional[int] = None,
#     ):
#         super().__init__()
#         self.input_mode = input_mode
#         self.num_mst_classes = num_mst_classes
#         self.num_bin_classes = num_bin_classes
#         self.is_multitask = num_bin_classes is not None

#         # Load base VGG16
#         base = (
#             models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
#             if use_bn
#             else models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
#         )

#         features = base.features

#         # Adjust first conv for different input modes
#         if input_mode == "hybrid":
#             # Hybrid: 6 channels (RGB + LAB)
#             old_conv = features[0]
#             assert isinstance(old_conv, nn.Conv2d)
#             new_conv = nn.Conv2d(
#                 in_channels=6,
#                 out_channels=old_conv.out_channels,
#                 kernel_size=old_conv.kernel_size,
#                 stride=old_conv.stride,
#                 padding=old_conv.padding,
#                 bias=(old_conv.bias is not None),
#             )
#             with torch.no_grad():
#                 # Duplicate RGB weights for both RGB and LAB channels
#                 new_conv.weight[:, :3, :, :] = old_conv.weight.clone()
#                 new_conv.weight[:, 3:, :, :] = old_conv.weight.clone()
#                 if old_conv.bias is not None:
#                     new_conv.bias.copy_(old_conv.bias)
#             features[0] = new_conv
#             print("[Model] Hybrid mode: First conv initialized with duplicated ImageNet weights")
        
#         elif input_mode == "lab":
#             # LAB: 3 channels but different meaning than RGB
#             # Adapt ImageNet weights by averaging RGB channels
#             old_conv = features[0]
#             assert isinstance(old_conv, nn.Conv2d)
            
#             with torch.no_grad():
#                 # Average RGB weights to create LAB initialization
#                 avg_weight = old_conv.weight.mean(dim=1, keepdim=True)
                
#                 # L channel (lightness ~ grayscale) gets full averaged weight
#                 old_conv.weight[:, 0:1, :, :] = avg_weight.clone()
                
#                 # a, b channels (color) get half weight initially
#                 old_conv.weight[:, 1:2, :, :] = avg_weight.clone() * 0.5
#                 old_conv.weight[:, 2:3, :, :] = avg_weight.clone() * 0.5
            
#             print("[Model] LAB mode: First conv adapted from ImageNet RGB weights")

#         self.features = features

#         # Infer flattened feature dimension
#         with torch.no_grad():
#             c = 6 if input_mode == "hybrid" else 3
#             dummy = torch.zeros(1, c, 224, 224)
#             feat = self.features(dummy)
#             flat_dim = feat.view(1, -1).shape[1]

#         self.feature_extractor = nn.Sequential(
#             nn.Linear(flat_dim, 1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout_p),
#             nn.Linear(1024, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout_p),
#         )

#         self.mst_classifier = nn.Linear(512, num_mst_classes)
        
#         if self.is_multitask:
#             self.bin_classifier = nn.Linear(512, num_bin_classes)
#             # Projection head (may exist in some checkpoints)
#             self.projection = nn.Sequential(
#                 nn.Linear(512, 256),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(256, 128),
#             )

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         feats = self.feature_extractor(x)
#         mst_logits = self.mst_classifier(feats)
        
#         if self.is_multitask:
#             bin_logits = self.bin_classifier(feats)
#             return mst_logits, bin_logits
#         else:
#             return mst_logits


# # ============================================================
# # IMAGE TRANSFORMS (INFERENCE)
# # ============================================================
# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]


# class RGBTransform:
#     """Standard RGB transform with ImageNet normalization"""
#     def __init__(self):
#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
#         ])

#     def __call__(self, img_pil: Image.Image) -> torch.Tensor:
#         return self.transform(img_pil)


# class LABTransform:
#     """LAB color space transform with dataset-specific normalization"""
#     def __init__(self, lab_mean: np.ndarray, lab_std: np.ndarray):
#         self.resize = transforms.Resize((224, 224))
#         self.lab_mean = np.asarray(lab_mean, dtype=np.float32)
#         self.lab_std = np.asarray(lab_std, dtype=np.float32)

#     def __call__(self, img_pil: Image.Image) -> torch.Tensor:
#         img = self.resize(img_pil)
#         rgb = np.asarray(img).astype(np.float32) / 255.0
#         lab = rgb2lab(rgb).astype(np.float32)
#         lab_norm = (lab - self.lab_mean) / self.lab_std
#         return torch.from_numpy(lab_norm.transpose(2, 0, 1)).float()


# class HybridTransform:
#     """Concatenate RGB (ImageNet norm) and LAB (dataset norm) -> [6, H, W]"""
#     def __init__(self, lab_mean: np.ndarray, lab_std: np.ndarray):
#         self.resize = transforms.Resize((224, 224))
#         self.lab_mean = np.asarray(lab_mean, dtype=np.float32)
#         self.lab_std = np.asarray(lab_std, dtype=np.float32)
#         self.to_tensor = transforms.ToTensor()
#         self.rgb_norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

#     def __call__(self, img_pil: Image.Image) -> torch.Tensor:
#         img = self.resize(img_pil)
#         rgb_tensor = self.rgb_norm(self.to_tensor(img))
#         rgb_np = np.asarray(img).astype(np.float32) / 255.0
#         lab = rgb2lab(rgb_np).astype(np.float32)
#         lab_norm = (lab - self.lab_mean) / self.lab_std
#         lab_tensor = torch.from_numpy(lab_norm.transpose(2, 0, 1)).float()
#         return torch.cat([rgb_tensor, lab_tensor], dim=0)


# # ============================================================
# # SKIN TONE PREDICTOR (FLEXIBLE WITH LAB SUPPORT)
# # ============================================================
# class SkinTonePredictor:
#     """
#     Auto-detecting skin tone predictor with full RGB/LAB/Hybrid support.
    
#     FIXED: Properly handles LAB models with adapted ImageNet weights
    
#     Outputs:
#       - mst_label: 1..N (where N is num_mst_classes)
#       - bin_label: 0..2 (derived from MST or from bin_classifier if available)
#     """

#     def __init__(
#         self,
#         model_path: str,
#         input_mode: str = "rgb",
#         lab_mean: Optional[np.ndarray] = None,
#         lab_std: Optional[np.ndarray] = None,
#         device: str = "cuda",
#         dropout_p: float = 0.5,
#     ):
#         self.device = torch.device(device if torch.cuda.is_available() else "cpu")
#         self.input_mode = input_mode

#         # Inspect checkpoint
#         arch_info = inspect_checkpoint(model_path)
#         self.uses_bn = arch_info["uses_bn"]
#         self.has_bin_classifier = arch_info["has_bin_classifier"]
#         self.num_mst_classes = arch_info["num_mst_classes"]
#         self.num_bin_classes = arch_info["num_bin_classes"]
#         self.checkpoint_input_channels = arch_info["input_channels"]

#         # Validate input mode matches checkpoint
#         expected_channels = 6 if input_mode == "hybrid" else 3
#         if self.checkpoint_input_channels != expected_channels:
#             print(f"[WARN] Input mode '{input_mode}' expects {expected_channels} channels, "
#                   f"but checkpoint has {self.checkpoint_input_channels} channels")
#             if self.checkpoint_input_channels == 6 and input_mode in ["rgb", "lab"]:
#                 print("[WARN] Checkpoint is hybrid but you specified RGB/LAB - this will likely fail")
#             elif self.checkpoint_input_channels == 3 and input_mode == "hybrid":
#                 print("[WARN] Checkpoint is RGB/LAB but you specified hybrid - this will likely fail")

#         print(f"\n[SkinTone] Building model architecture...")
#         print(f"  - Input mode: {input_mode}")
#         print(f"  - Device: {self.device}")

#         # Build matching architecture
#         self.model = VGG16MSTFlexible(
#             input_mode=input_mode,
#             use_bn=self.uses_bn,
#             dropout_p=dropout_p,
#             num_mst_classes=self.num_mst_classes,
#             num_bin_classes=self.num_bin_classes,
#         )

#         # Load checkpoint
#         ckpt = torch.load(model_path, map_location=self.device)
#         if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
#             state_dict = ckpt["model_state_dict"]
#         else:
#             state_dict = ckpt

#         # Load with strict=False to handle optional projection head
#         missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        
#         # Only warn about truly problematic missing keys (not projection head)
#         critical_missing = [k for k in missing if not k.startswith("projection.")]
#         if critical_missing:
#             print(f"[WARN] Missing critical keys: {critical_missing}")
#         if unexpected:
#             print(f"[WARN] Unexpected keys: {unexpected}")

#         self.model.to(self.device)
#         self.model.eval()
#         print("[SkinTone] Model loaded successfully\n")

#         # Transform
#         if input_mode == "rgb":
#             self.transform = RGBTransform()
#         elif input_mode == "lab":
#             if lab_mean is None or lab_std is None:
#                 raise ValueError("LAB mode requires lab_mean and lab_std")
#             self.transform = LABTransform(lab_mean, lab_std)
#         elif input_mode == "hybrid":
#             if lab_mean is None or lab_std is None:
#                 raise ValueError("Hybrid mode requires lab_mean and lab_std")
#             self.transform = HybridTransform(lab_mean, lab_std)
#         else:
#             raise ValueError(f"Unknown input_mode: {input_mode}")

#     def predict(self, img_pil: Image.Image) -> Tuple[Optional[int], Optional[int]]:
#         """Predict single image"""
#         try:
#             x = self.transform(img_pil).unsqueeze(0).to(self.device)
#             with torch.no_grad():
#                 output = self.model(x)
                
#                 if self.has_bin_classifier:
#                     mst_logits, bin_logits = output
#                     bin_idx = int(bin_logits.argmax(dim=1).item())
#                 else:
#                     mst_logits = output
#                     bin_idx = None

#             mst_idx = int(mst_logits.argmax(dim=1).item())
#             mst_label = mst_idx + 1  # Convert to 1-indexed

#             # Derive bin if not from classifier
#             if bin_idx is None:
#                 bin_label = mst_to_bin(mst_label, self.num_mst_classes)
#             else:
#                 bin_label = bin_idx

#             return mst_label, bin_label
            
#         except Exception as e:
#             print(f"[WARN] Skin tone prediction failed: {e}")
#             return None, None

#     def predict_batch(self, img_pils: List[Image.Image]) -> List[Tuple[Optional[int], Optional[int]]]:
#         """Predict batch of images"""
#         if not img_pils:
#             return []

#         try:
#             tensors = [self.transform(img) for img in img_pils]
#             batch = torch.stack(tensors).to(self.device)

#             with torch.no_grad():
#                 output = self.model(batch)
                
#                 if self.has_bin_classifier:
#                     mst_logits, bin_logits = output
#                     bin_idxs = bin_logits.argmax(dim=1).cpu().numpy()
#                 else:
#                     mst_logits = output
#                     bin_idxs = None

#             mst_idxs = mst_logits.argmax(dim=1).cpu().numpy()

#             out = []
#             for i, mi in enumerate(mst_idxs):
#                 mst_label = int(mi) + 1
                
#                 if bin_idxs is None:
#                     bin_label = mst_to_bin(mst_label, self.num_mst_classes)
#                 else:
#                     bin_label = int(bin_idxs[i])
                
#                 out.append((mst_label, bin_label))
#             return out
            
#         except Exception as e:
#             print(f"[WARN] Batch skin tone prediction failed: {e}")
#             return [(None, None)] * len(img_pils)


# # ============================================================
# # GENDER MODEL
# # ============================================================
# class GenderPredictor:
#     """Wrapper for prithivMLmods/Realistic-Gender-Classification"""

#     def __init__(self, device: str = "cuda"):
#         from transformers import AutoImageProcessor, AutoModelForImageClassification

#         model_name = "prithivMLmods/Realistic-Gender-Classification"
#         self.device = torch.device(device if torch.cuda.is_available() else "cpu")

#         print(f"[Gender] Loading {model_name} on {self.device}...")
#         self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
#         self.model = AutoModelForImageClassification.from_pretrained(model_name)
#         self.model = self.model.to(self.device)
#         self.model.eval()
#         print("[Gender] Model loaded successfully\n")

#     def predict(self, img_pil: Image.Image) -> Tuple[Optional[str], Optional[float]]:
#         try:
#             inputs = self.processor(images=img_pil, return_tensors="pt")
#             inputs = {k: v.to(self.device) for k, v in inputs.items()}

#             with torch.no_grad():
#                 outputs = self.model(**inputs)
#                 probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
#                 pred_class = probs.argmax(dim=1).item()
#                 confidence = probs[0, pred_class].item()

#             label = self.model.config.id2label[pred_class]
#             gender = label.split()[0].capitalize()
#             return gender, float(confidence)

#         except Exception as e:
#             print(f"[WARN] Gender prediction failed: {e}")
#             return None, None

#     def predict_batch(self, img_pils: List[Image.Image]) -> List[Tuple[Optional[str], Optional[float]]]:
#         if not img_pils:
#             return []

#         try:
#             inputs = self.processor(images=img_pils, return_tensors="pt")
#             inputs = {k: v.to(self.device) for k, v in inputs.items()}

#             with torch.no_grad():
#                 outputs = self.model(**inputs)
#                 probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
#                 pred_classes = probs.argmax(dim=1).cpu().numpy()
#                 confidences = probs.max(dim=1).values.cpu().numpy()

#             out = []
#             for c, conf in zip(pred_classes, confidences):
#                 label = self.model.config.id2label[int(c)]
#                 gender = label.split()[0].capitalize()
#                 out.append((gender, float(conf)))
#             return out

#         except Exception as e:
#             print(f"[WARN] Batch gender prediction failed: {e}")
#             return [(None, None)] * len(img_pils)


# # ============================================================
# # ANNOTATION MANIFEST
# # ============================================================
# class AnnotationManifest:
#     """JSONL manifest tracking completed annotations"""
    
#     def __init__(self, path: str, flush_every: int = 128):
#         self.path = path
#         self.flush_every = flush_every
#         self.lock = Lock()
#         self.processed = set()
#         self.buffer = []

#         os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

#         if os.path.exists(path):
#             with open(path, "r", encoding="utf-8") as f:
#                 for line in f:
#                     record = json.loads(line)
#                     self.processed.add(record["image"])
#             print(f"[Manifest] Loaded {len(self.processed)} existing annotations")

#     def is_processed(self, image: str) -> bool:
#         return image in self.processed

#     def record(
#         self,
#         image: str,
#         gender: str,
#         gender_conf: float,
#         mst_label: int,
#         bin_label: int,
#         bin_name: str,
#         num_mst_classes: int,
#     ):
#         with self.lock:
#             record = {
#                 "image": image,
#                 "gender": gender,
#                 "gender_confidence": gender_conf,
#                 "mst_label": int(mst_label),
#                 "mst_max": num_mst_classes,
#                 "bin_label": int(bin_label),
#                 "bin_name": bin_name,
#             }
#             self.buffer.append(record)
#             self.processed.add(image)

#             if len(self.buffer) >= self.flush_every:
#                 self.flush()

#     def flush(self):
#         if not self.buffer:
#             return
#         with open(self.path, "a", encoding="utf-8") as f:
#             for record in self.buffer:
#                 f.write(json.dumps(record) + "\n")
#         self.buffer.clear()


# # ============================================================
# # IMAGE DATASET
# # ============================================================
# class ImageDataset:
#     """Enumerates images that need annotation"""

#     def __init__(
#         self,
#         image_dir: str,
#         manifest: AnnotationManifest,
#         exclude_dirs: List[str],
#         target_subdir: Optional[str] = None,
#     ):
#         self.image_dir = image_dir
#         exclude_dirs = {d.lower() for d in exclude_dirs}
#         self.paths = []

#         if target_subdir:
#             print(f"[Dataset] Looking for images in '{target_subdir}' subdirectories...")
#             for root, _, files in os.walk(image_dir):
#                 if os.path.basename(root).lower() == target_subdir.lower():
#                     path_parts = set(os.path.normpath(root).lower().split(os.sep))
#                     if path_parts & exclude_dirs:
#                         continue

#                     for fname in files:
#                         if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
#                             continue
#                         abs_path = os.path.join(root, fname)
#                         rel = os.path.relpath(abs_path, image_dir).replace("\\", "/")
#                         if manifest.is_processed(rel):
#                             continue
#                         self.paths.append((abs_path, rel))
#         else:
#             for p in glob.glob(os.path.join(image_dir, "**", "*.*"), recursive=True):
#                 if not p.lower().endswith((".jpg", ".jpeg", ".png")):
#                     continue
#                 if set(os.path.normpath(p).lower().split(os.sep)) & exclude_dirs:
#                     continue

#                 rel = os.path.relpath(p, image_dir).replace("\\", "/")
#                 if manifest.is_processed(rel):
#                     continue

#                 self.paths.append((p, rel))

#         print(f"[Dataset] Found {len(self.paths)} images to process\n")

#     def __len__(self):
#         return len(self.paths)

#     def __getitem__(self, idx):
#         return self.paths[idx]

#     def get_batch(self, start_idx: int, batch_size: int):
#         end_idx = min(start_idx + batch_size, len(self.paths))
#         batch_paths = self.paths[start_idx:end_idx]

#         images = []
#         paths = []
#         for abs_path, rel_path in batch_paths:
#             try:
#                 img_pil = Image.open(abs_path).convert("RGB")
#                 images.append(img_pil)
#                 paths.append((abs_path, rel_path))
#             except Exception as e:
#                 print(f"[WARN] Failed to load {abs_path}: {e}")
#                 images.append(None)
#                 paths.append((abs_path, rel_path))

#         return images, paths


# # ============================================================
# # MAIN
# # ============================================================
# def main(args):
#     manifest_path = os.path.join(args.output_dir, "annotations.jsonl")
#     manifest = AnnotationManifest(manifest_path)

#     dataset = ImageDataset(args.image_dir, manifest, args.exclude_dirs, args.target_subdir)
#     if len(dataset) == 0:
#         print("No images to process. Exiting.")
#         return

#     gender_model = GenderPredictor(device=args.device)

#     # Load LAB stats if needed
#     lab_mean = None
#     lab_std = None
#     if args.skin_input_mode in ["lab", "hybrid"]:
#         if args.lab_stats_path is None:
#             raise ValueError(
#                 f"LAB/Hybrid mode requires --lab_stats_path (JSON or NPZ file with LAB statistics)"
#             )
#         lab_mean, lab_std = load_lab_stats(args.lab_stats_path)

#     skin_model = SkinTonePredictor(
#         model_path=args.skin_model_path,
#         input_mode=args.skin_input_mode,
#         lab_mean=lab_mean,
#         lab_std=lab_std,
#         device=args.device,
#         dropout_p=args.skin_dropout,
#     )

#     print(f"[Processing] Starting annotation of {len(dataset)} images...\n")

#     for i in tqdm(range(0, len(dataset), args.batch_size), desc="Annotating"):
#         images, paths = dataset.get_batch(i, args.batch_size)

#         valid_images = []
#         valid_paths = []
#         for img, path in zip(images, paths):
#             if img is not None:
#                 valid_images.append(img)
#                 valid_paths.append(path)

#         if not valid_images:
#             continue

#         gender_results = gender_model.predict_batch(valid_images)
#         skin_results = skin_model.predict_batch(valid_images)

#         for (abs_path, rel_path), (gender, gender_conf), (mst_label, bin_label) in zip(
#             valid_paths, gender_results, skin_results
#         ):
#             if gender is None or mst_label is None or bin_label is None:
#                 print(f"[WARN] Skipping {rel_path} due to prediction failure")
#                 continue

#             manifest.record(
#                 image=rel_path,
#                 gender=gender,
#                 gender_conf=gender_conf,
#                 mst_label=int(mst_label),
#                 bin_label=int(bin_label),
#                 bin_name=bin_to_name(bin_label, skin_model.num_mst_classes),
#                 num_mst_classes=skin_model.num_mst_classes,
#             )

#     manifest.flush()
#     print(f"\n[Complete] Annotations saved to {manifest_path}")


# # ============================================================
# # CLI
# # ============================================================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Annotate images with gender and skin tone predictions (RGB/LAB/Hybrid support)"
#     )

#     # Input/Output
#     parser.add_argument("--image_dir", required=True, help="Directory containing images")
#     parser.add_argument("--output_dir", required=True, help="Output directory for annotations")
#     parser.add_argument("--exclude_dirs", nargs="+", default=[], help="Subdirectories to exclude")
#     parser.add_argument("--target_subdir", type=str, help="Only process images under directories named this")

#     # Skin tone model (auto-detecting)
#     parser.add_argument("--skin_model_path", required=True, help="Path to model checkpoint (.pth)")
#     parser.add_argument(
#         "--skin_input_mode",
#         choices=["rgb", "lab", "hybrid"],
#         default="rgb",
#         help="Input color space for skin tone model",
#     )
#     parser.add_argument(
#         "--lab_stats_path", 
#         help="Path to LAB statistics (JSON from training or NPZ legacy format)"
#     )

#     # Inference parameters
#     parser.add_argument(
#         "--skin_dropout",
#         type=float,
#         default=0.5,
#         help="Dropout used in feature_extractor (should match training)",
#     )

#     # Hardware
#     parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
#     parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")

#     args = parser.parse_args()
    
#     # Validation
#     if args.skin_input_mode in ["lab", "hybrid"] and args.lab_stats_path is None:
#         parser.error(f"--lab_stats_path is required when using --skin_input_mode={args.skin_input_mode}")
    
#     os.makedirs(args.output_dir, exist_ok=True)
#     main(args)


"""
Universal Gender & Skin Tone Annotation Pipeline
================================================

Fully compatible with:

- VGG16 / ResNet18
- RGB / LAB
- Classification / CORAL / Regression
- Your new training script

Outputs JSONL manifest for downstream analysis.
"""

import os
import glob
import json
import argparse
from tqdm import tqdm
from threading import Lock
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from skimage.color import rgb2lab


# ============================================================
# MST BINNING
# ============================================================

def mst_to_bin(mst: int) -> int:
    if mst <= 3:
        return 0
    elif mst <= 7:
        return 1
    else:
        return 2


def bin_to_name(bin_id: int) -> str:
    return {
        0: "Light (1-3)",
        1: "Mid (4-7)",
        2: "Dark (8-10)"
    }.get(bin_id, "Unknown")


# ============================================================
# LAB STATS LOADING
# ============================================================

def load_lab_stats(path: str):
    with open(path, "r") as f:
        stats = json.load(f)

    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)

    print(f"[LAB] Mean: {mean.tolist()}")
    print(f"[LAB] Std:  {std.tolist()}")
    return mean, std


# ============================================================
# UNIVERSAL MODEL
# ============================================================

class UniversalSkinModel(nn.Module):
    def __init__(self, arch, num_outputs, dropout=0.5):
        super().__init__()
        self.arch = arch

        if arch == "vgg16":
            base = models.vgg16(weights=None)
            self.features = base.features
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            in_features = 512 * 7 * 7

            self.classifier = nn.Sequential(
                nn.Linear(in_features, 4096),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(4096, num_outputs),
            )

        elif arch == "resnet18":
            base = models.resnet18(weights=None)
            in_features = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(in_features, num_outputs)

        else:
            raise ValueError("Unsupported architecture")

    def forward(self, x):
        if self.arch == "vgg16":
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            logits = self.classifier(x)
        else:
            x = self.backbone(x)
            x = self.dropout(x)
            logits = self.head(x)

        return logits


# ============================================================
# CHECKPOINT INSPECTION
# ============================================================

def inspect_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    keys = list(state_dict.keys())

    if any(k.startswith("backbone.") for k in keys):
        arch = "resnet18"
        num_outputs = state_dict["head.weight"].shape[0]
    else:
        arch = "vgg16"
        last = [k for k in keys if "classifier" in k and "weight" in k][-1]
        num_outputs = state_dict[last].shape[0]

    if num_outputs == 1:
        mode = "regression"
    else:
        mode = "classification"  # may be CORAL (handled later)

    print(f"[Inspector] Arch: {arch}")
    print(f"[Inspector] Outputs: {num_outputs}")
    print(f"[Inspector] Mode guess: {mode}")

    return arch, num_outputs, mode


# ============================================================
# TRANSFORMS
# ============================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RGBTransform:
    def __init__(self):
        self.t = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __call__(self, img):
        return self.t(img)


class LABTransform:
    def __init__(self, mean, std):
        self.resize = transforms.Resize((224, 224))
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = self.resize(img)
        rgb = np.asarray(img).astype(np.float32) / 255.0
        lab = rgb2lab(rgb).astype(np.float32)
        lab = (lab - self.mean) / self.std
        return torch.from_numpy(lab.transpose(2, 0, 1)).float()


# ============================================================
# SKIN PREDICTOR
# ============================================================

class SkinTonePredictor:

    def __init__(self, model_path, input_space, lab_mean=None, lab_std=None, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        arch, num_outputs, mode_guess = inspect_checkpoint(model_path)

        self.arch = arch
        self.num_outputs = num_outputs
        self.mode = mode_guess

        self.model = UniversalSkinModel(
            arch=arch,
            num_outputs=num_outputs
        )

        ckpt = torch.load(model_path, map_location=self.device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        if input_space == "rgb":
            self.transform = RGBTransform()
        else:
            self.transform = LABTransform(lab_mean, lab_std)

        self.input_space = input_space

    def decode(self, logits):

        # Regression
        if self.num_outputs == 1:
            values = logits.squeeze(1).cpu().numpy()
            return np.clip(np.round(values), 1, 10).astype(int)

        # Possible CORAL
        if self.num_outputs <= 9:
            probs = torch.sigmoid(logits)
            probs = torch.cummin(probs, dim=1)[0]
            preds = torch.sum(probs > 0.5, dim=1).cpu().numpy()
            return preds + 1

        # Standard classification
        preds = logits.argmax(dim=1).cpu().numpy()
        return preds + 1

    def predict_batch(self, imgs):
        tensors = [self.transform(img) for img in imgs]
        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            logits = self.model(batch)

        mst_labels = self.decode(logits)

        results = []
        for mst in mst_labels:
            bin_label = mst_to_bin(int(mst))
            results.append((int(mst), int(bin_label)))

        return results


# ============================================================
# GENDER MODEL
# ============================================================

class GenderPredictor:
    def __init__(self, device="cuda"):
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        model_name = "prithivMLmods/Realistic-Gender-Classification"
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def predict_batch(self, imgs):
        inputs = self.processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            classes = probs.argmax(dim=1).cpu().numpy()
            confs = probs.max(dim=1).values.cpu().numpy()

        results = []
        for c, conf in zip(classes, confs):
            label = self.model.config.id2label[int(c)]
            gender = label.split()[0].capitalize()
            results.append((gender, float(conf)))

        return results


# ============================================================
# MAIN
# ============================================================

def main(args):

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, "annotations.jsonl")

    gender_model = GenderPredictor(args.device)

    lab_mean = lab_std = None
    if args.input_space == "lab":
        lab_mean, lab_std = load_lab_stats(args.lab_stats_path)

    skin_model = SkinTonePredictor(
        model_path=args.skin_model_path,
        input_space=args.input_space,
        lab_mean=lab_mean,
        lab_std=lab_std,
        device=args.device,
    )

    images = glob.glob(os.path.join(args.image_dir, "**", "*.jpg"), recursive=True)

    print(f"[Found] {len(images)} images")

    with open(manifest_path, "w", encoding="utf-8") as f:

        for i in tqdm(range(0, len(images), args.batch_size)):
            batch_paths = images[i:i + args.batch_size]
            batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]

            gender_results = gender_model.predict_batch(batch_imgs)
            skin_results = skin_model.predict_batch(batch_imgs)

            for path, (gender, conf), (mst, bin_label) in zip(batch_paths, gender_results, skin_results):

                record = {
                    "image": os.path.relpath(path, args.image_dir),
                    "gender": gender,
                    "gender_confidence": conf,
                    "mst_label": mst,
                    # "bin_label": bin_label,
                    # "bin_name": bin_to_name(bin_label)
                }

                f.write(json.dumps(record) + "\n")

    print("\n[Complete] Annotations saved.")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--skin_model_path", required=True)
    parser.add_argument("--input_space", choices=["rgb", "lab"], default="rgb")
    parser.add_argument("--lab_stats_path")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    if args.input_space == "lab" and args.lab_stats_path is None:
        parser.error("--lab_stats_path required for LAB mode")

    main(args)

# python "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\GenderSkinToneProxy\annotation_test_flexible.py" ^
# --image_dir "G:\Thesis\MonkSkinTone_Dataset\Segmented_MSTE_BGFixed" ^
# --output_dir "F:\VGG_MST_Testing\Models\ResNet18_4CCoral_LAB_FixedBG\Segmented_MSTE_BGFixed" ^
# --skin_model_path "F:\VGG_MST_Testing\Models\ResNet18_4CCoral_LAB_FixedBG\best_model.pth" ^
# --input_space lab ^
# --lab_stats_path "F:\VGG_MST_Testing\Models\ResNet18_4CCoral_LAB_FixedBG\lab_statistics.json" ^
# --batch_size 32

# python "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\GenderSkinToneProxy\annotation_test_flexible.py" ^
# --image_dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2_BGFixed" ^
# --output_dir "F:\VGG_MST_Testing\Models\ResNet18_4CCoral_LAB_FixedBG\Segmented_CCV2_BGFixed" ^
# --skin_model_path "F:\VGG_MST_Testing\Models\ResNet18_4CCoral_LAB_FixedBG\best_model.pth" ^
# --input_space lab ^
# --lab_stats_path "F:\VGG_MST_Testing\Models\ResNet18_4CCoral_LAB_FixedBG\lab_statistics.json" ^
# --batch_size 32

# python "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\GenderSkinToneProxy\annotation_test_flexible.py" ^
# --image_dir "F:\Thesis\Segmented_FairFace_BGFixed" ^
# --output_dir "F:\VGG_MST_Testing\Models\ResNet18_4CCoral_LAB_FixedBG\Segmented_FairFace_BGFixed" ^
# --skin_model_path "F:\VGG_MST_Testing\Models\ResNet18_4CCoral_LAB_FixedBG\best_model.pth" ^
# --input_space lab ^
# --lab_stats_path "F:\VGG_MST_Testing\Models\ResNet18_4CCoral_LAB_FixedBG\lab_statistics.json" ^
# --batch_size 32