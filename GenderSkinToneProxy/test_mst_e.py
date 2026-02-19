# """
# MST-E Dataset Evaluation Script with Comprehensive Diagnostics
# ================================================================

# Tests trained models on the Monk Skin Tone Examples (MST-E) dataset
# and provides detailed analysis of why generalization fails.

# Features:
# - Supports RGB, LAB, and Hybrid models
# - Per-class accuracy breakdown
# - Confusion matrix analysis
# - Distribution shift detection
# - Lighting condition analysis
# - Error visualization
# - Multi-model comparison
# - **NEW**: Domain alignment to training distribution

# Usage:
#     python test_mst_e.py \
#         --model_path "Models/LAB_4Class/vgg16_mst_best.pth" \
#         --mst_e_dir "path/to/MST-E" \
#         --label_mapping "label_mapping_4class.json" \
#         --input_mode lab \
#         --lab_stats_path "Models/LAB_4Class/lab_statistics.json" \
#         --align_to_training  # NEW: Normalize test data to training distribution
# """

# import argparse
# import json
# from pathlib import Path
# from collections import defaultdict, Counter
# from typing import Optional, Dict, List, Tuple

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from PIL import Image
# from torchvision import models, transforms
# from skimage.color import rgb2lab
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import seaborn as sns


# ##############################################################
# # LABEL MAPPER (FROM TRAINING SCRIPT)
# ##############################################################

# class LabelMapper:
#     """Handles flexible label mapping from MST values to class indices"""
#     def __init__(self, mapping_config=None):
#         if mapping_config is None:
#             self.mapping = {str(i): i-1 for i in range(1, 11)}
#             self.num_classes = 10
#             self.config_name = "mst10"
#         elif isinstance(mapping_config, str) or isinstance(mapping_config, Path):
#             with open(mapping_config, 'r') as f:
#                 config = json.load(f)
#             self._parse_config(config)
#             self.config_name = Path(mapping_config).stem
#         elif isinstance(mapping_config, dict):
#             self._parse_config(mapping_config)
#             self.config_name = "custom"
    
#     def _parse_config(self, config):
#         if "label_mapping" in config:
#             self.mapping = {str(k): int(v) for k, v in config["label_mapping"].items()}
#             self.num_classes = config.get("num_classes", max(self.mapping.values()) + 1)
#         elif "bins" in config:
#             self.mapping = {}
#             for bin_def in config["bins"]:
#                 mst_range = bin_def["range"]
#                 class_id = bin_def["class"]
#                 for mst in range(mst_range[0], mst_range[1] + 1):
#                     self.mapping[str(mst)] = class_id
#             self.num_classes = config.get("num_classes", max(self.mapping.values()) + 1)
        
#         for mst in range(1, 11):
#             if str(mst) not in self.mapping:
#                 raise ValueError(f"Mapping does not cover MST {mst}")
        
#         max_class = max(self.mapping.values())
#         if max_class >= self.num_classes:
#             raise ValueError(f"num_classes={self.num_classes} but mapping has class {max_class}")
    
#     def map_mst_to_class(self, mst_value):
#         return self.mapping[str(int(mst_value))]
    
#     def get_class_name(self, class_idx):
#         mst_values = [int(k) for k, v in self.mapping.items() if v == class_idx]
#         mst_values.sort()
        
#         if len(mst_values) == 0:
#             return f"Class{class_idx}"
#         elif len(mst_values) == 1:
#             return f"MST{mst_values[0]}"
#         else:
#             if mst_values == list(range(mst_values[0], mst_values[-1] + 1)):
#                 return f"MST{mst_values[0]}-{mst_values[-1]}"
#             else:
#                 return f"MST{','.join(map(str, mst_values))}"


# ##############################################################
# # MODEL DEFINITION (FROM TRAINING SCRIPT)
# ##############################################################

# ##############################################################
# # VGG16 MODEL (MATCHES TRAINING SCRIPT EXACTLY)
# ##############################################################

# class VGG16MSTModel(nn.Module):
#     def __init__(self, num_outputs, mode='classification',
#                  dropout=0.5, pretrained=True):
#         super().__init__()
#         self.mode = mode

#         vgg16 = models.vgg16(
#             weights=models.VGG16_Weights.IMAGENET1K_V1
#             if pretrained else None
#         )

#         self.features = vgg16.features
#         self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

#         self.projection = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 2048),
#             nn.ReLU(),
#             nn.Linear(2048, 128)
#         )

#         if mode == 'classification':
#             self.classifier = nn.Sequential(
#                 nn.Linear(512 * 7 * 7, 4096),
#                 nn.ReLU(True),
#                 nn.Dropout(dropout),
#                 nn.Linear(4096, 4096),
#                 nn.ReLU(True),
#                 nn.Dropout(dropout),
#                 nn.Linear(4096, num_outputs)
#             )
#         else:
#             self.classifier = nn.Sequential(
#                 nn.Linear(512 * 7 * 7, 4096),
#                 nn.ReLU(True),
#                 nn.Dropout(dropout),
#                 nn.Linear(4096, 1024),
#                 nn.ReLU(True),
#                 nn.Dropout(dropout),
#                 nn.Linear(1024, 1)
#             )

#     def forward(self, x, return_features=False):
#         x = self.features(x)
#         x = self.avgpool(x)
#         features = torch.flatten(x, 1)

#         logits = self.classifier(features)

#         if self.mode == 'regression':
#             logits = torch.clamp(logits, 1.0, 10.0)

#         if return_features:
#             proj = self.projection(features)
#             return logits, proj

#         return logits

# ##############################################################
# # RESNET18 MODEL (MATCHES TRAINING SCRIPT EXACTLY)
# ##############################################################

# class ResNet18MSTModel(nn.Module):
#     def __init__(self, num_outputs, mode='classification',
#                  dropout=0.5, pretrained=True):
#         super().__init__()
#         self.mode = mode

#         resnet = models.resnet18(
#             weights=models.ResNet18_Weights.IMAGENET1K_V1
#             if pretrained else None
#         )

#         in_features = resnet.fc.in_features
#         resnet.fc = nn.Identity()

#         self.backbone = resnet

#         self.projection = nn.Sequential(
#             nn.Linear(in_features, 512),
#             nn.ReLU(),
#             nn.Linear(512, 128)
#         )

#         self.dropout = nn.Dropout(dropout)

#         if mode == 'classification':
#             self.head = nn.Linear(in_features, num_outputs)
#         else:
#             self.head = nn.Linear(in_features, 1)

#     def forward(self, x, return_features=False):
#         features = self.backbone(x)
#         features = self.dropout(features)

#         logits = self.head(features)

#         if self.mode == 'regression':
#             logits = torch.clamp(logits, 1.0, 10.0)

#         if return_features:
#             proj = self.projection(features)
#             return logits, proj

#         return logits


# ##############################################################
# # TRANSFORMS
# ##############################################################

# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]


# class RGBTransform:
#     def __init__(self):
#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
#         ])
    
#     def __call__(self, img_pil):
#         return self.transform(img_pil)


# class LABTransform:
#     def __init__(self, lab_mean, lab_std, align_to_training=False, 
#                  test_mean=None, test_std=None):
#         """
#         Args:
#             lab_mean: Training set LAB mean
#             lab_std: Training set LAB std
#             align_to_training: If True, normalize test data to match training distribution
#             test_mean: Test set LAB mean (computed from first pass through data)
#             test_std: Test set LAB std (computed from first pass through data)
#         """
#         self.resize = transforms.Resize((224, 224))
#         self.lab_mean = np.asarray(lab_mean, dtype=np.float32)
#         self.lab_std = np.asarray(lab_std, dtype=np.float32)
#         self.align_to_training = align_to_training
#         self.test_mean = np.asarray(test_mean, dtype=np.float32) if test_mean is not None else None
#         self.test_std = np.asarray(test_std, dtype=np.float32) if test_std is not None else None
    
#     def __call__(self, img_pil):
#         img = self.resize(img_pil)
#         rgb = np.asarray(img).astype(np.float32) / 255.0
#         lab = rgb2lab(rgb).astype(np.float32)
        
#         # Domain alignment: shift test distribution to match training
#         if self.align_to_training and self.test_mean is not None and self.test_std is not None:
#             # Standardize using test statistics
#             lab_standardized = (lab - self.test_mean) / (self.test_std + 1e-8)
#             # Destandardize using training statistics
#             lab = (lab_standardized * self.lab_std) + self.lab_mean
        
#         # Normalize for model input
#         lab_norm = (lab - self.lab_mean) / self.lab_std
#         return torch.from_numpy(lab_norm.transpose(2, 0, 1)).float()


# class HybridTransform:
#     def __init__(self, lab_mean, lab_std, align_to_training=False,
#                  test_mean=None, test_std=None):
#         """
#         Args:
#             lab_mean: Training set LAB mean
#             lab_std: Training set LAB std
#             align_to_training: If True, normalize test data to match training distribution
#             test_mean: Test set LAB mean (for RGB->LAB conversion)
#             test_std: Test set LAB std (for RGB->LAB conversion)
#         """
#         self.resize = transforms.Resize((224, 224))
#         self.lab_mean = np.asarray(lab_mean, dtype=np.float32)
#         self.lab_std = np.asarray(lab_std, dtype=np.float32)
#         self.to_tensor = transforms.ToTensor()
#         self.rgb_norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
#         self.align_to_training = align_to_training
#         self.test_mean = np.asarray(test_mean, dtype=np.float32) if test_mean is not None else None
#         self.test_std = np.asarray(test_std, dtype=np.float32) if test_std is not None else None
    
#     def __call__(self, img_pil):
#         img = self.resize(img_pil)
#         rgb_tensor = self.rgb_norm(self.to_tensor(img))
#         rgb_np = np.asarray(img).astype(np.float32) / 255.0
#         lab = rgb2lab(rgb_np).astype(np.float32)
        
#         # Domain alignment for LAB channels
#         if self.align_to_training and self.test_mean is not None and self.test_std is not None:
#             lab_standardized = (lab - self.test_mean) / (self.test_std + 1e-8)
#             lab = (lab_standardized * self.lab_std) + self.lab_mean
        
#         lab_norm = (lab - self.lab_mean) / self.lab_std
#         lab_tensor = torch.from_numpy(lab_norm.transpose(2, 0, 1)).float()
#         return torch.cat([rgb_tensor, lab_tensor], dim=0)


# ##############################################################
# # MST-E DATASET
# ##############################################################

# class MSTEDataset:
#     """
#     MST-E Dataset Loader (Directory-based OR Annotation-based)

#     Annotation format (CSV or JSON):
#         filename,mst_label,subject_id
#     """

#     def __init__(self, mst_e_dir: str, transform, annotations_path: Optional[str] = None, pose_lighting_csv: Optional[str] = None, pose_filter: Optional[List[str]] = None, lighting_filter: Optional[List[str]] = None):

#         self.mst_e_dir = Path(mst_e_dir)
#         self.transform = transform
#         self.samples = []

#         if annotations_path:
#             self._load_from_annotations(annotations_path)
#         else:
#             self._load_from_directory()

#         if len(self.samples) == 0:
#             raise ValueError("No MST-E samples found")

#         self._print_distribution()

#     def _load_from_annotations(self, annotations_path, pose_lighting_csv=None, pose_filter=None, lighting_filter=None):

#         ann_path = Path(annotations_path)
#         print(f"[MST-E] Loading annotations from {ann_path}")

#         if ann_path.suffix.lower() == ".csv":
#             df = pd.read_csv(ann_path)
#         elif ann_path.suffix.lower() == ".json":
#             df = pd.read_json(ann_path)
#         else:
#             raise ValueError("Annotations must be CSV or JSON")

#         # --------------------------------------------------
#         # OPTIONAL: merge pose/lighting metadata
#         # --------------------------------------------------
#         if pose_lighting_csv:
#             print(f"[MST-E] Applying pose/lighting filters")
#             meta_df = pd.read_csv(pose_lighting_csv)

#             # Normalize filename matching
#             df["image_ID"] = df["filename"].apply(lambda x: Path(x).name)
#             meta_df["image_ID"] = meta_df["image_ID"].astype(str)

#             merged = df.merge(meta_df, on="image_ID", how="inner")

#             # Apply filters
#             if pose_filter:
#                 merged = merged[merged["pose"].isin(pose_filter)]
#                 print(f"  Pose filter: {pose_filter}")

#             if lighting_filter:
#                 merged = merged[merged["lighting"].isin(lighting_filter)]
#                 print(f"  Lighting filter: {lighting_filter}")

#             df = merged

#             print(f"[MST-E] After filtering: {len(df)} samples")

#         required_cols = {"filename", "mst_label"}
#         if not required_cols.issubset(df.columns):
#             raise ValueError(f"Annotations must contain {required_cols}")

#         for _, row in df.iterrows():
#             rel_path = Path(row["filename"])
#             img_path = rel_path if rel_path.is_absolute() else self.mst_e_dir / rel_path

#             if not img_path.exists():
#                 continue

#             mst = int(row["mst_label"])
#             subject_id = row.get("subject_id", None)
#             self.samples.append((img_path, mst, subject_id))

#         print(f"[MST-E] Loaded {len(self.samples)} annotated samples")


#     def _load_from_directory(self):
#         print("[MST-E] Loading from directory structure")

#         scale_dir = self.mst_e_dir / "scale-images"
#         if scale_dir.exists():
#             for mst in range(1, 11):
#                 for name in [f"mst-{mst}.jpg", f"MST-{mst}.jpg", f"{mst}.jpg"]:
#                     img_path = scale_dir / name
#                     if img_path.exists():
#                         self.samples.append((img_path, mst, None))
#                         break
#         else:
#             for mst in range(1, 11):
#                 mst_dir = self.mst_e_dir / str(mst)
#                 if mst_dir.exists():
#                     for img_path in mst_dir.glob("*.[jp][pn]g"):
#                         self.samples.append((img_path, mst, None))

#     def _print_distribution(self):
#         counts = Counter([mst for _, mst, _ in self.samples])
#         print("[MST-E] Distribution:")
#         for mst in range(1, 11):
#             print(f"  MST {mst:2d}: {counts.get(mst, 0):4d}")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         img_path, mst, subject_id = self.samples[idx]
#         img = Image.open(img_path).convert("RGB")
#         img_tensor = self.transform(img)
#         return img_tensor, mst, str(img_path), subject_id


# ##############################################################
# # COMPUTE TEST SET STATISTICS
# ##############################################################

# def compute_test_lab_statistics(mst_e_dir, annotations_path=None):
#     """
#     Compute LAB statistics for test set (before any normalization)
#     This is used for domain alignment.
#     """
#     print("\n[Computing] Test set LAB statistics for domain alignment...")
    
#     mst_e_dir = Path(mst_e_dir)
#     samples = []
    
#     # Load samples (same logic as MSTEDataset)
#     if annotations_path:
#         ann_path = Path(annotations_path)
#         if ann_path.suffix.lower() == ".csv":
#             df = pd.read_csv(ann_path)
#         elif ann_path.suffix.lower() == ".json":
#             df = pd.read_json(ann_path)
        
#         for _, row in df.iterrows():
#             rel_path = Path(row["filename"])
#             img_path = rel_path if rel_path.is_absolute() else mst_e_dir / rel_path
#             if img_path.exists():
#                 samples.append(img_path)
#     else:
#         scale_dir = mst_e_dir / "scale-images"
#         if scale_dir.exists():
#             for mst in range(1, 11):
#                 for name in [f"mst-{mst}.jpg", f"MST-{mst}.jpg", f"{mst}.jpg"]:
#                     img_path = scale_dir / name
#                     if img_path.exists():
#                         samples.append(img_path)
#                         break
#         else:
#             for mst in range(1, 11):
#                 mst_dir = mst_e_dir / str(mst)
#                 if mst_dir.exists():
#                     for img_path in mst_dir.glob("*.[jp][pn]g"):
#                         samples.append(img_path)
    
#     # Collect LAB values
#     lab_values = []
#     resize = transforms.Resize((224, 224))
    
#     for img_path in tqdm(samples, desc="Computing LAB stats"):
#         img = Image.open(img_path).convert("RGB")
#         img = resize(img)
#         rgb = np.asarray(img).astype(np.float32) / 255.0
#         lab = rgb2lab(rgb).astype(np.float32)
#         lab_values.append(lab.mean(axis=(0, 1)))  # Spatial average
    
#     lab_values = np.array(lab_values)
#     test_mean = lab_values.mean(axis=0)
#     test_std = lab_values.std(axis=0)
    
#     print(f"[Test Stats] Mean: L={test_mean[0]:.2f}, a={test_mean[1]:.2f}, b={test_mean[2]:.2f}")
#     print(f"[Test Stats] Std:  L={test_std[0]:.2f}, a={test_std[1]:.2f}, b={test_std[2]:.2f}")
    
#     return test_mean, test_std


# ##############################################################
# # EVALUATION METRICS
# ##############################################################

# def compute_confusion_matrix(predictions, targets, num_classes):
#     """Compute confusion matrix"""
#     confusion = np.zeros((num_classes, num_classes), dtype=int)
#     for pred, true in zip(predictions, targets):
#         confusion[true, pred] += 1
#     return confusion


# def analyze_errors(predictions, targets, image_paths, label_mapper):
#     """Analyze error patterns"""
#     errors = defaultdict(list)
    
#     for pred, true, path in zip(predictions, targets, image_paths):
#         if pred != true:
#             true_name = label_mapper.get_class_name(true)
#             pred_name = label_mapper.get_class_name(pred)
#             errors[(true, pred)].append(path)
    
#     return errors


# ##############################################################
# # DISTRIBUTION SHIFT ANALYSIS
# ##############################################################

# def analyze_distribution_shift(model, test_loader, train_stats_path, device):
#     """
#     Analyze if test data has different distribution than training data
#     """
#     print("\n" + "="*70)
#     print("DISTRIBUTION SHIFT ANALYSIS")
#     print("="*70 + "\n")
    
#     # Collect LAB statistics from test set
#     test_lab_values = []
#     for imgs, _, _, _ in test_loader:
#         imgs = imgs.to(device)
#         # Convert back to RGB then to LAB for analysis
#         for img_tensor in imgs:
#             img_data = img_tensor.cpu().numpy()
#             num_channels = img_data.shape[0]  # Check channels BEFORE transpose
            
#             # If LAB input, it's already in LAB space (but normalized)
#             # For RGB, convert to LAB
#             if num_channels == 3:  # RGB or LAB input
#                 img_rgb = img_data.transpose(1, 2, 0)
#                 # Assume RGB input for this analysis
#                 img_rgb = img_rgb * np.array(IMAGENET_STD).reshape(1, 1, 3)
#                 img_rgb = img_rgb + np.array(IMAGENET_MEAN).reshape(1, 1, 3)
#                 img_rgb = np.clip(img_rgb, 0, 1)
#                 lab = rgb2lab(img_rgb)
#                 test_lab_values.append(lab.mean(axis=(0, 1)))  # Average over spatial dims
#             elif num_channels == 6:  # Hybrid mode
#                 # Extract RGB channels (first 3) for analysis
#                 img_rgb = img_data[:3, :, :].transpose(1, 2, 0)
#                 img_rgb = img_rgb * np.array(IMAGENET_STD).reshape(1, 1, 3)
#                 img_rgb = img_rgb + np.array(IMAGENET_MEAN).reshape(1, 1, 3)
#                 img_rgb = np.clip(img_rgb, 0, 1)
#                 lab = rgb2lab(img_rgb)
#                 test_lab_values.append(lab.mean(axis=(0, 1)))
    
#     if len(test_lab_values) == 0:
#         print("[ERROR] No LAB values collected. Check your data loader and transforms.")
#         return
    
#     test_lab_values = np.array(test_lab_values)
#     test_lab_mean = test_lab_values.mean(axis=0)
#     test_lab_std = test_lab_values.std(axis=0)
    
#     print(f"Test Set LAB Statistics:")
#     print(f"  Mean: L={test_lab_mean[0]:.2f}, a={test_lab_mean[1]:.2f}, b={test_lab_mean[2]:.2f}")
#     print(f"  Std:  L={test_lab_std[0]:.2f}, a={test_lab_std[1]:.2f}, b={test_lab_std[2]:.2f}")
    
#     # Load training statistics if available
#     if train_stats_path and Path(train_stats_path).exists():
#         try:
#             with open(train_stats_path, 'r') as f:
#                 train_stats = json.load(f)
#             train_mean = np.array(train_stats['lab_mean'])
#             train_std = np.array(train_stats['lab_std'])
            
#             print(f"\nTraining Set LAB Statistics:")
#             print(f"  Mean: L={train_mean[0]:.2f}, a={train_mean[1]:.2f}, b={train_mean[2]:.2f}")
#             print(f"  Std:  L={train_std[0]:.2f}, a={train_std[1]:.2f}, b={train_std[2]:.2f}")
            
#             # Compute shift
#             mean_shift = np.abs(test_lab_mean - train_mean)
#             std_shift = np.abs(test_lab_std - train_std)
            
#             print(f"\nDistribution Shift:")
#             print(f"  Mean shift: L={mean_shift[0]:.2f}, a={mean_shift[1]:.2f}, b={mean_shift[2]:.2f}")
#             print(f"  Std shift:  L={std_shift[0]:.2f}, a={std_shift[1]:.2f}, b={std_shift[2]:.2f}")
            
#             if mean_shift[0] > 10:
#                 print(f"  ⚠️  WARNING: Large lightness shift detected (ΔL={mean_shift[0]:.1f})")
#                 print(f"      This suggests different lighting conditions between train and test")
            
#             if mean_shift[1] > 5 or mean_shift[2] > 5:
#                 print(f"  ⚠️  WARNING: Large color shift detected (Δa={mean_shift[1]:.1f}, Δb={mean_shift[2]:.1f})")
#                 print(f"      This suggests different demographics or color balance")
                
#         except Exception as e:
#             print(f"[WARN] Could not load training statistics: {e}")


# ##############################################################
# # VISUALIZATION
# ##############################################################

# def plot_confusion_matrix(confusion, label_mapper, save_path=None):
#     """Plot confusion matrix"""
#     num_classes = len(confusion)
#     class_names = [label_mapper.get_class_name(i) for i in range(num_classes)]
    
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=class_names, yticklabels=class_names,
#                 cbar_kws={'label': 'Count'})
#     plt.xlabel('Predicted Class')
#     plt.ylabel('True Class')
#     plt.title('Confusion Matrix - MST-E Dataset')
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         print(f"[Saved] Confusion matrix: {save_path}")
#     else:
#         plt.show()
#     plt.close()


# def plot_per_class_accuracy(accuracies, label_mapper, save_path=None):
#     """Plot per-class accuracy"""
#     num_classes = len(accuracies)
#     class_names = [label_mapper.get_class_name(i) for i in range(num_classes)]
    
#     plt.figure(figsize=(12, 6))
#     bars = plt.bar(range(num_classes), accuracies, color='steelblue', alpha=0.8)
    
#     # Color bars: red if <50%, yellow if 50-75%, green if >75%
#     for i, (bar, acc) in enumerate(zip(bars, accuracies)):
#         if acc < 50:
#             bar.set_color('crimson')
#         elif acc < 75:
#             bar.set_color('orange')
#         else:
#             bar.set_color('forestgreen')
    
#     plt.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
#     plt.axhline(y=75, color='orange', linestyle='--', alpha=0.5, label='75% threshold')
    
#     plt.xlabel('Class')
#     plt.ylabel('Accuracy (%)')
#     plt.title('Per-Class Accuracy - MST-E Dataset')
#     plt.xticks(range(num_classes), class_names, rotation=45, ha='right')
#     plt.ylim(0, 100)
#     plt.legend()
#     plt.grid(axis='y', alpha=0.3)
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         print(f"[Saved] Per-class accuracy: {save_path}")
#     else:
#         plt.show()
#     plt.close()


# ##############################################################
# # MAIN EVALUATION
# ##############################################################

# def evaluate_on_mst_e(
#     model,
#     test_loader,
#     label_mapper,
#     device,
#     output_dir=None,
#     suffix=""
# ):
#     """Evaluate model on MST-E dataset"""
#     model.eval()
    
#     all_predictions = []
#     all_targets = []
#     all_paths = []
#     all_probs = []
    
#     print("\n[Evaluating] Running inference on MST-E dataset...")
    
#     with torch.no_grad():
#         for imgs, mst_labels, paths, subject_ids in tqdm(test_loader, desc="Testing"):
#             imgs = imgs.to(device)
            
#             # Map MST labels to class indices
#             targets = torch.tensor([label_mapper.map_mst_to_class(mst.item()) 
#                                    for mst in mst_labels], dtype=torch.long)
            
#             logits = model(imgs)
#             if args.mode == "classification":
#                 probs = torch.softmax(logits, dim=1)
#                 preds = torch.argmax(logits, dim=1)
#             else:
#                 preds = logits.squeeze(1)
#                 probs = None

        
#             all_paths.extend(paths)
#             if args.mode == "classification":
#                 all_predictions.extend(preds.cpu().numpy())
#                 all_targets.extend(targets.numpy())
#                 all_probs.extend(probs.cpu().numpy())
#             else:
#                 all_predictions.extend(preds.cpu().numpy())
#                 all_targets.extend(mst_labels.numpy())  # true MST values

    
#     all_predictions = np.array(all_predictions)
#     all_targets = np.array(all_targets)
#     all_probs = np.array(all_probs)
    
#     # Overall accuracy
#     if args.mode == "classification":
#         correct = (all_predictions == all_targets).sum()
#         total = len(all_targets)
#         overall_acc = 100.0 * correct / total

#         print(f"\nOverall Accuracy: {overall_acc:.2f}%")

#     else:
#         all_predictions = np.array(all_predictions)
#         all_targets = np.array(all_targets)

#         mae = np.mean(np.abs(all_predictions - all_targets))
#         rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))

#         rounded_preds = np.round(all_predictions).clip(1, 10).astype(int)
#         off1 = 100.0 * np.mean(np.abs(rounded_preds - all_targets) <= 1)
#         off2 = 100.0 * np.mean(np.abs(rounded_preds - all_targets) <= 2)

#         print("\nREGRESSION METRICS")
#         print(f"MAE: {mae:.3f}")
#         print(f"RMSE: {rmse:.3f}")
#         print(f"Off-by-1: {off1:.2f}%")
#         print(f"Off-by-2: {off2:.2f}%")

#         return {
#             "mae": mae,
#             "rmse": rmse,
#             "off1": off1,
#             "off2": off2,
#         }

    
#     print(f"\n{'='*70}")
#     print(f"MST-E EVALUATION RESULTS{suffix}")
#     print(f"{'='*70}")
#     print(f"\nOverall Accuracy: {correct}/{total} = {overall_acc:.2f}%")
    
#     # Per-class accuracy
#     print(f"\nPer-Class Accuracy:")
#     class_accuracies = []
#     for class_idx in range(label_mapper.num_classes):
#         mask = all_targets == class_idx
#         if mask.sum() == 0:
#             print(f"  {label_mapper.get_class_name(class_idx)}: No samples")
#             class_accuracies.append(0.0)
#             continue
        
#         class_correct = (all_predictions[mask] == all_targets[mask]).sum()
#         class_total = mask.sum()
#         class_acc = 100.0 * class_correct / class_total
#         class_accuracies.append(class_acc)
        
#         print(f"  {label_mapper.get_class_name(class_idx)}: "
#               f"{class_correct}/{class_total} = {class_acc:.2f}%")
    
#     # Confusion matrix
#     confusion = compute_confusion_matrix(all_predictions, all_targets, 
#                                         label_mapper.num_classes)
    
#     print(f"\nConfusion Matrix:")
#     print("       " + "  ".join([f"{label_mapper.get_class_name(i):>6s}" 
#                                   for i in range(label_mapper.num_classes)]))
#     for i in range(label_mapper.num_classes):
#         row_label = label_mapper.get_class_name(i)
#         row_counts = "  ".join([f"{confusion[i, j]:6d}" 
#                                for j in range(label_mapper.num_classes)])
#         print(f"{row_label:>6s} {row_counts}")
    
#     # Error analysis
#     print(f"\nMost Common Errors:")
#     error_counts = defaultdict(int)
#     for pred, true in zip(all_predictions, all_targets):
#         if pred != true:
#             error_counts[(true, pred)] += 1
    
#     sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
#     for (true_class, pred_class), count in sorted_errors[:10]:
#         true_name = label_mapper.get_class_name(true_class)
#         pred_name = label_mapper.get_class_name(pred_class)
#         pct = 100.0 * count / total
#         print(f"  {true_name} → {pred_name}: {count} times ({pct:.1f}%)")
    
#     # Confidence analysis
#     print(f"\nPrediction Confidence:")
#     correct_mask = all_predictions == all_targets
#     correct_probs = all_probs[correct_mask, all_predictions[correct_mask]]
#     incorrect_probs = all_probs[~correct_mask, all_predictions[~correct_mask]]
    
#     if len(correct_probs) > 0:
#         print(f"  Correct predictions: {correct_probs.mean():.3f} ± {correct_probs.std():.3f}")
#     if len(incorrect_probs) > 0:
#         print(f"  Incorrect predictions: {incorrect_probs.mean():.3f} ± {incorrect_probs.std():.3f}")
    
#     # Save results
#     if output_dir:
#         output_dir = Path(output_dir)
#         output_dir.mkdir(parents=True, exist_ok=True)
        
#         # Add suffix to filenames
#         suffix_clean = suffix.replace(" ", "_").replace("(", "").replace(")", "")
        
#         # Save confusion matrix plot
#         plot_confusion_matrix(confusion, label_mapper, 
#                             output_dir / f"confusion_matrix{suffix_clean}.png")
        
#         # Save per-class accuracy plot
#         plot_per_class_accuracy(class_accuracies, label_mapper,
#                                output_dir / f"per_class_accuracy{suffix_clean}.png")
        
#         # Save detailed results JSON
#         results = {
#             "overall_accuracy": float(overall_acc),
#             "per_class_accuracy": {
#                 label_mapper.get_class_name(i): float(acc)
#                 for i, acc in enumerate(class_accuracies)
#             },
#             "confusion_matrix": confusion.tolist(),
#             "total_samples": int(total),
#             "correct_predictions": int(correct),
#         }
        
#         with open(output_dir / f"results{suffix_clean}.json", 'w') as f:
#             json.dump(results, f, indent=2)
#         print(f"\n[Saved] Detailed results: {output_dir / f'results{suffix_clean}.json'}")
        
#         # Save predictions CSV
#         pred_df = pd.DataFrame({
#             'image_path': all_paths,
#             'true_mst': all_targets,
#             'true_class': all_targets,
#             'pred_class': all_predictions,
#             'confidence': [all_probs[i, all_predictions[i]] 
#                           for i in range(len(all_predictions))],
#             'correct': (all_predictions == all_targets).astype(int)
#         })
#         pred_df.to_csv(output_dir / f"predictions{suffix_clean}.csv", index=False)
#         print(f"[Saved] Predictions: {output_dir / f'predictions{suffix_clean}.csv'}")
    
#     print(f"{'='*70}\n")
    
#     return {
#         'overall_accuracy': overall_acc,
#         'per_class_accuracy': class_accuracies,
#         'confusion_matrix': confusion,
#         'predictions': all_predictions,
#         'targets': all_targets,
#         'probabilities': all_probs
#     }


# ##############################################################
# # MAIN
# ##############################################################

# def main(args):
#     device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
#     print(f"[Device] {device}")
    
#     # Load label mapper
#     label_mapper = LabelMapper(args.label_mapping)
#     print(f"[Label Mapper] {label_mapper.num_classes} classes")
    
#     # Load LAB stats if needed
#     lab_mean, lab_std = None, None
#     if args.input_mode in ["lab", "hybrid"]:
#         if args.lab_stats_path is None:
#             raise ValueError("LAB/Hybrid mode requires --lab_stats_path")
        
#         with open(args.lab_stats_path, 'r') as f:
#             stats = json.load(f)
#         lab_mean = np.array(stats['lab_mean'], dtype=np.float32)
#         lab_std = np.array(stats['lab_std'], dtype=np.float32)
#         print(f"[LAB Stats] Loaded from {args.lab_stats_path}")
#         print(f"  Mean: {lab_mean}")
#         print(f"  Std:  {lab_std}")
    
#     # Compute test set statistics for domain alignment (if requested)
#     test_mean, test_std = None, None
#     if args.align_to_training and args.input_mode in ["lab", "hybrid"]:
#         print("\n" + "="*70)
#         print("DOMAIN ALIGNMENT MODE ENABLED")
#         print("="*70)
#         print("\nThis will normalize test images to match training distribution.")
#         print("⚠️  WARNING: This improves performance but may hide real-world issues!")
#         print("="*70 + "\n")
        
#         test_mean, test_std = compute_test_lab_statistics(
#             args.mst_e_dir, 
#             args.mst_e_annotations
#         )
    
#     # Create transform
#     if args.input_mode == "rgb":
#         transform = RGBTransform()
#     elif args.input_mode == "lab":
#         transform = LABTransform(lab_mean, lab_std, 
#                                 args.align_to_training, test_mean, test_std)
#     elif args.input_mode == "hybrid":
#         transform = HybridTransform(lab_mean, lab_std,
#                                    args.align_to_training, test_mean, test_std)
#     else:
#         raise ValueError(f"Unknown input_mode: {args.input_mode}")
    
#     # Load MST-E dataset
#     mst_e_dataset = MSTEDataset(
#         args.mst_e_dir,
#         transform=transform,
#         annotations_path=args.mst_e_annotations,
#         pose_lighting_csv=args.pose_lighting_csv,
#         pose_filter=args.pose_filter,
#         lighting_filter=args.lighting_filter
#     )

#     test_loader = torch.utils.data.DataLoader(
#         mst_e_dataset, 
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=4
#     )
    
#     # Load model
#     print(f"\n[Model] Loading from {args.model_path}")
#     num_outputs = label_mapper.num_classes if args.mode == "classification" else 1

#     if args.arch == "vgg16":
#         model = VGG16MSTModel(
#             num_outputs=num_outputs,
#             mode=args.mode,
#             dropout=args.dropout,
#             pretrained=False  # important when loading trained model
#         )
#     elif args.arch == "resnet18":
#         model = ResNet18MSTModel(
#             num_outputs=num_outputs,
#             mode=args.mode,
#             dropout=args.dropout,
#             pretrained=False
#         )
#     else:
#         raise ValueError("Unsupported architecture")

#     model = model.to(device)

#     checkpoint = torch.load(args.model_path, map_location=device)

#     # Case 1: Full training checkpoint
#     if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
#         state_dict = checkpoint["model_state_dict"]
#     else:
#         state_dict = checkpoint  # direct state_dict

#     missing, unexpected = model.load_state_dict(state_dict, strict=False)

#     # Ignore projection head mismatches
#     filtered_missing = [k for k in missing if not k.startswith("projection.")]
#     if filtered_missing:
#         print(f"[WARN] Missing keys: {filtered_missing}")

#     if unexpected:
#         print(f"[WARN] Unexpected keys: {unexpected}")

    
#     model.to(device)
#     model.eval()
#     print(f"[Model] Loaded successfully")
    
#     # Analyze distribution shift (only if NOT aligning)
#     if args.analyze_distribution and not args.align_to_training:
#         analyze_distribution_shift(model, test_loader, args.lab_stats_path, device)
#     elif args.analyze_distribution and args.align_to_training:
#         print("\n[INFO] Skipping distribution shift analysis (alignment enabled)")
    
#     # Evaluate
#     suffix = " (Domain Aligned)" if args.align_to_training else ""
#     results = evaluate_on_mst_e(
#         model=model,
#         test_loader=test_loader,
#         label_mapper=label_mapper,
#         device=device,
#         output_dir=args.output_dir,
#         suffix=suffix
#     )
    
#     # Additional diagnostics
#     if args.save_failure_cases and args.output_dir:
#         print("\n[Diagnostics] Analyzing failure cases...")
#         save_failure_examples(results, mst_e_dataset, label_mapper, 
#                             Path(args.output_dir) / "failure_cases")
    
#     # Final summary
#     if args.align_to_training:
#         print("\n" + "="*70)
#         print("DOMAIN ALIGNMENT SUMMARY")
#         print("="*70)
#         print(f"Overall Accuracy (aligned): {results['overall_accuracy']:.2f}%")
#         print("\nInterpretation:")
#         print("  • Higher accuracy indicates distribution shift was a major issue")
#         print("  • Lower accuracy suggests model has fundamental limitations")
#         print("  • For production, consider data augmentation instead of alignment")
#         print("="*70 + "\n")


# def save_failure_examples(results, dataset, label_mapper, output_dir):
#     """Save examples of failures for each error type"""
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     predictions = results['predictions']
#     targets = results['targets']
    
#     # Group errors
#     error_groups = defaultdict(list)
#     for i, (pred, true) in enumerate(zip(predictions, targets)):
#         if pred != true:
#             error_groups[(true, pred)].append(i)
    
#     # Save up to 5 examples per error type
#     for (true_class, pred_class), indices in error_groups.items():
#         true_name = label_mapper.get_class_name(true_class)
#         pred_name = label_mapper.get_class_name(pred_class)
        
#         error_dir = output_dir / f"{true_name}_predicted_as_{pred_name}"
#         error_dir.mkdir(exist_ok=True)
        
#         for idx in indices[:5]:  # Save up to 5 examples
#             img_path, mst, _ = dataset.samples[idx]
#             img = Image.open(img_path)
            
#             # Save copy
#             save_path = error_dir / f"example_{idx}_{img_path.name}"
#             img.save(save_path)
    
#     print(f"[Saved] Failure examples: {output_dir}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Evaluate trained model on MST-E dataset with comprehensive diagnostics"
#     )
    
#     # Model arguments
#     parser.add_argument("--arch", choices=["vgg16", "resnet18"], default="vgg16", help="Model architecture")
#     parser.add_argument("--mode", choices=["classification", "regression"], default="classification", help="Model mode used during training")
#     parser.add_argument("--model_path", required=True, help="Path to trained model (.pth)")
#     parser.add_argument("--label_mapping", required=True, help="Path to label mapping JSON")
#     parser.add_argument("--input_mode", choices=["rgb", "lab", "hybrid"], default="rgb")
#     parser.add_argument("--lab_stats_path", help="Path to LAB statistics (required for LAB/Hybrid)")
#     parser.add_argument("--use_bn", action="store_true", help="Model uses BatchNorm")
#     parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    
#     # Data arguments
#     parser.add_argument("--mst_e_dir", required=True, help="Path to MST-E dataset directory")
#     parser.add_argument("--mst_e_annotations", help="Path to MST-E annotations CSV/JSON (filename,mst_label,subject_id)")
#     parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    
#     # Filtering options
#     parser.add_argument("--pose_lighting_csv", help="CSV containing pose/lighting metadata")
#     parser.add_argument("--pose_filter", nargs="+",
#                         help="Filter by pose (e.g. frontal facing_camera side)")
#     parser.add_argument("--lighting_filter", nargs="+",
#                         help="Filter by lighting (e.g. well_lit poorly_lit)")

    
#     # Output arguments
#     parser.add_argument("--output_dir", help="Directory to save results and visualizations")
#     parser.add_argument("--save_failure_cases", action="store_true", 
#                        help="Save example images of failure cases")
    
#     # Analysis options
#     parser.add_argument("--analyze_distribution", action="store_true",
#                        help="Analyze distribution shift between train and test")
#     parser.add_argument("--align_to_training", action="store_true",
#                        help="Normalize test data to match training distribution (domain alignment)")
    
#     # Hardware
#     parser.add_argument("--gpu", type=int, default=0, help="GPU device")
    
#     args = parser.parse_args()
#     main(args)


"""
MST-E Dataset Evaluation Script with Comprehensive Diagnostics
================================================================

Supports *all* trained model variants from your training script:
- arch: vgg16 / resnet18
- mode: classification / regression
- input_mode: rgb / lab / hybrid (6-channel)
- checkpoints: raw state_dict OR full checkpoint w/ model_state_dict
- ignores projection-head mismatches safely
- domain alignment for LAB/Hybrid
- distribution shift analysis that is CORRECT for rgb vs lab vs hybrid
- failure-case exporting (classification; regression uses rounded bins)

Usage example:
python test_mst_e.py ^
  --model_path "...\best_model.pth" ^
  --label_mapping "...\label_mapping_5class.json" ^
  --mst_e_dir "...\Segmented_MSTE" ^
  --mst_e_annotations "...\annotations.csv" ^
  --input_mode lab ^
  --lab_stats_path "...\lab_statistics.json" ^
  --mode classification ^
  --arch vgg16 ^
  --output_dir "Results/MST-E_LAB_5Class" ^
  --analyze_distribution ^
  --save_failure_cases ^
  --align_to_training ^
  --gpu 0
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from skimage.color import rgb2lab
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


##############################################################
# LABEL MAPPER (FROM TRAINING SCRIPT)
##############################################################

class LabelMapper:
    """Handles flexible label mapping from MST values to class indices"""
    def __init__(self, mapping_config=None):
        if mapping_config is None:
            self.mapping = {str(i): i - 1 for i in range(1, 11)}
            self.num_classes = 10
            self.config_name = "mst10"
        elif isinstance(mapping_config, (str, Path)):
            with open(mapping_config, "r") as f:
                config = json.load(f)
            self._parse_config(config)
            self.config_name = Path(mapping_config).stem
        elif isinstance(mapping_config, dict):
            self._parse_config(mapping_config)
            self.config_name = "custom"
        else:
            raise ValueError("mapping_config must be None, path, or dict")

    def _parse_config(self, config):
        if "label_mapping" in config:
            self.mapping = {str(k): int(v) for k, v in config["label_mapping"].items()}
            self.num_classes = config.get("num_classes", max(self.mapping.values()) + 1)
        elif "bins" in config:
            self.mapping = {}
            for bin_def in config["bins"]:
                mst_range = bin_def["range"]
                class_id = bin_def["class"]
                for mst in range(mst_range[0], mst_range[1] + 1):
                    self.mapping[str(mst)] = class_id
            self.num_classes = config.get("num_classes", max(self.mapping.values()) + 1)
        else:
            raise ValueError("label mapping JSON must contain 'label_mapping' or 'bins'")

        for mst in range(1, 11):
            if str(mst) not in self.mapping:
                raise ValueError(f"Mapping does not cover MST {mst}")

        max_class = max(self.mapping.values())
        if max_class >= self.num_classes:
            raise ValueError(f"num_classes={self.num_classes} but mapping has class {max_class}")

    def map_mst_to_class(self, mst_value: int) -> int:
        return self.mapping[str(int(mst_value))]

    def get_class_name(self, class_idx: int) -> str:
        mst_values = [int(k) for k, v in self.mapping.items() if v == class_idx]
        mst_values.sort()

        if len(mst_values) == 0:
            return f"Class{class_idx}"
        if len(mst_values) == 1:
            return f"MST{mst_values[0]}"

        if mst_values == list(range(mst_values[0], mst_values[-1] + 1)):
            return f"MST{mst_values[0]}-{mst_values[-1]}"

        return f"MST{','.join(map(str, mst_values))}"


##############################################################
# MODELS (MATCH TRAINING SCRIPT)
##############################################################

class VGG16MSTModel(nn.Module):
    def __init__(self, num_outputs, mode="classification", dropout=0.5, pretrained=True):
        super().__init__()
        self.mode = mode

        vgg16 = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.features = vgg16.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.projection = nn.Sequential(
            nn.Linear(512 * 7 * 7, 2048),
            nn.ReLU(),
            nn.Linear(2048, 128),
        )

        if mode == "classification":
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(4096, num_outputs),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(4096, 1024),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(1024, 1),
            )

    def forward(self, x, return_features=False):
        x = self.features(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)

        logits = self.classifier(features)

        if self.mode == "regression":
            logits = torch.clamp(logits, 1.0, 10.0)

        if return_features:
            proj = self.projection(features)
            return logits, proj

        return logits


class ResNet18MSTModel(nn.Module):
    def __init__(self, num_outputs, mode="classification", dropout=0.5, pretrained=True):
        super().__init__()
        self.mode = mode

        resnet = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
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
        else:
            self.head = nn.Linear(in_features, 1)

    def forward(self, x, return_features=False):
        features = self.backbone(x)
        features = self.dropout(features)

        logits = self.head(features)

        if self.mode == "regression":
            logits = torch.clamp(logits, 1.0, 10.0)

        if return_features:
            proj = self.projection(features)
            return logits, proj

        return logits


##############################################################
# TRANSFORMS
##############################################################

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RGBTransform:
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __call__(self, img_pil):
        return self.transform(img_pil)


class LABTransform:
    def __init__(self, lab_mean, lab_std, align_to_training=False, test_mean=None, test_std=None):
        self.resize = transforms.Resize((224, 224))
        self.lab_mean = np.asarray(lab_mean, dtype=np.float32)
        self.lab_std = np.asarray(lab_std, dtype=np.float32)
        self.align_to_training = align_to_training
        self.test_mean = np.asarray(test_mean, dtype=np.float32) if test_mean is not None else None
        self.test_std = np.asarray(test_std, dtype=np.float32) if test_std is not None else None

    def __call__(self, img_pil):
        img = self.resize(img_pil)
        rgb = np.asarray(img).astype(np.float32) / 255.0
        lab = rgb2lab(rgb).astype(np.float32)

        if self.align_to_training and self.test_mean is not None and self.test_std is not None:
            lab_standardized = (lab - self.test_mean) / (self.test_std + 1e-8)
            lab = (lab_standardized * self.lab_std) + self.lab_mean

        lab_norm = (lab - self.lab_mean) / self.lab_std
        return torch.from_numpy(lab_norm.transpose(2, 0, 1)).float()


class HybridTransform:
    """Returns 6-channel tensor: [RGB_norm(3), LAB_norm(3)]"""
    def __init__(self, lab_mean, lab_std, align_to_training=False, test_mean=None, test_std=None):
        self.resize = transforms.Resize((224, 224))
        self.lab_mean = np.asarray(lab_mean, dtype=np.float32)
        self.lab_std = np.asarray(lab_std, dtype=np.float32)
        self.to_tensor = transforms.ToTensor()
        self.rgb_norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.align_to_training = align_to_training
        self.test_mean = np.asarray(test_mean, dtype=np.float32) if test_mean is not None else None
        self.test_std = np.asarray(test_std, dtype=np.float32) if test_std is not None else None

    def __call__(self, img_pil):
        img = self.resize(img_pil)

        rgb_tensor = self.rgb_norm(self.to_tensor(img))

        rgb_np = np.asarray(img).astype(np.float32) / 255.0
        lab = rgb2lab(rgb_np).astype(np.float32)

        if self.align_to_training and self.test_mean is not None and self.test_std is not None:
            lab_standardized = (lab - self.test_mean) / (self.test_std + 1e-8)
            lab = (lab_standardized * self.lab_std) + self.lab_mean

        lab_norm = (lab - self.lab_mean) / self.lab_std
        lab_tensor = torch.from_numpy(lab_norm.transpose(2, 0, 1)).float()

        return torch.cat([rgb_tensor, lab_tensor], dim=0)


##############################################################
# MST-E DATASET
##############################################################

class MSTEDataset:
    """
    MST-E Dataset Loader (Directory-based OR Annotation-based)
    Annotation format (CSV or JSON): filename,mst_label,subject_id
    """

    def __init__(
        self,
        mst_e_dir: str,
        transform,
        annotations_path: Optional[str] = None,
        pose_lighting_csv: Optional[str] = None,
        pose_filter: Optional[List[str]] = None,
        lighting_filter: Optional[List[str]] = None,
    ):
        self.mst_e_dir = Path(mst_e_dir)
        self.transform = transform
        self.samples: List[Tuple[Path, int, Optional[str]]] = []

        if annotations_path:
            self._load_from_annotations(annotations_path, pose_lighting_csv, pose_filter, lighting_filter)
        else:
            self._load_from_directory()

        if len(self.samples) == 0:
            raise ValueError("No MST-E samples found")

        self._print_distribution()

    def _load_from_annotations(self, annotations_path, pose_lighting_csv=None, pose_filter=None, lighting_filter=None):
        ann_path = Path(annotations_path)
        print(f"[MST-E] Loading annotations from {ann_path}")

        if ann_path.suffix.lower() == ".csv":
            df = pd.read_csv(ann_path)
        elif ann_path.suffix.lower() == ".json":
            df = pd.read_json(ann_path)
        else:
            raise ValueError("Annotations must be CSV or JSON")

        if pose_lighting_csv:
            print(f"[MST-E] Applying pose/lighting filters")
            meta_df = pd.read_csv(pose_lighting_csv)

            df["image_ID"] = df["filename"].apply(lambda x: Path(x).name)
            meta_df["image_ID"] = meta_df["image_ID"].astype(str)

            merged = df.merge(meta_df, on="image_ID", how="inner")

            if pose_filter:
                merged = merged[merged["pose"].isin(pose_filter)]
                print(f"  Pose filter: {pose_filter}")

            if lighting_filter:
                merged = merged[merged["lighting"].isin(lighting_filter)]
                print(f"  Lighting filter: {lighting_filter}")

            df = merged
            print(f"[MST-E] After filtering: {len(df)} samples")

        required_cols = {"filename", "mst_label"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Annotations must contain {required_cols}")

        for _, row in df.iterrows():
            rel_path = Path(row["filename"])
            img_path = rel_path if rel_path.is_absolute() else self.mst_e_dir / rel_path
            if not img_path.exists():
                continue

            mst = int(float(row["mst_label"]))
            mst = int(np.clip(mst, 1, 10))
            subject_id = row.get("subject_id", None)
            subject_id = None if pd.isna(subject_id) else str(subject_id)

            self.samples.append((img_path, mst, subject_id))

        print(f"[MST-E] Loaded {len(self.samples)} annotated samples")

    def _load_from_directory(self):
        print("[MST-E] Loading from directory structure")

        scale_dir = self.mst_e_dir / "scale-images"
        if scale_dir.exists():
            for mst in range(1, 11):
                for name in [f"mst-{mst}.jpg", f"MST-{mst}.jpg", f"{mst}.jpg"]:
                    img_path = scale_dir / name
                    if img_path.exists():
                        self.samples.append((img_path, mst, None))
                        break
        else:
            for mst in range(1, 11):
                mst_dir = self.mst_e_dir / str(mst)
                if mst_dir.exists():
                    for img_path in mst_dir.glob("*.[jp][pn]g"):
                        self.samples.append((img_path, mst, None))

    def _print_distribution(self):
        counts = Counter([mst for _, mst, _ in self.samples])
        print("[MST-E] Distribution:")
        for mst in range(1, 11):
            print(f"  MST {mst:2d}: {counts.get(mst, 0):4d}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mst, subject_id = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor, mst, str(img_path), subject_id


##############################################################
# DOMAIN ALIGNMENT STATS
##############################################################

def compute_test_lab_statistics(mst_e_dir, annotations_path=None):
    """
    Compute LAB statistics for test set (before any normalization)
    Returns mean/std over per-image spatial averages.
    """
    print("\n[Computing] Test set LAB statistics for domain alignment...")

    mst_e_dir = Path(mst_e_dir)
    samples: List[Path] = []

    if annotations_path:
        ann_path = Path(annotations_path)
        if ann_path.suffix.lower() == ".csv":
            df = pd.read_csv(ann_path)
        elif ann_path.suffix.lower() == ".json":
            df = pd.read_json(ann_path)
        else:
            raise ValueError("Annotations must be CSV or JSON")

        for _, row in df.iterrows():
            rel_path = Path(row["filename"])
            img_path = rel_path if rel_path.is_absolute() else mst_e_dir / rel_path
            if img_path.exists():
                samples.append(img_path)
    else:
        scale_dir = mst_e_dir / "scale-images"
        if scale_dir.exists():
            for mst in range(1, 11):
                for name in [f"mst-{mst}.jpg", f"MST-{mst}.jpg", f"{mst}.jpg"]:
                    img_path = scale_dir / name
                    if img_path.exists():
                        samples.append(img_path)
                        break
        else:
            for mst in range(1, 11):
                mst_dir = mst_e_dir / str(mst)
                if mst_dir.exists():
                    for img_path in mst_dir.glob("*.[jp][pn]g"):
                        samples.append(img_path)

    if len(samples) == 0:
        raise ValueError("No images found to compute test LAB statistics")

    lab_values = []
    resize = transforms.Resize((224, 224))

    for img_path in tqdm(samples, desc="Computing LAB stats"):
        img = Image.open(img_path).convert("RGB")
        img = resize(img)
        rgb = np.asarray(img).astype(np.float32) / 255.0
        lab = rgb2lab(rgb).astype(np.float32)
        lab_values.append(lab.mean(axis=(0, 1)))

    lab_values = np.asarray(lab_values, dtype=np.float32)
    test_mean = lab_values.mean(axis=0)
    test_std = lab_values.std(axis=0)

    print(f"[Test Stats] Mean: L={test_mean[0]:.2f}, a={test_mean[1]:.2f}, b={test_mean[2]:.2f}")
    print(f"[Test Stats] Std:  L={test_std[0]:.2f}, a={test_std[1]:.2f}, b={test_std[2]:.2f}")

    return test_mean, test_std


##############################################################
# ANALYSIS: DISTRIBUTION SHIFT
##############################################################

def _denorm_imagenet(rgb_chw: np.ndarray) -> np.ndarray:
    """CHW normalized RGB -> HWC float in [0,1] approx."""
    img = rgb_chw.transpose(1, 2, 0)
    img = img * np.asarray(IMAGENET_STD).reshape(1, 1, 3) + np.asarray(IMAGENET_MEAN).reshape(1, 1, 3)
    return np.clip(img, 0.0, 1.0)


def analyze_distribution_shift(
    test_loader,
    input_mode: str,
    train_lab_mean: Optional[np.ndarray],
    train_lab_std: Optional[np.ndarray],
):
    """
    Correctly computes LAB summary stats from the *actual* representation:
    - rgb: denorm RGB -> rgb2lab -> mean/std
    - lab: denorm LAB using training stats -> mean/std
    - hybrid: uses LAB channels (last 3), denorm using training stats -> mean/std
    """
    print("\n" + "="*70)
    print("DISTRIBUTION SHIFT ANALYSIS")
    print("="*70 + "\n")

    if input_mode in ("lab", "hybrid") and (train_lab_mean is None or train_lab_std is None):
        print("[WARN] LAB/Hybrid distribution analysis needs training LAB stats; skipping.")
        return

    test_lab_values = []
    for imgs, _, _, _ in tqdm(test_loader, desc="Collecting stats", leave=False):
        imgs_np = imgs.cpu().numpy()  # BxCxHxW

        for img in imgs_np:
            c = img.shape[0]

            if input_mode == "rgb":
                if c != 3:
                    print(f"[WARN] Expected 3 channels for rgb mode, got {c}. Skipping sample.")
                    continue
                rgb = _denorm_imagenet(img)
                lab = rgb2lab(rgb).astype(np.float32)
                test_lab_values.append(lab.mean(axis=(0, 1)))

            elif input_mode == "lab":
                if c != 3:
                    print(f"[WARN] Expected 3 channels for lab mode, got {c}. Skipping sample.")
                    continue
                # img is normalized LAB (CHW). Denormalize back to LAB.
                lab = img.transpose(1, 2, 0) * train_lab_std.reshape(1, 1, 3) + train_lab_mean.reshape(1, 1, 3)
                test_lab_values.append(lab.mean(axis=(0, 1)))

            elif input_mode == "hybrid":
                if c != 6:
                    print(f"[WARN] Expected 6 channels for hybrid mode, got {c}. Skipping sample.")
                    continue
                lab_norm = img[3:, :, :].transpose(1, 2, 0)
                lab = lab_norm * train_lab_std.reshape(1, 1, 3) + train_lab_mean.reshape(1, 1, 3)
                test_lab_values.append(lab.mean(axis=(0, 1)))

            else:
                raise ValueError(f"Unknown input_mode: {input_mode}")

    if len(test_lab_values) == 0:
        print("[ERROR] No LAB values collected.")
        return

    test_lab_values = np.asarray(test_lab_values, dtype=np.float32)
    test_mean = test_lab_values.mean(axis=0)
    test_std = test_lab_values.std(axis=0)

    print("Test Set LAB Statistics:")
    print(f"  Mean: L={test_mean[0]:.2f}, a={test_mean[1]:.2f}, b={test_mean[2]:.2f}")
    print(f"  Std:  L={test_std[0]:.2f}, a={test_std[1]:.2f}, b={test_std[2]:.2f}")

    if train_lab_mean is not None and train_lab_std is not None:
        train_mean = np.asarray(train_lab_mean, dtype=np.float32)
        train_std = np.asarray(train_lab_std, dtype=np.float32)

        print("\nTraining Set LAB Statistics:")
        print(f"  Mean: L={train_mean[0]:.2f}, a={train_mean[1]:.2f}, b={train_mean[2]:.2f}")
        print(f"  Std:  L={train_std[0]:.2f}, a={train_std[1]:.2f}, b={train_std[2]:.2f}")

        mean_shift = np.abs(test_mean - train_mean)
        std_shift = np.abs(test_std - train_std)

        print("\nDistribution Shift:")
        print(f"  Mean shift: L={mean_shift[0]:.2f}, a={mean_shift[1]:.2f}, b={mean_shift[2]:.2f}")
        print(f"  Std shift:  L={std_shift[0]:.2f}, a={std_shift[1]:.2f}, b={std_shift[2]:.2f}")

        if mean_shift[0] > 10:
            print(f"  ⚠️  WARNING: Large lightness shift detected (ΔL={mean_shift[0]:.1f})")
        if mean_shift[1] > 5 or mean_shift[2] > 5:
            print(f"  ⚠️  WARNING: Large color shift detected (Δa={mean_shift[1]:.1f}, Δb={mean_shift[2]:.1f})")


##############################################################
# VISUALIZATION
##############################################################

def plot_confusion_matrix(confusion, label_mapper, save_path=None):
    num_classes = len(confusion)
    class_names = [label_mapper.get_class_name(i) for i in range(num_classes)]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
    )
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Confusion Matrix - MST-E Dataset")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] Confusion matrix: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_per_class_accuracy(accuracies, label_mapper, save_path=None):
    num_classes = len(accuracies)
    class_names = [label_mapper.get_class_name(i) for i in range(num_classes)]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(num_classes), accuracies)

    # highlight poor classes
    for bar, acc in zip(bars, accuracies):
        if acc < 50:
            bar.set_color("crimson")
        elif acc < 75:
            bar.set_color("orange")
        else:
            bar.set_color("forestgreen")

    plt.axhline(y=50, linestyle="--", alpha=0.5, label="50% threshold")
    plt.axhline(y=75, linestyle="--", alpha=0.5, label="75% threshold")

    plt.xlabel("Class")
    plt.ylabel("Accuracy (%)")
    plt.title("Per-Class Accuracy - MST-E Dataset")
    plt.xticks(range(num_classes), class_names, rotation=45, ha="right")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] Per-class accuracy: {save_path}")
    else:
        plt.show()
    plt.close()


##############################################################
# MODEL INPUT ADAPTERS (HYBRID SUPPORT)
##############################################################

def adapt_model_input_channels(model: nn.Module, arch: str, input_mode: str):
    """
    If input_mode == 'hybrid', adjust first conv to accept 6 channels.
    This MUST match how you trained hybrid models.
    """
    if input_mode != "hybrid":
        return model

    if arch == "vgg16":
        old = model.features[0]
        if not isinstance(old, nn.Conv2d):
            raise RuntimeError("Unexpected VGG16 first layer type")
        if old.in_channels == 6:
            return model
        model.features[0] = nn.Conv2d(
            6,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=(old.bias is not None),
        )
        print("[Hybrid] Patched VGG16 first conv: 3 -> 6 channels")
        return model

    if arch == "resnet18":
        old = model.backbone.conv1
        if old.in_channels == 6:
            return model
        model.backbone.conv1 = nn.Conv2d(
            6,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        print("[Hybrid] Patched ResNet18 conv1: 3 -> 6 channels")
        return model

    raise ValueError(f"Unsupported arch for hybrid patch: {arch}")


##############################################################
# EVALUATION
##############################################################

def evaluate_on_mst_e(
    model,
    test_loader,
    label_mapper,
    device,
    mode: str = "classification",
    output_dir: Optional[str] = None,
    suffix: str = "",
):
    model.eval()

    all_paths: List[str] = []
    all_true_mst: List[int] = []
    all_true_class: List[int] = []
    all_pred_class: List[int] = []
    all_pred_mst: List[float] = []
    all_probs: List[np.ndarray] = []

    print("\n[Evaluating] Running inference on MST-E dataset...")

    with torch.no_grad():
        for imgs, mst_labels, paths, _subject_ids in tqdm(test_loader, desc="Testing"):
            imgs = imgs.to(device)

            mst_labels_np = np.asarray(mst_labels, dtype=np.float32)
            mst_int = np.round(mst_labels_np).clip(1, 10).astype(int)

            logits = model(imgs)

            all_paths.extend(paths)
            all_true_mst.extend(mst_int.tolist())

            if mode == "classification":
                targets = np.array([label_mapper.map_mst_to_class(m) for m in mst_int], dtype=np.int64)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                all_true_class.extend(targets.tolist())
                all_pred_class.extend(preds.cpu().numpy().astype(np.int64).tolist())
                all_probs.extend(probs.cpu().numpy())

            else:  # regression
                preds = logits.squeeze(1).cpu().numpy().astype(np.float32)
                all_pred_mst.extend(preds.tolist())

    if mode == "regression":
        pred = np.asarray(all_pred_mst, dtype=np.float32)
        true = np.asarray(all_true_mst, dtype=np.int32)

        mae = float(np.mean(np.abs(pred - true)))
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))

        rounded_preds = np.round(pred).clip(1, 10).astype(int)
        off1 = float(100.0 * np.mean(np.abs(rounded_preds - true) <= 1))
        off2 = float(100.0 * np.mean(np.abs(rounded_preds - true) <= 2))

        print("\n" + "=" * 70)
        print(f"MST-E EVALUATION RESULTS{suffix} (REGRESSION)")
        print("=" * 70)
        print(f"MAE: {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"Off-by-1: {off1:.2f}%")
        print(f"Off-by-2: {off2:.2f}%")
        print("=" * 70 + "\n")

        # Save CSV for regression
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame({
                "image_path": all_paths,
                "true_mst": true,
                "pred_mst": pred,
                "pred_mst_rounded": rounded_preds,
                "abs_error": np.abs(pred - true),
            })
            df.to_csv(out / f"predictions_regression{suffix.replace(' ', '_').replace('(', '').replace(')', '')}.csv", index=False)
            print(f"[Saved] Regression predictions CSV: {out}")

        return {"mae": mae, "rmse": rmse, "off1": off1, "off2": off2}

    # classification
    true_class = np.asarray(all_true_class, dtype=np.int64)
    pred_class = np.asarray(all_pred_class, dtype=np.int64)
    probs = np.asarray(all_probs, dtype=np.float32)

    correct = int((pred_class == true_class).sum())
    total = int(len(true_class))
    overall_acc = float(100.0 * correct / max(1, total))

    print("\n" + "=" * 70)
    print(f"MST-E EVALUATION RESULTS{suffix}")
    print("=" * 70)
    print(f"\nOverall Accuracy: {correct}/{total} = {overall_acc:.2f}%")

    # per-class accuracy
    print(f"\nPer-Class Accuracy:")
    class_accuracies = []
    for class_idx in range(label_mapper.num_classes):
        mask = true_class == class_idx
        if mask.sum() == 0:
            print(f"  {label_mapper.get_class_name(class_idx)}: No samples")
            class_accuracies.append(0.0)
            continue
        cls_correct = int((pred_class[mask] == true_class[mask]).sum())
        cls_total = int(mask.sum())
        cls_acc = float(100.0 * cls_correct / cls_total)
        class_accuracies.append(cls_acc)
        print(f"  {label_mapper.get_class_name(class_idx)}: {cls_correct}/{cls_total} = {cls_acc:.2f}%")

    # confusion matrix
    confusion = np.zeros((label_mapper.num_classes, label_mapper.num_classes), dtype=int)
    for p, t in zip(pred_class, true_class):
        confusion[t, p] += 1

    print(f"\nMost Common Errors:")
    error_counts = defaultdict(int)
    for p, t in zip(pred_class, true_class):
        if p != t:
            error_counts[(t, p)] += 1
    for (t, p), count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        pct = 100.0 * count / max(1, total)
        print(f"  {label_mapper.get_class_name(t)} → {label_mapper.get_class_name(p)}: {count} times ({pct:.1f}%)")

    # confidence
    idx = np.arange(len(pred_class))
    pred_conf = probs[idx, pred_class]
    correct_mask = pred_class == true_class
    correct_probs = pred_conf[correct_mask]
    incorrect_probs = pred_conf[~correct_mask]

    print(f"\nPrediction Confidence:")
    if len(correct_probs) > 0:
        print(f"  Correct predictions: {correct_probs.mean():.3f} ± {correct_probs.std():.3f}")
    if len(incorrect_probs) > 0:
        print(f"  Incorrect predictions: {incorrect_probs.mean():.3f} ± {incorrect_probs.std():.3f}")

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        suffix_clean = suffix.replace(" ", "_").replace("(", "").replace(")", "")

        plot_confusion_matrix(confusion, label_mapper, out / f"confusion_matrix{suffix_clean}.png")
        plot_per_class_accuracy(class_accuracies, label_mapper, out / f"per_class_accuracy{suffix_clean}.png")

        results = {
            "overall_accuracy": overall_acc,
            "total_samples": total,
            "correct_predictions": correct,
            "per_class_accuracy": {label_mapper.get_class_name(i): float(a) for i, a in enumerate(class_accuracies)},
            "confusion_matrix": confusion.tolist(),
        }
        with open(out / f"results{suffix_clean}.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Saved] Detailed results: {out / f'results{suffix_clean}.json'}")

        df = pd.DataFrame({
            "image_path": all_paths,
            "true_mst": np.asarray(all_true_mst, dtype=int),
            "true_class": true_class,
            "pred_class": pred_class,
            "confidence": pred_conf,
            "correct": (pred_class == true_class).astype(int),
        })
        df.to_csv(out / f"predictions{suffix_clean}.csv", index=False)
        print(f"[Saved] Predictions CSV: {out / f'predictions{suffix_clean}.csv'}")

    print("=" * 70 + "\n")

    return {
        "overall_accuracy": overall_acc,
        "per_class_accuracy": class_accuracies,
        "confusion_matrix": confusion,
        "predictions": pred_class,
        "targets": true_class,
        "probabilities": probs,
        "true_mst": np.asarray(all_true_mst, dtype=int),
    }


def save_failure_examples(results, dataset: MSTEDataset, label_mapper: LabelMapper, output_dir: Path, mode: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    # For regression, group by rounded prediction vs true MST
    if mode == "regression":
        # You only get metrics dict; if you want regression failure saving, use the saved CSV.
        print("[INFO] Regression failure-case export: use the saved regression CSV (contains abs_error).")
        return

    predictions = results["predictions"]
    targets = results["targets"]

    error_groups = defaultdict(list)
    for i, (pred, true) in enumerate(zip(predictions, targets)):
        if int(pred) != int(true):
            error_groups[(int(true), int(pred))].append(i)

    for (true_class, pred_class), indices in error_groups.items():
        true_name = label_mapper.get_class_name(true_class)
        pred_name = label_mapper.get_class_name(pred_class)

        error_dir = output_dir / f"{true_name}_predicted_as_{pred_name}"
        error_dir.mkdir(exist_ok=True)

        for idx in indices[:5]:
            img_path, mst, subject_id = dataset.samples[idx]
            img = Image.open(img_path)
            save_path = error_dir / f"example_{idx}_mst{mst}_{img_path.name}"
            img.save(save_path)

    print(f"[Saved] Failure examples: {output_dir}")


##############################################################
# MAIN
##############################################################

def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    label_mapper = LabelMapper(args.label_mapping)
    print(f"[Label Mapper] {label_mapper.num_classes} classes")

    # training LAB stats
    lab_mean, lab_std = None, None
    if args.input_mode in ["lab", "hybrid"]:
        if args.lab_stats_path is None:
            raise ValueError("LAB/Hybrid mode requires --lab_stats_path")
        with open(args.lab_stats_path, "r") as f:
            stats = json.load(f)
        # Support both old and new training formats
        if "lab_mean" in stats:
            lab_mean = np.asarray(stats["lab_mean"], dtype=np.float32)
            lab_std = np.asarray(stats["lab_std"], dtype=np.float32)
        else:
            lab_mean = np.asarray(stats["mean"], dtype=np.float32)
            lab_std = np.asarray(stats["std"], dtype=np.float32)
        print(f"[LAB Stats] Loaded from {args.lab_stats_path}")
        print(f"  Mean: {lab_mean}")
        print(f"  Std:  {lab_std}")

    # domain alignment: compute test stats
    test_mean, test_std = None, None
    if args.align_to_training and args.input_mode in ["lab", "hybrid"]:
        print("\n" + "=" * 70)
        print("DOMAIN ALIGNMENT MODE ENABLED")
        print("=" * 70)
        print("\nThis will normalize test images to match training distribution.")
        print("⚠️  WARNING: This may hide real-world issues; use for diagnostics.")
        print("=" * 70 + "\n")
        test_mean, test_std = compute_test_lab_statistics(args.mst_e_dir, args.mst_e_annotations)

    # transforms
    if args.input_mode == "rgb":
        transform = RGBTransform()
    elif args.input_mode == "lab":
        transform = LABTransform(lab_mean, lab_std, args.align_to_training, test_mean, test_std)
    elif args.input_mode == "hybrid":
        transform = HybridTransform(lab_mean, lab_std, args.align_to_training, test_mean, test_std)
    else:
        raise ValueError(f"Unknown input_mode: {args.input_mode}")

    # dataset
    mst_e_dataset = MSTEDataset(
        args.mst_e_dir,
        transform=transform,
        annotations_path=args.mst_e_annotations,
        pose_lighting_csv=args.pose_lighting_csv,
        pose_filter=args.pose_filter,
        lighting_filter=args.lighting_filter,
    )

    test_loader = torch.utils.data.DataLoader(
        mst_e_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # model
    print(f"\n[Model] Loading from {args.model_path}")
    num_outputs = label_mapper.num_classes if args.mode == "classification" else 1

    if args.arch == "vgg16":
        model = VGG16MSTModel(num_outputs=num_outputs, mode=args.mode, dropout=args.dropout, pretrained=False)
    elif args.arch == "resnet18":
        model = ResNet18MSTModel(num_outputs=num_outputs, mode=args.mode, dropout=args.dropout, pretrained=False)
    else:
        raise ValueError("Unsupported architecture")

    # hybrid patch (must happen BEFORE loading state_dict)
    model = adapt_model_input_channels(model, args.arch, args.input_mode)

    model = model.to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # filter projection-related messages (common)
    filtered_missing = [k for k in missing if not k.startswith("projection.")]
    filtered_unexpected = [k for k in unexpected if not k.startswith("projection.")]

    if filtered_missing:
        print(f"[WARN] Missing keys (non-projection): {filtered_missing[:25]}{' ...' if len(filtered_missing) > 25 else ''}")
    if filtered_unexpected:
        print(f"[WARN] Unexpected keys (non-projection): {filtered_unexpected[:25]}{' ...' if len(filtered_unexpected) > 25 else ''}")

    model.eval()
    print("[Model] Loaded successfully")

    # distribution shift analysis (skip if alignment enabled, per your original behavior)
    if args.analyze_distribution and not args.align_to_training:
        analyze_distribution_shift(
            test_loader=test_loader,
            input_mode=args.input_mode,
            train_lab_mean=lab_mean,
            train_lab_std=lab_std,
        )
    elif args.analyze_distribution and args.align_to_training:
        print("\n[INFO] Skipping distribution shift analysis (alignment enabled)")

    # evaluation
    suffix = " (Domain Aligned)" if args.align_to_training else ""
    results = evaluate_on_mst_e(
        model=model,
        test_loader=test_loader,
        label_mapper=label_mapper,
        device=device,
        mode=args.mode,
        output_dir=args.output_dir,
        suffix=suffix,
    )

    # failure cases
    if args.save_failure_cases and args.output_dir:
        print("\n[Diagnostics] Saving failure examples...")
        save_failure_examples(
            results=results,
            dataset=mst_e_dataset,
            label_mapper=label_mapper,
            output_dir=Path(args.output_dir) / "failure_cases",
            mode=args.mode,
        )

    # summary
    if args.align_to_training and args.mode == "classification":
        print("\n" + "=" * 70)
        print("DOMAIN ALIGNMENT SUMMARY")
        print("=" * 70)
        print(f"Overall Accuracy (aligned): {results['overall_accuracy']:.2f}%")
        print("Interpretation:")
        print("  • Higher accuracy indicates distribution shift was a major issue")
        print("  • Lower accuracy suggests model has fundamental limitations")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained model on MST-E dataset with comprehensive diagnostics"
    )

    # Model arguments
    parser.add_argument("--arch", choices=["vgg16", "resnet18"], default="vgg16", help="Model architecture")
    parser.add_argument("--mode", choices=["classification", "regression"], default="classification", help="Model mode used during training")
    parser.add_argument("--model_path", required=True, help="Path to trained model (.pth)")
    parser.add_argument("--label_mapping", required=True, help="Path to label mapping JSON")
    parser.add_argument("--input_mode", choices=["rgb", "lab", "hybrid"], default="rgb")
    parser.add_argument("--lab_stats_path", help="Path to LAB statistics (required for LAB/Hybrid)")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")

    # Data arguments
    parser.add_argument("--mst_e_dir", required=True, help="Path to MST-E dataset directory")
    parser.add_argument("--mst_e_annotations", help="Path to MST-E annotations CSV/JSON (filename,mst_label,subject_id)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    # Filtering options
    parser.add_argument("--pose_lighting_csv", help="CSV containing pose/lighting metadata")
    parser.add_argument("--pose_filter", nargs="+", help="Filter by pose (e.g. frontal facing_camera side)")
    parser.add_argument("--lighting_filter", nargs="+", help="Filter by lighting (e.g. well_lit poorly_lit)")

    # Output
    parser.add_argument("--output_dir", help="Directory to save results and visualizations")
    parser.add_argument("--save_failure_cases", action="store_true", help="Save example images of failure cases")

    # Analysis
    parser.add_argument("--analyze_distribution", action="store_true", help="Analyze distribution shift between train and test")
    parser.add_argument("--align_to_training", action="store_true", help="Normalize test data to match training distribution (domain alignment)")

    # Hardware
    parser.add_argument("--gpu", type=int, default=0, help="GPU device")

    args = parser.parse_args()
    main(args)




# python test_mst_e.py --model_path "F:\VGG_MST_Testing\Models\DEL_ResNet18_Ultimate_NoContrastive_FG0\best_model.pth" --arch "resnet18" --mode "classification" --label_mapping "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\DatasetAnnotation\VGG16_MST_Testing\label_mapping_4class.json" --mst_e_dir "G:\Thesis\MonkSkinTone_Dataset\Segmented_MSTE" --mst_e_annotations "G:\Thesis\MonkSkinTone_Dataset\Segmented_MSTE\\annotations.csv" --input_mode lab --pose_lighting_csv "G:\Thesis\MonkSkinTone_Dataset\mst-e_data\mst-e_image_details.csv" --lighting_filter "well_lit" --lab_stats_path "F:\VGG_MST_Testing\Models\DEL_ResNet18_Ultimate_NoContrastive_FG0\lab_statistics.json" --output_dir "Results2/MST-E_LAB_5Class" --analyze_distribution --save_failure_cases --gpu 0

# python test_mst_e.py --model_path "F:\VGG_MST_Testing\Models\DEL_ResNet18_Ultimate_NoContrastive_FG0\best_model.pth" --arch "resnet18" --mode "classification" --label_mapping "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\DatasetAnnotation\VGG16_MST_Testing\label_mapping_4class.json" --mst_e_dir "G:\Thesis\MonkSkinTone_Dataset\Segmented_MSTE" --mst_e_annotations "G:\Thesis\MonkSkinTone_Dataset\Segmented_MSTE\\annotations.csv" --input_mode lab --lab_stats_path "F:\VGG_MST_Testing\Models\DEL_ResNet18_Ultimate_NoContrastive_FG0\lab_statistics.json" --output_dir "Results2/MST-E_LAB_5Class_NO_FILTER" --analyze_distribution --save_failure_cases --gpu 0

# python test_mst_e.py ^
# --model_path "F:\Thesis\CasualConversationv2_Dataset\Models\5Class_Dark_Balanced_Lab\vgg16_mst_best.pth" ^
# --label_mapping "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\DatasetAnnotation\VGG16_MST_Testing\label_mapping_5class.json" ^
# --mst_e_dir "G:\Thesis\MonkSkinTone_Dataset\Segmented_MSTE" ^
#   --mst_e_annotations "G:\Thesis\MonkSkinTone_Dataset\Segmented_MSTE\annotations.csv" ^
# --input_mode lab ^
# --lab_stats_path "F:\Thesis\CasualConversationv2_Dataset\Models\5Class_Dark_Balanced_Lab\lab_statistics.json" ^
# --output_dir "Results/MST-E_LAB_5Class" ^
# --analyze_distribution ^
# --save_failure_cases ^
# --use_bn ^
# --align_to_training ^
# --gpu 0
