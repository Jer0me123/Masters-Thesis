# https://chatgpt.com/c/692dd562-59cc-8326-83ab-806664347a6f
##############################################################
#  VGG16 MST CLASSIFIER (RGB / LAB / HYBRID) - FLEXIBLE LABELS
#  - Modes:
#       * --input-mode rgb    : 3-channel RGB (ImageNet normalized)
#       * --input-mode lab    : 3-channel LAB (dataset mean/std)
#       * --input-mode hybrid : 6-channel [RGB || LAB]
#  - FLEXIBLE classification: Auto-detect number of unique labels
#  - Subject-level stratified split (person_id)
#  - Class-weighted CrossEntropy (weights capped)
#  - Metrics computed based on detected classes
#  - Person-level evaluation (majority vote over frames)
##############################################################

import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from skimage.color import rgb2lab
from sklearn.model_selection import train_test_split


##############################################################
# 0. REPRODUCIBILITY
##############################################################

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

##############################################################
# FLEXIBLE LABEL MAPPING
##############################################################

def build_label_mapping(df):
    """
    Auto-detect unique labels and create bidirectional mapping.
    
    Returns:
        label_to_idx: dict mapping original label -> class index
        idx_to_label: dict mapping class index -> original label
        num_classes: int, number of unique classes
    """
    unique_labels = sorted(df['label'].unique())
    # label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    label_to_idx = {int(label): int(idx) for idx, label in enumerate(unique_labels)}
    # idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    idx_to_label = {int(idx): int(label) for label, idx in label_to_idx.items()}
    num_classes = len(unique_labels)
    
    print(f"\n[INFO] Detected {num_classes} unique labels: {unique_labels}")
    print(f"[INFO] Label mapping: {label_to_idx}\n")
    
    return label_to_idx, idx_to_label, num_classes


##############################################################
# 1. MONK → LAB utilities (for optional L2-LAB metrics)
##############################################################

MONK_COLORS_RGB = np.array([
    (246, 237, 228),
    (243, 231, 219),
    (247, 234, 208),
    (243, 218, 186),
    (215, 189, 150),
    (160, 126, 86),
    (130, 92, 67),
    (96, 65, 52),
    (58, 49, 42),
    (41, 36, 32),
], dtype=np.float32)


def _rgb_to_lab(rgb_vec):
    rgb = rgb_vec.reshape(-1, 1, 1, 3) / 255.0
    lab = rgb2lab(rgb)
    return lab.reshape(-1, 3).astype(np.float32)


MONK_COLORS_LAB = _rgb_to_lab(MONK_COLORS_RGB)


def monk_scalar_to_lab(values):
    """
    values: array-like of MST scores in [1,10] (possibly fractional).
    Returns LAB triplets interpolated between canonical Monk colors.
    """
    vals = np.asarray(values, dtype=np.float32)
    labs = []
    for v in vals:
        v = float(max(1.0, min(10.0, v)))
        base = int(v)
        if base >= 10:
            labs.append(MONK_COLORS_LAB[-1])
            continue
        lab0 = MONK_COLORS_LAB[base - 1]
        lab1 = MONK_COLORS_LAB[base]
        labs.append(lab0 + (v - base) * (lab1 - lab0))
    return np.stack(labs, axis=0)


def calc_l2_lab_distance(pred_labels, tgt_labels):
    """
    pred_labels, tgt_labels: numpy arrays of original label values.
    For MST labels (1-10), map to Monk LAB and compute L2 distance.
    For other labels, return zeros (not applicable).
    """
    # Check if labels are MST-like (1-10 range)
    if np.all((pred_labels >= 1) & (pred_labels <= 10)) and \
       np.all((tgt_labels >= 1) & (tgt_labels <= 10)):
        pred_lab = monk_scalar_to_lab(pred_labels.astype(np.float32))
        tgt_lab = monk_scalar_to_lab(tgt_labels.astype(np.float32))
        return np.sqrt(((pred_lab - tgt_lab) ** 2).sum(axis=1))
    else:
        # L2-LAB not applicable for non-MST labels
        return np.zeros_like(pred_labels, dtype=np.float32)


##############################################################
# 2. LAB DATASET STATISTICS
##############################################################

def compute_dataset_lab_stats(image_dir, csv_path):
    df = pd.read_csv(csv_path)
    img_names = df.iloc[:, 0].astype(str).tolist()

    print(f"[INFO] Computing LAB statistics on {len(img_names)} images...")

    resize_op = transforms.Resize((224, 224))

    sum_lab = np.zeros(3, dtype=np.float64)
    sum_sq_lab = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for name in tqdm(img_names, desc="LAB stats"):
        fpath = Path(image_dir) / name
        try:
            img = Image.open(fpath).convert("RGB")
        except Exception as e:
            print(f"[WARN] Could not read {fpath}: {e}")
            continue

        img = resize_op(img)

        rgb = np.asarray(img).astype(np.float32) / 255.0
        lab = rgb2lab(rgb).astype(np.float64)

        lab_flat = lab.reshape(-1, 3)
        sum_lab += lab_flat.sum(axis=0)
        sum_sq_lab += (lab_flat ** 2).sum(axis=0)
        total_pixels += lab_flat.shape[0]

    if total_pixels == 0:
        raise RuntimeError("No valid pixels found while computing LAB statistics.")

    mean = sum_lab / total_pixels
    var = (sum_sq_lab / total_pixels) - (mean ** 2)
    std = np.sqrt(np.maximum(var, 1e-8))

    print("\n========== LAB DATASET STATISTICS ==========")
    print("LAB_MEAN =", mean.tolist())
    print("LAB_STD  =", std.tolist())
    print("============================================\n")

    return mean.astype(np.float32), std.astype(np.float32)

##############################################################
#  VALIDATE LAB MEAN / STD
##############################################################

def verify_lab_stats(image_dir, lab_mean, lab_std, sample_size=200):
    """
    Randomly samples N images and checks LAB normalization ranges.
    Ensures values are not exploding or collapsing.
    """
    print("\n[VERIFY] Checking LAB normalization validity ...")

    files = list(Path(image_dir).rglob("*"))
    files = [f for f in files if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]

    if len(files) == 0:
        print("[WARN] No images to verify.")
        return

    sample_files = np.random.choice(files, size=min(sample_size, len(files)), replace=False)

    L_vals = []
    A_vals = []
    B_vals = []

    for f in sample_files:
        try:
            img = Image.open(f).convert("RGB")
        except:
            continue

        rgb = np.asarray(img).astype(np.float32) / 255.0
        lab = rgb2lab(rgb).astype(np.float32)  # [H,W,3]

        lab_norm = (lab - lab_mean) / lab_std

        L_vals.append(lab_norm[..., 0].flatten())
        A_vals.append(lab_norm[..., 1].flatten())
        B_vals.append(lab_norm[..., 2].flatten())

    L_vals = np.concatenate(L_vals)
    A_vals = np.concatenate(A_vals)
    B_vals = np.concatenate(B_vals)

    print(f"  L channel: mean={L_vals.mean():.3f}, std={L_vals.std():.3f}, min={L_vals.min():.2f}, max={L_vals.max():.2f}")
    print(f"  A channel: mean={A_vals.mean():.3f}, std={A_vals.std():.3f}, min={A_vals.min():.2f}, max={A_vals.max():.2f}")
    print(f"  B channel: mean={B_vals.mean():.3f}, std={B_vals.std():.3f}, min={B_vals.min():.2f}, max={B_vals.max():.2f}")

    print("[VERIFY] LAB stats check complete.\n")


# Default LAB stats (update with your own if desired)
LAB_MEAN_DEFAULT = np.array([33.618656158447266,
                             8.958210945129395,
                             8.925719261169434], dtype=np.float32)

LAB_STD_DEFAULT = np.array([26.940208435058594,
                            8.05940055847168,
                            9.126977920532227], dtype=np.float32)

##############################################################
# 3. Transforms: RGB / LAB / HYBRID
##############################################################

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RGBTransform:
    def __init__(self, is_train=True):
        if is_train:
            self.geom = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.geom = transforms.Resize((224, 224))

        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, img_pil):
        img_pil = self.geom(img_pil)
        t = self.to_tensor(img_pil)  # [3,H,W], 0..1
        t = self.norm(t)
        return t


class LABTransform:
    def __init__(self, is_train=True, lab_mean=None, lab_std=None):
        self.lab_mean = np.asarray(lab_mean, dtype=np.float32)
        self.lab_std = np.asarray(lab_std, dtype=np.float32)

        if is_train:
            self.geom = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.geom = transforms.Resize((224, 224))

    def __call__(self, img_pil):
        img_pil = self.geom(img_pil)
        rgb = np.asarray(img_pil).astype(np.float32) / 255.0
        lab = rgb2lab(rgb).astype(np.float32)  # [H,W,3]
        lab_norm = (lab - self.lab_mean) / self.lab_std
        lab_chw = torch.from_numpy(lab_norm.transpose(2, 0, 1)).float()
        return lab_chw


class HybridTransform:
    """
    Concatenates RGB (ImageNet normalized) and LAB (dataset normalized):
    Output shape: [6, H, W] = [R,G,B,L,a,b]
    """
    def __init__(self, is_train=True, lab_mean=None, lab_std=None):
        self.lab_mean = np.asarray(lab_mean, dtype=np.float32)
        self.lab_std = np.asarray(lab_std, dtype=np.float32)

        if is_train:
            self.geom = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.geom = transforms.Resize((224, 224))

        self.to_tensor = transforms.ToTensor()
        self.rgb_norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, img_pil):
        # Apply SAME geometric transforms once
        img_pil = self.geom(img_pil)

        # RGB branch
        rgb_tensor = self.rgb_norm(self.to_tensor(img_pil))  # [3,H,W]

        # LAB branch
        rgb_np = np.asarray(img_pil).astype(np.float32) / 255.0
        lab = rgb2lab(rgb_np).astype(np.float32)
        lab_norm = (lab - self.lab_mean) / self.lab_std
        lab_tensor = torch.from_numpy(lab_norm.transpose(2, 0, 1)).float()  # [3,H,W]

        # Concatenate: [RGB || LAB] -> [6,H,W]
        hybrid = torch.cat([rgb_tensor, lab_tensor], dim=0)
        return hybrid


##############################################################
# 4. Dataset + Dataloaders
##############################################################

class SkinToneDataset(Dataset):
    """
    Unified dataset for both classification and regression.
    
    For classification: returns class_idx based on label_to_idx mapping
    For regression: returns normalized label in [0,1] based on label range
    """
    def __init__(self, df, img_dir, transform, task_mode, label_to_idx=None, 
                 label_min=None, label_max=None, weights=None):
        self.df = df.reset_index(drop=True).copy()
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.task_mode = task_mode
        self.label_to_idx = label_to_idx  # For classification
        self.label_min = label_min  # For regression normalization
        self.label_max = label_max  # For regression normalization

        if self.df.shape[1] < 3:
            raise ValueError("Dataset df must have at least 3 columns: filename, label, person_id")

        # Handle weights for regression
        if weights is None and task_mode == "regression":
            print("[INFO] Calculating dynamic sample weights for regression...")
            raw_labels = self.df['label'].astype(float).values
            binned_labels = np.round(raw_labels).astype(int)
            
            counts = Counter(binned_labels)
            total_samples = len(binned_labels)
            
            weights_map = {k: total_samples / v for k, v in counts.items()}
            weights = [weights_map[int(round(x))] for x in raw_labels]
            weights = np.array(weights)
            weights = weights / weights.mean()
            
            print(f"[INFO] Weights calculated. Min: {weights.min():.2f}, Max: {weights.max():.2f}")
        
        self.weights = torch.FloatTensor(weights) if weights is not None else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = str(row['filename'])
        label = row['label']
        person_id = str(row['person_id'])

        img_path = self.img_dir / fname
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to load {img_path}: {e}")
            img = Image.new("RGB", (224, 224), color="black")

        img_t = self.transform(img)

        if self.task_mode == "classification":
            class_idx = self.label_to_idx[label]
            return img_t, torch.tensor(class_idx, dtype=torch.long), person_id
        
        else:  # regression
            # Normalize to [0,1] based on label range
            label_normalized = (float(label) - self.label_min) / (self.label_max - self.label_min)
            weight = self.weights[idx] if self.weights is not None else torch.tensor(1.0)
            return img_t, torch.tensor(label_normalized, dtype=torch.float32), weight


def build_dataloaders(csv_path, image_dir, batch_size, val_ratio,
                      input_mode, lab_mean, lab_std,
                      task_mode="classification",
                      num_workers=4,
                      balance_strategy="sampling",
                      save_split_path=None):

    df = pd.read_csv(csv_path).dropna()

    if df.shape[1] < 3:
        raise ValueError("CSV must have 3 columns: filename, label, person_id")

    print(f"[INFO] Loaded CSV with columns: {list(df.columns)}")

    # Ensure standardized column names
    df = df.iloc[:, :3]
    df.columns = ["filename", "label", "person_id"]
    df["person_id"] = df["person_id"].astype(str)

    # Build label mapping for classification
    if task_mode == "classification":
        label_to_idx, idx_to_label, num_classes = build_label_mapping(df)
        label_min, label_max = None, None
    else:  # regression
        label_to_idx, idx_to_label = None, None
        label_min = df['label'].min()
        label_max = df['label'].max()
        num_classes = None
        print(f"[INFO] Regression label range: [{label_min}, {label_max}]")

    # One label per person (first occurrence)
    person_labels = df.groupby("person_id").agg({
        "label": "first"
    }).reset_index()

    if task_mode == "classification":
        labels_per_person = person_labels["label"].map(label_to_idx).values
    else:
        # For regression, bin labels for stratification
        labels_per_person = np.digitize(
            person_labels["label"].values,
            bins=np.linspace(label_min, label_max, 6)  # Create 5 bins
        )

    print(f"[INFO] Found {len(person_labels)} unique persons")

    # Person distribution
    person_counts = Counter(labels_per_person)
    print("\n[INFO] Person distribution across labels (persons):")
    for label in sorted(person_counts.keys()):
        if task_mode == "classification":
            orig_label = idx_to_label[label]
            print(f"  Label {orig_label} (class {label}): {person_counts[label]} persons")
        else:
            print(f"  Bin {label}: {person_counts[label]} persons")
    print()

    # Stratified split on person-level labels
    train_persons, val_persons = train_test_split(
        person_labels["person_id"].values,
        test_size=val_ratio,
        shuffle=True,
        stratify=labels_per_person,
        random_state=42
    )

    # SAVE SPLIT TO DISK
    split_info = {
        "train": train_persons.tolist(),
        "val": val_persons.tolist(),
        "random_state": 42,
        "val_ratio": val_ratio
    }
    
    if task_mode == "classification":
        split_info["label_to_idx"] = label_to_idx
        split_info["idx_to_label"] = idx_to_label
        split_info["num_classes"] = num_classes
    else:
        split_info["label_min"] = float(label_min)
        split_info["label_max"] = float(label_max)

    split_path = Path(save_split_path) if save_split_path is not None else Path("train_val_split.json")
    with open(split_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"[INFO] Saved train/val split to {split_path}")

    train_df = df[df["person_id"].isin(train_persons)].reset_index(drop=True)
    val_df = df[df["person_id"].isin(val_persons)].reset_index(drop=True)

    print(f"[INFO] Train: {len(train_persons)} persons ({len(train_df)} images)")
    print(f"[INFO] Val:   {len(val_persons)} persons ({len(val_df)} images)")

    # Image-level distributions
    if task_mode == "classification":
        train_class_idx = train_df["label"].map(label_to_idx)
        val_class_idx = val_df["label"].map(label_to_idx)

        train_counts = Counter(train_class_idx)
        val_counts = Counter(val_class_idx)

        print("\n[INFO] Train label distribution (images):")
        for class_idx in sorted(train_counts.keys()):
            orig_label = idx_to_label[class_idx]
            print(f"  Label {orig_label} (class {class_idx}): {train_counts[class_idx]} images")

        print("\n[INFO] Val label distribution (images):")
        for class_idx in sorted(val_counts.keys()):
            orig_label = idx_to_label[class_idx]
            print(f"  Label {orig_label} (class {class_idx}): {val_counts[class_idx]} images")
        print()

        # ---------- CLASS WEIGHTS (for CE) ----------
        class_counts = Counter(train_class_idx)

        raw_weights = {}
        for k in range(num_classes):
            c = class_counts.get(k, 1)  # avoid div-by-zero
            raw_weights[k] = 1.0 / float(c)

        # Normalize to mean=1 and cap at max_weight
        raw_vals = np.array(list(raw_weights.values()), dtype=np.float32)
        mean_raw = raw_vals.mean()
        norm_weights = {k: (v / mean_raw) for k, v in raw_weights.items()}

        max_weight = 5.0
        capped_weights = {k: min(max_weight, w) for k, w in norm_weights.items()}

        print("[INFO] Class weights (after normalization & capping):")
        for k in range(num_classes):
            print(f"  Class {k}: weight={capped_weights[k]:.3f}")
        print()

        # Build weight tensor for CE
        class_weight_tensor = torch.zeros(num_classes, dtype=torch.float32)
        for k in range(num_classes):
            class_weight_tensor[k] = capped_weights.get(k, 1.0)
    else:
        class_weight_tensor = None

    # ---------- Transforms ----------
    if input_mode == "rgb":
        train_tf = RGBTransform(is_train=True)
        val_tf = RGBTransform(is_train=False)
    elif input_mode == "lab":
        train_tf = LABTransform(is_train=True, lab_mean=lab_mean, lab_std=lab_std)
        val_tf = LABTransform(is_train=False, lab_mean=lab_mean, lab_std=lab_std)
    elif input_mode == "hybrid":
        train_tf = HybridTransform(is_train=True, lab_mean=lab_mean, lab_std=lab_std)
        val_tf = HybridTransform(is_train=False, lab_mean=lab_mean, lab_std=lab_std)
    else:
        raise ValueError(f"Unknown input_mode: {input_mode}")

    train_dataset = SkinToneDataset(
        train_df, image_dir, train_tf, task_mode, 
        label_to_idx=label_to_idx, label_min=label_min, label_max=label_max
    )
    val_dataset = SkinToneDataset(
        val_df, image_dir, val_tf, task_mode,
        label_to_idx=label_to_idx, label_min=label_min, label_max=label_max
    )

    # =============================================================
    #  SAMPLER FOR BALANCED TRAINING (classification only)
    # =============================================================
    
    if task_mode == "classification" and balance_strategy == "sampling":
        class_freq = Counter(train_class_idx)

        img_weights = []
        for _, row in train_df.iterrows():
            label = row["label"]
            class_idx = label_to_idx[label]
            freq = class_freq[class_idx]
            img_weights.append(1.0 / freq)

        img_weights = torch.DoubleTensor(img_weights)

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=img_weights,
            num_samples=len(img_weights),  
            replacement=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2,
    )

    return train_loader, val_loader, class_weight_tensor, label_to_idx, idx_to_label, num_classes, label_min, label_max


##############################################################
# FOCAL LOSS FOR CLASSIFICATION
##############################################################

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification:
    FL = -alpha * (1 - p)^gamma * log(p)

    alpha: tensor of shape [num_classes] for class reweighting
    gamma: focusing parameter (default 2.0)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha  # tensor [C] or None

    def forward(self, logits, targets):
        """
        logits: [B, C]
        targets: [B]
        """
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.softmax(logits, dim=1)[range(len(targets)), targets]  # p_t

        focal = (1 - pt) ** self.gamma * ce

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        else:
            return focal


##############################################################
# 5. VGG16 MST Classifier & Regressor
##############################################################

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

##############################################################
# 6. Metrics
##############################################################

def compute_classification_metrics(preds, targets, idx_to_label):
    """
    preds, targets: 1D torch tensors of class indices
    idx_to_label: mapping from class index to original label
    
    Returns dict with accuracy and L2-LAB if applicable
    """
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()

    correct = (preds == targets).float()
    accuracy = correct.mean().item() * 100.0

    # Convert to original labels for L2-LAB calculation
    pred_labels = np.array([idx_to_label[int(p)] for p in preds])
    target_labels = np.array([idx_to_label[int(t)] for t in targets])
    
    l2 = calc_l2_lab_distance(pred_labels, target_labels)
    l2_mean = float(l2.mean()) if l2.any() else 0.0
    l2_std = float(l2.std()) if l2.any() else 0.0

    return {
        "accuracy": accuracy,
        "l2_mean": l2_mean,
        "l2_std": l2_std,
    }


##############################################################
# 7. Training / Validation
##############################################################

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, idx_to_label):
    model.train()

    running_loss = 0.0
    correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", ncols=100)

    for imgs, labels, person_ids in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)

        batch_size = imgs.size(0)
        running_loss += loss.item() * batch_size
        correct += (preds == labels).sum().item()
        total_samples += batch_size

        batch_acc = 100.0 * (preds == labels).float().mean().item()
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{batch_acc:.2f}",
        })

    avg_loss = running_loss / total_samples
    avg_acc = 100.0 * correct / total_samples

    return avg_loss, avg_acc

def validate(model, loader, criterion, device, epoch, idx_to_label):
    model.eval()

    running_loss = 0.0
    total_samples = 0

    all_preds = []
    all_targets = []
    all_person_ids = []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Validate Epoch {epoch}", ncols=100)
        for imgs, labels, person_ids in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(imgs)
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)

            batch_size = imgs.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())
            all_person_ids.extend(person_ids)

            batch_acc = 100.0 * (preds == labels).float().mean().item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{batch_acc:.2f}",
            })

    avg_loss = running_loss / total_samples
    preds_all = torch.cat(all_preds, dim=0)
    targets_all = torch.cat(all_targets, dim=0)

    # Image-level metrics
    img_metrics = compute_classification_metrics(preds_all, targets_all, idx_to_label)

    # Person-level metrics (majority vote)
    person_true = {}
    person_vote_counts = defaultdict(Counter)

    for pred, tgt, pid in zip(preds_all.numpy(), targets_all.numpy(), all_person_ids):
        person_vote_counts[pid][int(pred)] += 1
        if pid not in person_true:
            person_true[pid] = int(tgt)

    person_preds = []
    person_targets = []
    for pid, vote_counter in person_vote_counts.items():
        pred_class = vote_counter.most_common(1)[0][0]
        tgt_class = person_true[pid]
        person_preds.append(pred_class)
        person_targets.append(tgt_class)

    person_preds = torch.tensor(person_preds, dtype=torch.long)
    person_targets = torch.tensor(person_targets, dtype=torch.long)
    person_metrics = compute_classification_metrics(person_preds, person_targets, idx_to_label)

    return avg_loss, img_metrics, person_metrics

def train_one_epoch_regression(model, loader, optimizer, device, epoch, label_min, label_max, threshold=0.5):
    model.train()
    total_loss = 0.0
    total_acc = 0.0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", ncols=100)

    for i, (imgs, labels, weights) in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)

        preds = model(imgs)

        # Weighted MSE loss
        loss_unreduced = F.mse_loss(preds, labels, reduction="none")
        weights_t = torch.sqrt(weights)
        loss_weighted = loss_unreduced * weights_t
        loss = loss_weighted.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy (convert back to original scale)
        label_range = label_max - label_min
        pred_orig = preds.detach().cpu().numpy() * label_range + label_min
        label_orig = labels.detach().cpu().numpy() * label_range + label_min
        abs_diff = np.abs(pred_orig - label_orig)
        acc = (abs_diff <= threshold).mean() * 100.0

        total_loss += loss.item() * imgs.size(0)
        total_acc += acc * imgs.size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{acc:.2f}"
        })

    n = len(loader.dataset)
    return total_loss / n, total_acc / n

def validate_regression(model, loader, device, epoch, label_min, label_max, threshold=0.5):
    model.eval()
    mse = nn.MSELoss()
    total_loss = 0.0
    total_acc = 0.0
    all_l2 = []

    label_range = label_max - label_min

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Validate Epoch {epoch}", ncols=100)

        for imgs, labels, _ in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            preds = model(imgs)

            loss = mse(preds, labels)
            
            # Convert back to original scale
            pred_orig = preds.cpu().numpy() * label_range + label_min
            label_orig = labels.cpu().numpy() * label_range + label_min
            abs_diff = np.abs(pred_orig - label_orig)
            acc = (abs_diff <= threshold).mean() * 100.0
            
            # L2-LAB if applicable
            l2 = calc_l2_lab_distance(
                np.round(pred_orig).astype(int),
                np.round(label_orig).astype(int)
            )

            total_loss += loss.item() * imgs.size(0)
            total_acc += acc * imgs.size(0)
            all_l2.extend(l2)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{acc:.2f}"
            })

    n = len(loader.dataset)
    l2_mean = float(np.mean(all_l2)) if all_l2 and np.any(all_l2) else 0.0
    l2_std = float(np.std(all_l2)) if all_l2 and np.any(all_l2) else 0.0
    
    return total_loss / n, total_acc / n, l2_mean, l2_std

##############################################################
# 8. Checkpoint Management
##############################################################

def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, val_loss, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_acc": val_acc,
        "val_loss": val_loss,
    }, path)
    print(f"[INFO] Checkpoint saved -> {path}")


def load_checkpoint(model, optimizer, scheduler, path, device):
    if not Path(path).exists():
        print(f"[INFO] No checkpoint found at {path}, starting from scratch.")
        return 1, float("inf"), 0.0

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint["val_loss"]
    best_val_acc = checkpoint["val_acc"]

    print(f"[INFO] Resuming from epoch {checkpoint['epoch']}")
    print(f"[INFO] Previous best - ValLoss: {best_val_loss:.4f}, ValAcc: {best_val_acc:.2f}%")

    return start_epoch, best_val_loss, best_val_acc


##############################################################
# 9. Early Stopping
##############################################################

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = 0.0
        self.should_stop = False

    def __call__(self, val_acc):
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.counter = 0
        else:
            self.counter += 1
            print(f"[EARLY STOP] No improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                self.should_stop = True
                print(f"[EARLY STOP] Triggered! Best ValAcc: {self.best_acc:.2f}%")

        return self.should_stop


##############################################################
# 10. MAIN
##############################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--save-dir", default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--additional-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use-bn", action="store_true")
    parser.add_argument("--compute-lab-stats", action="store_true")
    parser.add_argument("--input-mode", choices=["rgb", "lab", "hybrid"], default="rgb")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--balance-strategy", choices=["weights", "sampling", "none"], default="sampling")

    parser.add_argument("--task-mode", choices=["classification", "regression"], default="classification")
    parser.add_argument("--regression-threshold", type=float, default=0.5)

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-delta", type=float, default=0.1)
    parser.add_argument("--verify-lab-stats", action="store_true")

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("\n================= VGG16 FLEXIBLE LABEL TRAINING =================")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Input mode: {args.input_mode}")
    print(f"[INFO] Task mode: {args.task_mode}")
    print("=================================================================\n")

    # LAB stats
    if args.input_mode in ("lab", "hybrid"):
        if args.compute_lab_stats:
            print("[INFO] Computing LAB statistics from dataset...")
            lab_mean, lab_std = compute_dataset_lab_stats(args.image_dir, args.csv_path)
        else:
            print("[INFO] Using hardcoded LAB normalization values.")
            lab_mean, lab_std = LAB_MEAN_DEFAULT, LAB_STD_DEFAULT
        print(f"[INFO] Active LAB_MEAN: {lab_mean.tolist()}")
        print(f"[INFO] Active LAB_STD : {lab_std.tolist()}")
    else:
        lab_mean, lab_std = LAB_MEAN_DEFAULT, LAB_STD_DEFAULT

    if args.input_mode in ("lab", "hybrid") and args.verify_lab_stats:
        verify_lab_stats(args.image_dir, lab_mean, lab_std)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_path = save_dir / "vgg16_mst_best.pth"
    final_path = save_dir / "vgg16_mst_final.pth"
    checkpoint_path = save_dir / "vgg16_mst_checkpoint.pth"
    save_split_path = save_dir / "train_val_split.json"

    # Data
    train_loader, val_loader, class_weight_tensor, label_to_idx, idx_to_label, num_classes, label_min, label_max = build_dataloaders(
        args.csv_path,
        args.image_dir,
        args.batch_size,
        args.val_ratio,
        args.input_mode,
        lab_mean,
        lab_std,
        task_mode=args.task_mode,
        num_workers=args.num_workers,
        balance_strategy=args.balance_strategy,
        save_split_path=save_split_path
    )

    # Model
    if args.task_mode == "classification":
        model = VGG16MSTClassifier(
            input_mode=args.input_mode,
            use_bn=args.use_bn,
            num_classes=num_classes,
            dropout_p=0.5,
        ).to(device)
    else: 
        model = VGG16MSTRegressor(
            input_mode=args.input_mode,
            use_bn=args.use_bn,
            dropout_p=0.5,
        ).to(device)

    # Optimizer
    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("features"):
            backbone_params.append(p)
        else:
            head_params.append(p)

    base_lr = args.lr
    optimizer = torch.optim.Adam(
        [
            {"params": backbone_params, "lr": base_lr * 0.1},
            {"params": head_params, "lr": base_lr},
        ],
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.3,
        patience=3,
        min_lr=1e-6,
        verbose=True,
    )

    # Loss
    if args.task_mode == "classification":
        class_weight_tensor = class_weight_tensor.to(device)
        if args.balance_strategy == "weights":
            criterion = FocalLoss(alpha=class_weight_tensor, gamma=2.0)
            print("[INFO] Using FocalLoss with class weights")
        elif args.balance_strategy == "sampling":
            criterion = FocalLoss(alpha=None, gamma=2.0)
            print("[INFO] Using FocalLoss without weights (balanced via sampling)")
        else:
            criterion = FocalLoss(alpha=None, gamma=2.0)
            print("[INFO] Using FocalLoss without any balancing")
    else:
        criterion = None
        print("[INFO] Using weighted MSE for training, standard MSE for validation")

    # Resume or start fresh
    if args.resume:
        start_epoch, best_val_loss, best_val_acc = load_checkpoint(
            model, optimizer, scheduler, checkpoint_path, device
        )
        if args.additional_epochs is not None:
            max_epoch = start_epoch - 1 + args.additional_epochs
            print(f"[INFO] Will train for {args.additional_epochs} additional epochs (until epoch {max_epoch})")
        else:
            max_epoch = args.epochs
            if start_epoch > max_epoch:
                print(f"[WARNING] Checkpoint is at epoch {start_epoch - 1}, already exceeds --epochs {max_epoch}")
                return
            print(f"[INFO] Will train until epoch {max_epoch}")
    else:
        start_epoch = 1
        max_epoch = args.epochs
        best_val_loss = float("inf")
        best_val_acc = 0.0
        print(f"[INFO] Starting fresh training for {max_epoch} epochs")

    early_stopper = EarlyStopping(
        patience=args.early_stop_patience,
        min_delta=args.early_stop_delta,
    )

    # Training Loop
    for epoch in range(start_epoch, max_epoch + 1):
        print(f"\n===== EPOCH {epoch}/{max_epoch} =====")

        if args.task_mode == "classification":
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch, idx_to_label
            )

            va_loss, img_metrics, person_metrics = validate(
                model, val_loader, criterion, device, epoch, idx_to_label
            )

            print(
                f"[EPOCH {epoch}] "
                f"TrainLoss={tr_loss:.4f} TrainAcc={tr_acc:.2f}% | "
                f"ValLoss={va_loss:.4f} | "
                f"ValAcc_img={img_metrics['accuracy']:.2f}% "
                f"ValAcc_person={person_metrics['accuracy']:.2f}%"
            )

            current_val_acc = person_metrics["accuracy"]
        
        else:
            tr_loss, tr_acc = train_one_epoch_regression(
                model, train_loader, optimizer, device, epoch, label_min, label_max, args.regression_threshold
            )
            va_loss, va_acc, l2_mean, l2_std = validate_regression(
                model, val_loader, device, epoch, label_min, label_max, args.regression_threshold
            )
            
            print(
                f"[EPOCH {epoch}] "
                f"TrainLoss={tr_loss:.4f} TrainAcc={tr_acc:.2f}% | "
                f"ValLoss={va_loss:.4f} ValAcc={va_acc:.2f}% | "
                f"L2 mean={l2_mean:.3f}, std={l2_std:.3f}"
            )
            current_val_acc = va_acc

        scheduler.step(va_loss)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            torch.save(model.state_dict(), best_path)
            print(f"[INFO] Saved BEST model -> {best_path}")

        save_checkpoint(
            model, optimizer, scheduler, epoch, current_val_acc, va_loss, checkpoint_path
        )

        if early_stopper(current_val_acc):
            print(f"[INFO] Early stopping triggered at epoch {epoch}")
            break

    torch.save(model.state_dict(), final_path)
    print(f"[INFO] Saved FINAL model -> {final_path}")
    print(
        f"[INFO] Training complete. Best ValAcc: {best_val_acc:.2f}%, "
        f"Best ValLoss: {best_val_loss:.4f}"
    )

if __name__ == "__main__":
    main()