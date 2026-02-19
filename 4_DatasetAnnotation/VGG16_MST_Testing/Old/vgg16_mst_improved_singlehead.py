##############################################################
#  VGG16 MST CLASSIFIER - SINGLE-TASK VERSION
#  
#  *** CLEAN SINGLE-TASK ARCHITECTURE ***
#  
#  Removed multi-task learning to focus purely on 10-class MST.
#  This fixes the issue where models predict mostly middle bins (4-7).
#  
#  Changes from multi-task version:
#  1. Removed 3-bin auxiliary classifier head
#  2. Removed bin_loss and all bin-related code
#  3. Simplified model architecture
#  4. Removed lambda_bin parameter
#  5. Focus purely on fine-grained 10-class MST classification
#  
#  Bug Fixes Applied:
#  - Fixed LabelSmoothingFocalLoss (F.softmax instead of torch.exp)
#  - Added numerical stability (clamping, NaN checks)
#  - Fixed SupervisedContrastiveLoss (temperature 0.5, clamping)
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
# LABEL MODE UTILITIES
##############################################################

def mst_to_3bin(mst):
    # 1–3 -> 0, 4–7 -> 1, 8–10 -> 2
    if mst <= 3:
        return 0
    elif mst <= 7:
        return 1
    else:
        return 2


def map_mst_to_label(mst, label_mode):
    if label_mode == "mst10":
        return mst - 1        # 0..9
    elif label_mode == "mst3":
        return mst_to_3bin(mst)  # 0..2
    else:
        raise ValueError(f"Unknown label mode: {label_mode}")


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


def calc_l2_lab_distance_classes(pred_classes, tgt_classes):
    """
    pred_classes, tgt_classes: numpy arrays of class indices 0-9.
    Convert to MST [1..10], map to Monk LAB, compute L2 distance.
    """
    pred_mst = pred_classes.astype(np.float32) + 1.0
    tgt_mst = tgt_classes.astype(np.float32) + 1.0
    pred_lab = monk_scalar_to_lab(pred_mst)
    tgt_lab = monk_scalar_to_lab(tgt_mst)
    return np.sqrt(((pred_lab - tgt_lab) ** 2).sum(axis=1))


##############################################################
# 2. LAB DATASET STATISTICS
##############################################################

def compute_dataset_lab_stats(image_dir, csv_path):
    df = pd.read_csv(csv_path)
    img_names = df.iloc[:, 0].astype(str).tolist()

    print(f"[INFO] Computing LAB dataset statistics on {len(img_names)} images...")

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


# Default LAB stats
LAB_MEAN_DEFAULT = np.array([33.618656158447266,
                             8.958210945129395,
                             8.925719261169434], dtype=np.float32)

LAB_STD_DEFAULT = np.array([26.940208435058594,
                            8.05940055847168,
                            9.126977920532227], dtype=np.float32)

##############################################################
# 3. IMPROVED Transforms: RGB / LAB / HYBRID
##############################################################

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RGBTransform:
    def __init__(self, is_train=True):
        if is_train:
            self.geom = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                # Careful with color jitter for skin tone
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.05,  # Very conservative
                    hue=0.02          # Very conservative
                ),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
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
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.05,
                    hue=0.02
                ),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
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
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.05,
                    hue=0.02
                ),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
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
    Dataset for MST classification with person ID tracking
    """
    def __init__(self, df, img_dir, transform, label_mode):
        self.df = df.reset_index(drop=True).copy()
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.label_mode = label_mode

        if self.df.shape[1] < 3:
            raise ValueError("Dataset df must have at least 3 columns: filename, label, person_id")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = str(row.iloc[0])
        label = float(row.iloc[1])
        person_id = str(row.iloc[2])

        img_path = self.img_dir / fname
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to load {img_path}: {e}")
            img = Image.new("RGB", (224, 224), color="black")

        img_t = self.transform(img)

        # MST 1..10 -> class index
        mst = int(round(label))
        mst = max(1, min(10, mst))
        class_idx = map_mst_to_label(mst, self.label_mode)
        
        return img_t, torch.tensor(class_idx, dtype=torch.long), person_id


def create_balanced_subset(df, max_per_class=5000, label_col='label', person_col='person_id', 
                          move_excess_to_val=False):
    """
    Create balanced subset by capping maximum samples per MST class.
    Preserves person-level integrity (doesn't split a person's images).
    
    Args:
        df: Input dataframe with columns [filename, label, person_id]
        max_per_class: Maximum images per MST class
        label_col: Name of label column
        person_col: Name of person ID column
        move_excess_to_val: If True, return excess images separately for validation
    
    Returns:
        If move_excess_to_val=False: balanced_df
        If move_excess_to_val=True: (balanced_df, excess_df)
    """
    df['mst_int'] = df[label_col].astype(float).round().astype(int)
    df['mst_int'] = np.clip(df['mst_int'], 1, 10)
    
    balanced_dfs = []
    excess_dfs = [] if move_excess_to_val else None
    
    print("\n" + "="*70)
    print("[INFO] Creating balanced subset:")
    print(f"[INFO] Maximum {max_per_class} images per class")
    if move_excess_to_val:
        print("[INFO] Moving excess images to validation set")
    print("="*70 + "\n")
    
    for mst in range(1, 11):
        class_df = df[df['mst_int'] == mst].copy()
        n_original = len(class_df)
        
        if n_original <= max_per_class:
            # Keep all samples if below threshold
            balanced_dfs.append(class_df)
            print(f"  MST {mst:2d}: {n_original:6d} → {n_original:6d} (kept all)")
        else:
            # Sample at PERSON level to maintain integrity
            persons = class_df[person_col].unique()
            person_img_counts = class_df.groupby(person_col).size()
            
            # Shuffle persons for randomness
            np.random.seed(42)
            persons_shuffled = np.random.permutation(persons)
            
            selected_persons = []
            excess_persons = []
            total_images = 0
            
            for person in persons_shuffled:
                person_n_images = person_img_counts[person]
                if total_images + person_n_images <= max_per_class:
                    selected_persons.append(person)
                    total_images += person_n_images
                else:
                    excess_persons.append(person)
            
            # Get selected samples
            sampled_df = class_df[class_df[person_col].isin(selected_persons)]
            balanced_dfs.append(sampled_df)
            
            # Handle excess
            if move_excess_to_val and len(excess_persons) > 0:
                excess_df = class_df[class_df[person_col].isin(excess_persons)]
                excess_dfs.append(excess_df)
                print(f"  MST {mst:2d}: {n_original:6d} → {len(sampled_df):6d} "
                      f"({len(selected_persons)} persons, {len(excess_df)} moved to val)")
            else:
                print(f"  MST {mst:2d}: {n_original:6d} → {len(sampled_df):6d} "
                      f"({len(selected_persons)} persons)")
    
    df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    df_balanced = df_balanced.drop(columns=['mst_int'])
    
    print(f"\n[INFO] Original dataset: {len(df):,} images")
    print(f"[INFO] Balanced dataset: {len(df_balanced):,} images")
    print(f"[INFO] Reduction: {100*(1-len(df_balanced)/len(df)):.1f}%")
    
    if move_excess_to_val and excess_dfs:
        df_excess = pd.concat(excess_dfs, ignore_index=True)
        df_excess = df_excess.drop(columns=['mst_int'])
        print(f"[INFO] Excess moved to val: {len(df_excess):,} images")
        print("="*70 + "\n")
        return df_balanced, df_excess
    else:
        print("="*70 + "\n")
        return df_balanced


def build_dataloaders(csv_path, image_dir, batch_size, val_ratio,
                      input_mode, lab_mean, lab_std,
                      label_mode,
                      num_workers=4,
                      balance_strategy="sampling",
                      save_split_path=None,
                      max_samples_per_class=None,
                      move_excess_to_val=False):

    df = pd.read_csv(csv_path).dropna()

    print(f"[INFO] Loaded CSV with columns: {list(df.columns)}")

    # Auto-detect column names
    # Try common column name variations
    filename_col = None
    label_col = None
    person_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if filename_col is None and any(x in col_lower for x in ['filename', 'cropped', 'image', 'file']):
            # Prefer 'cropped_image' if it exists, otherwise 'filename'
            if 'cropped' in col_lower:
                filename_col = col
            elif filename_col is None or 'filename' in col_lower:
                filename_col = col
        if label_col is None and any(x in col_lower for x in ['label', 'mst', 'score']):
            label_col = col
        if person_col is None and any(x in col_lower for x in ['person', 'subject', 'user', 'id']):
            person_col = col
    
    # Fallback to positional if auto-detect fails
    if filename_col is None:
        filename_col = df.columns[0]
        print(f"[WARN] Could not auto-detect filename column, using: {filename_col}")
    if label_col is None:
        label_col = df.columns[1] if len(df.columns) > 1 else None
        print(f"[WARN] Could not auto-detect label column, using: {label_col}")
    if person_col is None:
        person_col = df.columns[2] if len(df.columns) > 2 else None
        print(f"[WARN] Could not auto-detect person_id column, using: {person_col}")
    
    if label_col is None or person_col is None:
        raise ValueError(f"CSV must have at least filename, label, and person_id columns. Found: {list(df.columns)}")
    
    print(f"[INFO] Using columns: filename='{filename_col}', label='{label_col}', person_id='{person_col}'")
    
    # Create standardized dataframe
    df = df[[filename_col, label_col, person_col]].copy()
    df.columns = ["filename", "label", "person_id"]
    df["person_id"] = df["person_id"].astype(str)

    # ============= APPLY CLASS BALANCING IF REQUESTED =============
    excess_df = None
    if max_samples_per_class is not None:
        if move_excess_to_val:
            df, excess_df = create_balanced_subset(
                df, 
                max_per_class=max_samples_per_class,
                label_col="label",
                person_col="person_id",
                move_excess_to_val=True
            )
        else:
            df = create_balanced_subset(
                df, 
                max_per_class=max_samples_per_class,
                label_col="label",
                person_col="person_id",
                move_excess_to_val=False
            )
    # ===============================================================

    # One label per person (first occurrence)
    person_labels = df.groupby("person_id").agg({
        "label": "first"
    }).reset_index()

    person_labels["label_int"] = person_labels["label"].astype(float).round().astype(int)
    person_labels["label_int"] = np.clip(person_labels["label_int"], 1, 10)

    labels_per_person = person_labels["label_int"].apply(
        lambda x: map_mst_to_label(x, label_mode)
    ).values

    print(f"[INFO] Found {len(person_labels)} unique persons (after balancing)")
    print(f"[INFO] Person MST label range: {labels_per_person.min()} to {labels_per_person.max()}")

    # Person distribution
    person_counts = Counter(labels_per_person)
    print("\n[INFO] Person distribution across MST labels (after balancing):")
    for mst in sorted(person_counts.keys()):
        print(f"  Class {mst} (MST {mst+1}): {person_counts[mst]} persons")
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
        "train_persons": train_persons.tolist(),
        "val_persons": val_persons.tolist(),
        "random_state": 42,
        "val_ratio": val_ratio,
        "max_samples_per_class": max_samples_per_class,
        "move_excess_to_val": move_excess_to_val
    }

    split_path = Path(save_split_path) if save_split_path is not None else Path("train_val_split.json")
    with open(split_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"[INFO] Saved train/val split to {split_path}")

    train_df = df[df["person_id"].isin(train_persons)].reset_index(drop=True)
    val_df = df[df["person_id"].isin(val_persons)].reset_index(drop=True)

    # Add excess images to validation if requested
    if excess_df is not None and len(excess_df) > 0:
        print(f"[INFO] Adding {len(excess_df)} excess images to validation set")
        val_df = pd.concat([val_df, excess_df], ignore_index=True)

    print(f"[INFO] Train: {len(train_persons)} persons ({len(train_df)} images)")
    print(f"[INFO] Val:   {len(val_persons)} persons ({len(val_df)} images)")

    # Image-level distributions
    def mst_from_label_series(s):
        arr = s.astype(float).round().astype(int)
        return np.clip(arr, 1, 10)

    train_mst = mst_from_label_series(train_df["label"])
    val_mst = mst_from_label_series(val_df["label"])

    train_counts = Counter(train_mst)
    val_counts = Counter(val_mst)

    print("\n[INFO] Train MST label distribution (images):")
    for mst in sorted(train_counts.keys()):
        print(f"  MST {mst}: {train_counts[mst]} images")

    print("\n[INFO] Val MST label distribution (images):")
    for mst in sorted(val_counts.keys()):
        print(f"  MST {mst}: {val_counts[mst]} images")
    print()

    # ---------- CLASS WEIGHTS (with dampening for better minority class handling) ----------
    train_bins = train_mst.apply(lambda x: map_mst_to_label(x, label_mode))
    class_counts = Counter(train_bins)

    num_classes = 10 if label_mode == "mst10" else 3

    # Compute inverse frequency weights
    total_samples = len(train_bins)
    raw_weights = {}
    for k in range(num_classes):
        c = class_counts.get(k, 1)  # avoid div-by-zero
        raw_weights[k] = total_samples / (num_classes * c)

    # Apply square root dampening to prevent extreme weights
    dampened_weights = {}
    for k in range(num_classes):
        dampened_weights[k] = np.sqrt(raw_weights[k])
    
    # Optional: Cap at maximum value
    max_weight = 5.0
    for k in range(num_classes):
        dampened_weights[k] = min(dampened_weights[k], max_weight)

    print("[INFO] Class weights (sqrt-dampened, capped at 5.0):")
    for k in range(num_classes):
        print(f"  Class {k} (MST {k+1}): raw={raw_weights[k]:.3f}, dampened={dampened_weights[k]:.3f}")
    print()

    # Build weight tensor for loss functions
    class_weight_tensor = torch.zeros(num_classes, dtype=torch.float32)
    for k in range(num_classes):
        class_weight_tensor[k] = dampened_weights.get(k, 1.0)

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

    train_dataset = SkinToneDataset(train_df, image_dir, train_tf, label_mode)
    val_dataset = SkinToneDataset(val_df, image_dir, val_tf, label_mode)

    # =============================================================
    #  SAMPLER FOR BALANCED TRAINING
    # =============================================================
    if balance_strategy == "sampling":
        class_freq = Counter(train_bins)

        img_weights = []
        for _, row in train_df.iterrows():
            mst = int(round(float(row["label"])))
            mst = max(1, min(10, mst))
            bin_id = map_mst_to_label(mst, label_mode)
            freq = class_freq[bin_id]
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

    return train_loader, val_loader, class_weight_tensor


##############################################################
# LABEL SMOOTHING FOCAL LOSS (FIXED)
##############################################################

class LabelSmoothingFocalLoss(nn.Module):
    """
    FIXED: Numerically stable focal loss with label smoothing
    """
    def __init__(self, num_classes, alpha=None, gamma=2.0, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, targets):
        """
        logits: [B, C]
        targets: [B]
        """
        # Create smoothed labels
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)

        # FIXED: Use F.softmax directly instead of torch.exp(log_probs)
        log_probs = F.log_softmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)

        # Get probability of true class for focal weight
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Clamp pt to prevent numerical instability
        pt = torch.clamp(pt, min=1e-7, max=1.0 - 1e-7)

        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_weight = alpha_t * (1 - pt) ** self.gamma
        else:
            focal_weight = (1 - pt) ** self.gamma
        
        # Clamp focal weight to prevent extreme values
        focal_weight = torch.clamp(focal_weight, max=100.0)

        # Compute loss with smoothed labels
        loss = -(focal_weight.unsqueeze(1) * log_probs * true_dist).sum(dim=1)
        
        # Safety check for NaN/Inf
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("[WARNING] NaN/Inf in focal loss, using cross-entropy fallback")
            loss = -(log_probs * true_dist).sum(dim=1)

        return loss.mean()


##############################################################
# SUPERVISED CONTRASTIVE LOSS (FIXED)
##############################################################

class SupervisedContrastiveLoss(nn.Module):
    """
    FIXED: Stable contrastive loss with proper temperature
    """
    def __init__(self, temperature=0.5, base_temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """
        features: [B, D] - normalized feature vectors
        labels: [B] - class labels (0-9 for MST)
        """
        device = features.device
        batch_size = features.shape[0]

        # Normalize features
        features = F.normalize(features, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # FIXED: Clamp similarity matrix to prevent overflow
        similarity_matrix = torch.clamp(similarity_matrix, min=-10, max=10)

        # Create mask for positive pairs (same class)
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Mask out self-similarity
        logits_mask = torch.ones_like(mask).fill_diagonal_(0)
        mask = mask * logits_mask

        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Compute mean of log-likelihood over positive pairs
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


##############################################################
# 5. VGG16 MST Classifier - SINGLE TASK
##############################################################

class VGG16MSTClassifier(nn.Module):
    """
    SINGLE-TASK VGG16 for 10-class MST classification
    
    Removed:
    - 3-bin auxiliary classifier
    - bin_logits output
    - Multi-task complexity
    
    Focused on:
    - Clean 10-class MST prediction
    - Optional contrastive learning
    """
    def __init__(self, input_mode="rgb", use_bn=True, dropout_p=0.5):
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

        # Initial freeze (will be dynamically unfrozen during training)
        for p in self.features.parameters():
            p.requires_grad = False

        # Special handling for LAB mode - reinitialize first conv
        if input_mode == "lab":
            nn.init.kaiming_normal_(features[0].weight, mode='fan_out', nonlinearity='relu')
            if features[0].bias is not None:
                nn.init.zeros_(features[0].bias)
            # Make first conv trainable
            for p in features[0].parameters():
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

        # Shared feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(flat_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
        )

        # SINGLE-TASK: Only MST classifier (10 classes)
        self.mst_classifier = nn.Linear(512, 10)

        # Projection head for contrastive learning (optional)
        self.projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )

        self._init_head_weights()

    def _init_head_weights(self):
        for m in [self.feature_extractor, self.mst_classifier, self.projection]:
            for module in m.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, 0, 0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, x, return_features=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        features = self.feature_extractor(x)

        # SINGLE OUTPUT: Only MST logits
        mst_logits = self.mst_classifier(features)

        if return_features:
            # Return normalized projection for contrastive loss
            proj = F.normalize(self.projection(features), dim=1)
            return mst_logits, proj

        return mst_logits

    def unfreeze_gradually(self, epoch, total_epochs):
        """
        Gradually unfreeze layers during training for better fine-tuning.
        
        Phases:
        - Phase 1 (0-30%): Only head layers trainable
        - Phase 2 (30-60%): Last 2 VGG blocks trainable
        - Phase 3 (60-100%): All layers trainable
        """
        progress = epoch / total_epochs

        if progress < 0.3:  # First 30%: only head
            for p in self.features.parameters():
                if self.input_mode == "lab":
                    # Keep first conv trainable for LAB
                    if p in list(self.features[0].parameters()):
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
                else:
                    p.requires_grad = False
            status = "Phase 1: Head only"

        elif progress < 0.6:  # Next 30%: last 2 blocks
            for i, module in enumerate(self.features):
                if i >= 17:  # Last 2 blocks in VGG16
                    for p in module.parameters():
                        p.requires_grad = True
                elif self.input_mode == "lab" and i == 0:
                    # Keep first conv trainable for LAB
                    for p in module.parameters():
                        p.requires_grad = True
                else:
                    for p in module.parameters():
                        p.requires_grad = False
            status = "Phase 2: Last 2 blocks"

        else:  # Final 40%: all layers
            for p in self.features.parameters():
                p.requires_grad = True
            status = "Phase 3: All layers"

        return status


##############################################################
# 6. Metrics
##############################################################

def tone_bin_from_class_idx(class_idx, label_mode):
    if label_mode == "mst3":
        return int(class_idx)  # Already binned
    elif label_mode == "mst10":
        idx = int(class_idx)
        if idx <= 2:
            return 0
        elif idx <= 6:
            return 1
        else:
            return 2
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")


def compute_classification_metrics(preds, targets, label_mode):
    """
    preds, targets: 1D torch tensors of class idx 0..9
    Returns dict with various accuracy metrics
    """
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()

    correct = (preds == targets).float()
    top1 = correct.mean().item() * 100.0

    off1 = (preds - targets).abs() <= 1
    off1_acc = off1.float().mean().item() * 100.0

    # 3-bin accuracy
    preds_bin = torch.tensor([tone_bin_from_class_idx(i.item(), label_mode) for i in preds])
    targets_bin = torch.tensor([tone_bin_from_class_idx(i.item(), label_mode) for i in targets])
    three_bin = (preds_bin == targets_bin).float().mean().item() * 100.0

    # L2-LAB (Monk-based)
    preds_np = preds.numpy()
    targets_np = targets.numpy()
    l2 = calc_l2_lab_distance_classes(preds_np, targets_np)
    l2_mean = float(l2.mean())
    l2_std = float(l2.std())

    return {
        "top1": top1,
        "off1": off1_acc,
        "three_bin": three_bin,
        "l2_mean": l2_mean,
        "l2_std": l2_std,
    }


##############################################################
# 7. DIAGNOSTIC TOOLS
##############################################################

def analyze_predictions(all_preds, all_targets, all_person_ids, label_mode, split_name="Val"):
    """
    Comprehensive analysis of model predictions including:
    - Per-class accuracy
    - Confusion patterns
    - Person-level vs image-level comparison
    """
    print(f"\n{'='*60}")
    print(f"PREDICTION ANALYSIS - {split_name}")
    print(f"{'='*60}")

    # Convert to numpy
    preds = all_preds.numpy()
    targets = all_targets.numpy()

    # Overall metrics
    metrics = compute_classification_metrics(all_preds, all_targets, label_mode)
    print(f"\nOverall Metrics:")
    print(f"  Top-1 Accuracy: {metrics['top1']:.2f}%")
    print(f"  Off-by-1 Accuracy: {metrics['off1']:.2f}%")
    print(f"  3-Bin Accuracy: {metrics['three_bin']:.2f}%")
    print(f"  L2-LAB Distance: {metrics['l2_mean']:.3f} ± {metrics['l2_std']:.3f}")

    # Per-class analysis
    print(f"\nPer-Class Analysis:")
    predictions_by_true_label = defaultdict(list)
    for pred, true in zip(preds, targets):
        predictions_by_true_label[true].append(pred)

    for true_label in sorted(predictions_by_true_label.keys()):
        preds_for_class = predictions_by_true_label[true_label]
        correct = sum(1 for p in preds_for_class if p == true_label)
        total = len(preds_for_class)
        acc = 100.0 * correct / total if total > 0 else 0.0

        pred_dist = Counter(preds_for_class)
        mode_pred = pred_dist.most_common(1)[0][0] if pred_dist else -1

        print(f"  MST {true_label+1} (n={total}):")
        print(f"    Accuracy: {acc:.2f}%")
        print(f"    Most predicted: MST {mode_pred+1} ({pred_dist[mode_pred]} times)")
        
        # Show distribution if accuracy is low
        if acc < 50:
            top3 = pred_dist.most_common(3)
            dist_str = ", ".join([f"MST{p+1}:{c}" for p, c in top3])
            print(f"    Distribution: {dist_str}")

    # Confusion matrix visualization (simplified)
    print(f"\nConfusion Summary (most common mistakes):")
    confusion_pairs = defaultdict(int)
    for pred, true in zip(preds, targets):
        if pred != true:
            confusion_pairs[(true, pred)] += 1

    for (true, pred), count in sorted(confusion_pairs.items(), 
                                      key=lambda x: x[1], reverse=True)[:10]:
        print(f"  MST {true+1} → MST {pred+1}: {count} times")

    print(f"{'='*60}\n")


##############################################################
# 8. Training / Validation - SINGLE TASK
##############################################################

def train_one_epoch(model, loader, mst_criterion, contrastive_criterion,
                    optimizer, device, epoch, total_epochs, label_mode,
                    use_contrastive=True, lambda_contrast=0.1):
    """
    SINGLE-TASK training with:
    - Main MST classification loss
    - Optional supervised contrastive loss
    """
    model.train()

    # Gradually unfreeze layers
    unfreeze_status = model.unfreeze_gradually(epoch, total_epochs)

    running_loss = 0.0
    running_mst_loss = 0.0
    running_contrast_loss = 0.0
    
    correct_top1 = 0
    correct_off1 = 0
    correct_three_bin = 0
    total_samples = 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch} [{unfreeze_status}]", ncols=120)

    for imgs, labels, person_ids in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass
        if use_contrastive:
            mst_logits, features = model(imgs, return_features=True)
        else:
            mst_logits = model(imgs, return_features=False)
            features = None

        # Main MST loss
        mst_loss = mst_criterion(mst_logits, labels)

        # Contrastive loss (optional)
        if use_contrastive and features is not None:
            contrast_loss = contrastive_criterion(features, labels)
            total_loss = mst_loss + lambda_contrast * contrast_loss
        else:
            contrast_loss = torch.tensor(0.0, device=device)
            total_loss = mst_loss

        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        # Metrics
        preds = torch.argmax(mst_logits, dim=1)
        batch_size = imgs.size(0)
        
        running_loss += total_loss.item() * batch_size
        running_mst_loss += mst_loss.item() * batch_size
        running_contrast_loss += contrast_loss.item() * batch_size
        
        correct_top1 += (preds == labels).sum().item()
        correct_off1 += ((preds - labels).abs() <= 1).sum().item()

        preds_bin = torch.tensor([tone_bin_from_class_idx(i.item(), label_mode) 
                                 for i in preds.cpu()])
        labels_bin = torch.tensor([tone_bin_from_class_idx(i.item(), label_mode) 
                                  for i in labels.cpu()])
        correct_three_bin += (preds_bin == labels_bin).sum().item()

        total_samples += batch_size

        pbar.set_postfix({
            "loss": f"{total_loss.item():.4f}",
            "mst": f"{mst_loss.item():.3f}",
            "top1": f"{100.0*correct_top1/total_samples:.1f}",
        })

    avg_loss = running_loss / total_samples
    avg_mst_loss = running_mst_loss / total_samples
    avg_contrast_loss = running_contrast_loss / total_samples
    avg_top1 = 100.0 * correct_top1 / total_samples
    avg_off1 = 100.0 * correct_off1 / total_samples
    avg_three_bin = 100.0 * correct_three_bin / total_samples

    return {
        "loss": avg_loss,
        "mst_loss": avg_mst_loss,
        "contrast_loss": avg_contrast_loss,
        "top1": avg_top1,
        "off1": avg_off1,
        "three_bin": avg_three_bin,
    }


def validate(model, loader, mst_criterion, device, epoch, label_mode, analyze=False):
    """
    SINGLE-TASK validation with image-level and person-level metrics
    """
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

            # SINGLE OUTPUT
            mst_logits = model(imgs, return_features=False)

            # MST loss only
            mst_loss = mst_criterion(mst_logits, labels)

            preds = torch.argmax(mst_logits, dim=1)

            batch_size = imgs.size(0)
            running_loss += mst_loss.item() * batch_size
            total_samples += batch_size

            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())
            all_person_ids.extend(person_ids)

            batch_metrics = compute_classification_metrics(preds, labels, label_mode)
            pbar.set_postfix({
                "loss": f"{mst_loss.item():.4f}",
                "top1": f"{batch_metrics['top1']:.2f}",
            })

    avg_loss = running_loss / total_samples
    
    preds_all = torch.cat(all_preds, dim=0)
    targets_all = torch.cat(all_targets, dim=0)

    # Image-level metrics
    img_metrics = compute_classification_metrics(preds_all, targets_all, label_mode)

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
    person_metrics = compute_classification_metrics(person_preds, person_targets, label_mode)

    # Optional detailed analysis
    if analyze:
        analyze_predictions(preds_all, targets_all, all_person_ids, label_mode, "Validation")

    return {
        "loss": avg_loss,
        "img_metrics": img_metrics,
        "person_metrics": person_metrics,
    }


##############################################################
# 9. Checkpoint Management
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
# 10. Early Stopping
##############################################################

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.1):
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
# 11. MAIN
##############################################################

def main():
    parser = argparse.ArgumentParser(description="Single-Task VGG16 MST Classifier")
    
    # Data arguments
    parser.add_argument("--csv-path", required=True, help="Path to CSV with image labels")
    parser.add_argument("--image-dir", required=True, help="Directory containing images")
    parser.add_argument("--save-dir", default="./checkpoints", help="Directory to save models")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--additional-epochs", type=int, default=None, 
                        help="Additional epochs when resuming")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=1e-4, 
                        help="Learning rate for head; backbone uses lr*0.1")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    
    # Model arguments
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use-bn", action="store_true", help="Use batch normalization")
    parser.add_argument("--input-mode", choices=["rgb", "lab", "hybrid"], default="rgb")
    parser.add_argument("--dropout", type=float, default=0.5)
    
    # LAB color space arguments
    parser.add_argument("--compute-lab-stats", action="store_true",
                        help="Compute LAB statistics from dataset")
    
    # Loss and balancing arguments
    parser.add_argument("--label-mode", choices=["mst10", "mst3"], default="mst10")
    parser.add_argument("--balance-strategy", choices=["weights", "sampling", "none"], 
                        default="sampling")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing factor (0.0-1.0)")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Gamma parameter for focal loss")
    
    # Dataset balancing arguments
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Maximum images per MST class for training (e.g., 5000). None = use all data")
    parser.add_argument("--move-excess-to-val", action="store_true",
                        help="Move excess images (above max-per-class) to validation set instead of discarding")
    
    # Contrastive learning arguments
    parser.add_argument("--lambda-contrast", type=float, default=0.1,
                        help="Weight for contrastive loss")
    parser.add_argument("--no-contrastive", action="store_true",
                        help="Disable contrastive learning")
    
    # Optimization arguments
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--early-stop-delta", type=float, default=0.1)
    
    # Analysis arguments
    parser.add_argument("--analyze-every", type=int, default=5,
                        help="Run detailed analysis every N epochs")

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*70)
    print("VGG16 MST CLASSIFIER - SINGLE-TASK VERSION")
    print("="*70)
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Input mode: {args.input_mode}")
    print(f"[INFO] Label smoothing: {args.label_smoothing}")
    print(f"[INFO] Balance strategy: {args.balance_strategy}")
    print(f"[INFO] Contrastive learning: {'Enabled' if not args.no_contrastive else 'Disabled'}")
    if not args.no_contrastive:
        print(f"[INFO] Contrastive weight: {args.lambda_contrast}")
    if args.max_per_class is not None:
        print(f"[INFO] Dataset balancing: max {args.max_per_class} images per class")
        print(f"[INFO] Move excess to val: {args.move_excess_to_val}")
    print("="*70 + "\n")

    # LAB stats
    if args.input_mode in ("lab", "hybrid"):
        if args.compute_lab_stats:
            print("[INFO] Computing LAB statistics from dataset...")
            lab_mean, lab_std = compute_dataset_lab_stats(args.image_dir, args.csv_path)
        else:
            print("[INFO] Using default LAB normalization values.")
            lab_mean, lab_std = LAB_MEAN_DEFAULT, LAB_STD_DEFAULT
        print(f"[INFO] LAB_MEAN: {lab_mean.tolist()}")
        print(f"[INFO] LAB_STD: {lab_std.tolist()}\n")
    else:
        lab_mean, lab_std = LAB_MEAN_DEFAULT, LAB_STD_DEFAULT

    # Setup save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_path = save_dir / "vgg16_mst_best.pth"
    final_path = save_dir / "vgg16_mst_final.pth"
    checkpoint_path = save_dir / "vgg16_mst_checkpoint.pth"
    save_split_path = save_dir / "train_val_split.json"

    # Build dataloaders
    train_loader, val_loader, class_weight_tensor = build_dataloaders(
        args.csv_path,
        args.image_dir,
        args.batch_size,
        args.val_ratio,
        args.input_mode,
        lab_mean,
        lab_std,
        args.label_mode,
        num_workers=args.num_workers,
        balance_strategy=args.balance_strategy,
        save_split_path=save_split_path,
        max_samples_per_class=args.max_per_class,
        move_excess_to_val=args.move_excess_to_val
    )

    # Build SINGLE-TASK model
    model = VGG16MSTClassifier(
        input_mode=args.input_mode,
        use_bn=args.use_bn,
        dropout_p=args.dropout,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters: {total_params:,}")
    print(f"[INFO] Trainable parameters: {trainable_params:,}\n")

    # Optimizer with parameter groups
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
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": base_lr * 0.1},
            {"params": head_params, "lr": base_lr},
        ],
        weight_decay=args.weight_decay,
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=True,
    )

    # Loss functions
    class_weight_tensor = class_weight_tensor.to(device)
    
    # SINGLE LOSS: MST classification only
    if args.balance_strategy == "weights":
        mst_criterion = LabelSmoothingFocalLoss(
            num_classes=10,
            alpha=class_weight_tensor,
            gamma=args.focal_gamma,
            smoothing=args.label_smoothing
        )
        print("[INFO] Using Label Smoothing Focal Loss WITH class weights")
    else:
        mst_criterion = LabelSmoothingFocalLoss(
            num_classes=10,
            alpha=None,
            gamma=args.focal_gamma,
            smoothing=args.label_smoothing
        )
        print("[INFO] Using Label Smoothing Focal Loss WITHOUT class weights")
    
    # Contrastive loss (optional)
    contrastive_criterion = SupervisedContrastiveLoss(temperature=0.5)

    # Resume or start fresh
    if args.resume:
        start_epoch, best_val_loss, best_val_acc = load_checkpoint(
            model, optimizer, scheduler, checkpoint_path, device
        )
        if args.additional_epochs is not None:
            max_epoch = start_epoch - 1 + args.additional_epochs
            print(f"[INFO] Training for {args.additional_epochs} additional epochs (until {max_epoch})")
        else:
            max_epoch = args.epochs
            if start_epoch > max_epoch:
                print(f"[WARNING] Checkpoint at epoch {start_epoch-1} exceeds --epochs {max_epoch}")
                return
            print(f"[INFO] Resuming until epoch {max_epoch}")
    else:
        start_epoch = 1
        max_epoch = args.epochs
        best_val_loss = float("inf")
        best_val_acc = 0.0
        print(f"[INFO] Starting fresh training for {max_epoch} epochs\n")

    # Early stopping
    early_stopper = EarlyStopping(
        patience=args.early_stop_patience,
        min_delta=args.early_stop_delta,
    )

    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")

    for epoch in range(start_epoch, max_epoch + 1):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/{max_epoch}")
        print(f"{'='*70}")

        # Training (SINGLE-TASK)
        train_results = train_one_epoch(
            model, train_loader,
            mst_criterion, contrastive_criterion,
            optimizer, device, epoch, max_epoch, args.label_mode,
            use_contrastive=not args.no_contrastive,
            lambda_contrast=args.lambda_contrast
        )

        # Validation (SINGLE-TASK)
        analyze_this_epoch = (epoch % args.analyze_every == 0) or (epoch == max_epoch)
        val_results = validate(
            model, val_loader,
            mst_criterion,
            device, epoch, args.label_mode,
            analyze=analyze_this_epoch
        )

        # Print epoch summary
        print(f"\n[EPOCH {epoch} SUMMARY]")
        print(f"Train: Loss={train_results['loss']:.4f} "
              f"(MST={train_results['mst_loss']:.4f}, "
              f"Contrast={train_results['contrast_loss']:.4f})")
        print(f"       Top1={train_results['top1']:.2f}% "
              f"Off1={train_results['off1']:.2f}% "
              f"3bin={train_results['three_bin']:.2f}%")
        
        print(f"Val:   Loss={val_results['loss']:.4f}")
        print(f"       Image-level: Top1={val_results['img_metrics']['top1']:.2f}% "
              f"Off1={val_results['img_metrics']['off1']:.2f}% "
              f"3bin={val_results['img_metrics']['three_bin']:.2f}%")
        print(f"       Person-level: Top1={val_results['person_metrics']['top1']:.2f}% "
              f"Off1={val_results['person_metrics']['off1']:.2f}% "
              f"3bin={val_results['person_metrics']['three_bin']:.2f}%")
        print(f"       L2-LAB: {val_results['person_metrics']['l2_mean']:.3f} ± "
              f"{val_results['person_metrics']['l2_std']:.3f}")

        # Update scheduler
        scheduler.step(val_results['loss'])

        # Track best model (person-level top-1)
        current_val_acc = val_results['person_metrics']['top1']
        
        if val_results['loss'] < best_val_loss:
            best_val_loss = val_results['loss']
        
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            torch.save(model.state_dict(), best_path)
            print(f"\n[INFO] ✓ New best model saved (ValAcc: {best_val_acc:.2f}%)")

        # Save checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            current_val_acc, val_results['loss'], checkpoint_path
        )

        # Early stopping check
        if early_stopper(current_val_acc):
            print(f"\n[INFO] Early stopping triggered at epoch {epoch}")
            break

    # Save final model
    torch.save(model.state_dict(), final_path)
    print(f"\n[INFO] Final model saved -> {final_path}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best Person-Level Accuracy: {best_val_acc:.2f}%")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

# python vgg16_mst_improved_singlehead.py ^
#   --csv-path "G:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\annotations.csv" ^
#   --image-dir "G:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2" ^
#   --save-dir "G:\Thesis\CasualConversationv2_Dataset\Models\SingleTask_v1" ^
#   --gpu 0 ^
#   --input-mode rgb ^
#   --use-bn ^
#   --max-per-class 8000 ^
#   --balance-strategy weights ^
#   --label-smoothing 0.15 ^
#   --focal-gamma 0.5 ^
#   --lambda-contrast 0.3 ^
#   --dropout 0.5 ^
#   --weight-decay 2e-4 ^
#   --lr 3e-4 ^
#   --batch-size 32 ^
#   --epochs 40