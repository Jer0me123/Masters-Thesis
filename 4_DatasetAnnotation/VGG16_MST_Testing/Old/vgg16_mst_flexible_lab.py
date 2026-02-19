##############################################################
#  VGG16 MST CLASSIFIER - FLEXIBLE BINNING VERSION
#  
#  *** FIXED LAB COLOR SPACE IMPLEMENTATION ***
#  
#  CHANGES FROM ORIGINAL:
#  1. LABTransform: Removed RGB ColorJitter, added LAB-native augmentation
#  2. compute_dataset_lab_stats: Added include_augmentation option
#  3. VGG16MSTClassifier: Fixed LAB initialization (adapted from ImageNet)
#  4. Added undersampling diagnostics
#  
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
# LABEL MAPPING UTILITIES
##############################################################

class LabelMapper:
    """
    Handles flexible label mapping from MST values to class indices.
    Supports loading from JSON files with custom binning configurations.
    """
    def __init__(self, mapping_config=None):
        """
        Args:
            mapping_config: Path to JSON file or dict with mapping configuration
        """
        if mapping_config is None:
            # Default: MST 1-10 → class 0-9
            self.mapping = {str(i): i-1 for i in range(1, 11)}
            self.num_classes = 10
            self.config_name = "mst10"
        elif isinstance(mapping_config, str) or isinstance(mapping_config, Path):
            # Load from JSON file
            with open(mapping_config, 'r') as f:
                config = json.load(f)
            self._parse_config(config)
            self.config_name = Path(mapping_config).stem
        elif isinstance(mapping_config, dict):
            # Direct dict config
            self._parse_config(mapping_config)
            self.config_name = "custom"
        else:
            raise ValueError("mapping_config must be None, path to JSON, or dict")
    
    def _parse_config(self, config):
        """Parse configuration from dict"""
        if "label_mapping" in config:
            # Direct mapping format: {"label_mapping": {"1": 0, "2": 0, ...}}
            self.mapping = {str(k): int(v) for k, v in config["label_mapping"].items()}
            self.num_classes = config.get("num_classes", max(self.mapping.values()) + 1)
        elif "bins" in config:
            # Bins format: {"bins": [{"range": [1, 2], "class": 0}, ...]}
            self.mapping = {}
            for bin_def in config["bins"]:
                mst_range = bin_def["range"]
                class_id = bin_def["class"]
                for mst in range(mst_range[0], mst_range[1] + 1):
                    self.mapping[str(mst)] = class_id
            self.num_classes = config.get("num_classes", max(self.mapping.values()) + 1)
        else:
            raise ValueError("Config must have 'label_mapping' or 'bins' key")
        
        # Validate mapping covers MST 1-10
        for mst in range(1, 11):
            if str(mst) not in self.mapping:
                raise ValueError(f"Mapping does not cover MST {mst}")
        
        # Validate num_classes matches mapping
        max_class = max(self.mapping.values())
        if max_class >= self.num_classes:
            raise ValueError(f"num_classes={self.num_classes} but mapping has class {max_class}")
    
    def map_mst_to_class(self, mst_value):
        """Map MST value (1-10) to class index"""
        return self.mapping[str(int(mst_value))]
    
    def get_num_classes(self):
        """Get number of classes"""
        return self.num_classes
    
    def get_class_name(self, class_idx):
        """Get human-readable name for class"""
        # Find all MST values that map to this class
        mst_values = [int(k) for k, v in self.mapping.items() if v == class_idx]
        mst_values.sort()
        
        if len(mst_values) == 0:
            return f"Class{class_idx}"
        elif len(mst_values) == 1:
            return f"MST{mst_values[0]}"
        else:
            # Check if continuous range
            if mst_values == list(range(mst_values[0], mst_values[-1] + 1)):
                return f"MST{mst_values[0]}-{mst_values[-1]}"
            else:
                return f"MST{','.join(map(str, mst_values))}"
    
    def print_mapping(self):
        """Print the mapping configuration"""
        print("\n" + "="*70)
        print("LABEL MAPPING CONFIGURATION")
        print("="*70)
        print(f"Configuration: {self.config_name}")
        print(f"Number of classes: {self.num_classes}")
        print("\nMapping:")
        for class_idx in range(self.num_classes):
            mst_values = [int(k) for k, v in self.mapping.items() if v == class_idx]
            mst_values.sort()
            print(f"  Class {class_idx}: MST {mst_values} ({self.get_class_name(class_idx)})")
        print("="*70 + "\n")


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


def calc_l2_lab_distance_classes(pred_classes, tgt_classes, label_mapper):
    """
    pred_classes, tgt_classes: numpy arrays of class indices.
    Convert to MST values, map to Monk LAB, compute L2 distance.
    """
    # For binned classes, use the midpoint of the MST range
    pred_mst = []
    tgt_mst = []
    
    for p_class, t_class in zip(pred_classes, tgt_classes):
        # Get MST values for this class
        pred_mst_vals = [int(k) for k, v in label_mapper.mapping.items() if v == int(p_class)]
        tgt_mst_vals = [int(k) for k, v in label_mapper.mapping.items() if v == int(t_class)]
        
        # Use midpoint
        pred_mst.append(np.mean(pred_mst_vals))
        tgt_mst.append(np.mean(tgt_mst_vals))
    
    pred_mst = np.array(pred_mst, dtype=np.float32)
    tgt_mst = np.array(tgt_mst, dtype=np.float32)
    
    pred_lab = monk_scalar_to_lab(pred_mst)
    tgt_lab = monk_scalar_to_lab(tgt_mst)
    return np.sqrt(((pred_lab - tgt_lab) ** 2).sum(axis=1))


##############################################################
# 2. LAB DATASET STATISTICS - FIXED
##############################################################

def compute_dataset_lab_stats(image_dir, csv_path, include_augmentation=True, save_path=None):
    """
    FIXED: Compute LAB statistics with option to include augmentation.
    
    CHANGE 1: Added include_augmentation parameter
    CHANGE 2: If True, samples multiple augmented versions per image
    
    Args:
        image_dir: Directory containing images
        csv_path: Path to CSV with image filenames
        include_augmentation: If True, compute stats on augmented images
    """
    df = pd.read_csv(csv_path)
    img_names = df.iloc[:, 0].astype(str).tolist()

    print(f"[INFO] Computing LAB dataset statistics on {len(img_names)} images...")
    if include_augmentation:
        print("[INFO] Including LAB augmentation in statistics (RECOMMENDED)")
    else:
        print("[INFO] Computing on clean images only")

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

        if include_augmentation:
            # CHANGE: Sample multiple augmented versions
            for _ in range(3):  # 3 augmented versions per image
                rgb = np.asarray(img).astype(np.float32) / 255.0
                lab = rgb2lab(rgb).astype(np.float64)
                
                # Apply LAB augmentation
                lab = _augment_lab_for_stats(lab)
                
                lab_flat = lab.reshape(-1, 3)
                sum_lab += lab_flat.sum(axis=0)
                sum_sq_lab += (lab_flat ** 2).sum(axis=0)
                total_pixels += lab_flat.shape[0]
        else:
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

    # ========== NEW: SAVE TO FILE ==========
    if save_path is not None:
        stats = {
            "lab_mean": mean.tolist(),
            "lab_std": std.tolist(),
            "num_images": len(img_names),
            "include_augmentation": include_augmentation,
            "computed_date": str(pd.Timestamp.now())
        }
        
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"[INFO] LAB statistics saved to: {save_path}")
    # ========================================

    return mean.astype(np.float32), std.astype(np.float32)

def load_lab_stats(stats_path):
    """
    Load LAB statistics from saved JSON file
    
    Args:
        stats_path: Path to LAB statistics JSON file
        
    Returns:
        lab_mean, lab_std as numpy arrays
    """
    if not Path(stats_path).exists():
        return None, None
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    lab_mean = np.array(stats["lab_mean"], dtype=np.float32)
    lab_std = np.array(stats["lab_std"], dtype=np.float32)
    
    print(f"[INFO] Loaded LAB statistics from: {stats_path}")
    print(f"[INFO] Computed on {stats['num_images']} images")
    print(f"[INFO] With augmentation: {stats['include_augmentation']}")
    print(f"[INFO] LAB_MEAN: {lab_mean.tolist()}")
    print(f"[INFO] LAB_STD: {lab_std.tolist()}\n")
    
    return lab_mean, lab_std

def _augment_lab_for_stats(lab):
    """
    CHANGE: Helper function for LAB augmentation during stats computation
    Same augmentation as used in training
    """
    # Augment L (lightness) - simulates lighting changes
    l_shift = np.random.uniform(-10, 10)
    l_scale = np.random.uniform(0.85, 1.15)
    lab[:, :, 0] = np.clip(lab[:, :, 0] * l_scale + l_shift, 0, 100)
    
    # Augment a, b (color) - small changes only
    ab_shift = np.random.uniform(-5, 5, size=2)
    ab_scale = np.random.uniform(0.95, 1.05, size=2)
    lab[:, :, 1] = np.clip(lab[:, :, 1] * ab_scale[0] + ab_shift[0], -128, 127)
    lab[:, :, 2] = np.clip(lab[:, :, 2] * ab_scale[1] + ab_shift[1], -128, 127)
    
    return lab


# Default LAB stats (computed without augmentation - should be recomputed)
LAB_MEAN_DEFAULT = np.array([33.618656158447266,
                             8.958210945129395,
                             8.925719261169434], dtype=np.float32)

LAB_STD_DEFAULT = np.array([26.940208435058594,
                            8.05940055847168,
                            9.126977920532227], dtype=np.float32)

##############################################################
# 3. IMPROVED Transforms: RGB / LAB (FIXED) / HYBRID
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
        self.norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, img_pil):
        img_pil = self.geom(img_pil)
        t = self.to_tensor(img_pil)
        t = self.norm(t)
        return t


class LABTransform:
    """
    FIXED LAB TRANSFORM
    
    CHANGES:
    1. Removed ColorJitter from geometric transforms
    2. Added LAB-native augmentation (_augment_lab)
    3. Augmentation applied AFTER LAB conversion, BEFORE normalization
    """
    def __init__(self, is_train=True, lab_mean=None, lab_std=None):
        self.lab_mean = np.asarray(lab_mean, dtype=np.float32)
        self.lab_std = np.asarray(lab_std, dtype=np.float32)

        if is_train:
            # CHANGE 1: Removed ColorJitter - doesn't make sense for LAB
            self.geom = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                # NO ColorJitter here!
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            ])
            self.use_lab_augmentation = True
        else:
            self.geom = transforms.Resize((224, 224))
            self.use_lab_augmentation = False

    def _augment_lab(self, lab):
        """
        CHANGE 2: LAB-native augmentation
        
        Augment directly in LAB space (more principled than RGB ColorJitter)
        - L channel: Simulates lighting variations
        - a, b channels: Small color shifts
        """
        if not self.use_lab_augmentation:
            return lab
        
        # Augment L (lightness) - simulates lighting changes
        l_shift = np.random.uniform(-10, 10)
        l_scale = np.random.uniform(0.85, 1.15)
        lab[:, :, 0] = np.clip(lab[:, :, 0] * l_scale + l_shift, 0, 100)
        
        # Augment a, b (color) - small changes only
        ab_shift = np.random.uniform(-5, 5, size=2)
        ab_scale = np.random.uniform(0.95, 1.05, size=2)
        lab[:, :, 1] = np.clip(lab[:, :, 1] * ab_scale[0] + ab_shift[0], -128, 127)
        lab[:, :, 2] = np.clip(lab[:, :, 2] * ab_scale[1] + ab_shift[1], -128, 127)
        
        return lab

    def __call__(self, img_pil):
        # Apply geometric transforms (NO ColorJitter)
        img_pil = self.geom(img_pil)
        
        # Convert to LAB
        rgb = np.asarray(img_pil).astype(np.float32) / 255.0
        lab = rgb2lab(rgb).astype(np.float32)
        
        # CHANGE 3: Augment in LAB space (AFTER conversion, BEFORE normalization)
        lab = self._augment_lab(lab)
        
        # Normalize with LAB statistics
        lab_norm = (lab - self.lab_mean) / self.lab_std
        
        # Convert to tensor
        lab_chw = torch.from_numpy(lab_norm.transpose(2, 0, 1)).float()
        return lab_chw


class HybridTransform:
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
        img_pil = self.geom(img_pil)
        rgb_tensor = self.rgb_norm(self.to_tensor(img_pil))
        rgb_np = np.asarray(img_pil).astype(np.float32) / 255.0
        lab = rgb2lab(rgb_np).astype(np.float32)
        lab_norm = (lab - self.lab_mean) / self.lab_std
        lab_tensor = torch.from_numpy(lab_norm.transpose(2, 0, 1)).float()
        hybrid = torch.cat([rgb_tensor, lab_tensor], dim=0)
        return hybrid


##############################################################
# 4. Dataset + Dataloaders
##############################################################

class SkinToneDataset(Dataset):
    """Dataset for MST classification with flexible label mapping"""
    def __init__(self, df, img_dir, transform, label_mapper):
        self.df = df.reset_index(drop=True).copy()
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.label_mapper = label_mapper

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

        # MST 1..10 -> class index using mapper
        mst = int(round(label))
        mst = max(1, min(10, mst))
        class_idx = self.label_mapper.map_mst_to_class(mst)
        
        return img_t, torch.tensor(class_idx, dtype=torch.long), person_id


def create_balanced_subset(df, max_per_class=5000, label_col='label', person_col='person_id', 
                          move_excess_to_val=False):
    """Create balanced subset by capping maximum samples per MST class"""
    df['mst_int'] = df[label_col].astype(float).round().astype(int)
    df['mst_int'] = np.clip(df['mst_int'], 1, 10)
    
    balanced_dfs = []
    excess_dfs = [] if move_excess_to_val else None
    
    print("\n" + "="*70)
    print("[INFO] Creating balanced subset:")
    print(f"[INFO] Maximum {max_per_class} images per MST class")
    if move_excess_to_val:
        print("[INFO] Moving excess images to validation set")
    print("="*70 + "\n")
    
    for mst in range(1, 11):
        class_df = df[df['mst_int'] == mst].copy()
        n_original = len(class_df)
        
        if n_original <= max_per_class:
            balanced_dfs.append(class_df)
            print(f"  MST {mst:2d}: {n_original:6d} → {n_original:6d} (kept all)")
        else:
            persons = class_df[person_col].unique()
            person_img_counts = class_df.groupby(person_col).size()
            
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
            
            sampled_df = class_df[class_df[person_col].isin(selected_persons)]
            balanced_dfs.append(sampled_df)
            
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
                      label_mapper,
                      num_workers=4,
                      balance_strategy="sampling",
                      save_split_path=None,
                      max_samples_per_class=None,
                      move_excess_to_val=False):

    df = pd.read_csv(csv_path).dropna()
    print(f"[INFO] Loaded CSV with columns: {list(df.columns)}")

    # Auto-detect column names
    filename_col = None
    label_col = None
    person_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if filename_col is None and any(x in col_lower for x in ['filename', 'cropped', 'image', 'file']):
            if 'cropped' in col_lower:
                filename_col = col
            elif filename_col is None or 'filename' in col_lower:
                filename_col = col
        if label_col is None and any(x in col_lower for x in ['label', 'mst', 'score']):
            label_col = col
        if person_col is None and any(x in col_lower for x in ['person', 'subject', 'user', 'id']):
            person_col = col
    
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
    
    df = df[[filename_col, label_col, person_col]].copy()
    df.columns = ["filename", "label", "person_id"]
    df["person_id"] = df["person_id"].astype(str)

    # Apply class balancing if requested
    excess_df = None
    if max_samples_per_class is not None:
        if move_excess_to_val:
            df, excess_df = create_balanced_subset(
                df, max_per_class=max_samples_per_class,
                label_col="label", person_col="person_id",
                move_excess_to_val=True
            )
        else:
            df = create_balanced_subset(
                df, max_per_class=max_samples_per_class,
                label_col="label", person_col="person_id",
                move_excess_to_val=False
            )

    # One label per person
    person_labels = df.groupby("person_id").agg({"label": "first"}).reset_index()
    person_labels["label_int"] = person_labels["label"].astype(float).round().astype(int)
    person_labels["label_int"] = np.clip(person_labels["label_int"], 1, 10)

    labels_per_person = person_labels["label_int"].apply(
        lambda x: label_mapper.map_mst_to_class(x)
    ).values

    print(f"[INFO] Found {len(person_labels)} unique persons (after balancing)")
    print(f"[INFO] Person class label range: {labels_per_person.min()} to {labels_per_person.max()}")

    # Person distribution
    person_counts = Counter(labels_per_person)
    print(f"\n[INFO] Person distribution across {label_mapper.num_classes} classes (after balancing):")
    for class_idx in sorted(person_counts.keys()):
        class_name = label_mapper.get_class_name(class_idx)
        print(f"  Class {class_idx} ({class_name}): {person_counts[class_idx]} persons")
    print()

    # Stratified split
    train_persons, val_persons = train_test_split(
        person_labels["person_id"].values,
        test_size=val_ratio,
        shuffle=True,
        stratify=labels_per_person,
        random_state=42
    )

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

    if excess_df is not None and len(excess_df) > 0:
        print(f"[INFO] Adding {len(excess_df)} excess images to validation set")
        val_df = pd.concat([val_df, excess_df], ignore_index=True)

    print(f"[INFO] Train: {len(train_persons)} persons ({len(train_df)} images)")
    print(f"[INFO] Val:   {len(val_persons)} persons ({len(val_df)} images)")

    # Image-level distributions
    def get_class_from_label(s):
        arr = s.astype(float).round().astype(int)
        arr = np.clip(arr, 1, 10)
        return arr.apply(lambda x: label_mapper.map_mst_to_class(x))

    train_classes = get_class_from_label(train_df["label"])
    val_classes = get_class_from_label(val_df["label"])

    train_counts = Counter(train_classes)
    val_counts = Counter(val_classes)

    print(f"\n[INFO] Train class distribution (images):")
    for class_idx in sorted(train_counts.keys()):
        class_name = label_mapper.get_class_name(class_idx)
        print(f"  Class {class_idx} ({class_name}): {train_counts[class_idx]} images")

    print(f"\n[INFO] Val class distribution (images):")
    for class_idx in sorted(val_counts.keys()):
        class_name = label_mapper.get_class_name(class_idx)
        print(f"  Class {class_idx} ({class_name}): {val_counts[class_idx]} images")
    print()

    # Class weights
    class_counts = Counter(train_classes)
    num_classes = label_mapper.get_num_classes()

    total_samples = len(train_classes)
    raw_weights = {}
    for k in range(num_classes):
        c = class_counts.get(k, 1)
        raw_weights[k] = total_samples / (num_classes * c)

    dampened_weights = {}
    for k in range(num_classes):
        dampened_weights[k] = np.sqrt(raw_weights[k])
    
    max_weight = 5.0
    for k in range(num_classes):
        dampened_weights[k] = min(dampened_weights[k], max_weight)

    print("[INFO] Class weights (sqrt-dampened, capped at 5.0):")
    for k in range(num_classes):
        class_name = label_mapper.get_class_name(k)
        print(f"  Class {k} ({class_name}): raw={raw_weights[k]:.3f}, dampened={dampened_weights[k]:.3f}")
    print()

    class_weight_tensor = torch.zeros(num_classes, dtype=torch.float32)
    for k in range(num_classes):
        class_weight_tensor[k] = dampened_weights.get(k, 1.0)

    # Transforms
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

    train_dataset = SkinToneDataset(train_df, image_dir, train_tf, label_mapper)
    val_dataset = SkinToneDataset(val_df, image_dir, val_tf, label_mapper)

    # Sampler for balanced training
    if balance_strategy == "sampling":
        class_freq = Counter(train_classes)

        img_weights = []
        for _, row in train_df.iterrows():
            mst = int(round(float(row["label"])))
            mst = max(1, min(10, mst))
            class_id = label_mapper.map_mst_to_class(mst)
            freq = class_freq[class_id]
            img_weights.append(1.0 / freq)

        img_weights = torch.DoubleTensor(img_weights)

        # CHANGE: Undersampling - match smallest class
        num_samples = num_classes * min(class_freq.values())
        
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=img_weights,
            num_samples=num_samples,
            replacement=True
        )

        # ========== SAMPLING DIAGNOSTICS (FIXED) ==========
        print("\n" + "="*70)
        print("WEIGHTED SAMPLING DIAGNOSTICS (UNDERSAMPLING)")
        print("="*70)
        
        # FIX: Use num_samples instead of len(img_weights)
        total_samples_per_epoch = num_samples
        samples_per_class = total_samples_per_epoch / num_classes
        
        print(f"\n[SAMPLING STRATEGY] Undersampling (match smallest class)")
        print(f"Total samples per epoch: {total_samples_per_epoch:,}")
        print(f"Expected samples per class: ~{samples_per_class:,.0f}")
        print(f"\nPer-Class Sampling Breakdown:\n")
        
        for class_idx in sorted(class_freq.keys()):
            class_name = label_mapper.get_class_name(class_idx)
            unique_images = class_freq[class_idx]
            expected_samples = samples_per_class
            repetition_rate = expected_samples / unique_images
            
            print(f"  Class {class_idx} ({class_name}):")
            print(f"    Unique images available: {unique_images:,}")
            print(f"    Expected samples per epoch: ~{expected_samples:,.0f}")
            print(f"    Repetition rate: {repetition_rate:.2f}× per epoch")
            
            # Get unique people count for this class
            class_mask = train_classes == class_idx
            class_persons = train_df[class_mask]["person_id"].unique()
            num_people = len(class_persons)
            samples_per_person = expected_samples / num_people
            
            print(f"    Unique people: {num_people}")
            print(f"    Samples per person: ~{samples_per_person:.0f} per epoch")
            
            # MST breakdown within this class
            class_df = train_df[class_mask]
            mst_distribution = defaultdict(int)
            for _, row in class_df.iterrows():
                mst = int(round(float(row["label"])))
                mst = max(1, min(10, mst))
                mst_distribution[mst] += 1
            
            if len(mst_distribution) > 1:
                print(f"    MST breakdown:")
                for mst in sorted(mst_distribution.keys()):
                    count = mst_distribution[mst]
                    percentage = 100.0 * count / unique_images
                    # Expected samples for this MST within the class
                    expected_mst_samples = expected_samples * (count / unique_images)
                    print(f"      MST {mst}: {count:,} images ({percentage:.1f}%) "
                          f"→ ~{expected_mst_samples:,.0f} samples/epoch")
            
            print()
        
        print("="*70 + "\n")
        # ========== END SAMPLING DIAGNOSTICS ==========

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0), prefetch_factor=2
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0), prefetch_factor=2
        )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0), prefetch_factor=2
    )

    return train_loader, val_loader, class_weight_tensor


##############################################################
# LABEL SMOOTHING FOCAL LOSS (FIXED)
##############################################################

class LabelSmoothingFocalLoss(nn.Module):
    """FIXED: Numerically stable focal loss with label smoothing"""
    def __init__(self, num_classes, alpha=None, gamma=2.0, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, targets):
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)

        log_probs = F.log_softmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)

        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = torch.clamp(pt, min=1e-7, max=1.0 - 1e-7)

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_weight = alpha_t * (1 - pt) ** self.gamma
        else:
            focal_weight = (1 - pt) ** self.gamma
        
        focal_weight = torch.clamp(focal_weight, max=100.0)
        loss = -(focal_weight.unsqueeze(1) * log_probs * true_dist).sum(dim=1)
        
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("[WARNING] NaN/Inf in focal loss, using cross-entropy fallback")
            loss = -(log_probs * true_dist).sum(dim=1)

        return loss.mean()


##############################################################
# SUPERVISED CONTRASTIVE LOSS (FIXED)
##############################################################

class SupervisedContrastiveLoss(nn.Module):
    """FIXED: Stable contrastive loss with proper temperature"""
    def __init__(self, temperature=0.5, base_temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        similarity_matrix = torch.clamp(similarity_matrix, min=-10, max=10)

        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.ones_like(mask).fill_diagonal_(0)
        mask = mask * logits_mask

        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()


##############################################################
# 5. VGG16 MST Classifier - FIXED LAB INITIALIZATION
##############################################################

class VGG16MSTClassifier(nn.Module):
    """
    VGG16 with flexible number of output classes
    
    CHANGE: Fixed LAB initialization to preserve ImageNet weights
    """
    def __init__(self, num_classes, input_mode="rgb", use_bn=True, dropout_p=0.5):
        super().__init__()

        self.input_mode = input_mode
        self.num_classes = num_classes

        base = (
            models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
            if use_bn
            else models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        )

        features = base.features

        if input_mode == "hybrid":
            old_conv = features[0]
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

        for p in self.features.parameters():
            p.requires_grad = False

        # ========== CHANGE: FIXED LAB INITIALIZATION ==========
        if input_mode == "lab":
            # DON'T destroy ImageNet weights with random initialization!
            # Instead, adapt them for LAB by averaging RGB channels
            
            old_conv = self.features[0]
            
            with torch.no_grad():
                # Average RGB weights to create LAB initialization
                # This preserves learned edge detectors from ImageNet
                avg_weight = old_conv.weight.mean(dim=1, keepdim=True)
                
                # Assign to each LAB channel intelligently:
                # L channel (lightness ~ grayscale) gets full averaged weight
                old_conv.weight[:, 0:1, :, :] = avg_weight.clone()
                
                # a, b channels (color) get half weight initially
                # (color less important than luminance for initial features)
                old_conv.weight[:, 1:2, :, :] = avg_weight.clone() * 0.5
                old_conv.weight[:, 2:3, :, :] = avg_weight.clone() * 0.5
            
            # Make first layer trainable so it can adapt to LAB
            for p in self.features[0].parameters():
                p.requires_grad = True
            
            print("[INFO] LAB mode: Adapted first conv layer from ImageNet RGB weights")

        with torch.no_grad():
            if input_mode == "hybrid":
                dummy = torch.zeros(1, 6, 224, 224)
            else:
                dummy = torch.zeros(1, 3, 224, 224)
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

        # FLEXIBLE: num_classes instead of hardcoded 10
        self.mst_classifier = nn.Linear(512, num_classes)

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
        mst_logits = self.mst_classifier(features)

        if return_features:
            proj = F.normalize(self.projection(features), dim=1)
            return mst_logits, proj

        return mst_logits

    # def unfreeze_gradually(self, epoch, total_epochs):
    #     progress = epoch / total_epochs

    #     if progress < 0.3:
    #         for p in self.features.parameters():
    #             if self.input_mode == "lab":
    #                 if p in list(self.features[0].parameters()):
    #                     p.requires_grad = True
    #                 else:
    #                     p.requires_grad = False
    #             else:
    #                 p.requires_grad = False
    #         status = "Phase 1: Head only"

    #     elif progress < 0.6:
    #         for i, module in enumerate(self.features):
    #             if i >= 17:
    #                 for p in module.parameters():
    #                     p.requires_grad = True
    #             elif self.input_mode == "lab" and i == 0:
    #                 for p in module.parameters():
    #                     p.requires_grad = True
    #             else:
    #                 for p in module.parameters():
    #                     p.requires_grad = False
    #         status = "Phase 2: Last 2 blocks"

    #     else:
    #         for p in self.features.parameters():
    #             p.requires_grad = True
    #         status = "Phase 3: All layers"

    #     return status

    def unfreeze_gradually(self, epoch, total_epochs):
        progress = epoch / total_epochs

        if progress < 0.3:
            # FIXED: Use parameter identity (id) instead of tensor comparison
            first_layer_param_ids = set(id(p) for p in self.features[0].parameters())
            
            for p in self.features.parameters():
                if self.input_mode == "lab":
                    if id(p) in first_layer_param_ids:  # ← FIXED: Compare IDs
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
                else:
                    p.requires_grad = False
            status = "Phase 1: Head only"

        elif progress < 0.6:
            # FIXED: Same fix for Phase 2
            first_layer_param_ids = set(id(p) for p in self.features[0].parameters())
            
            for i, module in enumerate(self.features):
                if i >= 17:
                    for p in module.parameters():
                        p.requires_grad = True
                elif self.input_mode == "lab" and i == 0:
                    for p in module.parameters():
                        p.requires_grad = True
                else:
                    for p in module.parameters():
                        p.requires_grad = False
            status = "Phase 2: Last 2 blocks"

        else:
            for p in self.features.parameters():
                p.requires_grad = True
            status = "Phase 3: All layers"

        return status


##############################################################
# 6. Metrics
##############################################################

def compute_classification_metrics(preds, targets, label_mapper):
    """Compute classification metrics with flexible binning"""
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()

    correct = (preds == targets).float()
    top1 = correct.mean().item() * 100.0

    off1 = (preds - targets).abs() <= 1
    off1_acc = off1.float().mean().item() * 100.0

    # L2-LAB
    preds_np = preds.numpy()
    targets_np = targets.numpy()
    l2 = calc_l2_lab_distance_classes(preds_np, targets_np, label_mapper)
    l2_mean = float(l2.mean())
    l2_std = float(l2.std())

    return {
        "top1": top1,
        "off1": off1_acc,
        "l2_mean": l2_mean,
        "l2_std": l2_std,
    }


##############################################################
# 7. DIAGNOSTIC TOOLS
##############################################################

def analyze_predictions(all_preds, all_targets, all_person_ids, label_mapper, split_name="Val"):
    """Comprehensive analysis of model predictions"""
    print(f"\n{'='*60}")
    print(f"PREDICTION ANALYSIS - {split_name}")
    print(f"{'='*60}")

    preds = all_preds.numpy()
    targets = all_targets.numpy()

    metrics = compute_classification_metrics(all_preds, all_targets, label_mapper)
    print(f"\nOverall Metrics:")
    print(f"  Top-1 Accuracy: {metrics['top1']:.2f}%")
    print(f"  Off-by-1 Accuracy: {metrics['off1']:.2f}%")
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

        class_name = label_mapper.get_class_name(true_label)
        mode_name = label_mapper.get_class_name(mode_pred)
        
        print(f"  Class {true_label} ({class_name}, n={total}):")
        print(f"    Accuracy: {acc:.2f}%")
        print(f"    Most predicted: Class {mode_pred} ({mode_name}) - {pred_dist[mode_pred]} times")
        
        if acc < 50:
            top3 = pred_dist.most_common(3)
            dist_str = ", ".join([f"C{p}:{c}" for p, c in top3])
            print(f"    Distribution: {dist_str}")

    # Confusion matrix
    print(f"\nConfusion Summary (most common mistakes):")
    confusion_pairs = defaultdict(int)
    for pred, true in zip(preds, targets):
        if pred != true:
            confusion_pairs[(true, pred)] += 1

    for (true, pred), count in sorted(confusion_pairs.items(), 
                                      key=lambda x: x[1], reverse=True)[:10]:
        true_name = label_mapper.get_class_name(true)
        pred_name = label_mapper.get_class_name(pred)
        print(f"  {true_name} → {pred_name}: {count} times")

    print(f"{'='*60}\n")


##############################################################
# 8. Training / Validation
##############################################################

def train_one_epoch(model, loader, mst_criterion, contrastive_criterion,
                    optimizer, device, epoch, total_epochs, label_mapper,
                    use_contrastive=True, lambda_contrast=0.1):
    """Single-task training with flexible classes"""
    model.train()
    unfreeze_status = model.unfreeze_gradually(epoch, total_epochs)

    running_loss = 0.0
    running_mst_loss = 0.0
    running_contrast_loss = 0.0
    
    correct_top1 = 0
    correct_off1 = 0
    total_samples = 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch} [{unfreeze_status}]", ncols=120)

    for imgs, labels, person_ids in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_contrastive:
            mst_logits, features = model(imgs, return_features=True)
        else:
            mst_logits = model(imgs, return_features=False)
            features = None

        mst_loss = mst_criterion(mst_logits, labels)

        if use_contrastive and features is not None:
            contrast_loss = contrastive_criterion(features, labels)
            total_loss = mst_loss + lambda_contrast * contrast_loss
        else:
            contrast_loss = torch.tensor(0.0, device=device)
            total_loss = mst_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds = torch.argmax(mst_logits, dim=1)
        batch_size = imgs.size(0)
        
        running_loss += total_loss.item() * batch_size
        running_mst_loss += mst_loss.item() * batch_size
        running_contrast_loss += contrast_loss.item() * batch_size
        
        correct_top1 += (preds == labels).sum().item()
        correct_off1 += ((preds - labels).abs() <= 1).sum().item()
        total_samples += batch_size

        pbar.set_postfix({
            "loss": f"{total_loss.item():.4f}",
            "mst": f"{mst_loss.item():.3f}",
            "top1": f"{100.0*correct_top1/total_samples:.1f}",
        })

    return {
        "loss": running_loss / total_samples,
        "mst_loss": running_mst_loss / total_samples,
        "contrast_loss": running_contrast_loss / total_samples,
        "top1": 100.0 * correct_top1 / total_samples,
        "off1": 100.0 * correct_off1 / total_samples,
    }


def validate(model, loader, mst_criterion, device, epoch, label_mapper, analyze=False):
    """Validation with flexible classes"""
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

            mst_logits = model(imgs, return_features=False)
            mst_loss = mst_criterion(mst_logits, labels)
            preds = torch.argmax(mst_logits, dim=1)

            batch_size = imgs.size(0)
            running_loss += mst_loss.item() * batch_size
            total_samples += batch_size

            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())
            all_person_ids.extend(person_ids)

            batch_metrics = compute_classification_metrics(preds, labels, label_mapper)
            pbar.set_postfix({
                "loss": f"{mst_loss.item():.4f}",
                "top1": f"{batch_metrics['top1']:.2f}",
            })

    avg_loss = running_loss / total_samples
    
    preds_all = torch.cat(all_preds, dim=0)
    targets_all = torch.cat(all_targets, dim=0)

    img_metrics = compute_classification_metrics(preds_all, targets_all, label_mapper)

    # Person-level metrics
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
    person_metrics = compute_classification_metrics(person_preds, person_targets, label_mapper)

    if analyze:
        analyze_predictions(preds_all, targets_all, all_person_ids, label_mapper, "Validation")

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

    if checkpoint.get("optimizer_state_dict") and checkpoint["optimizer_state_dict"]:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        print("[INFO] Optimizer state not found in checkpoint - will be re-initialized")
    
    if checkpoint.get("scheduler_state_dict") and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    else:
        print("[INFO] Scheduler state not found in checkpoint - will be re-initialized")

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
    parser = argparse.ArgumentParser(description="Flexible VGG16 MST Classifier with Custom Binning")
    
    # Data arguments
    parser.add_argument("--csv-path", required=True, help="Path to CSV with image labels")
    parser.add_argument("--image-dir", required=True, help="Directory containing images")
    parser.add_argument("--save-dir", default="./checkpoints", help="Directory to save models")
    parser.add_argument("--label-mapping", default=None, 
                        help="Path to JSON file with label mapping configuration")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--additional-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    
    # Model arguments
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use-bn", action="store_true")
    parser.add_argument("--input-mode", choices=["rgb", "lab", "hybrid"], default="rgb")
    parser.add_argument("--dropout", type=float, default=0.5)
    
    # LAB color space
    parser.add_argument("--compute-lab-stats", action="store_true")
    
    # Loss and balancing
    parser.add_argument("--balance-strategy", choices=["weights", "sampling", "none"], default="sampling")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    
    # Dataset balancing
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--move-excess-to-val", action="store_true")
    
    # Contrastive learning
    parser.add_argument("--lambda-contrast", type=float, default=0.1)
    parser.add_argument("--no-contrastive", action="store_true")
    
    # Optimization
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--early-stop-delta", type=float, default=0.1)
    
    # Analysis
    parser.add_argument("--analyze-every", type=int, default=5)

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Initialize label mapper
    label_mapper = LabelMapper(args.label_mapping)
    label_mapper.print_mapping()

    # Setup save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_path = save_dir / "vgg16_mst_best.pth"
    final_path = save_dir / "vgg16_mst_final.pth"
    checkpoint_path = save_dir / "vgg16_mst_checkpoint.pth"
    save_split_path = save_dir / "train_val_split.json"

    print("\n" + "="*70)
    print("VGG16 MST CLASSIFIER - FLEXIBLE BINNING VERSION (LAB FIXED)")
    print("="*70)
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Input mode: {args.input_mode}")
    print(f"[INFO] Number of classes: {label_mapper.get_num_classes()}")
    print(f"[INFO] Label smoothing: {args.label_smoothing}")
    print(f"[INFO] Balance strategy: {args.balance_strategy}")
    print(f"[INFO] Contrastive learning: {'Enabled' if not args.no_contrastive else 'Disabled'}")
    if not args.no_contrastive:
        print(f"[INFO] Contrastive weight: {args.lambda_contrast}")
    if args.max_per_class is not None:
        print(f"[INFO] Dataset balancing: max {args.max_per_class} images per MST class")
        print(f"[INFO] Move excess to val: {args.move_excess_to_val}")
    print("="*70 + "\n")

    # LAB stats - CHANGE: Include augmentation by default
    if args.input_mode in ("lab", "hybrid"):

        # Define save path
        stats_save_path = save_dir / "lab_statistics.json"

        if args.compute_lab_stats:
            print("[INFO] Computing LAB statistics from dataset...")
            print("[INFO] Including LAB augmentation in statistics (RECOMMENDED)")
            lab_mean, lab_std = compute_dataset_lab_stats(
                args.image_dir, 
                args.csv_path,
                include_augmentation=True,  # CHANGE: Default to True
                save_path=stats_save_path  # ← Save to file
            )
        elif stats_save_path.exists():
            # Load from saved file
            print("[INFO] Loading LAB statistics from saved file...")
            lab_mean, lab_std = load_lab_stats(stats_save_path)
            
            if lab_mean is None:
                print("[WARN] Failed to load stats, using defaults")
                lab_mean, lab_std = LAB_MEAN_DEFAULT, LAB_STD_DEFAULT
        else:
            print("[INFO] Using default LAB normalization values.")
            print("[WARN] For best results, run with --compute-lab-stats")
            lab_mean, lab_std = LAB_MEAN_DEFAULT, LAB_STD_DEFAULT
        print(f"[INFO] LAB_MEAN: {lab_mean.tolist()}")
        print(f"[INFO] LAB_STD: {lab_std.tolist()}\n")
    else:
        lab_mean, lab_std = LAB_MEAN_DEFAULT, LAB_STD_DEFAULT

    # Build dataloaders
    train_loader, val_loader, class_weight_tensor = build_dataloaders(
        args.csv_path, args.image_dir, args.batch_size, args.val_ratio,
        args.input_mode, lab_mean, lab_std, label_mapper,
        num_workers=args.num_workers,
        balance_strategy=args.balance_strategy,
        save_split_path=save_split_path,
        max_samples_per_class=args.max_per_class,
        move_excess_to_val=args.move_excess_to_val
    )

    # Build model with flexible num_classes
    model = VGG16MSTClassifier(
        num_classes=label_mapper.get_num_classes(),
        input_mode=args.input_mode,
        use_bn=args.use_bn,
        dropout_p=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters: {total_params:,}")
    print(f"[INFO] Trainable parameters: {trainable_params:,}\n")

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
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": base_lr * 0.1},
            {"params": head_params, "lr": base_lr},
        ],
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7, verbose=True
    )

    # Loss functions
    class_weight_tensor = class_weight_tensor.to(device)
    
    if args.balance_strategy == "weights":
        mst_criterion = LabelSmoothingFocalLoss(
            num_classes=label_mapper.get_num_classes(),
            alpha=class_weight_tensor,
            gamma=args.focal_gamma,
            smoothing=args.label_smoothing
        )
        print("[INFO] Using Label Smoothing Focal Loss WITH class weights")
    else:
        mst_criterion = LabelSmoothingFocalLoss(
            num_classes=label_mapper.get_num_classes(),
            alpha=None,
            gamma=args.focal_gamma,
            smoothing=args.label_smoothing
        )
        print("[INFO] Using Label Smoothing Focal Loss WITHOUT class weights")
    
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

        train_results = train_one_epoch(
            model, train_loader, mst_criterion, contrastive_criterion,
            optimizer, device, epoch, max_epoch, label_mapper,
            use_contrastive=not args.no_contrastive,
            lambda_contrast=args.lambda_contrast
        )

        analyze_this_epoch = (epoch % args.analyze_every == 0) or (epoch == max_epoch)
        val_results = validate(
            model, val_loader, mst_criterion, device, epoch, label_mapper,
            analyze=analyze_this_epoch
        )

        print(f"\n[EPOCH {epoch} SUMMARY]")
        print(f"Train: Loss={train_results['loss']:.4f} "
              f"(MST={train_results['mst_loss']:.4f}, Contrast={train_results['contrast_loss']:.4f})")
        print(f"       Top1={train_results['top1']:.2f}% Off1={train_results['off1']:.2f}%")
        
        print(f"Val:   Loss={val_results['loss']:.4f}")
        print(f"       Image-level: Top1={val_results['img_metrics']['top1']:.2f}% "
              f"Off1={val_results['img_metrics']['off1']:.2f}%")
        print(f"       Person-level: Top1={val_results['person_metrics']['top1']:.2f}% "
              f"Off1={val_results['person_metrics']['off1']:.2f}%")
        print(f"       L2-LAB: {val_results['person_metrics']['l2_mean']:.3f} ± "
              f"{val_results['person_metrics']['l2_std']:.3f}")

        scheduler.step(val_results['loss'])

        current_val_acc = val_results['person_metrics']['top1']
        
        if val_results['loss'] < best_val_loss:
            best_val_loss = val_results['loss']
        
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            torch.save(model.state_dict(), best_path)
            print(f"\n[INFO] ✓ New best model saved (ValAcc: {best_val_acc:.2f}%)")

        save_checkpoint(
            model, optimizer, scheduler, epoch,
            current_val_acc, val_results['loss'], checkpoint_path
        )

        if early_stopper(current_val_acc):
            print(f"\n[INFO] Early stopping triggered at epoch {epoch}")
            break

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

# Balance Strategy Sampling + Added extra Black Examples + LAB -> ✓ New best model saved (ValAcc: 70.50%)
# python vgg16_mst_flexible_lab.py ^
#   --csv-path "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\annotations_with_dark.csv" ^
#   --image-dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2" ^
#   --save-dir "F:\Thesis\CasualConversationv2_Dataset\Models\5Class_Dark_Balanced_Lab" ^
#   --label-mapping "label_mapping_5class.json" ^
#   --gpu 0 ^
#   --input-mode lab ^
#   --compute-lab-stats ^
#   --use-bn ^
#   --balance-strategy sampling ^
#   --no-contrastive ^
#   --label-smoothing 0.15 ^
#   --focal-gamma 0.5 ^
#   --dropout 0.5 ^
#   --weight-decay 2e-4 ^
#   --lr 3e-4 ^
#   --batch-size 32 ^
#   --epochs 40 ^
#   --analyze-every 1 ^
#   --resume

# python vgg16_mst_flexible_lab.py ^
#   --csv-path "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\annotations_with_dark.csv" ^
#   --image-dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2" ^
#   --save-dir "F:\Thesis\CasualConversationv2_Dataset\Models\5Class_Dark_Balanced_Lab" ^
#   --label-mapping "label_mapping_5class.json" ^
#   --gpu 0 ^
#   --input-mode lab ^
#   --use-bn ^
#   --balance-strategy sampling ^
#   --no-contrastive ^
#   --label-smoothing 0.15 ^
#   --focal-gamma 0.5 ^
#   --dropout 0.5 ^
#   --weight-decay 2e-4 ^
#   --lr 3e-4 ^
#   --batch-size 32 ^
#   --epochs 40 ^
#   --analyze-every 1 ^
#   --resume





# python vgg16_mst_flexible_lab.py ^
#   --csv-path "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\annotations_with_dark.csv" ^
#   --image-dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2" ^
#   --save-dir "F:\Thesis\CasualConversationv2_Dataset\Models\4Class_Dark_Balanced_Lab" ^
#   --label-mapping "label_mapping_4class.json" ^
#   --gpu 0 ^
#   --input-mode lab ^
#   --use-bn ^
#   --balance-strategy sampling ^
#   --no-contrastive ^
#   --label-smoothing 0.15 ^
#   --focal-gamma 0.5 ^
#   --dropout 0.5 ^
#   --weight-decay 2e-4 ^
#   --lr 3e-4 ^
#   --batch-size 32 ^
#   --epochs 40 ^
#   --analyze-every 1 ^
#   --resume