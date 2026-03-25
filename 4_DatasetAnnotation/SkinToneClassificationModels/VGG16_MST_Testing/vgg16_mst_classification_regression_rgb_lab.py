##############################################################
#  VGG16 MST CLASSIFIER - ULTIMATE VERSION
#  
#  COMBINES ALL FEATURES:
#  - Original: Contrastive learning, Focal loss, Label smoothing,
#              Person-level metrics, Resume training, etc.
#  - New: Regression mode, Ordinal loss, Distance penalties
#  
#  This is the complete, production-ready version.
##############################################################

import argparse
from pathlib import Path
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models

from skimage.color import rgb2lab, lab2rgb, deltaE_ciede2000
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error
)


##############################################################
# REPRODUCIBILITY
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
# LOSS FUNCTIONS
##############################################################

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Focuses on hard examples.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', weight=self.alpha
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class OrdinalCrossEntropyLoss(nn.Module):
    """
    Cross-Entropy with distance-based penalty for ordinal classes.
    """
    def __init__(self, class_weights=None, distance_weight=0.5):
        super().__init__()
        self.class_weights = class_weights
        self.distance_weight = distance_weight
        self.ce = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
    
    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        preds = logits.argmax(dim=1)
        distance = torch.abs(preds.float() - targets.float())
        total_loss = ce_loss + self.distance_weight * distance
        return total_loss.mean()


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss for better feature separation.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        """
        Args:
            features: (batch_size, feature_dim) - normalized features
            labels: (batch_size,) - class labels
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs (same class)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Remove diagonal (self-similarity)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
        
        # Loss
        loss = -mean_log_prob_pos.mean()
        
        return loss


##############################################################
# LABEL MAPPER
##############################################################

class LabelMapper:
    """Handles flexible label mapping for classification or regression"""
    def __init__(self, mapping_config=None, mode='classification'):
        self.mode = mode
        
        if mapping_config is None:
            if mode == 'classification':
                self.mapping = {str(i): i-1 for i in range(1, 11)}
                self.num_classes = 10
            else:
                self.mapping = {str(i): float(i) for i in range(1, 11)}
                self.num_classes = 1
            self.config_name = "mst10"
        elif isinstance(mapping_config, (str, Path)):
            with open(mapping_config, 'r') as f:
                config = json.load(f)
            self._parse_config(config)
            self.config_name = Path(mapping_config).stem
        elif isinstance(mapping_config, dict):
            self._parse_config(mapping_config)
            self.config_name = "custom"
    
    def _parse_config(self, config):
        if "label_mapping" in config:
            if self.mode == 'classification':
                self.mapping = {str(k): int(v) for k, v in config["label_mapping"].items()}
                self.num_classes = config.get("num_classes", max(self.mapping.values()) + 1)
            else:
                self.mapping = {str(k): float(v) for k, v in config["label_mapping"].items()}
                self.num_classes = 1
        elif "bins" in config:
            if self.mode != 'classification':
                raise ValueError("Bins format only for classification")
            self.mapping = {}
            for bin_def in config["bins"]:
                for mst in range(bin_def["range"][0], bin_def["range"][1] + 1):
                    self.mapping[str(mst)] = bin_def["class"]
            self.num_classes = config.get("num_classes", max(self.mapping.values()) + 1)
        
        for mst in range(1, 11):
            if str(mst) not in self.mapping:
                raise ValueError(f"Mapping missing MST {mst}")
    
    def map_mst_to_target(self, mst_value):
        return self.mapping[str(int(mst_value))]
    
    def get_class_name(self, class_idx):
        if self.mode != 'classification':
            return f"MST{class_idx:.1f}"
        
        mst_values = [int(k) for k, v in self.mapping.items() if v == class_idx]
        mst_values.sort()
        
        if len(mst_values) == 1:
            return f"MST{mst_values[0]}"
        elif mst_values == list(range(mst_values[0], mst_values[-1] + 1)):
            return f"MST{mst_values[0]}-{mst_values[-1]}"
        else:
            return f"MST{','.join(map(str, mst_values))}"


##############################################################
# DATA LOADING WITH BALANCING
##############################################################

def load_and_split_data(csv_path, label_mapper, val_ratio=0.2,
                        max_samples_per_class=None, balance_strategy='weighting',
                        save_split_path=None):
    """
    Load and split data with various balancing strategies.
    
    Args:
        balance_strategy: 'none', 'weighting', or 'sampling'
    """
    df = pd.read_csv(csv_path).dropna()
    print(f"[INFO] Loaded CSV: {list(df.columns)}")
    
    # Auto-detect columns
    filename_col = None
    label_col = None
    person_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if filename_col is None and any(x in col_lower for x in ['filename', 'cropped', 'image']):
            filename_col = col
        if label_col is None and any(x in col_lower for x in ['label', 'mst', 'score']):
            label_col = col
        if person_col is None and any(x in col_lower for x in ['person', 'subject', 'id']):
            person_col = col
    
    if not all([filename_col, label_col, person_col]):
        raise ValueError("CSV needs filename, label, person_id columns")
    
    print(f"[INFO] Columns: filename='{filename_col}', label='{label_col}', person='{person_col}'")
    
    df = df[[filename_col, label_col, person_col]].copy()
    df.columns = ["filename", "label", "person_id"]
    df["person_id"] = df["person_id"].astype(str)
    
    # Balancing via sampling
    # if balance_strategy == 'sampling' and max_samples_per_class and label_mapper.mode == 'classification':
    if balance_strategy == 'sampling' and max_samples_per_class:
        print(f"\n[Balancing] Sampling strategy: {max_samples_per_class} per MST class")
        
        df['mst_int'] = df['label'].astype(float).round().astype(int).clip(0, 10)
        balanced_dfs = []
        
        for mst in range(1, 11):
            class_df = df[df['mst_int'] == mst].copy()
            n_original = len(class_df)
            
            if n_original == 0:
                continue
            
            if n_original <= max_samples_per_class:
                balanced_dfs.append(class_df)
                print(f"  MST {mst:2d}: {n_original:6d} (kept all)")
            else:
                persons = class_df['person_id'].unique()
                n_persons = len(persons)
                images_per_person = n_original / n_persons
                target_persons = int(max_samples_per_class / images_per_person)
                target_persons = max(1, min(target_persons, n_persons))
                
                selected_persons = np.random.choice(persons, size=target_persons, replace=False)
                sampled_df = class_df[class_df['person_id'].isin(selected_persons)]
                balanced_dfs.append(sampled_df)
                print(f"  MST {mst:2d}: {n_original:6d} → {len(sampled_df):6d}")
        
        df = pd.concat(balanced_dfs, ignore_index=True)
        df = df.drop(columns=['mst_int'])
        print(f"\n[INFO] After balancing: {len(df):,} images\n")
    
    # Person-level split
    person_labels = df.groupby("person_id").agg({"label": "first"}).reset_index()
    person_labels["label_int"] = person_labels["label"].astype(float).round().astype(int).clip(0, 10)
    
    if label_mapper.mode == 'classification':
        labels_per_person = person_labels["label_int"].apply(
            lambda x: label_mapper.map_mst_to_target(x)
        ).values
    else:
        labels_per_person = person_labels["label_int"].values
    
    print(f"[INFO] {len(person_labels)} unique persons")
    
    train_persons, val_persons = train_test_split(
        person_labels["person_id"].values,
        test_size=val_ratio,
        shuffle=True,
        stratify=labels_per_person,
        random_state=42
    )
    
    if save_split_path:
        split_info = {
            "train_persons": train_persons.tolist(),
            "val_persons": val_persons.tolist(),
            "random_state": 42,
            "val_ratio": val_ratio,
        }
        Path(save_split_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_split_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        print(f"[INFO] Split saved: {save_split_path}")
    
    train_df = df[df["person_id"].isin(train_persons)].reset_index(drop=True)
    val_df = df[df["person_id"].isin(val_persons)].reset_index(drop=True)
    
    print(f"[INFO] Train: {len(train_persons)} persons ({len(train_df)} images)")
    print(f"[INFO] Val:   {len(val_persons)} persons ({len(val_df)} images)\n")
    
    return train_df, val_df


##############################################################
# LAB COLOR SPACE
##############################################################

def compute_lab_stats_full_dataset(csv_path, image_dir):
    """
    Compute LAB mean/std across entire dataset using streaming accumulation.
    Memory safe.
    """
    print("[LAB Stats] Computing from entire dataset (streaming)...")

    df = pd.read_csv(csv_path).dropna()

    # Auto-detect filename column
    filename_col = None
    for col in df.columns:
        if any(x in col.lower() for x in ['filename', 'cropped', 'image']):
            filename_col = col
            break

    if filename_col is None:
        raise ValueError("Could not detect filename column")

    total_pixels = 0
    sum_channels = np.zeros(3, dtype=np.float64)
    sum_sq_channels = np.zeros(3, dtype=np.float64)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing LAB stats"):
        img_path = Path(image_dir) / row[filename_col]
        if not img_path.exists():
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((224, 224))
            rgb_array = np.array(img).astype(np.float32) / 255.0
            lab_array = rgb2lab(rgb_array)

            pixels = lab_array.reshape(-1, 3)

            sum_channels += pixels.sum(axis=0)
            sum_sq_channels += (pixels ** 2).sum(axis=0)
            total_pixels += pixels.shape[0]

        except:
            continue

    mean = sum_channels / total_pixels
    var = (sum_sq_channels / total_pixels) - (mean ** 2)
    std = np.sqrt(var)

    print(f"[LAB Stats] Mean: L={mean[0]:.2f}, a={mean[1]:.2f}, b={mean[2]:.2f}")
    print(f"[LAB Stats] Std:  L={std[0]:.2f}, a={std[1]:.2f}, b={std[2]:.2f}\n")

    return mean.tolist(), std.tolist()


# class LABTransform:
#     """LAB color space transform with augmentation"""
#     def __init__(self, mean, std, mode='train', input_size=224):
#         self.mean = torch.tensor(mean).view(3, 1, 1)
#         self.std = torch.tensor(std).view(3, 1, 1)
#         self.mode = mode
#         self.input_size = input_size
    
#     def __call__(self, pil_image):
#         pil_image = pil_image.resize((self.input_size, self.input_size), Image.BILINEAR)
        
#         rgb_array = np.array(pil_image).astype(np.float32) / 255.0
#         lab_array = rgb2lab(rgb_array)
        
#         # Augmentation
#         if self.mode == 'train':
#             lab_array[:, :, 0] += np.random.uniform(-5, 5)
#             lab_array[:, :, 1] += np.random.uniform(-3, 3)
#             lab_array[:, :, 2] += np.random.uniform(-3, 3)
#             lab_array[:, :, 0] = np.clip(lab_array[:, :, 0], 0, 100)
#             lab_array[:, :, 1] = np.clip(lab_array[:, :, 1], -128, 127)
#             lab_array[:, :, 2] = np.clip(lab_array[:, :, 2], -128, 127)
        
#         lab_tensor = torch.from_numpy(lab_array).permute(2, 0, 1).float()
#         lab_tensor = (lab_tensor - self.mean) / self.std
        
#         return lab_tensor

class LABTransform:
    """
    LAB color space transform with optional luminance control.
    """
    def __init__(self, mean, std, mode='train',
                 input_size=224,
                 drop_l=False,
                 normalize_l=False):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.mode = mode
        self.input_size = input_size
        self.drop_l = drop_l
        self.normalize_l = normalize_l

    def __call__(self, pil_image):
        pil_image = pil_image.resize((self.input_size, self.input_size), Image.BILINEAR)

        rgb_array = np.array(pil_image).astype(np.float32) / 255.0
        lab_array = rgb2lab(rgb_array)

        # -----------------------------
        # Luminance handling
        # -----------------------------
        if self.drop_l:
            # Remove luminance completely
            lab_array[:, :, 0] = 50.0  # constant mid-lightness

        elif self.normalize_l:
            # Per-image normalization of L
            L = lab_array[:, :, 0]
            L = (L - L.mean()) / (L.std() + 1e-6)
            L = L * 10 + 50  # restore scale
            lab_array[:, :, 0] = np.clip(L, 0, 100)

        # -----------------------------
        # Mild chroma augmentation
        # -----------------------------
        if self.mode == 'train':
            lab_array[:, :, 1] += np.random.uniform(-3, 3)
            lab_array[:, :, 2] += np.random.uniform(-3, 3)

            lab_array[:, :, 1] = np.clip(lab_array[:, :, 1], -128, 127)
            lab_array[:, :, 2] = np.clip(lab_array[:, :, 2], -128, 127)

        lab_tensor = torch.from_numpy(lab_array).permute(2, 0, 1).float()
        lab_tensor = (lab_tensor - self.mean) / self.std

        return lab_tensor


# class RGBTransform:
#     """
#     Standard RGB transform with ImageNet normalization.
#     """
#     def __init__(self, mode='train', input_size=224):
#         self.mode = mode
#         self.input_size = input_size

#         # ImageNet normalization
#         self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#         self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

#     def __call__(self, pil_image):
#         pil_image = pil_image.resize((self.input_size, self.input_size), Image.BILINEAR)

#         rgb_array = np.array(pil_image).astype(np.float32) / 255.0

#         # Light augmentation
#         if self.mode == 'train':
#             rgb_array += np.random.uniform(-0.02, 0.02)
#             rgb_array = np.clip(rgb_array, 0.0, 1.0)

#         rgb_tensor = torch.from_numpy(rgb_array).permute(2, 0, 1).float()
#         rgb_tensor = (rgb_tensor - self.mean) / self.std

#         return rgb_tensor

class RGBTransform:
    def __init__(self, mode='train', input_size=224, normalize_brightness=False):
        self.mode = mode
        self.input_size = input_size
        self.normalize_brightness = normalize_brightness

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __call__(self, pil_image):
        pil_image = pil_image.resize((self.input_size, self.input_size), Image.BILINEAR)

        rgb_array = np.array(pil_image).astype(np.float32) / 255.0

        if self.normalize_brightness:
            mean_intensity = rgb_array.mean()
            rgb_array = rgb_array / (mean_intensity + 1e-6)
            rgb_array = np.clip(rgb_array, 0.0, 1.0)

        if self.mode == 'train':
            rgb_array += np.random.uniform(-0.02, 0.02)
            rgb_array = np.clip(rgb_array, 0.0, 1.0)

        rgb_tensor = torch.from_numpy(rgb_array).permute(2, 0, 1).float()
        rgb_tensor = (rgb_tensor - self.mean) / self.std

        return rgb_tensor



##############################################################
# DATASET
##############################################################

class SkinToneDataset(Dataset):
    def __init__(self, df, image_dir, label_mapper, transform, mode='train'):
        self.image_dir = Path(image_dir)
        self.label_mapper = label_mapper
        self.transform = transform
        self.mode = mode
        
        self.samples = []
        for _, row in df.iterrows():
            img_path = self.image_dir / row['filename']
            
            try:
                mst = int(float(row['label']))
            except:
                continue
            
            if img_path.exists() and 1 <= mst <= 10:
                target = label_mapper.map_mst_to_target(mst)
                person_id = row['person_id']
                self.samples.append((str(img_path), target, mst, person_id))
        
        print(f"[{mode}] {len(self.samples)} samples")
        
        if label_mapper.mode == 'classification':
            class_counts = Counter([t for _, t, _, _ in self.samples])
            for class_idx in range(label_mapper.num_classes):
                count = class_counts.get(class_idx, 0)
                print(f"  {label_mapper.get_class_name(class_idx)}: {count}")
        else:
            targets = [t for _, t, _, _ in self.samples]
            print(f"  Range: [{min(targets):.1f}, {max(targets):.1f}]")
            print(f"  Mean: {np.mean(targets):.2f}, Std: {np.std(targets):.2f}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, target, mst, person_id = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = self.transform(image)
            return image_tensor, target, mst, person_id
        except:
            return self.__getitem__((idx + 1) % len(self))


##############################################################
# VGG16 MODEL
##############################################################

class VGG16MSTModel(nn.Module):
    """VGG16 with contrastive learning support"""
    def __init__(self, num_outputs, mode='classification', dropout=0.5,
                 use_bn=False, pretrained=True):
        super().__init__()
        self.mode = mode
        self.num_outputs = num_outputs

        if mode == 'classification':
            out_dim = num_outputs

        elif mode == 'coral':
            # CORAL uses K-1 logits
            out_dim = num_outputs - 1

        else:  # regression
            out_dim = 1


        vgg16 = (
            models.vgg16_bn(weights='IMAGENET1K_V1' if pretrained else None)
            if use_bn
            else models.vgg16(weights='IMAGENET1K_V1' if pretrained else None)
        )
        self.features = vgg16.features
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(512 * 7 * 7, 2048),
            nn.ReLU(),
            nn.Linear(2048, 128)
        )
        
        # Main classifier
        if mode in ['classification', 'coral']:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(4096, out_dim)
            )
        else:  # regression
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(4096, 1024),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(1024, out_dim)
            )
        
        print(f"[VGG16] Mode: {mode}, Outputs: {num_outputs}, Dropout: {dropout}, Pretrained: {pretrained}")
    
    def forward(self, x, return_features=False):
        x = self.features(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        
        logits = self.classifier(features)
        
        # if self.mode == 'regression':
        #     logits = torch.clamp(logits, 0.0, 10.0)

        if self.mode == 'regression' and not self.training:
            logits = torch.clamp(logits, 0.0, 10.0)
        
        if return_features:
            proj_features = self.projection(features)
            return logits, proj_features
        else:
            return logits

##############################################################
# ResNet18 MODEL
##############################################################

class ResNet18MSTModel(nn.Module):
    """
    ResNet18 with optional contrastive projection head.
    Supports classification & regression.
    """
    def __init__(self, num_outputs, mode='classification',
                 dropout=0.5, pretrained=True):
        super().__init__()
        self.mode = mode

        resnet = models.resnet18(
            weights='IMAGENET1K_V1' if pretrained else None
        )

        in_features = resnet.fc.in_features
        resnet.fc = nn.Identity()  # remove original classifier

        self.backbone = resnet

        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

        self.dropout = nn.Dropout(dropout)

        if mode == 'classification':
            self.head = nn.Linear(in_features, num_outputs)
        elif mode == 'coral':
            self.head = nn.Linear(in_features, num_outputs - 1)
        else:
            self.head = nn.Linear(in_features, 1)


        print(f"[ResNet18] Mode: {mode}, Outputs: {num_outputs}, "
              f"Dropout: {dropout}, Pretrained: {pretrained}")

    def forward(self, x, return_features=False):
        features = self.backbone(x)
        features = self.dropout(features)

        logits = self.head(features)

        # if self.mode == 'regression':
        #     logits = torch.clamp(logits, 0.0, 10.0)

        if self.mode == 'regression' and not self.training:
            logits = torch.clamp(logits, 0.0, 10.0)

        if return_features:
            proj_features = self.projection(features)
            return logits, proj_features

        return logits


##############################################################
# EARLY STOPPING
##############################################################

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, val_metric):
        if self.best_score is None:
            self.best_score = val_metric
            return False
        
        if val_metric > self.best_score + self.min_delta:
            self.best_score = val_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


##############################################################
# CHECKPOINT MANAGEMENT
##############################################################

def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, val_loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
    }, path)


def load_checkpoint(model, optimizer, scheduler, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint.get('val_acc', 0.0), checkpoint.get('val_loss', float('inf'))


class CORALLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # logits shape: (B, K-1)
        # targets shape: (B,)
        num_classes = logits.size(1) + 1

        # Create ordinal targets
        target_matrix = []
        for k in range(num_classes - 1):
            target_matrix.append((targets > k).float())
        target_matrix = torch.stack(target_matrix, dim=1)

        return self.bce(logits, target_matrix)


##############################################################
# TRAINING
##############################################################

def train_epoch(model, train_loader, mst_criterion, contrastive_criterion,
                optimizer, device, epoch, mode, use_contrastive=True,
                lambda_contrast=0.5):
    model.train()
    
    total_mst_loss = 0
    total_contrast_loss = 0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for images, targets, _, _ in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        if use_contrastive:
            outputs, features = model(images, return_features=True)
        else:
            outputs = model(images)
        
        # MST loss
        if mode == 'classification':
            mst_loss = mst_criterion(outputs, targets.long())
            preds = outputs.argmax(dim=1)

        elif mode == 'coral':
            mst_loss = mst_criterion(outputs, targets.long())

            probs = torch.sigmoid(outputs)

            # enforce non-increasing constraint
            # probs = torch.minimum(probs, torch.cat([probs[:, :1], probs[:, :-1]], dim=1))
            probs = torch.cummin(probs, dim=1)[0]
            preds = torch.sum(probs > 0.5, dim=1)

        else:  # regression
            targets_float = targets.float().unsqueeze(1)
            mst_loss = mst_criterion(outputs, targets_float)
            preds = outputs.squeeze(1)
        
        # Contrastive loss
        if use_contrastive:
            if mode in ['classification', 'coral']:
                contrast_loss = contrastive_criterion(features, targets.long())
            else:
                # For regression, discretize targets for contrastive learning
                discrete_targets = torch.round(targets).long().clamp(0, 9)
                contrast_loss = contrastive_criterion(features, discrete_targets)
        else:
            contrast_loss = torch.tensor(0.0).to(device)
        
        # Combined loss
        loss = mst_loss + lambda_contrast * contrast_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_mst_loss += mst_loss.item()
        total_contrast_loss += contrast_loss.item()
        all_preds.extend(preds.cpu().detach().numpy())
        all_targets.extend(targets.cpu().numpy() if mode == 'regression' else targets.cpu().numpy())
        
        pbar.set_postfix({'mst': f'{total_mst_loss/(pbar.n+1):.3f}',
                         'con': f'{total_contrast_loss/(pbar.n+1):.3f}'})
    
    avg_mst_loss = total_mst_loss / len(train_loader)
    avg_contrast_loss = total_contrast_loss / len(train_loader)
    
    if mode in ['classification', 'coral']:
        # Top-1 and Off-by-1
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        top1 = 100.0 * accuracy_score(all_targets, all_preds)
        off1 = 100.0 * np.mean(np.abs(all_preds - all_targets) <= 1)
        return {
            'mst_loss': avg_mst_loss,
            'contrast_loss': avg_contrast_loss,
            'total_loss': avg_mst_loss + lambda_contrast * avg_contrast_loss,
            'top1': top1,
            'off1': off1
        }
    else:  # regression
        mae = mean_absolute_error(all_targets, all_preds)
        return {
            'mst_loss': avg_mst_loss,
            'contrast_loss': avg_contrast_loss,
            'total_loss': avg_mst_loss + lambda_contrast * avg_contrast_loss,
            'mae': mae
        }


def evaluate(model, val_loader, criterion, device, label_mapper,
             tolerance=0.5, show_detailed=True):
    """Comprehensive evaluation with person-level metrics"""
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_targets = []
    all_mst_values = []
    all_person_ids = []
    
    with torch.no_grad():
        for images, targets, mst_values, person_ids in tqdm(val_loader, desc="Evaluating", leave=False):
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            
            if args.mode == 'classification':
                loss = criterion(outputs, targets.long())
                preds = outputs.argmax(dim=1).cpu().numpy()

            elif args.mode == 'coral':
                loss = criterion(outputs, targets.long())

                # CORAL decoding
                probs = torch.sigmoid(outputs)
                # probs = torch.minimum(probs, torch.cat([probs[:, :1], probs[:, :-1]], dim=1))
                probs = torch.cummin(probs, dim=1)[0]
                preds = torch.sum(probs > 0.5, dim=1).cpu().numpy()

            else:  # regression
                targets_float = targets.float().unsqueeze(1)
                loss = criterion(outputs, targets_float)
                preds = outputs.squeeze(1).cpu().numpy()

            
            total_loss += loss.item()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            all_mst_values.extend(mst_values.numpy())
            all_person_ids.extend(person_ids)
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_mst_values = np.array(all_mst_values)
    
    avg_loss = total_loss / len(val_loader)
    
    # Classification metrics
    if args.mode in ['classification', 'coral']:
        # Image-level
        top1_img = 100.0 * accuracy_score(all_targets, all_preds)
        off1_img = 100.0 * np.mean(np.abs(all_preds - all_targets) <= 1)
        
        # Person-level
        person_data = defaultdict(lambda: {'preds': [], 'targets': []})
        for pred, target, person_id in zip(all_preds, all_targets, all_person_ids):
            person_data[person_id]['preds'].append(pred)
            person_data[person_id]['targets'].append(target)
        
        person_preds = []
        person_targets = []
        for person_id, data in person_data.items():
            # Majority vote
            pred_mode = Counter(data['preds']).most_common(1)[0][0]
            target_mode = Counter(data['targets']).most_common(1)[0][0]
            person_preds.append(pred_mode)
            person_targets.append(target_mode)
        
        person_preds = np.array(person_preds)
        person_targets = np.array(person_targets)
        
        top1_person = 100.0 * accuracy_score(person_targets, person_preds)
        off1_person = 100.0 * np.mean(np.abs(person_preds - person_targets) <= 1)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            all_targets, all_preds, average=None, zero_division=0
        )
        cm = confusion_matrix(all_targets, all_preds)
        
        metrics_dict = {
            'loss': avg_loss,
            'img_top1': top1_img,
            'img_off1': off1_img,
            'person_top1': top1_person,
            'person_off1': off1_person,
            'per_class': {},
            'confusion_matrix': cm
        }
        
        for class_idx in range(label_mapper.num_classes):
            metrics_dict['per_class'][class_idx] = {
                'name': label_mapper.get_class_name(class_idx),
                'accuracy': 100.0 * (cm[class_idx, class_idx] / support[class_idx]) if support[class_idx] > 0 else 0,
                'precision': precision[class_idx] * 100,
                'recall': recall[class_idx] * 100,
                'f1': f1[class_idx] * 100,
                'support': int(support[class_idx])
            }
        
        if show_detailed:
            print("\n" + "="*70)
            print("VALIDATION METRICS")
            print("="*70)
            print(f"\nImage-Level:  Top-1: {top1_img:.2f}%, Off-by-1: {off1_img:.2f}%")
            print(f"Person-Level: Top-1: {top1_person:.2f}%, Off-by-1: {off1_person:.2f}%")
            
            print("\nPer-Class:")
            print(f"{'Class':<15} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Sup':>7}")
            print("-" * 70)
            for class_idx in range(label_mapper.num_classes):
                m = metrics_dict['per_class'][class_idx]
                print(f"{m['name']:<15} {m['accuracy']:>6.1f}% {m['precision']:>6.1f}% "
                      f"{m['recall']:>6.1f}% {m['f1']:>6.1f}% {m['support']:>7d}")
            print("="*70 + "\n")
        
        return avg_loss, top1_person, metrics_dict

    else:  # regression
        mae = mean_absolute_error(all_targets, all_preds)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        within_tol = 100.0 * np.mean(np.abs(all_preds - all_targets) <= tolerance)

        # --------------------------------------------------
        # Ordinal diagnostic metrics (classification-style)
        # --------------------------------------------------

        min_label = int(all_targets.min())
        max_label = int(all_targets.max())

        rounded_preds = np.round(all_preds).clip(min_label, max_label).astype(int)
        true_mst = all_targets.astype(int)

        label_range = list(range(min_label, max_label + 1))

        cm = confusion_matrix(true_mst, rounded_preds, labels=label_range)


        # Off-by-k accuracy
        off1 = 100.0 * np.mean(np.abs(rounded_preds - true_mst) <= 1)
        off2 = 100.0 * np.mean(np.abs(rounded_preds - true_mst) <= 2)

        # Per-MST stats
        per_mst_stats = {}
        for mst in label_range:
            mask = true_mst == mst
            if mask.sum() == 0:
                continue

            mst_mae = mean_absolute_error(true_mst[mask], all_preds[mask])
            mst_acc = 100.0 * np.mean(rounded_preds[mask] == mst)
            mst_off1 = 100.0 * np.mean(np.abs(rounded_preds[mask] - mst) <= 1)

            per_mst_stats[mst] = {
                "mae": mst_mae,
                "exact_acc": mst_acc,
                "off1_acc": mst_off1,
                "support": int(mask.sum())
            }

        # Person-level regression
        person_data = defaultdict(lambda: {'preds': [], 'targets': []})
        for pred, target, person_id in zip(all_preds, all_targets, all_person_ids):
            person_data[person_id]['preds'].append(pred)
            person_data[person_id]['targets'].append(target)

        person_preds = []
        person_targets = []
        for person_id, data in person_data.items():
            person_preds.append(np.mean(data['preds']))
            person_targets.append(np.mean(data['targets']))

        person_mae = mean_absolute_error(person_targets, person_preds)

        metrics_dict = {
            'loss': avg_loss,
            'mae': mae,
            'rmse': rmse,
            'within_tol': within_tol,
            'off1': off1,
            'off2': off2,
            'person_mae': person_mae,
            'per_mst': per_mst_stats,
            'confusion_matrix': cm
        }

        if show_detailed:
            print("\n" + "="*70)
            print("VALIDATION METRICS (REGRESSION - ORDINAL ANALYSIS)")
            print("="*70)
            print(f"\nImage-Level:")
            print(f"  MAE: {mae:.3f}")
            print(f"  RMSE: {rmse:.3f}")
            print(f"  ±{tolerance}: {within_tol:.2f}%")
            print(f"  Off-by-1: {off1:.2f}%")
            print(f"  Off-by-2: {off2:.2f}%")

            print(f"\nPerson-Level:")
            print(f"  MAE: {person_mae:.3f}")

            print("\nPer-MST Performance:")
            print(f"{'MST':<5} {'MAE':>7} {'Exact%':>8} {'Off1%':>8} {'Sup':>6}")
            print("-" * 50)
            for mst, stats in per_mst_stats.items():
                print(f"{mst:<5} {stats['mae']:>7.3f} "
                    f"{stats['exact_acc']:>7.2f}% "
                    f"{stats['off1_acc']:>7.2f}% "
                    f"{stats['support']:>6d}")

            print("="*70 + "\n")

        return avg_loss, person_mae, metrics_dict

##############################################################
# MAIN
##############################################################

def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}\n")
    
    # Label mapper
    lm_mode = 'classification' if args.mode in ['classification', 'coral'] else 'regression'
    label_mapper = LabelMapper(args.label_mapping, mode=lm_mode)
    print(f"[Mode] {args.mode}")
    
    if args.mode in ['classification', 'coral']:
        print(f"[Classes] {label_mapper.num_classes}\n")
    else:
        print(f"[Regression] Tolerance: ±{args.tolerance}\n")
    
    # Load data
    print("="*70)
    print("DATA LOADING")
    print("="*70 + "\n")
    
    train_df, val_df = load_and_split_data(
        csv_path=args.csv_path,
        label_mapper=label_mapper,
        val_ratio=args.val_ratio,
        max_samples_per_class=args.max_samples_per_class,
        balance_strategy=args.balance_strategy,
        save_split_path=args.save_split_path
    )
    
    # ======================================================
    # COLOR SPACE SETUP
    # ======================================================

    if args.input_space == "lab":

        stats_path = Path(args.save_dir) / "lab_statistics.json"

        if stats_path.exists():
            print("[LAB Stats] Loading from file")
            with open(stats_path, "r") as f:
                stats = json.load(f)
            lab_mean = stats["mean"]
            lab_std = stats["std"]
        else:
            if args.compute_lab_stats:
                lab_mean, lab_std = compute_lab_stats_full_dataset(
                    args.csv_path,
                    args.image_dir
                )

                stats_path.parent.mkdir(parents=True, exist_ok=True)
                with open(stats_path, "w") as f:
                    json.dump({"mean": lab_mean, "std": lab_std}, f, indent=2)

                print(f"[LAB Stats] Saved to {stats_path}\n")
            else:
                lab_mean = [50.0, 0.0, 0.0]
                lab_std = [25.0, 50.0, 50.0]
                print("[LAB Stats] Using defaults\n")

        # train_transform = LABTransform(lab_mean, lab_std, mode='train')
        # val_transform = LABTransform(lab_mean, lab_std, mode='val')

        train_transform = LABTransform(
            lab_mean, lab_std,
            mode='train',
            drop_l=args.drop_l,
            normalize_l=args.normalize_l
        )

        val_transform = LABTransform(
            lab_mean, lab_std,
            mode='val',
            drop_l=args.drop_l,
            normalize_l=args.normalize_l
        )


    else:
        print("[Input Space] RGB (ImageNet normalization)\n")
        train_transform = RGBTransform(mode='train', normalize_brightness = args.normalize_rgb_brightness)
        val_transform = RGBTransform(mode='val', normalize_brightness = args.normalize_rgb_brightness)
    
    # Datasets
    train_dataset = SkinToneDataset(train_df, args.image_dir, label_mapper, train_transform, 'train')
    val_dataset = SkinToneDataset(val_df, args.image_dir, label_mapper, val_transform, 'val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print("\n" + "="*70)
    print("MODEL & LOSS")
    print("="*70 + "\n")
    
    # Model
    num_outputs = label_mapper.num_classes if (args.mode == 'classification' or args.mode == 'coral') else 1

    if args.arch == "vgg16":
        model = VGG16MSTModel(
            num_outputs=num_outputs,
            mode=args.mode,
            dropout=args.dropout,
            use_bn=args.use_bn,
            pretrained=args.pretrained
        )
    elif args.arch == "resnet18":
        model = ResNet18MSTModel(
            num_outputs=num_outputs,
            mode=args.mode,
            dropout=args.dropout,
            pretrained=args.pretrained
        )

    model = model.to(device)

    
    # Loss functions
    if args.mode == 'classification':
        # Compute class weights
        train_targets = [t for _, t, _, _ in train_dataset]
        class_counts = Counter(train_targets)
        total = len(train_targets)
        class_weights = []
        for i in range(label_mapper.num_classes):
            count = class_counts.get(i, 1)
            weight = total / (label_mapper.num_classes * count)
            class_weights.append(weight)
        
        class_weights = np.array(class_weights)
        if args.balance_strategy == 'weighting':
            class_weights = np.sqrt(class_weights)
            class_weights = np.clip(class_weights, 0.5, 2.0)
        else:
            class_weights = np.ones(label_mapper.num_classes)
        
        class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        
        print(f"[Loss] ", end='')
        if args.focal_gamma > 0:
            print(f"Focal (gamma={args.focal_gamma})")
            if args.use_ordinal_loss:
                print(f"       + Ordinal penalty (weight={args.distance_weight})")
                # Combine focal with ordinal - use ordinal
                mst_criterion = OrdinalCrossEntropyLoss(
                    class_weights=class_weight_tensor,
                    distance_weight=args.distance_weight
                )
            else:
                mst_criterion = FocalLoss(
                    alpha=class_weight_tensor,
                    gamma=args.focal_gamma
                )
        elif args.use_ordinal_loss:
            print(f"Ordinal CE (distance_weight={args.distance_weight})")
            mst_criterion = OrdinalCrossEntropyLoss(
                class_weights=class_weight_tensor,
                distance_weight=args.distance_weight
            )
        else:
            if args.label_smoothing > 0:
                print(f"CrossEntropy (label_smoothing={args.label_smoothing})")
                mst_criterion = nn.CrossEntropyLoss(
                    weight=class_weight_tensor,
                    label_smoothing=args.label_smoothing
                )
            else:
                print(f"CrossEntropy")
                mst_criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
        
        if args.balance_strategy == 'weighting':
            print(f"\n[Class Weights]")
            for i in range(label_mapper.num_classes):
                print(f"  {label_mapper.get_class_name(i)}: {class_weights[i]:.3f}")

    elif args.mode == 'coral':
        print("[Loss] CORAL Ordinal Loss")
        mst_criterion = CORALLoss()
    
    else:  # regression
        # print(f"[Loss] MAE (L1)")
        # mst_criterion = nn.L1Loss()
        print(f"[Loss] Huber (delta=1.0)")
        mst_criterion = nn.HuberLoss(delta=1.0)
    
    # Contrastive loss
    contrastive_criterion = SupervisedContrastiveLoss(temperature=0.07)
    use_contrastive = not args.no_contrastive
    if use_contrastive:
        print(f"[Contrastive] Enabled (λ={args.lambda_contrast})")
    else:
        print(f"[Contrastive] Disabled")
    
    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    print(f"\n[Optimizer] Adam (lr={args.lr}, wd={args.weight_decay})")
    print(f"[Scheduler] ReduceLROnPlateau")
    print(f"[Epochs] {args.epochs}")
    print(f"[Patience] {args.early_stop_patience}\n")
    
    # Resume if checkpoint exists
    start_epoch = 1
    best_val_metric = 0.0 if args.mode in ['classification', 'coral'] else float('inf')
    
    checkpoint_path = Path(args.save_dir) / "checkpoint_latest.pth"
    best_path = Path(args.save_dir) / "best_model.pth"
    
    if args.resume and checkpoint_path.exists():
        print(f"[Resume] Loading from {checkpoint_path}")
        start_epoch, val_acc, val_loss = load_checkpoint(
            model, optimizer, scheduler, checkpoint_path, device
        )
        best_val_metric = val_acc if args.mode in ['classification', 'coral'] else val_loss
        start_epoch += 1
        print(f"[Resume] Starting from epoch {start_epoch}\n")
    
    # Early stopping
    early_stopper = EarlyStopping(
        patience=args.early_stop_patience,
        min_delta=args.early_stop_delta
    )
    
    # Training loop
    print("="*70)
    print("TRAINING")
    print("="*70 + "\n")
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print(f"{'='*70}")
        
        train_results = train_epoch(
            model, train_loader, mst_criterion, contrastive_criterion,
            optimizer, device, epoch, args.mode,
            use_contrastive=use_contrastive,
            lambda_contrast=args.lambda_contrast
        )
        
        show_detailed = (epoch % args.show_metrics_every == 0) or (epoch == args.epochs)
        val_loss, val_metric, val_metrics = evaluate(
            model, val_loader, mst_criterion, device, label_mapper,
            tolerance=args.tolerance,
            show_detailed=show_detailed
        )
        
        scheduler.step(val_loss)
        
        # Print summary
        print(f"\n[EPOCH {epoch}]")
        if args.mode in ['classification', 'coral']:
            print(f"Train: Loss={train_results['total_loss']:.4f}, "
                  f"Top1={train_results['top1']:.2f}%, Off1={train_results['off1']:.2f}%")
            print(f"Val:   Loss={val_loss:.4f}, "
                  f"Person-Top1={val_metric:.2f}%, Person-Off1={val_metrics['person_off1']:.2f}%")
        else:
            print(f"Train: Loss={train_results['total_loss']:.4f}, MAE={train_results['mae']:.3f}")
            print(f"Val:   Loss={val_loss:.4f}, Person-MAE={val_metric:.3f}")
        
        # Save checkpoint
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(model, optimizer, scheduler, epoch, val_metric, val_loss, checkpoint_path)
        
        # Save best
        is_best = False
        if args.mode in ['classification', 'coral']:
            if val_metric > best_val_metric:
                is_best = True
                best_val_metric = val_metric
        else:
            if val_metric < best_val_metric:
                is_best = True
                best_val_metric = val_metric
        
        if is_best:
            torch.save(model.state_dict(), best_path)
            print(f"✓ Best model saved (Metric: {best_val_metric:.2f})")
        
        # Early stopping
        if early_stopper(val_metric if args.mode in ['classification', 'coral'] else -val_metric):
            print(f"\n[Early Stopping] Triggered at epoch {epoch}")
            break
    
    # Save final
    final_path = Path(args.save_dir) / "final_model.pth"
    torch.save(model.state_dict(), final_path)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    metric_name = "Person-Top1" if args.mode == 'classification' else "Person-MAE"
    print(f"Best {metric_name}: {best_val_metric:.2f}{'%' if args.mode == 'classification' else ''}")
    print(f"Models saved in: {args.save_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VGG16 Ultimate - All Features")
    
    # Mode
    parser.add_argument("--mode", choices=['classification', 'regression', 'coral'],
                       default='classification')
    
    # Data
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--label_mapping", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--save_split_path", default="train_val_split.json")
    
    parser.add_argument("--arch", choices=["vgg16", "resnet18"], default="vgg16", help="Model architecture")
    parser.add_argument("--input_space", choices=["lab", "rgb"], default="lab", help="Color space to train on")
    parser.add_argument("--normalize_rgb_brightness", action="store_true")
    parser.add_argument("--drop_l", action="store_true", help="Remove luminance channel (set L constant)")
    parser.add_argument("--normalize_l", action="store_true", help="Per-image normalize luminance channel")

    # Balancing
    parser.add_argument("--balance_strategy", choices=['none', 'weighting', 'sampling'],
                       default='weighting')
    parser.add_argument("--max_samples_per_class", type=int, default=None)
    
    # Model
    parser.add_argument("--pretrained", action='store_true', default=True)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--use_bn", action='store_true')
    
    # Loss functions
    parser.add_argument("--focal_gamma", type=float, default=0.0,
                       help="Focal loss gamma (0 = disabled)")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--use_ordinal_loss", action='store_true')
    parser.add_argument("--distance_weight", type=float, default=0.5)
    
    # Contrastive learning
    parser.add_argument("--no_contrastive", action='store_true')
    parser.add_argument("--lambda_contrast", type=float, default=0.5)
    
    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    
    # Early stopping
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--early_stop_delta", type=float, default=0.0)
    
    # LAB color space
    parser.add_argument("--compute_lab_stats", action='store_true')
    
    # Regression specific
    parser.add_argument("--tolerance", type=float, default=0.5)
    
    # Display
    parser.add_argument("--show_metrics_every", type=int, default=5)
    
    # Resume
    parser.add_argument("--resume", action='store_true')
    
    # Output
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    
    args = parser.parse_args()
    main(args)


##############################################################
# ResNet18 LAB Classification
##############################################################

# Resnet18 4Class Classification -> ✓ Best model saved (Metric: 72.73)
# python vgg16_mst_classification_regression_rgb_lab.py ^
#   --mode classification ^
#   --arch "resnet18" ^
#   --input_space lab ^
#   --csv_path "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\annotations_filtered_MH_with_dark.csv" ^
#   --image_dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2" ^
#   --label_mapping "label_mapping_4class.json" ^
#   --use_ordinal_loss ^
#   --distance_weight 0.2 ^
#   --focal_gamma 0 ^
#   --label_smoothing 0 ^
#   --no_contrastive ^
#   --dropout 0.5 ^
#   --compute_lab_stats ^
#   --balance_strategy sampling ^
#   --max_samples_per_class 1500 ^
#   --batch_size 32 ^
#   --epochs 50 ^
#   --lr 1e-4 ^
#   --weight_decay 2e-4 ^
#   --save_dir  "F:/VGG_MST_Testing/Models/ResNet18_4Classification_LAB" ^
#   --save_split_path "F:/VGG_MST_Testing/Models/ResNet18_4Classification_LAB/train_val_split.json" ^
#   --show_metrics_every 1 ^
#   --resume ^
#   --gpu 0

# Resnet18 4Class Classification Fixed Background -> ✓ Best model saved (Metric: 75.32)
# python vgg16_mst_classification_regression_rgb_lab.py ^
#   --mode classification ^
#   --arch "resnet18" ^
#   --input_space lab ^
#   --csv_path "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\annotations_filtered_MH_with_dark.csv" ^
#   --image_dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2_BGFixed" ^
#   --label_mapping "label_mapping_4class.json" ^
#   --use_ordinal_loss ^
#   --distance_weight 0.2 ^
#   --focal_gamma 0 ^
#   --label_smoothing 0 ^
#   --no_contrastive ^
#   --dropout 0.5 ^
#   --compute_lab_stats ^
#   --balance_strategy sampling ^
#   --max_samples_per_class 1500 ^
#   --batch_size 32 ^
#   --epochs 50 ^
#   --lr 1e-4 ^
#   --weight_decay 2e-4 ^
#   --save_dir  "F:/VGG_MST_Testing/Models/ResNet18_4Classification_LAB_FixedBG" ^
#   --save_split_path "F:/VGG_MST_Testing/Models/ResNet18_4Classification_LAB_FixedBG/train_val_split.json" ^
#   --show_metrics_every 1 ^
#   --resume ^
#   --gpu 0

# Resnet18 3Class Classification Fixed Background -> ✓ Best model saved (Metric: 89.61)
# python vgg16_mst_classification_regression_rgb_lab.py ^
#   --mode classification ^
#   --arch "resnet18" ^
#   --input_space lab ^
#   --csv_path "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\annotations_filtered_MH_with_dark.csv" ^
#   --image_dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2_BGFixed" ^
#   --label_mapping "label_mapping_3class.json" ^
#   --use_ordinal_loss ^
#   --distance_weight 0.2 ^
#   --focal_gamma 0 ^
#   --label_smoothing 0 ^
#   --no_contrastive ^
#   --dropout 0.5 ^
#   --compute_lab_stats ^
#   --balance_strategy sampling ^
#   --max_samples_per_class 1500 ^
#   --batch_size 32 ^
#   --epochs 50 ^
#   --lr 1e-4 ^
#   --weight_decay 2e-4 ^
#   --save_dir  "F:/VGG_MST_Testing/Models/ResNet18_3Classification_LAB_FixedBG" ^
#   --save_split_path "F:/VGG_MST_Testing/Models/ResNet18_3Classification_LAB_FixedBG/train_val_split.json" ^
#   --show_metrics_every 1 ^
#   --resume ^
#   --gpu 0

# Resnet18 10Class Classification Fixed Background -> 
# python vgg16_mst_classification_regression_rgb_lab.py ^
#   --mode classification ^
#   --arch "resnet18" ^
#   --input_space lab ^
#   --csv_path "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\annotations_filtered_MH_with_dark.csv" ^
#   --image_dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2_BGFixed" ^
#   --label_mapping "label_mapping_10class.json" ^
#   --use_ordinal_loss ^
#   --distance_weight 0.2 ^
#   --focal_gamma 0 ^
#   --label_smoothing 0 ^
#   --no_contrastive ^
#   --dropout 0.5 ^
#   --compute_lab_stats ^
#   --balance_strategy sampling ^
#   --max_samples_per_class 1500 ^
#   --batch_size 32 ^
#   --epochs 50 ^
#   --lr 1e-4 ^
#   --weight_decay 2e-4 ^
#   --save_dir  "F:/VGG_MST_Testing/Models/ResNet18_10Classification_LAB_FixedBG" ^
#   --save_split_path "F:/VGG_MST_Testing/Models/ResNet18_10Classification_LAB_FixedBG/train_val_split.json" ^
#   --show_metrics_every 1 ^
#   --resume ^
#   --gpu 0



















# Resnet18 4Class Coral -> Best Person-MAE: 71.43
# python vgg16_mst_classification_regression_rgb_lab.py ^
#   --mode coral ^
#   --arch "resnet18" ^
#   --input_space lab ^
#   --csv_path "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\annotations_filtered_MH_with_dark.csv" ^
#   --image_dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2" ^
#   --label_mapping "label_mapping_4class.json" ^
#   --distance_weight 0.2 ^
#   --focal_gamma 0 ^
#   --label_smoothing 0 ^
#   --no_contrastive ^
#   --dropout 0.5 ^
#   --compute_lab_stats ^
#   --balance_strategy sampling ^
#   --max_samples_per_class 1500 ^
#   --batch_size 32 ^
#   --epochs 50 ^
#   --lr 1e-4 ^
#   --weight_decay 2e-4 ^
#   --save_dir  "F:/VGG_MST_Testing/Models/ResNet18_4CCoral_LAB" ^
#   --save_split_path "F:/VGG_MST_Testing/Models/ResNet18_4CCoral_LAB/train_val_split.json" ^
#   --show_metrics_every 1 ^
#   --resume ^
#   --gpu 0

# Resnet18 4Class Regression -> ✓ Best model saved (Metric: 0.45)
# python vgg16_mst_classification_regression_rgb_lab.py ^
#   --mode regression ^
#   --arch "resnet18" ^
#   --input_space lab ^
#   --csv_path "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\annotations_filtered_MH_with_dark.csv" ^
#   --image_dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2" ^
#   --label_mapping "label_mapping_4class.json" ^
#   --distance_weight 0.2 ^
#   --focal_gamma 0 ^
#   --label_smoothing 0 ^
#   --no_contrastive ^
#   --dropout 0.5 ^
#   --compute_lab_stats ^
#   --balance_strategy sampling ^
#   --max_samples_per_class 1500 ^
#   --batch_size 32 ^
#   --epochs 50 ^
#   --lr 1e-4 ^
#   --weight_decay 2e-4 ^
#   --save_dir  "F:/VGG_MST_Testing/Models/ResNet18_4CRegression_LAB" ^
#   --save_split_path "F:/VGG_MST_Testing/Models/ResNet18_4CRegression_LAB/train_val_split.json" ^
#   --show_metrics_every 1 ^
#   --resume ^
#   --gpu 0




# Resnet18 4Class Coral Fixed Background -> Best Person-MAE: 72.73
# python vgg16_mst_classification_regression_rgb_lab.py ^
#   --mode coral ^
#   --arch "resnet18" ^
#   --input_space lab ^
#   --csv_path "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\annotations_filtered_MH_with_dark.csv" ^
#   --image_dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2_BGFixed" ^
#   --label_mapping "label_mapping_4class.json" ^
#   --distance_weight 0.2 ^
#   --focal_gamma 0 ^
#   --label_smoothing 0 ^
#   --no_contrastive ^
#   --dropout 0.5 ^
#   --compute_lab_stats ^
#   --balance_strategy sampling ^
#   --max_samples_per_class 1500 ^
#   --batch_size 32 ^
#   --epochs 50 ^
#   --lr 1e-4 ^
#   --weight_decay 2e-4 ^
#   --save_dir  "F:/VGG_MST_Testing/Models/ResNet18_4CCoral_LAB_FixedBG" ^
#   --save_split_path "F:/VGG_MST_Testing/Models/ResNet18_4CCoral_LAB_FixedBG/train_val_split.json" ^
#   --show_metrics_every 1 ^
#   --resume ^
#   --gpu 0


