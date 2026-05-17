##############################################################
#  DenseNet121-LAB TRAINING PIPELINE (MODE-SPECIFIC WEIGHTING)
#  - LAB input + dataset stats
#  - Person-wise stratified split
#  - MODE-SPECIFIC weighting:
#      * Classification: class weights only (PyTorch built-in)
#      * Regression: per-sample weights (for continuous labels)
#  - Modes: classification (default, 10 MST) OR regression
#  - Losses: CE / OCE (classification), MSE (regression)
#  - Fine-tuning: last block only OR full network
#  - Resume support:
#      * --resume
#      * --additional-epochs
#      * checkpoint: densenet121_lab_checkpoint.pth
##############################################################

import argparse
from pathlib import Path
from collections import Counter
import json

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

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
# 1. MONK -> LAB utilities (for L2 in LAB space)
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
    labs = []
    for v in values:
        v = float(max(1, min(10, v)))
        base = int(v)
        if base >= 10:
            labs.append(MONK_COLORS_LAB[-1])
            continue
        lab0 = MONK_COLORS_LAB[base - 1]
        lab1 = MONK_COLORS_LAB[base]
        labs.append(lab0 + (v - base) * (lab1 - lab0))
    return np.stack(labs)


def calc_l2_lab_distance(pred, target):
    """
    pred, target in [0,1] normalized MST regression space.
    Convert to MST [1,10], then to LAB, then L2.
    """
    p = pred.detach().cpu().numpy() * 9 + 1
    t = target.detach().cpu().numpy() * 9 + 1
    return np.sqrt(np.sum((monk_scalar_to_lab(p) - monk_scalar_to_lab(t)) ** 2, axis=1))

def save_split_json(train_persons, val_persons, val_ratio, save_path):
    split_data = {
        "train_persons": sorted(list(map(str, train_persons))),
        "val_persons": sorted(list(map(str, val_persons))),
        "random_state": 42,
        "val_ratio": val_ratio,
    }

    with open(save_path, "w") as f:
        json.dump(split_data, f, indent=4)

    print(f"[INFO] Train/Val split saved -> {save_path}")


##############################################################
# 2. LAB DATASET STATISTICS
##############################################################

def compute_dataset_lab_stats(image_dir, csv_path):
    df = pd.read_csv(csv_path)
    img_names = df.iloc[:, 0].tolist()

    print(f"[INFO] Computing LAB dataset statistics on {len(img_names)} images...")

    resize_op = transforms.Resize((224, 224))

    sum_lab = np.zeros(3, dtype=np.float64)
    sum_sq_lab = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for name in tqdm(img_names, desc="LAB stats"):
        fpath = Path(image_dir) / name

        try:
            img = Image.open(fpath).convert("RGB")
            img = resize_op(img)

            rgb = np.asarray(img).astype(np.float32) / 255.0
            lab = rgb2lab(rgb).astype(np.float64)

            lab_flat = lab.reshape(-1, 3)
            sum_lab += lab_flat.sum(axis=0)
            sum_sq_lab += (lab_flat ** 2).sum(axis=0)
            total_pixels += lab_flat.shape[0]

        except Exception as e:
            print(f"[WARN] Could not read {fpath}: {e}")

    mean = sum_lab / total_pixels
    var = (sum_sq_lab / total_pixels) - (mean ** 2)
    std = np.sqrt(var)

    print("\n========== LAB DATASET STATISTICS ==========")
    print("LAB_MEAN =", mean.tolist())
    print("LAB_STD  =", std.tolist())
    print("============================================\n")

    return mean.astype(np.float32), std.astype(np.float32)


##############################################################
# 3. LAB Transform
##############################################################

class RGBToLABTensorTransform:
    first_call = True

    def __init__(self, is_train=True, lab_mean=None, lab_std=None):
        self.lab_mean = lab_mean
        self.lab_std = lab_std

        if is_train:
            self.geom = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.geom = transforms.Compose([
                transforms.Resize((224, 224)),
            ])

    def __call__(self, img_pil):
        img_pil = self.geom(img_pil)

        rgb = np.asarray(img_pil).astype(np.float32) / 255.0
        lab = rgb2lab(rgb).astype(np.float32)
        lab_norm = (lab - self.lab_mean) / self.lab_std

        if RGBToLABTensorTransform.first_call:
            RGBToLABTensorTransform.first_call = False
            print("\n[DIAG] LAB BEFORE NORMALIZATION:")
            print("  L range:", lab[..., 0].min(), lab[..., 0].max())
            print("  a range:", lab[..., 1].min(), lab[..., 1].max())
            print("  b range:", lab[..., 2].min(), lab[..., 2].max())
            print("[DIAG] LAB AFTER NORMALIZATION:")
            print("  min:", lab_norm.min(), "max:", lab_norm.max())
            print("  mean:", lab_norm.mean(), "std:", lab_norm.std(), "\n")

        return torch.from_numpy(lab_norm.transpose(2, 0, 1)).float()


##############################################################
# 4. Dataset (MODE-SPECIFIC WEIGHTING)
##############################################################

class SkinToneLABDataset(Dataset):
    """
    MODE-SPECIFIC dataset:
    - mode='regression': returns (img, label, weight, mst_label)
                        Uses per-sample weights for imbalance handling
    - mode='classification': returns (img, label, mst_label)
                            No per-sample weights (uses class weights in loss)
    """
    def __init__(self, df, img_dir, transform, mode="classification", weights=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.mode = mode

        if self.mode not in {"classification", "regression"}:
            raise ValueError("mode must be 'classification' or 'regression'")

        # Precompute MST labels
        self.mst_vals = self.df.iloc[:, 1].astype(float).values

        # ONLY compute per-sample weights for REGRESSION mode
        if self.mode == "regression":
            if weights is None:
                print("[INFO] REGRESSION mode: Calculating per-sample weights (MST bins)...")
                raw_labels = self.mst_vals
                binned_labels = np.round(raw_labels).astype(int)
                binned_labels = np.clip(binned_labels, 1, 10)

                counts = Counter(binned_labels)
                total_samples = len(binned_labels)

                weights_map = {k: total_samples / v for k, v in counts.items()}
                weights = [weights_map[int(round(x))] for x in raw_labels]
                weights = np.array(weights, dtype=np.float32)
                weights = weights / weights.mean()

                print(f"[INFO] Per-sample weights calculated. Min: {weights.min():.2f}, "
                      f"Max: {weights.max():.2f}, Mean: {weights.mean():.2f}")

            self.weights = torch.FloatTensor(weights)
        else:
            # Classification: no per-sample weights
            print("[INFO] CLASSIFICATION mode: Using class weights only (no per-sample weights)")
            self.weights = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / str(row.iloc[0])

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to load {img_path}: {e}")
            img = Image.new('RGB', (224, 224), color='black')

        img_t = self.transform(img)

        mst_label = float(self.mst_vals[idx])
        mst_label_clipped = float(np.clip(round(mst_label), 1, 10))

        if self.mode == "regression":
            # Normalise MST [1,10] -> [0,1]
            label = (mst_label - 1.0) / 9.0
            label_t = torch.tensor(label, dtype=torch.float32)
            weight = self.weights[idx]
            return img_t, label_t, weight, torch.tensor(mst_label, dtype=torch.float32)
        else:
            # Classification: 0..9 class index, no weight
            cls_idx = int(mst_label_clipped) - 1
            label_t = torch.tensor(cls_idx, dtype=torch.long)
            return img_t, label_t, torch.tensor(mst_label, dtype=torch.float32)


def build_dataloaders(
    csv_path,
    image_dir,
    batch_size,
    val_ratio,
    lab_mean,
    lab_std,
    mode="classification",
    num_workers=4,
    save_split_path=None
):
    """
    Person-wise stratified split (shared for both modes).
    Returns class_weights for classification mode (used in loss functions).
    """
    df = pd.read_csv(csv_path).dropna()

    if df.shape[1] < 3:
        raise ValueError("CSV must have 3 columns: filename, label, person_id")

    print(f"[INFO] Loaded CSV with columns: {list(df.columns)}")

    df['person_id'] = df.iloc[:, 2].astype(str)

    # Person-level labels
    person_labels = df.groupby('person_id').agg({
        df.columns[1]: 'first'
    }).reset_index()
    person_labels.columns = ['person_id', 'label']
    labels_per_person = person_labels['label'].astype(float).values

    print(f"[INFO] Found {len(person_labels)} unique persons")
    print(f"[INFO] Label range: {labels_per_person.min():.2f} to {labels_per_person.max():.2f}")

    rounded_labels = np.round(labels_per_person).astype(int)
    rounded_labels = np.clip(rounded_labels, 1, 10)

    print(f"[INFO] Rounded MST labels: {np.unique(rounded_labels)}")

    label_counts = Counter(rounded_labels)
    print("\n[INFO] Person distribution across MST labels:")
    for mst in sorted(label_counts.keys()):
        print(f"  MST {mst}: {label_counts[mst]} persons")
    print()

    # Person-wise stratified split
    train_persons, val_persons = train_test_split(
        person_labels['person_id'].values,
        test_size=val_ratio,
        shuffle=True,
        stratify=rounded_labels,
        random_state=42,
    )

    if save_split_path is not None:
        save_split_json(
            train_persons=train_persons,
            val_persons=val_persons,
            val_ratio=val_ratio,
            save_path=save_split_path
        )


    train_df = df[df['person_id'].isin(train_persons)].reset_index(drop=True)
    val_df = df[df['person_id'].isin(val_persons)].reset_index(drop=True)

    train_df = train_df[[df.columns[0], df.columns[1]]]
    val_df = val_df[[df.columns[0], df.columns[1]]]

    print(f"[INFO] Train: {len(train_persons)} persons ({len(train_df)} images)")
    print(f"[INFO] Val:   {len(val_persons)} persons ({len(val_df)} images)")

    # Diagnostics on image-level label distributions
    train_labels = train_df.iloc[:, 1].astype(float).values
    val_labels = val_df.iloc[:, 1].astype(float).values

    train_rounded = np.clip(np.round(train_labels).astype(int), 1, 10)
    val_rounded = np.clip(np.round(val_labels).astype(int), 1, 10)

    train_counts = Counter(train_rounded)
    val_counts = Counter(val_rounded)

    print("\n[INFO] Train MST label distribution (images):")
    for mst in sorted(train_counts.keys()):
        print(f"  MST {mst}: {train_counts[mst]} images")

    print("\n[INFO] Val MST label distribution (images):")
    for mst in sorted(val_counts.keys()):
        print(f"  MST {mst}: {val_counts[mst]} images")
    print()

    train_tf = RGBToLABTensorTransform(is_train=True,  lab_mean=lab_mean, lab_std=lab_std)
    val_tf   = RGBToLABTensorTransform(is_train=False, lab_mean=lab_mean, lab_std=lab_std)

    train_ds = SkinToneLABDataset(train_df, image_dir, train_tf, mode=mode)
    
    # For regression: pass train weights to val set
    if mode == "regression":
        val_ds = SkinToneLABDataset(val_df, image_dir, val_tf, mode=mode,
                                    weights=train_ds.weights.numpy())
    else:
        val_ds = SkinToneLABDataset(val_df, image_dir, val_tf, mode=mode)

    # Calculate class weights (for classification CE/OCE)
    train_mst_rounded = train_rounded
    cls_counts = Counter(train_mst_rounded)
    total = len(train_mst_rounded)
    num_classes = 10
    cls_weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(1, num_classes + 1):
        if cls_counts[c] > 0:
            cls_weights[c - 1] = total / cls_counts[c]
        else:
            cls_weights[c - 1] = 0.0
    if cls_weights.sum() > 0:
        cls_weights = cls_weights / cls_weights[cls_weights > 0].mean()

    if mode == "classification":
        print("\n[INFO] Class weights for CLASSIFICATION (index 0..9):")
        for c in range(num_classes):
            print(f"  Class {c} (MST {c+1}): weight={cls_weights[c]:.3f}, count={cls_counts.get(c+1,0)}")
        print()
    else:
        print("[INFO] REGRESSION mode: Class weights will not be used.\n")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    return train_loader, val_loader, torch.FloatTensor(cls_weights)


##############################################################
# 5. DenseNet121 Model (LAB input, modes, finetune switch)
##############################################################

class DenseNet121LabImproved(nn.Module):
    def __init__(self, mode="classification", finetune_mode="last"):
        """
        mode: 'classification' or 'regression'
        finetune_mode:
          - 'last': only denseblock4 + norm5 + classifier trainable
          - 'all' : all parameters trainable
        """
        super().__init__()

        if mode not in {"classification", "regression"}:
            raise ValueError("mode must be 'classification' or 'regression'")
        if finetune_mode not in {"last", "all"}:
            raise ValueError("finetune_mode must be 'last' or 'all'")

        self.mode = mode

        base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_features = base.classifier.in_features

        if self.mode == "classification":
            base.classifier = nn.Linear(in_features, 10)
        else:
            base.classifier = nn.Sequential(
                nn.Linear(in_features, 1),
                nn.Sigmoid()
            )

        self.backbone = base

        # Freezing strategy
        if finetune_mode == "all":
            for p in self.backbone.parameters():
                p.requires_grad = True
        else:
            # Freeze everything
            for p in self.backbone.parameters():
                p.requires_grad = False

            # Unfreeze denseblock4, norm5, classifier
            for name, module in self.backbone.named_modules():
                if name.startswith("features.denseblock4") or name.startswith("features.norm5"):
                    for p in module.parameters():
                        p.requires_grad = True
            for p in self.backbone.classifier.parameters():
                p.requires_grad = True

    def forward(self, x):
        out = self.backbone(x)
        if self.mode == "classification":
            return out  # logits (N,10)
        else:
            return out.squeeze(-1)  # (N,) in [0,1]


##############################################################
# 6. Losses (MODE-SPECIFIC)
##############################################################

class OrdinalCrossEntropyWeighted(nn.Module):
    """
    OCE with class-weights ONLY (no per-sample weights):
    loss_i = (1 + |y - y_hat|/k) * CE_i
    Class weights handled internally by nn.CrossEntropyLoss
    """
    def __init__(self, num_classes=10, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss(weight=class_weights, reduction='none')

    def forward(self, logits, target_idx):
        ce_per_sample = self.ce(logits, target_idx)  # (N,) - already class-weighted
        with torch.no_grad():
            preds_idx = torch.argmax(logits, dim=1)
            y = target_idx.float() + 1.0
            y_hat = preds_idx.float() + 1.0
            dist = torch.abs(y - y_hat)
            factor = 1.0 + dist / float(self.num_classes)  # (N,)
        loss = ce_per_sample * factor  # Apply ordinal penalty
        return loss.mean()


def mse_weighted(preds, targets, sample_weights):
    """MSE with per-sample weights (for REGRESSION only)"""
    loss_unreduced = F.mse_loss(preds, targets, reduction="none")
    loss = loss_unreduced * sample_weights
    return loss.mean()


##############################################################
# 7. Metrics & Diagnostics
##############################################################

def compute_metrics_regression(pred, tgt, threshold=0.5):
    """
    Regression mode:
    pred, tgt in [0,1] MST-normalised space.
    """
    pred_np = pred.detach().cpu().numpy() * 9 + 1
    tgt_np = tgt.detach().cpu().numpy() * 9 + 1

    abs_diff = np.abs(pred_np - tgt_np)
    acc1 = (abs_diff <= threshold).mean() * 100
    # L2 in LAB
    l2 = calc_l2_lab_distance(pred, tgt)
    return acc1, l2


def compute_metrics_classification(pred_idx, tgt_mst):
    """
    Classification mode:
    pred_idx in [0..9], tgt_mst in MST space [1..10] (float).
    """
    pred_mst = pred_idx.detach().cpu().numpy() + 1
    tgt_mst_np = tgt_mst.detach().cpu().numpy()
    pred_mst_rounded = np.round(pred_mst).astype(int)
    tgt_mst_rounded = np.round(tgt_mst_np).astype(int)

    abs_diff = np.abs(pred_mst_rounded - tgt_mst_rounded)
    acc_exact = (abs_diff == 0).mean() * 100
    acc_pm1 = (abs_diff <= 1).mean() * 100

    # for L2, map classification outputs to [0,1]
    pred_norm = (pred_mst - 1.0) / 9.0
    tgt_norm = (tgt_mst_np - 1.0) / 9.0
    l2 = np.sqrt(np.sum(
        (monk_scalar_to_lab(pred_norm * 9 + 1) - monk_scalar_to_lab(tgt_norm * 9 + 1)) ** 2,
        axis=1
    ))

    return acc_exact, acc_pm1, l2


def print_diagnostics(imgs, labels, preds, mst_labels, mode, tag):
    print(f"\n=== [DIAG] {tag} ===")
    print("IMG mean/std:", imgs.mean().item(), imgs.std().item())
    print("IMG min/max:", imgs.min().item(), imgs.max().item())

    if mode == "regression":
        print("Labels (norm) min/max:", labels.min().item(), labels.max().item())
        print("Preds   (norm) min/max:", preds.min().item(), preds.max().item())
    else:
        print("Labels (cls idx) min/max:", labels.min().item(), labels.max().item())
        print("Preds  (cls idx) min/max:", preds.min().item(), preds.max().item())

    print("MST label range:", mst_labels.min().item(), mst_labels.max().item())

    # FIX: compute std safely for integer tensors
    if preds.dtype in (torch.float16, torch.float32, torch.float64):
        st = preds.std().item()
    else:
        st = preds.float().std().item()

    if st < 1e-4:
        print("WARNING: predictions collapsed to near-constant.")

    print("===========================================\n")


##############################################################
# 8. Train / Validate (MODE-SPECIFIC UNPACKING)
##############################################################

def train_one_epoch(
    model,
    loader,
    optim,
    device,
    epoch,
    mode,
    loss_type,
    threshold=0.5,
    class_weights=None
):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_samples = 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", ncols=100)

    # Setup loss functions for classification
    if mode == "classification":
        if loss_type == "ce":
            ce = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
        else:  # 'oce'
            oce = OrdinalCrossEntropyWeighted(num_classes=10, class_weights=class_weights)

    for i, batch in enumerate(pbar):
        # MODE-SPECIFIC UNPACKING
        if mode == "regression":
            imgs, labels, weights, mst_labels = batch
            weights = weights.to(device, non_blocking=True)
        else:
            imgs, labels, mst_labels = batch
        
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        mst_labels = mst_labels.to(device, non_blocking=True)

        preds = model(imgs)

        if mode == "regression":
            loss = mse_weighted(preds, labels, weights)
            acc, _ = compute_metrics_regression(preds, labels, threshold=threshold)
        else:
            if loss_type == "ce":
                loss = ce(preds, labels)
            else:  # 'oce'
                loss = oce(preds, labels)

            pred_idx = torch.argmax(preds, dim=1)
            acc_exact, _, _ = compute_metrics_classification(pred_idx, mst_labels)
            acc = acc_exact

        optim.zero_grad()
        loss.backward()
        optim.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_acc += acc * bs
        n_samples += bs

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{acc:.2f}"
        })

        if i == 0:
            if mode == "regression":
                print_diagnostics(imgs, labels, preds, mst_labels, mode, f"TRAIN EPOCH {epoch}")
            else:
                print_diagnostics(imgs, labels, pred_idx, mst_labels, mode, f"TRAIN EPOCH {epoch}")

    return total_loss / n_samples, total_acc / n_samples


def validate(
    model,
    loader,
    device,
    epoch,
    mode,
    loss_type,
    threshold=0.5,
    class_weights=None
):
    model.eval()
    
    total_loss = 0.0
    total_acc = 0.0
    all_l2 = []
    n_samples = 0

    # Setup loss functions
    if mode == "classification":
        if loss_type == "ce":
            ce = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
        else:
            oce = OrdinalCrossEntropyWeighted(num_classes=10, class_weights=class_weights)
    else:
        mse = nn.MSELoss(reduction='none')

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Validate Epoch {epoch}", ncols=100)

        for i, batch in enumerate(pbar):
            # MODE-SPECIFIC UNPACKING
            if mode == "regression":
                imgs, labels, weights, mst_labels = batch
                weights = weights.to(device, non_blocking=True)
            else:
                imgs, labels, mst_labels = batch
            
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mst_labels = mst_labels.to(device, non_blocking=True)

            preds = model(imgs)

            if mode == "regression":
                loss_unreduced = mse(preds, labels)
                loss = (loss_unreduced * weights).mean()
                acc, l2 = compute_metrics_regression(preds, labels, threshold=threshold)
            else:
                if loss_type == "ce":
                    loss = ce(preds, labels)
                else:
                    loss = oce(preds, labels)

                pred_idx = torch.argmax(preds, dim=1)
                acc_exact, acc_pm1, l2 = compute_metrics_classification(pred_idx, mst_labels)
                acc = acc_exact

            bs = imgs.size(0)
            total_loss += loss.item() * bs
            total_acc += acc * bs
            all_l2.extend(list(l2))
            n_samples += bs

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{acc:.2f}"
            })

            if i == 0:
                if mode == "regression":
                    print_diagnostics(imgs, labels, preds, mst_labels, mode, f"VALID EPOCH {epoch}")
                else:
                    print_diagnostics(imgs, labels, pred_idx, mst_labels, mode, f"VALID EPOCH {epoch}")

    return total_loss / n_samples, total_acc / n_samples, float(np.mean(all_l2)), float(np.std(all_l2))


##############################################################
# 9. Checkpoint Management (Resume Logic)
##############################################################

def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, val_loss, path):
    """Save training checkpoint with all state."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
    }, path)
    print(f"[INFO] Checkpoint saved -> {path}")


def load_checkpoint(model, optimizer, scheduler, path, device):
    """Load training checkpoint and return start epoch + best metrics."""
    if not Path(path).exists():
        print(f"[INFO] No checkpoint found at {path}, starting from scratch.")
        return 1, float('inf'), 0.0
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['val_loss']
    best_val_acc = checkpoint['val_acc']
    
    print(f"[INFO] Resuming from epoch {checkpoint['epoch']}")
    print(f"[INFO] Previous best - ValLoss: {best_val_loss:.4f}, ValAcc: {best_val_acc:.2f}%")
    
    return start_epoch, best_val_loss, best_val_acc


##############################################################
# 10. Early Stopping
##############################################################

class EarlyStopping:
    """
    Early stopping based on validation accuracy.
    """
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
# 11. MAIN
##############################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--save-dir", default="./checkpoints_densenet_lab")
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--additional-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)

    # LAB stats
    parser.add_argument("--compute-lab-stats", action="store_true")

    # Mode / loss
    parser.add_argument("--mode", choices=["classification", "regression"], default="classification",
                        help="Model mode: classification (10 MST classes) or regression (continuous MST)")
    parser.add_argument("--loss", choices=["ce", "oce", "mse"], default=None,
                        help="Loss: ce/oce (classification), mse (regression). If None, uses sensible default.")

    # Fine-tuning
    parser.add_argument("--finetune", choices=["last", "all"], default="last",
                        help="Fine-tuning mode: 'last' (paper-style) or 'all'")

    # Resume & Early stopping & scheduler
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-delta", type=float, default=0.1)
    parser.add_argument("--lr-patience", type=int, default=5)

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # LAB stats
    if args.compute_lab_stats:
        print("[INFO] Computing LAB statistics from dataset...")
        lab_mean, lab_std = compute_dataset_lab_stats(args.image_dir, args.csv_path)
        print(f"[INFO] Computed LAB_MEAN: {lab_mean.tolist()}")
        print(f"[INFO] Computed LAB_STD:  {lab_std.tolist()}")
    else:
        print("[INFO] Using pre-computed LAB normalization values.")
        # lab_mean = np.array([27.715821371226003, 10.521480987873188, 8.514460146640673], dtype=np.float32)
        # lab_std  = np.array([24.70048803837073, 8.827357389195186, 8.660419910058293], dtype=np.float32)

        lab_mean = np.array([33.618656158447266, 8.958210945129395, 8.925719261169434], dtype=np.float32)
        lab_std = np.array([26.940208435058594, 8.05940055847168, 9.126977920532227], dtype=np.float32)

    print(f"[INFO] Active LAB_MEAN: {lab_mean.tolist()}")
    print(f"[INFO] Active LAB_STD:  {lab_std.tolist()}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_path = save_dir / "densenet121_lab_best.pth"
    final_path = save_dir / "densenet121_lab_final.pth"
    checkpoint_path = save_dir / "densenet121_lab_checkpoint.pth"
    split_json_path = save_dir / "train_val_split.json"

    # Resolve loss default based on mode
    if args.mode == "classification":
        loss_type = args.loss or "oce"
        if loss_type not in {"ce", "oce"}:
            raise ValueError("For classification, loss must be 'ce' or 'oce'.")
    else:
        loss_type = args.loss or "mse"
        if loss_type != "mse":
            raise ValueError("For regression mode, only 'mse' is allowed.")

    print(f"[INFO] Mode: {args.mode}, Loss: {loss_type}, Finetune: {args.finetune}")

    # Data
    train_loader, val_loader, cls_weights = build_dataloaders(
        args.csv_path, args.image_dir, args.batch_size, args.val_ratio,
        lab_mean, lab_std, mode=args.mode, num_workers=args.num_workers,
        split_save_path=split_json_path
    )
    cls_weights = cls_weights.to(device)

    # Model
    model = DenseNet121LabImproved(mode=args.mode, finetune_mode=args.finetune).to(device)

    # Optimizer & scheduler
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.lr_patience, verbose=True
    )

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
                print(f"[WARNING] Checkpoint is at epoch {start_epoch-1}, already exceeds --epochs {max_epoch}")
                return
            print(f"[INFO] Will train until epoch {max_epoch}")
    else:
        start_epoch = 1
        max_epoch = args.epochs
        best_val_loss = float("inf")
        best_val_acc = 0.0
        print(f"[INFO] Starting fresh training for {max_epoch} epochs")

    # Early stopping on ValAcc
    early_stopper = EarlyStopping(
        patience=args.early_stop_patience,
        min_delta=args.early_stop_delta
    )

    # Training loop
    for epoch in range(start_epoch, max_epoch + 1):
        print(f"\n===== EPOCH {epoch}/{max_epoch} =====")
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            mode=args.mode, loss_type=loss_type,
            threshold=args.threshold, class_weights=cls_weights
        )
        va_loss, va_acc, l2_mean, l2_std = validate(
            model, val_loader, device, epoch,
            mode=args.mode, loss_type=loss_type,
            threshold=args.threshold, class_weights=cls_weights
        )

        print(
            f"[EPOCH {epoch}] "
            f"TrainLoss={tr_loss:.4f} TrainAcc={tr_acc:.2f}% | "
            f"ValLoss={va_loss:.4f} ValAcc={va_acc:.2f}% | "
            f"L2 mean={l2_mean:.3f}, std={l2_std:.3f}"
        )

        scheduler.step(va_loss)

        # Track best by ValLoss, but early stop by ValAcc
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(model.state_dict(), best_path)
            print(f"[INFO] Saved BEST model -> {best_path}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc

        # Save checkpoint each epoch (for resume)
        save_checkpoint(model, optimizer, scheduler, epoch, va_acc, va_loss, checkpoint_path)

        if early_stopper(va_acc):
            print(f"[INFO] Early stopping triggered at epoch {epoch}")
            break

    torch.save(model.state_dict(), final_path)
    print(f"[INFO] Saved FINAL model -> {final_path}")
    print(f"[INFO] Training complete. Best ValAcc: {best_val_acc:.2f}%, Best ValLoss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

# python DenseNet121_SkinTone_Training.py ^
#   --csv-path "" ^
#   --image-dir "" ^
#   --save-dir "" ^
#   --epochs 32 --batch-size 32 ^
#   --val-ratio 0.35 --lr 1e-4 --weight-decay 1e-4 ^
#   --mode regression --loss mse --finetune all --gpu 0
