"""
This script implements a complete, paper-aligned experimental
pipeline for probing demographic and appearance-related artifacts in
image datasets using supervised classification models. For each task
(gender or skin-tone), the pipeline trains ImageNet-pretrained
models using fixed train/validation/test splits, performs hyperparameter
selection via grid search on the validation set, and evaluates the
best-performing model exactly once on a held-out test set. Performance
is quantified primarily using ROC-AUC, with confidence intervals
estimated via bootstrapping to capture statistical uncertainty.

The implementation faithfully reproduces the core methodological
choices described in the paper, including the use of ResNet-50 probes,
standard ImageNet preprocessing, SGD optimization with momentum, and
validation-based model selection. Extensions beyond the paper—such as
support for ConvNeXt-Tiny backbones, early stopping, additional
diagnostic metrics, and configurable hyperparameter grids—are included
to improve computational efficiency, robustness analysis, and
interpretability, while preserving the original experimental logic.

Enhancements over the original script:
- Checkpoint / resume support at epoch level, per (seed, combo) pair
- Grid-search state saved to JSON so completed combos are skipped on resume
- Per-epoch history dict tracking loss, accuracy, AUC, and learning rate
- Per-seed output files: metrics_seed_{seed}.json, history_seed_{seed}.json,
  model_seed_{seed}.pt (full save dict with hyperparameters + history)
- summary.json aggregating results across all seeds (mean ± std)
- EarlyStopping as a proper class with serialisable state for checkpointing
- leave=True on tqdm so epoch bars are retained in the terminal
"""

import argparse
import copy
import json
import random
from pathlib import Path
import csv
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
from PIL import Image
from tqdm import tqdm

COULDNTLOADCOUNT = 0

MST3_MAPPING = {
    0: 0,  # light
    1: 0,
    2: 0,
    3: 1,  # medium
    4: 1,
    5: 1,
    6: 1,  
    7: 2,  # dark
    8: 2,
    9: 2
}

# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
    """
    Sets all relevant random seeds to ensure experimental reproducibility.

    This function fixes the random state for Python's random module, NumPy,
    and PyTorch (both CPU and CUDA). This ensures that data shuffling,
    weight initialization, and other stochastic components are repeatable
    across runs.

    Alignment with paper:
    The paper reports confidence intervals via bootstrapping and repeated
    training across multiple random seeds. Fixing the random seed enables
    controlled multi-seed experiments and reproducible comparisons.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ============================================================
# Dataset
# ============================================================

class ImageLabelDataset(Dataset):
    """
    Simple image-label dataset wrapper.

    This dataset takes pre-defined (image_path, label) pairs and applies
    a specified transform at load time. The dataset itself is agnostic
    to how the splits are created; it simply consumes the train/val/test
    partitions defined externally (via JSON).
    """
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        global COULDNTLOADCOUNT

        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (FileNotFoundError, Exception):
            # print(f"Couldn't load image: {path}")
            COULDNTLOADCOUNT+=1
            img = Image.new("RGB", (224, 224), (0, 0, 0))
            return self.transform(img), label
        img = self.transform(img)
        return img, label

def collapse_labels_to_mst3(labels):
    return np.array([MST3_MAPPING[l] for l in labels])

def collapse_probs_to_mst3(probs):

    mst3_probs = np.zeros((probs.shape[0], 3))

    for cls, group in MST3_MAPPING.items():
        if cls < probs.shape[1]:
            mst3_probs[:, group] += probs[:, cls]

    return mst3_probs

# ============================================================
# Transforms (paper-aligned)
# ============================================================

def train_transform():
    """
    Image preprocessing and augmentation used during training.

    Images are resized to 224x224, randomly flipped horizontally,
    converted to tensors, and normalized using ImageNet statistics.

    Alignment with paper:
    The paper trains ResNet-50 models using ImageNet-pretrained weights
    and standard ImageNet preprocessing. It resizes images to 224x224 as
    required by ResNet-50, whilst random horizontal flipping
    is explicitly described as the only augmentation applied.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

def eval_transform():
    """
    Image preprocessing used during validation and testing.

    This transform mirrors the training preprocessing but omits
    stochastic augmentation to ensure deterministic evaluation.

    Alignment with paper:
    Evaluation is performed without data augmentation, consistent
    with standard practice and the paper's methodology.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

# ============================================================
# Model (binary + multiclass)
# ============================================================

class ResNet50Probe(nn.Module):
    """
    ResNet-50 probe model for attribute prediction.

    This model uses a ResNet-50 backbone pretrained on ImageNet-1K.
    The final fully connected layer is replaced to predict either:
      - a single logit (binary gender task) - Aligned with paper
      - three logits (skin-tone classification task) - Extra addition

    Alignment with paper:
    The paper explicitly trains ResNet-50 gender classifiers with a final
    linear layer mapping the 2048-dimensional feature vector to the
    target attribute. This class directly implements that architecture.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        self.backbone = models.resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        out = self.backbone(x)
        return out.squeeze(1) if out.shape[1] == 1 else out

class ConvNeXtTinyProbe(nn.Module):
    """
    ConvNeXt-Tiny probe model for architecture robustness experiments.

    This model replaces the ResNet-50 backbone with a ConvNeXt-Tiny
    architecture pretrained on ImageNet-1K, while keeping the probe
    setup identical (linear classification head).

    Alignment with paper:
    This model is NOT part of the original paper.
    It is an extension introduced to test whether the observed
    artifact signals are robust to changes in backbone architecture.
    It was implemented as it served as the primary model of choice for the
    secondary relevant paper.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        self.backbone = models.convnext_tiny(weights=weights)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        out = self.backbone(x)
        return out.squeeze(1) if out.shape[1] == 1 else out

def build_model(backbone: str, num_classes: int):
    """
    Factory method for constructing the probe model.

    This function enables selecting different backbone architectures
    (e.g., ResNet-50 or ConvNeXt-Tiny) via command-line arguments while
    keeping the remainder of the training and evaluation pipeline fixed.
    """
    if backbone == "resnet50":
        return ResNet50Probe(num_classes)
    elif backbone == "convnext_tiny":
        return ConvNeXtTinyProbe(num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

# ============================================================
# Early Stopping
# ============================================================

class EarlyStopping:
    """
    Early stopping handler with serialisable state for checkpointing.

    Monitors a scalar metric each epoch and signals when training should
    stop because no meaningful improvement has been seen for `patience`
    consecutive epochs.

    The internal state (counter, best_value, best_epoch) is exposed as
    plain Python scalars so it can be embedded directly in a checkpoint
    dict and restored on resume without pickling the object itself.

    Enhancement over the original script (which used an inline counter):
    Using a class keeps the logic self-contained and makes the
    checkpoint/resume path simpler.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value: Optional[float] = None
        self.best_epoch: int = 0
        self.early_stop: bool = False

    def __call__(self, current_value: float, epoch: int) -> bool:
        if self.best_value is None:
            self.best_value = current_value
            self.best_epoch = epoch
            return False

        improved = (
            current_value > self.best_value + self.min_delta
            if self.mode == "max"
            else current_value < self.best_value - self.min_delta
        )

        if improved:
            self.best_value = current_value
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def state_dict(self) -> dict:
        return {
            "counter": self.counter,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "early_stop": self.early_stop,
        }

    def load_state_dict(self, state: dict):
        self.counter = state["counter"]
        self.best_value = state["best_value"]
        self.best_epoch = state["best_epoch"]
        self.early_stop = state["early_stop"]

# ============================================================
# Checkpoint utilities
# ============================================================

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: List[dict],
    best_metric: float,
    best_state: Optional[dict],
    early_stopper: Optional[EarlyStopping],
    total_epochs: int,
):
    """
    Saves a mid-training checkpoint to disk.

    Stores model weights, optimiser state, current epoch, running history,
    the best-so-far model weights, and (optionally) the early-stopping
    counter so that training can be resumed exactly from where it left off.

    Enhancement:
    Adapted from DatasetClassification_gradient_accum.py to work with
    the SGD-based grid-search training loop.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "best_metric": best_metric,
        "best_state": best_state,
        "total_epochs": total_epochs,
        "early_stopper_state": early_stopper.state_dict() if early_stopper is not None else None,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    early_stopper: Optional[EarlyStopping],
    device: str,
    total_epochs: int,
) -> Tuple[int, List[dict], float, Optional[dict]]:
    """
    Restores a mid-training checkpoint.

    Returns the epoch to resume from, the running history, the best
    validation metric seen so far, and the best model state dict.

    Raises ValueError if the checkpoint was saved with a different
    total_epochs to prevent silent mismatches.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if checkpoint["total_epochs"] != total_epochs:
        raise ValueError(
            f"Checkpoint has total_epochs={checkpoint['total_epochs']} "
            f"but current run has total_epochs={total_epochs}. "
            f"Delete the checkpoint or match --epochs."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if early_stopper is not None and checkpoint["early_stopper_state"] is not None:
        early_stopper.load_state_dict(checkpoint["early_stopper_state"])

    return (
        checkpoint["epoch"],
        checkpoint["history"],
        checkpoint["best_metric"],
        checkpoint["best_state"],
    )

# ============================================================
# Metrics
# ============================================================

def macro_ovr_auc(y_true, y_scores):
    """
    Computes a single summary metric for a multi-class task by averaging
    the one-vs-rest ROC-AUC values across classes.

    Each class is treated as positive against all others, and the
    resulting AUCs are averaged equally across classes.

    Alignment with paper:
    The paper reports one-vs-rest AUC for multi-class attribute prediction
    to assess separability of each group independently.
    """
    return roc_auc_score(
        y_true,
        y_scores,
        multi_class="ovr",
        average="macro",
    )

def per_class_ovr_auc(y_true, y_scores):
    """
    Computes the individual one-vs-rest ROC-AUC values for each class
    without aggregation.

    Alignment with paper:
    Per-class analysis is used to understand uneven performance across
    attribute categories.
    """
    num_classes = y_scores.shape[1]
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    return {
        f"class_{c}_auc": roc_auc_score(y_bin[:, c], y_scores[:, c])
        for c in range(num_classes)
    }

# ============================================================
# Evaluation
# ============================================================

def evaluate(model, loader, num_classes, device, split_name: str = "Val"):
    """
    Evaluates a trained model on a validation or test set.

    For binary tasks (gender), ROC-AUC is computed using sigmoid probabilities.
    For multi-class tasks (skin-tone), macro one-vs-rest ROC-AUC and per-class
    AUCs are computed using softmax probabilities.

    Additional metrics (accuracy, precision, recall, F1) are reported
    as supplementary diagnostics.

    Alignment with paper:
    ROC-AUC is the primary evaluation metric used in the paper.
    Accuracy and F1 are not emphasised in the paper and are treated
    as auxiliary metrics for interpretability.
    """
    model.eval()
    logits_all, labels_all = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"  [{split_name}]", leave=True):
            x = x.to(device)
            logits = model(x)
            logits_all.append(logits.cpu())
            labels_all.append(y)

    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all).numpy()

    if num_classes == 1:
        probs = torch.sigmoid(logits).numpy()
        preds = (probs >= 0.5).astype(int)

        auc = roc_auc_score(labels, probs)
        acc = accuracy_score(labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )

        return {
            "metric": auc,
            "auc": auc,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "labels": labels,
            "scores": probs,
            "per_class_auc": None,
        }
    else:
        probs = torch.softmax(logits, dim=1).numpy()
        preds = probs.argmax(axis=1)

        mst3_labels = collapse_labels_to_mst3(labels)
        mst3_probs = collapse_probs_to_mst3(probs)

        mst3_auc = roc_auc_score(
            mst3_labels,
            mst3_probs,
            multi_class="ovr",
            average="macro"
        )

        macro_auc = macro_ovr_auc(labels, probs)
        acc = accuracy_score(labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        per_class = per_class_ovr_auc(labels, probs)

        return {
            "metric": macro_auc,
            "auc": macro_auc,
            "mst3_auc": mst3_auc,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "labels": labels,
            "scores": probs,
            "per_class_auc": per_class,
        }

# ============================================================
# Bootstrapping
# ============================================================

def bootstrap_auc(y_true, y_scores, n_bootstrap=5000, seed=0):
    """
    Computes bootstrapped confidence intervals for binary ROC-AUC.

    The test set is resampled with replacement 5,000 times, and the
    distribution of AUC values is used to estimate a 95% confidence interval.

    Alignment with paper:
    The paper reports 95% confidence intervals using 5,000 bootstrap
    resamples to quantify uncertainty in test performance.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = [
        roc_auc_score(y_true[idx := rng.integers(0, n, n)], y_scores[idx])
        for _ in range(n_bootstrap)
    ]
    return np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def bootstrap_multiclass_auc(y_true, y_scores, n_bootstrap=5000, seed=0):
    """
    Computes bootstrapped confidence intervals for macro one-vs-rest AUC
    in multi-class settings.

    Alignment with paper:
    This mirrors the bootstrapping procedure used for binary tasks,
    extended to multi-class macro AUC.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = [
        macro_ovr_auc(y_true[idx := rng.integers(0, n, n)], y_scores[idx])
        for _ in range(n_bootstrap)
    ]
    return np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

# ============================================================
# Training (single model, one hyperparameter combo)
# ============================================================

def train_one_model(
    train_loader,
    val_loader,
    num_classes: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    device: str,
    backbone: str,
    earlystopping: bool = False,
    patience: int = 5,
    save_mode: str = "best",
    combo_idx: Optional[int] = None,
    combo_total: Optional[int] = None,
    # ── new checkpoint params ──────────────────────────────
    checkpoint_path: Optional[Path] = None,
):
    """
    Trains a single model for a given hyperparameter configuration.

    The model is optimised using stochastic gradient descent (SGD) with
    momentum, and performance is evaluated on the validation set after
    each training epoch. The best-performing epoch (by validation AUC)
    is retained. Validation ROC-AUC is used as the primary criterion
    for model selection.

    Alignment with paper:
    - Optimizer: SGD with momentum = 0.9
    - Hyperparameters: grid search over learning rate and weight decay
    - Model selection: based on validation performance

    Extensions beyond the paper:
    - Optional early stopping (EarlyStopping class with serialisable state)
    - Epoch-level checkpoint / resume via `checkpoint_path`
    - Per-epoch history dict (epoch, train_loss, train_acc, val_auc, val_acc)
    - leave=True on tqdm so completed epoch bars stay visible

    Returns:
        model          – best (or final) model
        best_metric    – best validation AUC seen during training
        history        – list of per-epoch metric dicts
    """

    model = build_model(backbone, num_classes).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )

    criterion = (
        nn.BCEWithLogitsLoss()
        if num_classes == 1
        else nn.CrossEntropyLoss()
    )

    # ── Early stopping ─────────────────────────────────────
    early_stopper: Optional[EarlyStopping] = None
    if earlystopping:
        early_stopper = EarlyStopping(patience=patience, mode="max")

    # ── Resume from checkpoint if one exists ───────────────
    start_epoch = 0
    history: List[dict] = []
    best_metric = -1.0
    best_state: Optional[dict] = None

    if checkpoint_path is not None and checkpoint_path.exists():
        print(f"  Found checkpoint — resuming combo {combo_idx} from {checkpoint_path.name}")
        start_epoch, history, best_metric, best_state = load_checkpoint(
            checkpoint_path, model, optimizer, early_stopper, device, epochs
        )
        print(f"    Resuming from epoch {start_epoch + 1}/{epochs}, best_auc so far = {best_metric:.4f}")

    # ── Training loop ──────────────────────────────────────
    for epoch in range(start_epoch, epochs):

        print("Couldn't load images count so far:", COULDNTLOADCOUNT)

        # ---------- train ----------
        model.train()

        desc = (
            f"combo={combo_idx}/{combo_total} "
            f"lr={lr} wd={weight_decay} "
            f"epoch={epoch + 1}/{epochs}"
        )

        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in tqdm(train_loader, desc=desc, leave=True):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)

            loss = (
                criterion(logits, y.float())
                if num_classes == 1
                else criterion(logits, y)
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            if num_classes == 1:
                preds = (torch.sigmoid(logits) >= 0.5).long()
                correct += (preds == y).sum().item()
            else:
                correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

        train_loss = total_loss / total
        train_acc  = correct / total

        # ---------- validate ----------
        val_metrics = evaluate(model, val_loader, num_classes, device, split_name="Val")
        val_auc = val_metrics["auc"]
        val_acc = val_metrics["accuracy"]

        print(
            f"  [VAL] epoch={epoch + 1} "
            f"auc={val_auc:.4f} "
            f"acc={val_acc:.4f} "
            f"f1={val_metrics['f1']:.4f} "
            f"| train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f}"
        )

        # ---------- history ----------
        history.append({
            "epoch": epoch + 1,
            "lr": lr,
            "weight_decay": weight_decay,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_auc": val_auc,
            "val_acc": val_acc,
            "val_f1": val_metrics["f1"],
        })

        # ---------- best model ----------
        if val_auc > best_metric:
            best_metric = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  New best val AUC: {best_metric:.4f}")

        # ---------- checkpoint ----------
        if checkpoint_path is not None:
            save_checkpoint(
                checkpoint_path,
                model, optimizer,
                epoch + 1,          # next epoch to start from
                history,
                best_metric,
                best_state,
                early_stopper,
                epochs,
            )

        # ---------- early stopping ----------
        if early_stopper is not None:
            should_stop = early_stopper(val_auc, epoch + 1)
            if should_stop:
                print(
                    f"  Early stopping at epoch {epoch + 1} "
                    f"(best epoch was {early_stopper.best_epoch}, "
                    f"auc={early_stopper.best_value:.4f})"
                )
                break

    # ── Restore best or keep final ─────────────────────────
    if save_mode == "best" and best_state is not None:
        model.load_state_dict(best_state)

    return model, best_metric, history

# ============================================================
# Grid Search
# ============================================================

def grid_search(
    train_loader,
    val_loader,
    num_classes: int,
    epochs: int,
    lrs: List[float],
    wds: List[float],
    device: str,
    backbone: str,
    earlystopping: bool,
    patience: int,
    save_mode: str,
    seed: int,
    out_dir: Path,
):
    """
    Performs grid search over learning rate and weight decay.

    Each hyperparameter combination is trained independently, and the
    model achieving the highest validation AUC is selected as the best.

    Resume logic:
    A JSON file `grid_state_seed{seed}.json` in `out_dir` records which
    combinations have already completed (with their best validation AUC).
    On resume, completed combinations are skipped and their saved result
    is used directly, so only the remaining combinations need to be trained.

    Alignment with paper:
    The paper performs hyperparameter tuning via grid search over
    learning rate and weight decay using values
    [1e-2, 1e-3, 1e-4, 1e-5] for both lr & wd over the validation set.

    Extensions:
    - CLI-configurable grids
    - Checkpoint / resume per combo (epoch-level)
    - Grid-state JSON for combo-level resume
    """

    combos = [(lr, wd) for lr in lrs for wd in wds]
    grid_state_path = out_dir / f"grid_state_seed{seed}.json"
    print(grid_state_path)

    # Load any previously completed combos
    completed: Dict[str, dict] = {}
    if grid_state_path.exists():
        with open(grid_state_path) as f:
            completed = json.load(f)
        print(f"  Loaded grid state: {len(completed)}/{len(combos)} combos already done")

    best_metric = -1.0
    best_model   = None
    best_cfg     = None
    best_history: List[dict] = []

    for idx, (lr, wd) in enumerate(combos, start=1):
        combo_key = f"lr{lr}_wd{wd}"

        # ── Skip if already done ───────────────────────────
        if combo_key in completed:
            val_score = completed[combo_key]["val_auc"]
            print(
                f"  [SKIP] combo={idx}/{len(combos)} lr={lr} wd={wd} "
                f"(already done, val_auc={val_score:.4f})"
            )
            # We don't have the model object in memory for skipped combos;
            # track the best score so final best_cfg reflects reality.
            if val_score > best_metric:
                best_metric = val_score
                best_cfg    = (lr, wd)
                # best_model stays None for skipped combos — handled below.
            continue

        print(f"\n  {'─'*60}")
        print(f"  combo={idx}/{len(combos)} | lr={lr} | wd={wd} | seed={seed}")
        print(f"  {'─'*60}")

        checkpoint_path = out_dir / f"checkpoint_seed{seed}_combo{idx}.pt"

        model, val_score, history = train_one_model(
            train_loader,
            val_loader,
            num_classes,
            lr,
            wd,
            epochs,
            device,
            backbone,
            earlystopping,
            patience,
            save_mode,
            combo_idx=idx,
            combo_total=len(combos),
            checkpoint_path=checkpoint_path,
        )

        # ── Save per-combo best model weights ─────────────
        # Saved immediately so the model is never lost if the run is
        # interrupted after this combo but before a later combo finishes.
        combo_model_path = out_dir / f"model_seed{seed}_combo{idx}_best.pt"
        torch.save(model.state_dict(), combo_model_path)
        print(f"  Combo {idx} best model saved → {combo_model_path.name}")

        # ── Save combo-level history ───────────────────────
        combo_history_path = out_dir / f"history_seed{seed}_combo{idx}.json"
        with open(combo_history_path, "w") as f:
            json.dump(history, f, indent=2)

        # ── Mark combo as done in grid state ──────────────
        # Store the model path so the resume path can reload it directly.
        completed[combo_key] = {
            "combo_idx": idx,
            "lr": lr,
            "wd": wd,
            "val_auc": val_score,
            "model_path": str(combo_model_path),
        }
        with open(grid_state_path, "w") as f:
            json.dump(completed, f, indent=2)
        print(f"  Combo {idx} complete — val_auc={val_score:.4f} — state saved")

        # ── Delete epoch-level checkpoint (no longer needed) ──
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        if val_score > best_metric:
            best_metric  = val_score
            best_model   = model
            best_cfg     = (lr, wd)
            best_history = history

    # ── If best combo was skipped (already done on a prior run), reload it ──
    # This happens when ALL remaining combos were skipped, meaning best_model
    # was never assigned in this run. We look up the saved model path from
    # the grid state JSON and reload it from disk.
    if best_model is None and best_cfg is not None:
        lr_best, wd_best = best_cfg
        best_combo_idx = next(
            i for i, (lr, wd) in enumerate(combos, 1)
            if lr == lr_best and wd == wd_best
        )
        combo_key  = f"lr{lr_best}_wd{wd_best}"
        entry      = completed[combo_key]

        # Prefer the explicit path stored in the grid state; fall back to
        # the expected filename pattern for backwards compatibility.
        combo_model_path = Path(entry.get(
            "model_path",
            str(out_dir / f"model_seed{seed}_combo{best_combo_idx}_best.pt")
        ))
        history_pt = out_dir / f"history_seed{seed}_combo{best_combo_idx}.json"

        if combo_model_path.exists():
            print(
                f"  ↩ Best combo was already done — "
                f"reloading model from {combo_model_path.name}"
            )
            best_model = build_model(backbone, num_classes).to(device)
            best_model.load_state_dict(
                torch.load(combo_model_path, map_location=device, weights_only=True)
            )
        else:
            raise RuntimeError(
                f"Best combo {combo_key} is marked as done in grid state but "
                f"its model file '{combo_model_path}' was not found.\n"
                f"Delete grid_state_seed{seed}.json and rerun to retrain it."
            )

        if history_pt.exists():
            with open(history_pt) as f:
                best_history = json.load(f)

    # Save best model weights separately for easy reloading
    best_model_pt = out_dir / f"model_seed{seed}_best.pt"
    torch.save(best_model.state_dict(), best_model_pt)
    print(f"  Best model weights saved → {best_model_pt.name}")

    return best_model, best_cfg, best_history

# ============================================================
# JSON loader
# ============================================================

def load_splits(json_path):
    """
    Loads train/validation/test splits from a JSON file.

    Each split consists of a list of dicts with "image" and "label" keys.
    """
    with open(json_path) as f:
        data = json.load(f)

    def parse(split):
        return [(x["image"], x["label"]) for x in data[split]]

    return parse("train"), parse("val"), parse("test")

# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--splits_json",  required=True)
    p.add_argument("--task",         choices=["gender", "skintone"], required=True)
    p.add_argument("--num_classes",  type=int) # Used only for skintone task; inferred from task for others
    p.add_argument("--epochs",       type=int,   default=15)           # paper default
    p.add_argument("--batch_size",   type=int,   default=64)           # paper default
    p.add_argument("--seeds",        type=int,   nargs="+", default=[0])
    p.add_argument("--bootstrap",    action="store_true")
    p.add_argument("--lrs",          type=float, nargs="+", default=[1e-2, 1e-3, 1e-4, 1e-5])  # paper default
    p.add_argument("--wds",          type=float, nargs="+", default=[1e-2, 1e-3, 1e-4, 1e-5])  # paper default
    p.add_argument("--out_dir",      default="outputs")
    p.add_argument("--backbone",     choices=["resnet50", "convnext_tiny"], default="resnet50")
    p.add_argument("--earlystopping", action="store_true")
    p.add_argument("--patience",     type=int, default=5)
    p.add_argument("--save_mode",    choices=["best", "final"], default="best",
                   help="Save best validation epoch or final epoch model")
    return p.parse_args()

# ============================================================
# Main
# ============================================================

def main():
    """
    Main experiment driver.

    For each random seed:
    1. Run hyperparameter grid search with checkpoint / resume support
    2. Select the best model via validation AUC
    3. Evaluate exactly once on the held-out test set
    4. Compute confidence intervals (optional)
    5. Write per-seed outputs:
         - metrics_seed_{seed}.json
         - history_seed_{seed}.json  (best combo's epoch history)
         - model_seed_{seed}.pt      (full save dict: weights + hyperparams + history)
    6. Append a row to results_{task}.csv (unchanged from original)

    After all seeds:
    7. Write summary.json with mean ± std across seeds for AUC, accuracy, F1.

    Alignment with paper:
    This structure mirrors the paper's experimental protocol:
    validation-based model selection, test-only reporting, and
    uncertainty estimation via bootstrapping.

    Enhancement:
    All intermediate state (epoch checkpoints, grid state JSON) is written
    to `out_dir` so that a crashed or interrupted run can be resumed by
    simply re-running the same command.
    """

    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_classes = 1 if args.task in ("gender") else args.num_classes

    train_s, val_s, test_s = load_splits(args.splits_json)

    # Pre-build loaders once (workers shared across seeds / combos)
    train_loader = DataLoader(
        ImageLabelDataset(train_s, train_transform()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        ImageLabelDataset(val_s, eval_transform()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        ImageLabelDataset(test_s, eval_transform()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=False,
    )

    # Pre-warm val_loader workers so epoch 1 → epoch 2 transition has no delay
    print("Pre-warming val_loader workers...")
    for _ in val_loader:
        break

    # ── CSV (results file, same structure as original) ──────
    csv_path   = out_dir / f"results_{args.task}.csv"
    fieldnames = [
        "task", "backbone", "seed", "lr", "weight_decay",
        "test_auc", "test_accuracy", "test_f1",
        "ci_low", "ci_high"
    ]

    if num_classes > 2:
        fieldnames += [f"class_{i}_auc" for i in range(num_classes)]
        fieldnames += ["mst3_auc"]

    # Open in append mode so resumed runs add rows rather than overwrite
    csv_existed = csv_path.exists()
    csv_file    = open(csv_path, "a", newline="")
    writer      = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not csv_existed:
        writer.writeheader()

    # ── Track per-seed summary info ──────────────────────────
    all_results: List[dict] = []

    print(f"\n{'='*70}")
    print(f"Task: {args.task} | Backbone: {args.backbone} | Device: {device}")
    print(f"Seeds: {args.seeds} | Epochs: {args.epochs} | Batch: {args.batch_size}")
    print(f"LR grid: {args.lrs}")
    print(f"WD grid: {args.wds}")
    print(f"{'='*70}")

    for seed in args.seeds:
        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print(f"{'='*70}")

        set_seed(seed)

        # ── Grid search (with checkpoint / resume) ───────────
        best_model, best_cfg, best_history = grid_search(
            train_loader,
            val_loader,
            num_classes,
            args.epochs,
            args.lrs,
            args.wds,
            device,
            args.backbone,
            args.earlystopping,
            args.patience,
            args.save_mode,
            seed=seed,
            out_dir=out_dir,
        )

        best_lr, best_wd = best_cfg
        print(f"\nBest config — lr={best_lr} | wd={best_wd}")

        # ── Test evaluation (once, on held-out set) ──────────
        test_metrics = evaluate(best_model, test_loader, num_classes, device, split_name="Test")

        print(
            f"  [TEST] auc={test_metrics['auc']:.4f} "
            f"acc={test_metrics['accuracy']:.4f} "
            f"f1={test_metrics['f1']:.4f}"
        )

        # ── Confidence intervals ─────────────────────────────
        ci_low = ci_high = None
        if args.bootstrap:
            if num_classes == 1:
                _, ci_low, ci_high = bootstrap_auc(
                    test_metrics["labels"],
                    test_metrics["scores"],
                    seed=seed,
                )
            else:
                _, ci_low, ci_high = bootstrap_multiclass_auc(
                    test_metrics["labels"],
                    test_metrics["scores"],
                    seed=seed,
                )
            print(f"  [BOOTSTRAP] 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

        # ── Per-seed outputs ─────────────────────────────────

        # 1. metrics_seed_{seed}.json
        metrics_dict = {
            "seed":          seed,
            "backbone":      args.backbone,
            "best_lr":       best_lr,
            "best_wd":       best_wd,
            "test_auc":      test_metrics["auc"],
            "test_accuracy": test_metrics["accuracy"],
            "test_f1":       test_metrics["f1"],
            "test_precision":test_metrics["precision"],
            "test_recall":   test_metrics["recall"],
            "ci_low":        ci_low,
            "ci_high":       ci_high,
            "num_epochs_run": len(best_history),
            "best_val_auc":  max(h["val_auc"] for h in best_history) if best_history else None,
            "per_class_auc": test_metrics["per_class_auc"],
        }
        metrics_path = out_dir / f"metrics_seed{seed}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"  Saved {metrics_path.name}")

        # 2. history_seed_{seed}.json  (best combo's epoch history)
        history_path = out_dir / f"history_seed{seed}.json"
        with open(history_path, "w") as f:
            json.dump(best_history, f, indent=2)
        print(f"  Saved {history_path.name}")

        # 3. model_seed_{seed}.pt  (full save dict)
        model_path = out_dir / f"model_seed{seed}.pt"
        torch.save(
            {
                "seed":             seed,
                "model_state_dict": best_model.state_dict(),
                "hyperparameters": {
                    "task":         args.task,
                    "backbone":     args.backbone,
                    "lr":           best_lr,
                    "weight_decay": best_wd,
                    "epochs":       args.epochs,
                    "batch_size":   args.batch_size,
                },
                "test_auc":      test_metrics["auc"],
                "test_accuracy": test_metrics["accuracy"],
                "test_f1":       test_metrics["f1"],
                "ci_low":        ci_low,
                "ci_high":       ci_high,
                "history":       best_history,
            },
            model_path,
        )
        print(f"  Saved {model_path.name}")

        # 4. CSV row (original behaviour preserved)
        row = {
            "task":          args.task,
            "backbone":      args.backbone,
            "seed":          seed,
            "lr":            best_lr,
            "weight_decay":  best_wd,
            "test_auc":      test_metrics["auc"],
            "test_accuracy": test_metrics["accuracy"],
            "test_f1":       test_metrics["f1"],
            "ci_low":        ci_low,
            "ci_high":       ci_high,
            **{f"class_{c}_auc": None for c in range(num_classes)},
            "mst3_auc":      test_metrics.get("mst3_auc"),
        }

        if num_classes > 1 and test_metrics["per_class_auc"] is not None:
            for k, v in test_metrics["per_class_auc"].items():
                row[k.replace("class_", "class_")] = v   # keys already match fieldnames
        writer.writerow(row)
        csv_file.flush()

        all_results.append(metrics_dict)

    csv_file.close()
    print(f"\nCSV results saved → {csv_path}")

    # ── Summary JSON (cross-seed aggregation) ─────────────
    test_aucs = [r["test_auc"]      for r in all_results]
    test_accs = [r["test_accuracy"] for r in all_results]
    test_f1s  = [r["test_f1"]       for r in all_results]

    summary = {
        "task":          args.task,
        "backbone":      args.backbone,
        "seeds":         args.seeds,
        "test_aucs":     test_aucs,
        "test_accuracies": test_accs,
        "test_f1s":      test_f1s,
        "test_auc_mean": float(np.mean(test_aucs)),
        "test_auc_std":  float(np.std(test_aucs, ddof=1)) if len(test_aucs) > 1 else 0.0,
        "test_acc_mean": float(np.mean(test_accs)),
        "test_acc_std":  float(np.std(test_accs, ddof=1)) if len(test_accs) > 1 else 0.0,
        "test_f1_mean":  float(np.mean(test_f1s)),
        "test_f1_std":   float(np.std(test_f1s,  ddof=1)) if len(test_f1s)  > 1 else 0.0,
        "best_cfgs":     [{"seed": r["seed"], "lr": r["best_lr"], "wd": r["best_wd"]} for r in all_results],
    }

    # Include per-seed CI if bootstrapping was used
    if args.bootstrap:
        summary["ci_low"]  = [r["ci_low"]  for r in all_results]
        summary["ci_high"] = [r["ci_high"] for r in all_results]

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print("Experiment complete!")
    print(f"  Test AUC:  {summary['test_auc_mean']:.4f} ± {summary['test_auc_std']:.4f}")
    print(f"  Test Acc:  {summary['test_acc_mean']:.4f} ± {summary['test_acc_std']:.4f}")
    print(f"  Test F1:   {summary['test_f1_mean']:.4f}  ± {summary['test_f1_std']:.4f}")
    print(f"  Summary  → {summary_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()