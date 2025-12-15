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

Taken together, this codebase enables controlled, reproducible
measurement of attribute predictability under varying architectural
and training conditions, supporting both direct replication of the
paper's results and principled extensions to new datasets and
attributes.
"""

import argparse
import json
import random
from pathlib import Path
import csv
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
from PIL import Image
from tqdm import tqdm

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
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label

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
# Training
# ============================================================

def train_one_model(
    train_loader,
    val_loader,
    num_classes,
    lr,
    weight_decay,
    epochs,
    device,
    backbone,
    earlystopping=False,
    patience=5,
    save_mode="best",
    combo_idx=None,
    combo_total=None,
):
    """
    Trains a single model for a given hyperparameter configuration.

    The model is optimized using stochastic gradient descent (SGD) with
    momentum, and performance is evaluated on the validation set after
    each training epoch. The best-performing epoch (by validation AUC)
    can be retained. Validation ROC-AUC is used as the primary
    criterion for model selection.

    Alignment with paper:
    - Optimizer: SGD with momentum = 0.9
    - Hyperparameters: grid search over learning rate and weight decay
    - Model selection: based on validation performance

    Extensions beyond the paper:
    - Optional early stopping to reduce unnecessary training
    - Ability to restore the best validation epoch based on highest validation AUC instead of the final epoch
    - Progress reporting including hyperparameter combination index
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

    best_metric = -1.0
    best_state = None
    patience_ctr = 0

    for epoch in range(epochs):
        model.train()

        desc = (
            f"combination={combo_idx}/{combo_total} "
            f"lr={lr}, wd={weight_decay}, epoch={epoch+1}/{epochs}"
        )

        for x, y in tqdm(train_loader, desc=desc, leave=False):
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

        # ----------------------------
        # Validation evaluation
        # ----------------------------
        val_metrics = evaluate(model, val_loader, num_classes, device)
        val_score = val_metrics["metric"]

        if val_score > best_metric:
            best_metric = val_score
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            patience_ctr = 0
        else:
            patience_ctr += 1

        if earlystopping and patience_ctr >= patience:
            break

    # ----------------------------
    # Restore best or final model
    # ----------------------------
    if save_mode == "best" and best_state is not None:
        model.load_state_dict(best_state)

    return model, best_metric

# ============================================================
# Metrics
# ============================================================

def macro_ovr_auc(y_true, y_scores):
    """
    Computes a single summary metric for a multi-class task by averaging the one-vs-rest ROC-AUC values across classes.

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
    Computes the individual one-vs-rest ROC-AUC values for each class without aggregation.

    This function returns individual AUC values for each class, allowing
    analysis of which classes are more or less separable.

    Alignment with paper:
    Per-class analysis is used to understand uneven performance across
    attribute categories.
    """
    num_classes = y_scores.shape[1]
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))

    aucs = {}
    for c in range(num_classes):
        aucs[f"class_{c}_auc"] = roc_auc_score(
            y_bin[:, c],
            y_scores[:, c]
        )
    return aucs

# ============================================================
# Evaluation
# ============================================================

def evaluate(model, loader, num_classes, device):
    """
    Evaluates a trained model on a validation or test set.

    For binary tasks (gender), ROC-AUC is computed using sigmoid probabilities.
    For multi-class tasks (skin-tone), macro one-vs-rest ROC-AUC and per-class AUCs
    are computed using softmax probabilities.

    Additional metrics (accuracy, precision, recall, F1) are reported
    as supplementary diagnostics.

    Alignment with paper:
    ROC-AUC is the primary evaluation metric used in the paper.
    Accuracy and F1 are not emphasized in the paper and are treated
    as auxiliary metrics for interpretability.
    """

    model.eval()
    logits_all, labels_all = [], []

    with torch.no_grad():
        for x, y in loader:
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

        macro_auc = macro_ovr_auc(labels, probs)
        acc = accuracy_score(labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )

        per_class = per_class_ovr_auc(labels, probs)

        return {
            "metric": macro_auc,
            "auc": macro_auc,
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
    aucs = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        aucs.append(roc_auc_score(y_true[idx], y_scores[idx]))

    return (
        np.mean(aucs),
        np.percentile(aucs, 2.5),
        np.percentile(aucs, 97.5),
    )

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
    aucs = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        aucs.append(
            macro_ovr_auc(y_true[idx], y_scores[idx])
        )

    return (
        np.mean(aucs),
        np.percentile(aucs, 2.5),
        np.percentile(aucs, 97.5),
    )

# ============================================================
# Grid Search
# ============================================================

def grid_search(
    train_loader,
    val_loader,
    num_classes,
    epochs,
    lrs,
    wds,
    device,
    backbone,
    earlystopping,
    patience,
    save_mode,
):
    """
    Performs grid search over learning rate and weight decay.

    Each hyperparameter combination is trained independently, and the
    model achieving the highest validation AUC is selected.

    Alignment with paper:
    The paper performs hyperparameter tuning via grid search over
    learning rate and weight decay using this set of values 
    [1e-2, 1e-3, 1e-4, 1e-5] for both lr & wd over the validation set.

    Extensions:
    - CLI-configurable grids
    - Early stopping support
    - Explicit progress reporting of grid combinations
    """

    combos = [(lr, wd) for lr in lrs for wd in wds]
    best_metric = -1
    best_model = None
    best_cfg = None

    for idx, (lr, wd) in enumerate(combos, start=1):
        model, val_score = train_one_model(
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
        )

        if val_score > best_metric:
            best_metric = val_score
            best_model = model
            best_cfg = (lr, wd)

    return best_model, best_cfg


# ============================================================
# JSON loader
# ============================================================

def load_splits(json_path):
    """
    Loads train/validation/test splits from a JSON file.

    Each split consists of a list of image paths and corresponding labels.
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
    p.add_argument("--splits_json", required=True)
    p.add_argument("--task", choices=["gender", "skintone"], required=True)
    p.add_argument("--epochs", type=int, default=15) # Default as per the paper
    p.add_argument("--batch_size", type=int, default=64) # Default as per the paper
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--lrs", type=float, nargs="+", default=[1e-2, 1e-3, 1e-4, 1e-5], help="Learning rates for grid search") # Default as per the paper
    p.add_argument("--wds", type=float, nargs="+", default=[1e-2, 1e-3, 1e-4, 1e-5], help="Weight decay values for grid search") # Default as per the paper
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--backbone", choices=["resnet50", "convnext_tiny"], default="resnet50")
    p.add_argument("--earlystopping", action="store_true")
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--save_mode", choices=["best", "final"], default="best", help="Save best validation epoch or final epoch model")
    return p.parse_args()

# ============================================================
# Main
# ============================================================

def main():
    """
    Main experiment driver.

    For each random seed:
    - perform hyperparameter grid search
    - select the best model via validation AUC
    - evaluate once on the held-out test set
    - compute confidence intervals (optional)
    - log results to CSV

    Alignment with paper:
    This structure mirrors the paper's experimental protocol:
    validation-based model selection, test-only reporting, and
    uncertainty estimation via bootstrapping.
    """

    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.out_dir).mkdir(exist_ok=True)

    num_classes = 1 if args.task == "gender" else 3

    train_s, val_s, test_s = load_splits(args.splits_json)

    train_loader = DataLoader(
        ImageLabelDataset(train_s, train_transform()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0, #4,
    )
    val_loader = DataLoader(
        ImageLabelDataset(val_s, eval_transform()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0, #4,
    )
    test_loader = DataLoader(
        ImageLabelDataset(test_s, eval_transform()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0, #4,
    )

    csv_path = Path(args.out_dir) / f"results_{args.task}.csv"

    fieldnames = [
        "task", "backbone", "seed", "lr", "weight_decay",
        "test_auc", "test_accuracy", "test_f1",
        "ci_low", "ci_high",
        "class_0_auc", "class_1_auc", "class_2_auc",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for seed in args.seeds:
            set_seed(seed)
           
            best_model, best_cfg = grid_search(
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
            )

            test_metrics = evaluate(
                best_model, test_loader, num_classes, device
            )

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

            torch.save(
                best_model.state_dict(),
                Path(args.out_dir) / f"{args.task}_best_seed_{seed}.pt"
            )

            row = {
                "task": args.task,
                "backbone": args.backbone,
                "seed": seed,
                "lr": best_cfg[0],
                "weight_decay": best_cfg[1],
                "test_auc": test_metrics["auc"],
                "test_accuracy": test_metrics["accuracy"],
                "test_f1": test_metrics["f1"],
                "ci_low": ci_low,
                "ci_high": ci_high,
                "class_0_auc": None,
                "class_1_auc": None,
                "class_2_auc": None,
            }

            # Populate for multi-class task
            if num_classes > 1 and test_metrics["per_class_auc"] is not None:
                for k, v in test_metrics["per_class_auc"].items():
                    row[k] = v

            writer.writerow(row)

    print(f"Saved results to {csv_path}")

if __name__ == "__main__":
    main()

# Example CLI:
# python ResNet-50_Convnext-Tiny_Classification.py --splits_json splits_gender.json --task gender --seeds 0 1 2 3 4 --bootstrap --backbone resnet50 --batch_size 64 --earlystopping --patience 5 --save_mode best --out_dir outputs_ResNet50_gender
# python ResNet-50_Convnext-Tiny_Classification.py --splits_json splits_gender.json --task gender --seeds 0 1 2 3 4 --bootstrap --backbone convnext_tiny --batch_size 16 --earlystopping --patience 5 --save_mode best --out_dir outputs_Convnext_Tiny_gender

# python ResNet-50_Convnext-Tiny_Classification.py --splits_json splits_skintone.json --task skintone --seeds 0 1 2 3 4 --bootstrap --backbone resnet50 --batch_size 64 --earlystopping --patience 5 --save_mode best --out_dir outputs_ResNet50_skinTone
# python ResNet-50_Convnext-Tiny_Classification.py --splits_json splits_skintone.json --task skintone --seeds 0 1 2 3 4 --bootstrap --backbone convnext_tiny --batch_size 16 --earlystopping --patience 5 --save_mode best --out_dir outputs_Convnext_Tiny_skinTone