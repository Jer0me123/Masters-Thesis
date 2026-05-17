"""
This script implements a paper-aligned Multi-Layer Perceptron (MLP) probe
used to evaluate whether human pose information alone is sufficient to
predict demographic attributes, such as gender and skin tone.

In contrast to image-based probes (e.g., ResNet-50), this model operates
exclusively on pre-extracted numeric features (e.g., body keypoints),
thereby removing all pixel-level appearance cues. This allows the model
to isolate and measure the contribution of pose-related artifacts.

The experimental protocol mirrors the paper exactly:
- Fixed train / test splits
- No hyperparameter tuning (by default)
- Fixed number of training iterations (default: 100)
- ROC-AUC as the primary metric
- Confidence intervals estimated via bootstrapping
- Repeated runs across multiple random seeds

Extensions beyond the paper include:
- Automatic task detection from JSON metadata
- Support for multi-class skin-tone prediction
- Explicit saving of trained models
- Additional diagnostic metrics (accuracy, F1)
- Optional early stopping and hyperparameter tuning
- Progress bars for training visibility

Overall, this script provides a controlled, low-capacity probing setup
designed to test *artifact predictability*, not semantic understanding.
"""


import argparse
import json
from pathlib import Path
import csv
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize, StandardScaler

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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# ============================================================
# Data loading with auto-detection
# ============================================================

def load_split(split):
    """
    Loads features and labels from a split.

    Features are assumed to be:
    - Pose keypoints only
    - Flattened numeric vectors (e.g., 34-dim for COCO)

    This directly matches the paper's pose-only setting,
    where no pixel information is used.
    """
    X = np.array([x["features"] for x in split], dtype=np.float32)
    y = np.array([x["label"] for x in split], dtype=np.int64)
    return X, y

def detect_task_and_classes(data):
    """
    Automatically detects task name and number of classes from JSON metadata.
    
    Priority order:
    1. If 'task' field exists in JSON → use it
    2. If 'num_classes' field exists in JSON → use it
    3. Otherwise, infer from unique labels in train split
    
    Returns:
        task (str): Task name (e.g., "gender", "skintone", "unknown")
        num_classes (int): 1 for binary, >1 for multi-class
        is_binary (bool): True if binary classification
    """
    # Try to get task name from metadata
    task = data.get("task", None)
    
    # Try to get num_classes from metadata
    num_classes_meta = data.get("num_classes", None)
    
    # Infer from labels in train split
    train_labels = [x["label"] for x in data["train"]]
    unique_labels = len(set(train_labels))
    
    # Determine num_classes
    if num_classes_meta is not None:
        num_classes = num_classes_meta
    else:
        # For binary tasks, use 1 output (BCE loss)
        # For multi-class, use N outputs (CE loss)
        num_classes = 1 if unique_labels == 2 else unique_labels
    
    # Determine task name if not provided
    if task is None:
        if unique_labels == 2:
            task = "binary_classification"
        elif unique_labels == 3:
            task = "skintone"  # Assume 3-class is skin-tone
        else:
            task = f"{unique_labels}_class_classification"
    
    is_binary = (unique_labels == 2)
    
    return task, num_classes, is_binary, unique_labels

def load_dataset(json_path):
    """
    Loads train / val / test splits and automatically detects task configuration.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test: Feature and label arrays
        task (str): Task name
        num_classes (int): Number of output neurons (1 for binary, N for multi-class)
        is_binary (bool): Whether this is a binary classification task
        unique_labels (int): Actual number of unique labels in the data
        
    Note:
    - The validation split is intentionally unused by default. The paper does 
      not perform model selection or hyperparameter tuning for the MLP probe, 
      as the goal is not to optimize performance but to measure whether pose 
      alone contains predictive signal under a fixed-capacity classifier.
    - When --patience is specified, validation is used for early stopping.
    """
    with open(json_path) as f:
        data = json.load(f)

    X_train, y_train = load_split(data["train"])
    X_val, y_val     = load_split(data["val"])
    X_test, y_test   = load_split(data["test"])
    
    task, num_classes, is_binary, unique_labels = detect_task_and_classes(data)

    return X_train, y_train, X_val, y_val, X_test, y_test, task, num_classes, is_binary, unique_labels

# ============================================================
# Metrics
# ============================================================

def binary_metrics(y_true, scores):
    """
    Computes binary classification metrics.

    The paper primarily reports AUC; accuracy and F1
    are logged for completeness and comparison with
    other tasks in your pipeline.
    """

    preds = (scores >= 0.5).astype(int)

    auc = roc_auc_score(y_true, scores)
    acc = accuracy_score(y_true, preds)
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0
    )

    return auc, acc, f1

def multiclass_metrics(y_true, probs, num_classes):
    """
    For multi-class skin-tone prediction, ROC-AUC is computed in a one-vs-rest
    fashion. This is necessary because ROC-AUC is inherently a binary metric.
    The macro-average ensures that each skin-tone category contributes
    equally, regardless of class imbalance.
    """

    preds = probs.argmax(axis=1)

    acc = accuracy_score(y_true, preds)
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="macro", zero_division=0
    )

    # Handle case where not all classes are present
    if len(np.unique(y_true)) < num_classes:
        return np.nan, acc, f1, None

    macro_auc = roc_auc_score(
        y_true, probs, multi_class="ovr", average="macro"
    )

    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    per_class_auc = {
        f"class_{i}_auc": roc_auc_score(y_bin[:, i], probs[:, i])
        for i in range(num_classes)
    }

    return macro_auc, acc, f1, per_class_auc

# ============================================================
# Bootstrapping
# ============================================================

def bootstrap_auc_binary(y_true, scores, n=5000, seed=0):
    """
    Bootstrap estimation of AUC confidence intervals.

    The ± value reported in the paper corresponds to half the width of this
    95% confidence interval (i.e., (upper - lower) / 2).
    """
    rng = np.random.default_rng(seed)
    aucs = []

    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        aucs.append(roc_auc_score(y_true[idx], scores[idx]))

    return (
        np.mean(aucs),
        np.percentile(aucs, 2.5),
        np.percentile(aucs, 97.5),
    )

def bootstrap_auc_multiclass(y_true, probs, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    aucs = []

    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        aucs.append(
            roc_auc_score(
                y_true[idx],
                probs[idx],
                multi_class="ovr",
                average="macro",
            )
        )

    return (
        np.mean(aucs),
        np.percentile(aucs, 2.5),
        np.percentile(aucs, 97.5),
    )

# ============================================================
# MLP model (paper-aligned, extended)
# ============================================================

class ProbeMLP(nn.Module):
    """
    Lightweight MLP probe used for artifact analysis.

    The intentionally small capacity of this network is critical. A deeper
    or wider model could potentially learn semantic correlations unrelated
    to pose artifacts, undermining the probing objective. The MLP is therefore
    designed to be expressive enough to detect linear and mildly nonlinear
    relationships, but insufficient for high-level reasoning.

    Alignment with paper:
    - Used for pose-based gender prediction
    - Low-capacity network to avoid learning semantics

    Extension:
    - Supports multi-class outputs for skin-tone prediction
    """

    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        out = self.net(x)
        return out.squeeze(1) if out.shape[1] == 1 else out

# ============================================================
# Training / Evaluation
# ============================================================

def train_and_eval_mlp(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    num_classes,
    is_binary,
    seed,
    device,
    num_epochs=100,
    batch_size=None,
    lr=1e-3,
    patience=None,
):
    """
    Training is performed for a fixed number of epochs with no early stopping
    by default (paper-aligned). Optional early stopping can be enabled via
    the patience parameter.
    
    When patience is None (default), this strictly follows the paper's
    protocol and ensures that performance differences cannot be attributed
    to tuning or optimization strategies.
    
    When patience is set, the model will use validation-based early stopping,
    which is useful for hyperparameter exploration but deviates from the
    paper's fixed-iteration protocol.
    """

    set_seed(seed)

    model = ProbeMLP(
        input_dim=X_train.shape[1],
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = (
        nn.BCEWithLogitsLoss()
        if is_binary
        else nn.CrossEntropyLoss()
    )

    Xtr = torch.tensor(X_train, device=device)
    ytr = torch.tensor(y_train, device=device)
    Xva = torch.tensor(X_val, device=device)
    yva = torch.tensor(y_val, device=device)
    Xte = torch.tensor(X_test, device=device)

    # Default to full-batch training (paper-aligned)
    if batch_size is None:
        batch_size = len(Xtr)

    # Early stopping tracking
    best_val_auc = -1.0
    best_state = None
    patience_ctr = 0

    # --------------------------------------------------------
    # Training: fixed iterations (paper-aligned by default)
    # --------------------------------------------------------

    model.train()
    epoch_bar = tqdm(range(1, num_epochs + 1), desc="Epochs")

    for epoch in epoch_bar:
        # Create batches (if batch_size < len(Xtr), otherwise single batch)
        n_samples = len(Xtr)
        
        # For full-batch training, no shuffling needed (paper-aligned)
        # For mini-batch training, use deterministic shuffling
        if batch_size >= n_samples:
            indices = torch.arange(n_samples)
        else:
            generator = torch.Generator().manual_seed(seed + epoch)
            indices = torch.randperm(n_samples, generator=generator)
        
        batch_bar = tqdm(
            range(0, n_samples, batch_size),
            desc=f"Train (epoch {epoch})",
            leave=False,
        )

        for i in batch_bar:
            idx = indices[i:i + batch_size]
            logits = model(Xtr[idx])

            loss = (
                criterion(logits, ytr[idx].float())
                if is_binary
                else criterion(logits, ytr[idx])
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        # ---------------- Validation (only if patience is set) ----------------
        if patience is not None:
            model.eval()
            with torch.no_grad():
                val_logits = model(Xva)

            yva_np = yva.cpu().numpy()

            if is_binary:
                scores = torch.sigmoid(val_logits).cpu().numpy()
                val_auc = roc_auc_score(yva_np, scores)
            else:
                probs = torch.softmax(val_logits, dim=1).cpu().numpy()
                if len(np.unique(yva_np)) < num_classes:
                    val_auc = np.nan
                else:
                    val_auc = roc_auc_score(
                        yva_np, probs, multi_class="ovr", average="macro"
                    )

            epoch_bar.set_postfix(
                val_auc="nan" if np.isnan(val_auc) else f"{val_auc:.4f}"
            )

            if not np.isnan(val_auc) and val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
                tqdm.write(f"✓ New best model (val AUC = {val_auc:.4f})")
            else:
                patience_ctr += 1
                tqdm.write(
                    "✗ No improvement "
                    + ("(val AUC undefined)" if np.isnan(val_auc)
                       else f"(patience {patience_ctr}/{patience})")
                )

            if patience_ctr >= patience:
                tqdm.write("⏹ Early stopping triggered")
                break

            model.train()

    # Restore best model if early stopping was used
    # if patience is not None and best_state is not None:
    #     model.load_state_dict(best_state)
    #     tqdm.write("✓ Restored best validation model")

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------
    
    """
    Evaluation is performed exactly once on the held-out test set. The test
    data is never used for training decisions, preserving a clean separation
    between probing and evaluation.
    """

    tqdm.write("▶ Evaluating on test set")
    model.eval()
    with torch.no_grad():
        logits = model(Xte)

    if is_binary:
        scores = torch.sigmoid(logits).cpu().numpy()
        auc, acc, f1 = binary_metrics(y_test, scores)
        return model, scores, auc, acc, f1, None

    else:
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        auc, acc, f1, per_class = multiclass_metrics(
            y_test, probs, num_classes
        )
        return model, probs, auc, acc, f1, per_class

# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train MLP probe with automatic task detection from JSON"
    )
    p.add_argument("--splits_json", required=True,
                   help="Path to JSON file containing train/val/test splits")
    p.add_argument("--seeds", type=int, nargs="+", default=[0],
                   help="Random seeds for multiple runs (default: [0])")
    p.add_argument("--bootstrap", action="store_true",
                   help="Compute bootstrap confidence intervals")
    p.add_argument("--out_dir", default="outputs",
                   help="Output directory for results and models")
    
    # Hyperparameters (optional, defaults are paper-aligned)
    p.add_argument("--epochs", type=int, default=100,
                   help="Number of training epochs (default: 100, paper-aligned)")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Batch size (default: None = full batch, paper-aligned)")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate (default: 1e-3)")
    p.add_argument("--patience", type=int, default=None,
                   help="Early stopping patience (default: None = no early stopping, paper-aligned)")
    
    return p.parse_args()

# ============================================================
# Main
# ============================================================

def main():
    """
    This setup ensures that any predictive performance above chance can be
    attributed to systematic structure in the pose representation itself,
    rather than confounding visual or contextual cues.
    
    The task and number of classes are automatically detected from the JSON file,
    eliminating the need for manual specification.
    """

    args = parse_args()
    Path(args.out_dir).mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data and auto-detect task configuration
    tqdm.write("▶ Loading data and detecting task configuration...")
    X_train, y_train, X_val, y_val, X_test, y_test, task, num_classes, is_binary, unique_labels = \
        load_dataset(args.splits_json)
    
    
    tqdm.write(f"✓ Task detected: {task}")
    tqdm.write(f"✓ Number of unique labels: {unique_labels}")
    tqdm.write(f"✓ Classification type: {'Binary (2 classes)' if is_binary else f'Multi-class ({unique_labels} classes)'}")
    tqdm.write(f"✓ Network output neurons: {num_classes}")
    tqdm.write(f"✓ Loss function: {'BCEWithLogitsLoss' if is_binary else 'CrossEntropyLoss'}")
    tqdm.write(f"✓ Feature dimension: {X_train.shape[1]}")
    tqdm.write(f"✓ Train samples: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    csv_path = Path(args.out_dir) / f"mlp_results_{task}.csv"

    fieldnames = [
        "task", "seed",
        "test_auc", "test_accuracy", "test_f1",
        "ci_low", "ci_high",
        *[f"class_{i}_auc" for i in range(unique_labels)]
        # "class_0_auc", "class_1_auc", "class_2_auc",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for seed in args.seeds:
            tqdm.write(f"\n{'='*60}")
            tqdm.write(f"Seed {seed}")
            tqdm.write(f"{'='*60}")

            model, scores_or_probs, auc, acc, f1, per_class = \
                train_and_eval_mlp(
                    X_train, y_train,
                    X_val, y_val,
                    X_test, y_test,
                    num_classes,
                    is_binary,
                    seed,
                    device,
                    num_epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    patience=args.patience,
                )

            ci_low = ci_high = None
            if args.bootstrap:
                tqdm.write("▶ Computing bootstrap confidence intervals...")
                if is_binary:
                    _, ci_low, ci_high = bootstrap_auc_binary(
                        y_test, scores_or_probs, seed=seed
                    )
                else:
                    _, ci_low, ci_high = bootstrap_auc_multiclass(
                        y_test, scores_or_probs, seed=seed
                    )

            model_path = (
                Path(args.out_dir)
                / f"mlp_{task}_seed_{seed}.pt"
            )
            torch.save(model.state_dict(), model_path)
            tqdm.write(f"✓ Saved model to {model_path}")

            row = {
                "task": task,
                "seed": seed,
                "test_auc": auc,
                "test_accuracy": acc,
                "test_f1": f1,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }

            # Initialize class AUC fields dynamically
            for i in range(unique_labels):
                row[f"class_{i}_auc"] = None

            if not is_binary and per_class is not None:
                for k, v in per_class.items():
                    row[k] = v

            writer.writerow(row)

    tqdm.write(f"\n✓ Results saved to {csv_path}")

if __name__ == "__main__":
    main()