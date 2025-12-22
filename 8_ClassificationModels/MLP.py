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
- No hyperparameter tuning
- Fixed number of training iterations -> 100
- ROC-AUC as the primary metric
- Confidence intervals estimated via bootstrapping
- Repeated runs across multiple random seeds

Extensions beyond the paper include:
- Support for multi-class skin-tone prediction
- Explicit saving of trained models
- Additional diagnostic metrics (accuracy, F1)

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

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize

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

# ============================================================
# Data loading
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

def load_dataset(json_path):
    """
    Loads train / val / test splits.

    Note:
    - The validation split is intentionally unused. The paper does not perform
    model selection or hyperparameter tuning for the MLP probe, as the goal
    is not to optimize performance but to measure whether pose alone contains
    predictive signal under a fixed-capacity classifier.

    """
    with open(json_path) as f:
        data = json.load(f)

    X_train, y_train = load_split(data["train"])
    X_val, y_val     = load_split(data["val"])   # unused (paper-aligned) since their is no hyper-param tuning
    X_test, y_test   = load_split(data["test"])

    return X_train, y_train, X_test, y_test

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

    macro_auc = roc_auc_score(
        y_true, probs, multi_class="ovr", average="macro"
    )
    acc = accuracy_score(y_true, preds)
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="macro", zero_division=0
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
    X_test,
    y_test,
    num_classes,
    seed,
    device,
    num_epochs = 100 # Aligned with the paper
):
    """
    Training is performed for a fixed number of epochs with no early stopping
    and no validation-based model selection. This strictly follows the paper’s
    protocol and ensures that performance differences cannot be attributed
    to tuning or optimization strategies.
    
    Training is performed for a fixed number of epochs with no early stopping
    and no validation-based model selection. This strictly follows the paper’s
    protocol and ensures that performance differences cannot be attributed
    to tuning or optimization strategies.
    """

    set_seed(seed)

    model = ProbeMLP(
        input_dim=X_train.shape[1],
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    criterion = (
        nn.BCEWithLogitsLoss()
        if num_classes == 1
        else nn.CrossEntropyLoss()
    )

    Xtr = torch.tensor(X_train, device=device)
    ytr = torch.tensor(y_train, device=device)
    Xte = torch.tensor(X_test, device=device)

    # --------------------------------------------------------
    # Training: fixed iterations (paper-aligned)
    # --------------------------------------------------------

    model.train()
    for _ in range(num_epochs):
        logits = model(Xtr)

        loss = (
            criterion(logits, ytr.float())
            if num_classes == 1
            else criterion(logits, ytr)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------
    
    """
    Evaluation is performed exactly once on the held-out test set. The test
    data is never used for training decisions, preserving a clean separation
    between probing and evaluation.
    """

    model.eval()
    with torch.no_grad():
        logits = model(Xte)

    if num_classes == 1:
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
    p = argparse.ArgumentParser()
    p.add_argument("--splits_json", required=True)
    p.add_argument("--task", choices=["gender", "skintone"], required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--out_dir", default="outputs")
    return p.parse_args()

# ============================================================
# Main
# ============================================================

def main():
    """
    This setup ensures that any predictive performance above chance can be
    attributed to systematic structure in the pose representation itself,
    rather than confounding visual or contextual cues.
    """

    args = parse_args()
    Path(args.out_dir).mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 1 if args.task == "gender" else 3

    csv_path = Path(args.out_dir) / f"mlp_results_{args.task}.csv"

    fieldnames = [
        "task", "seed",
        "test_auc", "test_accuracy", "test_f1",
        "ci_low", "ci_high",
        "class_0_auc", "class_1_auc", "class_2_auc",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for seed in args.seeds:
            X_train, y_train, X_test, y_test = \
                load_dataset(args.splits_json)

            model, scores_or_probs, auc, acc, f1, per_class = \
                train_and_eval_mlp(
                    X_train, y_train,
                    X_test, y_test,
                    num_classes,
                    seed,
                    device,
                )

            ci_low = ci_high = None
            if args.bootstrap:
                if num_classes == 1:
                    _, ci_low, ci_high = bootstrap_auc_binary(
                        y_test, scores_or_probs, seed=seed
                    )
                else:
                    _, ci_low, ci_high = bootstrap_auc_multiclass(
                        y_test, scores_or_probs, seed=seed
                    )

            model_path = (
                Path(args.out_dir)
                / f"mlp_{args.task}_seed_{seed}.pt"
            )
            torch.save(model.state_dict(), model_path)

            row = {
                "task": args.task,
                "seed": seed,
                "test_auc": auc,
                "test_accuracy": acc,
                "test_f1": f1,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "class_0_auc": None,
                "class_1_auc": None,
                "class_2_auc": None,
            }

            if num_classes > 1 and per_class is not None:
                for k, v in per_class.items():
                    row[k] = v

            writer.writerow(row)

    print(f"Saved results to {csv_path}")

if __name__ == "__main__":
    main()

# python MLP.py --splits_json test_mlp_gender.json --task gender --seeds 0 --out_dir test_mlp_gender 
# python MLP.py --splits_json test_mlp_skintone.json --task skintone --seeds 0 --out_dir test_mlp_skintone