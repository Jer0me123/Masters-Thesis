# import argparse
# import json
# from pathlib import Path
# import csv
# import random
# import numpy as np

# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import (
#     roc_auc_score,
#     accuracy_score,
#     precision_recall_fscore_support,
# )
# from sklearn.preprocessing import label_binarize
# from sklearn.utils import resample

# # ============================================================
# # Reproducibility
# # ============================================================

# def set_seed(seed: int):
#     random.seed(seed)
#     np.random.seed(seed)

# # ============================================================
# # Data loading
# # ============================================================

# def load_split(split):
#     X = np.array([x["features"] for x in split], dtype=np.float32)
#     y = np.array([x["label"] for x in split], dtype=np.int64)
#     return X, y

# def load_dataset(json_path):
#     with open(json_path) as f:
#         data = json.load(f)

#     X_train, y_train = load_split(data["train"])
#     X_val, y_val     = load_split(data["val"])
#     X_test, y_test   = load_split(data["test"])

#     return X_train, y_train, X_val, y_val, X_test, y_test

# # ============================================================
# # Metrics
# # ============================================================

# def binary_metrics(y_true, scores):
#     preds = (scores >= 0.5).astype(int)

#     auc = roc_auc_score(y_true, scores)
#     acc = accuracy_score(y_true, preds)
#     prec, rec, f1, _ = precision_recall_fscore_support(
#         y_true, preds, average="binary", zero_division=0
#     )

#     return {
#         "auc": auc,
#         "accuracy": acc,
#         "f1": f1,
#     }

# def multiclass_metrics(y_true, probs, num_classes):
#     preds = probs.argmax(axis=1)

#     macro_auc = roc_auc_score(
#         y_true, probs, multi_class="ovr", average="macro"
#     )
#     acc = accuracy_score(y_true, preds)
#     prec, rec, f1, _ = precision_recall_fscore_support(
#         y_true, preds, average="macro", zero_division=0
#     )

#     # Per-class OVR AUC
#     y_bin = label_binarize(y_true, classes=list(range(num_classes)))
#     per_class_auc = {
#         f"class_{i}_auc": roc_auc_score(y_bin[:, i], probs[:, i])
#         for i in range(num_classes)
#     }

#     return {
#         "auc": macro_auc,
#         "accuracy": acc,
#         "f1": f1,
#         "per_class_auc": per_class_auc,
#     }

# # ============================================================
# # Bootstrapping
# # ============================================================

# def bootstrap_auc_binary(y_true, scores, n=5000, seed=0):
#     rng = np.random.default_rng(seed)
#     aucs = []

#     for _ in range(n):
#         idx = rng.integers(0, len(y_true), len(y_true))
#         aucs.append(roc_auc_score(y_true[idx], scores[idx]))

#     return (
#         np.mean(aucs),
#         np.percentile(aucs, 2.5),
#         np.percentile(aucs, 97.5),
#     )

# def bootstrap_auc_multiclass(y_true, probs, n=5000, seed=0):
#     rng = np.random.default_rng(seed)
#     aucs = []

#     for _ in range(n):
#         idx = rng.integers(0, len(y_true), len(y_true))
#         aucs.append(
#             roc_auc_score(
#                 y_true[idx],
#                 probs[idx],
#                 multi_class="ovr",
#                 average="macro",
#             )
#         )

#     return (
#         np.mean(aucs),
#         np.percentile(aucs, 2.5),
#         np.percentile(aucs, 97.5),
#     )

# # ============================================================
# # Grid search
# # ============================================================

# def grid_search(
#     X_train, y_train,
#     X_val, y_val,
#     lrs, wds,
#     task,
# ):
#     best_auc = -1
#     best_model = None
#     best_cfg = None

#     for lr in lrs:
#         for wd in wds:
#             C = 1.0 / wd if wd > 0 else 1e6

#             model = LogisticRegression(
#                 penalty="l2",
#                 C=C,
#                 solver="lbfgs",
#                 max_iter=500,
#                 n_jobs=-1,
#                 multi_class="auto",
#             )

#             model.fit(X_train, y_train)

#             if task == "gender":
#                 scores = model.predict_proba(X_val)[:, 1]
#                 auc = roc_auc_score(y_val, scores)
#             else:
#                 probs = model.predict_proba(X_val)
#                 auc = roc_auc_score(
#                     y_val, probs, multi_class="ovr", average="macro"
#                 )

#             if auc > best_auc:
#                 best_auc = auc
#                 best_model = model
#                 best_cfg = (lr, wd)

#     return best_model, best_cfg

# # ============================================================
# # CLI
# # ============================================================

# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--splits_json", required=True)
#     p.add_argument("--task", choices=["gender", "skintone"], required=True)
#     p.add_argument("--seeds", type=int, nargs="+", default=[0])
#     p.add_argument("--lrs", type=float, nargs="+",
#                    default=[1e-2, 1e-3, 1e-4, 1e-5])
#     p.add_argument("--wds", type=float, nargs="+",
#                    default=[1e-2, 1e-3, 1e-4, 1e-5])
#     p.add_argument("--bootstrap", action="store_true")
#     p.add_argument("--out_dir", default="outputs")
#     return p.parse_args()

# # ============================================================
# # Main
# # ============================================================

# def main():
#     args = parse_args()
#     Path(args.out_dir).mkdir(exist_ok=True)

#     csv_path = Path(args.out_dir) / f"logreg_results_{args.task}.csv"

#     fieldnames = [
#         "task", "seed", "lr", "weight_decay",
#         "test_auc", "test_accuracy", "test_f1",
#         "ci_low", "ci_high",
#         "class_0_auc", "class_1_auc", "class_2_auc",
#     ]

#     with open(csv_path, "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()

#         for seed in args.seeds:
#             set_seed(seed)

#             X_train, y_train, X_val, y_val, X_test, y_test = \
#                 load_dataset(args.splits_json)

#             model, (lr, wd) = grid_search(
#                 X_train, y_train,
#                 X_val, y_val,
#                 args.lrs, args.wds,
#                 args.task,
#             )

#             if args.task == "gender":
#                 scores = model.predict_proba(X_test)[:, 1]
#                 metrics = binary_metrics(y_test, scores)

#                 ci_low = ci_high = None
#                 if args.bootstrap:
#                     _, ci_low, ci_high = bootstrap_auc_binary(
#                         y_test, scores, seed=seed
#                     )

#                 row = {
#                     "task": args.task,
#                     "seed": seed,
#                     "lr": lr,
#                     "weight_decay": wd,
#                     "test_auc": metrics["auc"],
#                     "test_accuracy": metrics["accuracy"],
#                     "test_f1": metrics["f1"],
#                     "ci_low": ci_low,
#                     "ci_high": ci_high,
#                     "class_0_auc": None,
#                     "class_1_auc": None,
#                     "class_2_auc": None,
#                 }

#             else:
#                 probs = model.predict_proba(X_test)
#                 metrics = multiclass_metrics(y_test, probs, num_classes=3)

#                 ci_low = ci_high = None
#                 if args.bootstrap:
#                     _, ci_low, ci_high = bootstrap_auc_multiclass(
#                         y_test, probs, seed=seed
#                     )

#                 row = {
#                     "task": args.task,
#                     "seed": seed,
#                     "lr": lr,
#                     "weight_decay": wd,
#                     "test_auc": metrics["auc"],
#                     "test_accuracy": metrics["accuracy"],
#                     "test_f1": metrics["f1"],
#                     "ci_low": ci_low,
#                     "ci_high": ci_high,
#                     "class_0_auc": metrics["per_class_auc"]["class_0_auc"],
#                     "class_1_auc": metrics["per_class_auc"]["class_1_auc"],
#                     "class_2_auc": metrics["per_class_auc"]["class_2_auc"],
#                 }

#             writer.writerow(row)

#     print(f"Saved results to {csv_path}")

# if __name__ == "__main__":
#     main()


import argparse
import json
from pathlib import Path
import csv
import random
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
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
    random.seed(seed)
    np.random.seed(seed)

# ============================================================
# Data loading
# ============================================================

def load_split(split):
    X = np.array([x["features"] for x in split], dtype=np.float32)
    y = np.array([x["label"] for x in split], dtype=np.int64)
    return X, y

def load_dataset(json_path):
    with open(json_path) as f:
        data = json.load(f)

    X_train, y_train = load_split(data["train"])
    X_val, y_val     = load_split(data["val"])   # unused (paper-aligned)
    X_test, y_test   = load_split(data["test"])

    return X_train, y_train, X_test, y_test

# ============================================================
# Metrics
# ============================================================

def binary_metrics(y_true, scores):
    preds = (scores >= 0.5).astype(int)

    auc = roc_auc_score(y_true, scores)
    acc = accuracy_score(y_true, preds)
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0
    )

    return auc, acc, f1

def multiclass_metrics(y_true, probs, num_classes):
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
# Model factory (paper-aligned)
# ============================================================

def build_logistic_regression(task, feature_type):
    """
    Paper-aligned logistic regression configuration.

    - Mean RGB  -> L2, λ=1
    - Objects   -> L1, λ=1
    - Skin tone -> L2, λ=1 (multiclass OVR)
    """

    if feature_type == "mean_rgb":
        return LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
        )

    if feature_type == "objects":
        return LogisticRegression(
            penalty="l1",
            C=1.0,
            solver="liblinear",
            max_iter=1000,
        )

    raise ValueError(f"Unsupported feature type: {feature_type}")

# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--splits_json", required=True)
    p.add_argument("--task", choices=["gender", "skintone"], required=True)
    p.add_argument("--feature_type",
                   choices=["mean_rgb", "objects"],
                   required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--out_dir", default="outputs")
    return p.parse_args()

# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    Path(args.out_dir).mkdir(exist_ok=True)

    csv_path = Path(args.out_dir) / f"logreg_results_{args.task}_{args.feature_type}.csv"

    fieldnames = [
        "task", "feature_type", "seed",
        "test_auc", "test_accuracy", "test_f1",
        "ci_low", "ci_high",
        "class_0_auc", "class_1_auc", "class_2_auc",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for seed in args.seeds:
            set_seed(seed)

            X_train, y_train, X_test, y_test = \
                load_dataset(args.splits_json)

            model = build_logistic_regression(
                args.task, args.feature_type
            )
            model.fit(X_train, y_train)

            model_path = (
                Path(args.out_dir)
                / f"logreg_{args.task}_{args.feature_type}_seed_{seed}.joblib"
            )

            joblib.dump(model, model_path)


            if args.task == "gender":
                scores = model.predict_proba(X_test)[:, 1]
                auc, acc, f1 = binary_metrics(y_test, scores)

                ci_low = ci_high = None
                if args.bootstrap:
                    _, ci_low, ci_high = bootstrap_auc_binary(
                        y_test, scores, seed=seed
                    )

                row = {
                    "task": args.task,
                    "feature_type": args.feature_type,
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

            else:
                probs = model.predict_proba(X_test)
                auc, acc, f1, per_class = \
                    multiclass_metrics(y_test, probs, num_classes=3)

                ci_low = ci_high = None
                if args.bootstrap:
                    _, ci_low, ci_high = bootstrap_auc_multiclass(
                        y_test, probs, seed=seed
                    )

                row = {
                    "task": args.task,
                    "feature_type": args.feature_type,
                    "seed": seed,
                    "test_auc": auc,
                    "test_accuracy": acc,
                    "test_f1": f1,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    **per_class,
                }

            writer.writerow(row)

    print(f"Saved results to {csv_path}")

if __name__ == "__main__":
    main()
