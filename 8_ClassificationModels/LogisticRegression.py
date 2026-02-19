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
from sklearn.model_selection import GridSearchCV
from itertools import product

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
    X_val, y_val     = load_split(data["val"])
    X_test, y_test   = load_split(data["test"])

    return X_train, y_train, X_val, y_val, X_test, y_test

def detect_task_info(y_train, y_val, y_test):
    """
    Auto-detect number of classes and task type from labels.
    
    Returns:
        num_classes: int
        task_type: 'binary' or 'multiclass'
    """
    all_labels = np.concatenate([y_train, y_val, y_test])
    num_classes = len(np.unique(all_labels))
    
    task_type = 'binary' if num_classes == 2 else 'multiclass'
    
    print(f"Auto-detected: {num_classes} classes ({task_type} classification)")
    print(f"Class distribution in training set: {np.bincount(y_train)}")
    
    return num_classes, task_type

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
    num_classes = probs.shape[1]

    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        y_sample = y_true[idx]

        # Skip resamples that drop a class
        if len(np.unique(y_sample)) < num_classes:
            continue

        aucs.append(
            roc_auc_score(
                y_sample,
                probs[idx],
                multi_class="ovr",
                average="macro",
            )
        )

    if len(aucs) == 0:
        raise RuntimeError("No valid bootstrap samples contained all classes")

    return (
        np.mean(aucs),
        np.percentile(aucs, 2.5),
        np.percentile(aucs, 97.5),
    )

# ============================================================
# Model factory (paper-aligned)
# ============================================================

def build_logistic_regression(feature_type, task_type, C=1.0, max_iter=1000, tol=0.0001):
    """
    Paper-aligned logistic regression configuration.

    - Mean RGB  -> L2, λ=1 (C=1.0)
    - Objects   -> L1, λ=1 (C=1.0)
    
    Args:
        feature_type: 'mean_rgb' or 'objects'
        task_type: 'binary' or 'multiclass'
        C: Inverse of regularization strength (1/λ)
        max_iter: Maximum iterations
        tol: Tolerance for stopping
    """
    
    if feature_type == "mean_rgb":
        # Paper: L2 regularization, λ=1
        return LogisticRegression(
            penalty="l2",
            C=C,
            solver="lbfgs",
            max_iter=max_iter,
            tol=tol,
            verbose=1,
            random_state=42
        )

    elif feature_type == "objects":
        # Paper: L1 regularization, λ=1, liblinear solver
        return LogisticRegression(
            penalty="l1",
            C=C,
            solver="liblinear",
            max_iter=max_iter,
            tol=tol,
            verbose=1,
            random_state=42
        )
    
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")

def tune_hyperparameters(X_train, y_train, X_val, y_val, feature_type, task_type):
    """
    Perform grid search for hyperparameters on validation set.
    
    Following the paper's approach:
    - Grid search over regularization strengths
    - Evaluate on validation set
    - Return best hyperparameters
    
    Note: Paper mentions grid search for CNN (learning rate, weight decay)
    but uses fixed λ=1 for logistic regression. This function provides
    optional tuning capability.
    """
    
    # Paper-inspired grid (converting from λ to C = 1/λ)
    # λ values: {10^-2, 10^-3, 10^-4, 10^-5} → C values: {100, 1000, 10000, 100000}
    # Also include λ=1 (C=1.0) and nearby values
    C_values = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    
    param_grid = {
        'C': C_values,
    }
    
    base_model = build_logistic_regression(
        feature_type=feature_type,
        task_type=task_type,
        C=1.0  # Will be overridden by grid search
    )
    
    # Use AUC as scoring metric (paper's primary metric)
    if task_type == 'binary':
        scoring = 'roc_auc'
    else:
        scoring = 'roc_auc_ovr'  # One-vs-Rest for multiclass
    
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        scoring=scoring,
        cv=[(np.arange(len(X_train)), np.arange(len(X_train), len(X_train) + len(X_val)))],
        # Use predefined split (train as train, validation as test)
        verbose=2,
        n_jobs=-1
    )
    
    # Combine train and validation for grid search
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])
    
    grid_search.fit(X_combined, y_combined)
    
    print(f"\nGrid Search Results:")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best validation AUC: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_

# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Logistic regression for classification with auto-detection of task type"
    )
    p.add_argument("--splits_json", required=True, 
                   help="Path to JSON file with train/val/test splits")
    p.add_argument("--feature_type",
                   choices=["mean_rgb", "objects"],
                   required=True,
                   help="Type of features: mean_rgb (3 features) or objects (binary vector)")
    
    p.add_argument("--task_name", type=str, default="classification",
                   help="Name for this classification task (for output files)")
    
    # Hyperparameter options
    p.add_argument("--tune_hyperparameters", action="store_true",
                   help="Perform grid search for hyperparameters on validation set")
    p.add_argument("--C", type=float, default=1.0,
                   help="Regularization strength (1/lambda). Default: 1.0 (paper-aligned)")
    p.add_argument("--max_iter", type=int, default=1000,
                   help="Maximum number of iterations. Default: 1000")
    p.add_argument("--tol", type=float, default=0.0001,
                   help="Tolerance for stopping. Default: 0.0001")
    
    # Evaluation options
    p.add_argument("--seeds", type=int, nargs="+", default=[0],
                   help="Random seeds for multiple runs")
    p.add_argument("--bootstrap", action="store_true",
                   help="Compute bootstrapped confidence intervals (5000 resamples)")
    p.add_argument("--out_dir", default="outputs",
                   help="Output directory for results and models")
    
    return p.parse_args()

# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    Path(args.out_dir).mkdir(exist_ok=True, parents=True)

    # Load data
    print(f"Loading data from {args.splits_json}...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(args.splits_json)
    
    # Auto-detect task type
    num_classes, task_type = detect_task_info(y_train, y_val, y_test)
    
    print(f"\nFeature type: {args.feature_type}")
    print(f"Task type: {task_type} ({num_classes} classes)")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Hyperparameter tuning (optional)
    if args.tune_hyperparameters:
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING")
        print("="*60)
        best_params = tune_hyperparameters(
            X_train, y_train, X_val, y_val,
            args.feature_type, task_type
        )
        C_to_use = best_params['C']
        print(f"\nUsing tuned C={C_to_use}")
    else:
        C_to_use = args.C
        print(f"\nUsing fixed C={C_to_use} (paper-aligned default)")
    
    # Prepare CSV output
    csv_path = Path(args.out_dir) / f"logreg_results_{args.task_name}_{args.feature_type}.csv"
    
    # Dynamic fieldnames based on number of classes
    fieldnames = [
        "task_name", "feature_type", "task_type", "num_classes", "seed",
        "test_auc", "test_accuracy", "test_f1",
        "ci_low", "ci_high",
    ]
    # Add per-class AUC fields for multiclass
    if task_type == 'multiclass':
        fieldnames.extend([f"class_{i}_auc" for i in range(num_classes)])
    
    # Train models with different seeds
    print("\n" + "="*60)
    print("TRAINING AND EVALUATION")
    print("="*60)
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for seed_idx, seed in enumerate(args.seeds):
            print(f"\n--- Seed {seed_idx + 1}/{len(args.seeds)}: {seed} ---")
            set_seed(seed)

            # Build and train model
            model = build_logistic_regression(
                feature_type=args.feature_type,
                task_type=task_type,
                C=C_to_use,
                max_iter=args.max_iter,
                tol=args.tol
            )
            
            print("Training model...")
            model.fit(X_train, y_train)
            
            # Save model
            model_path = (
                Path(args.out_dir)
                / f"logreg_{args.task_name}_{args.feature_type}_seed_{seed}.joblib"
            )
            joblib.dump(model, model_path)
            print(f"Model saved to {model_path}")

            # Evaluate
            if task_type == 'binary':
                scores = model.predict_proba(X_test)[:, 1]
                auc, acc, f1 = binary_metrics(y_test, scores)
                
                print(f"Test AUC: {auc:.4f}")
                print(f"Test Accuracy: {acc:.4f}")
                print(f"Test F1: {f1:.4f}")

                ci_low = ci_high = None
                if args.bootstrap:
                    print("Computing bootstrapped confidence intervals...")
                    _, ci_low, ci_high = bootstrap_auc_binary(
                        y_test, scores, seed=seed
                    )
                    print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

                row = {
                    "task_name": args.task_name,
                    "feature_type": args.feature_type,
                    "task_type": task_type,
                    "num_classes": num_classes,
                    "seed": seed,
                    "test_auc": auc,
                    "test_accuracy": acc,
                    "test_f1": f1,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }

            else:  # multiclass
                probs = model.predict_proba(X_test)
                auc, acc, f1, per_class = multiclass_metrics(
                    y_test, probs, num_classes=num_classes
                )
                
                print(f"Test AUC (macro): {auc:.4f}")
                print(f"Test Accuracy: {acc:.4f}")
                print(f"Test F1 (macro): {f1:.4f}")
                for class_id, class_auc in per_class.items():
                    print(f"  {class_id}: {class_auc:.4f}")

                ci_low = ci_high = None
                if args.bootstrap:
                    print("Computing bootstrapped confidence intervals...")
                    _, ci_low, ci_high = bootstrap_auc_multiclass(
                        y_test, probs, seed=seed
                    )
                    print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

                row = {
                    "task_name": args.task_name,
                    "feature_type": args.feature_type,
                    "task_type": task_type,
                    "num_classes": num_classes,
                    "seed": seed,
                    "test_auc": auc,
                    "test_accuracy": acc,
                    "test_f1": f1,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    **per_class,
                }

            writer.writerow(row)

    print(f"\n{'='*60}")
    print(f"Results saved to {csv_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()


# ============================================================
# USAGE EXAMPLES
# ============================================================

# Example 1: Basic usage with paper-aligned defaults (C=1.0, no tuning)
# python LogisticRegression.py ^
#     --splits_json "test_splits_gender_mean_rgb.json" ^
#     --feature_type "mean_rgb" ^
#     --task_name "gender" ^
#     --bootstrap ^
#     --out_dir "outputs"

# python LogisticRegression.py ^
#     --splits_json "test_splits_skintone_object_detection.json" ^
#     --feature_type "mean_rgb" ^
#     --task_name "skintone" ^
#     --bootstrap ^
#     --out_dir "outputs"

# Example 2: With hyperparameter tuning
# python LogisticRegression_improved.py \
#     --splits_json "path/to/splits.json" \
#     --feature_type "objects" \
#     --task_name "skintone" \
#     --tune_hyperparameters \
#     --bootstrap \
#     --out_dir "outputs"

# Example 3: Multiple seeds (paper uses 5 for alternative CI method)
# python LogisticRegression_improved.py \
#     --splits_json "path/to/splits.json" \
#     --feature_type "mean_rgb" \
#     --task_name "age" \
#     --seeds 0 1 2 3 4 \
#     --out_dir "outputs"

# Example 4: Custom regularization strength
# python LogisticRegression_improved.py \
#     --splits_json "path/to/splits.json" \
#     --feature_type "mean_rgb" \
#     --task_name "emotion" \
#     --C 10.0 \
#     --bootstrap \
#     --out_dir "outputs"