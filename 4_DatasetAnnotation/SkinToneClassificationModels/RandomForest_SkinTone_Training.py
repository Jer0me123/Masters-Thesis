##############################################################
#  RANDOM FOREST MST CLASSIFIER (SINGLE FILE, EXTENDED)
#  - Subject-level stratified split (same as VGG16 pipeline)
#  - Paper-spec feature extractor (RGB + Y + V + L, default 256 bins)
#  - Configurable bins for ablation (32, 64, 128, 256, ...)
#  - Dual modes:
#       * Classification (RandomForestClassifier)
#       * Regression (RandomForestRegressor)
#  - Optional hyperparameter tuning via RandomizedSearchCV
#    * NOW USING GroupKFold (person-level CV, no identity leakage)
#  - Outputs:
#       * CLF: ACC, Off-by-one ACC, MSE
#       * REG: RMSE, ACC (±0.5), rounded bin ACC, off-by-one bin ACC
#  - Auto-named model files with metrics + timestamp
##############################################################

import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GroupKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

import joblib
import json


##############################################################
# 0. REPRODUCIBILITY
##############################################################

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)

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
# 1. SUBJECT-LEVEL STRATIFIED SPLIT (SAME AS VGG16)
##############################################################

def subject_stratified_split(csv_path, val_ratio, save_split_path=None):
    """
    Reads CSV with columns: [filename, label, person_id, ...]
    Performs subject-level stratified split:
      - Persons appear exclusively in either train or val set
      - Stratification is done on rounded MST labels (1..10)
    Returns:
      train_df, val_df
    Where each has columns: [filename, label, person_id]
    """
    df = pd.read_csv(csv_path).dropna()

    # --- Flexible column resolution -----------------------------------
    # Supports both the original 3-column CSV (filename, label, person_id)
    # and wider CSVs such as Casual Conversation v2 annotations.
    FILENAME_ALIASES = ["filename", "cropped_image", "image", "file"]
    LABEL_ALIASES    = ["mst_label", "label", "mst", "skin_tone"]
    PERSON_ALIASES   = ["subject_id", "person_id", "person", "id"]

    def _resolve(aliases, cols):
        for a in aliases:
            if a in cols:
                return a
        return None

    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols  # normalise to lowercase

    filename_col = _resolve(FILENAME_ALIASES, cols)
    label_col    = _resolve(LABEL_ALIASES, cols)
    person_col   = _resolve(PERSON_ALIASES, cols)

    if filename_col is None or label_col is None or person_col is None:
        raise ValueError(
            f"Cannot resolve required columns. Found: {list(df.columns)}. "
            f"Need one of each: {FILENAME_ALIASES}, {LABEL_ALIASES}, {PERSON_ALIASES}"
        )

    # Ensure person_id is a string (for grouping)
    df["person_id"] = df[person_col].astype(str)

    # Aggregate label per PERSON (label assigned based on first image)
    person_labels = df.groupby("person_id").agg({label_col: "first"}).reset_index()
    labels_per_person = person_labels[label_col].astype(float).values

    # Rounded labels for stratification (MST 1..10)
    rounded = np.round(labels_per_person).astype(int)
    rounded = np.clip(rounded, 1, 10)

    # Person-level stratified split
    train_persons, val_persons = train_test_split(
        person_labels["person_id"],
        test_size=val_ratio,
        shuffle=True,
        stratify=rounded,
        random_state=42
    )

    if save_split_path is not None:
        save_split_json(
            train_persons=train_persons,
            val_persons=val_persons,
            val_ratio=val_ratio,
            save_path=save_split_path
        )

    # Filter original df by person_id
    train_df = df[df["person_id"].isin(train_persons)].reset_index(drop=True)
    val_df   = df[df["person_id"].isin(val_persons)].reset_index(drop=True)

    # Keep only filename, label, person_id (Option A: person_id is retained)
    train_df = train_df[[filename_col, label_col, "person_id"]]
    val_df   = val_df[[filename_col, label_col, "person_id"]]

    # ============================
    # DEBUG SPLIT INFORMATION
    # ============================
    print("\n================ SPLIT DIAGNOSTICS ================")
    print(f"Total persons: {len(person_labels)}")
    print(f"Train persons: {len(train_persons)}")
    print(f"Val persons:   {len(val_persons)}")
    print(f"Train images:  {len(train_df)}")
    print(f"Val images:    {len(val_df)}\n")

    # Raw labels (image-level)
    train_labels_raw = train_df[label_col].astype(float).values
    val_labels_raw   = val_df[label_col].astype(float).values

    # Rounded MST bins
    train_bins = np.clip(np.round(train_labels_raw).astype(int), 1, 10)
    val_bins   = np.clip(np.round(val_labels_raw).astype(int), 1, 10)

    train_counts = Counter(train_bins)
    val_counts   = Counter(val_bins)

    print("[INFO] Train MST Distribution (images):")
    for k in range(1, 11):
        print(f"  MST {k}: {train_counts.get(k, 0)} images")

    print("\n[INFO] Val MST Distribution (images):")
    for k in range(1, 11):
        print(f"  MST {k}: {val_counts.get(k, 0)} images")

    print("====================================================\n")

    # Safety check: no person appears in both splits
    overlap = set(train_persons).intersection(set(val_persons))
    if len(overlap) > 0:
        print("[WARN] Some person_ids appear in both train and val splits!")
    else:
        print("[INFO] Person-level exclusivity between TRAIN and VAL is satisfied.\n")

    return train_df, val_df


##############################################################
# 2. FEATURE EXTRACTOR (PAPER SPEC + CONFIGURABLE BINS)
##############################################################

def extract_hist_features(image_bgr, bins=256):
    """
    Compute concatenated histograms:
      - RGB (3 channels)
      - Y (YCbCr)
      - V (HSV)
      - L (Lab)
    Each with `bins` bins → total feature dim = 6 * bins.

    A binary mask is generated to exclude near-black background
    pixels (common in segmented face images where the non-skin
    region is zero-filled).  Only pixels whose maximum BGR
    channel value exceeds `bg_threshold` contribute to the
    histograms.  This prevents the large (0,0,0) background
    from dominating the feature vector after L1 normalisation.
    """
    BG_THRESHOLD = 10          # max(B,G,R) must exceed this

    if image_bgr.dtype != np.uint8:
        image_bgr = (image_bgr * 255).astype(np.uint8)

    # ----------------------------------------------------------
    # Build foreground mask: exclude near-black background pixels
    # ----------------------------------------------------------
    max_channel = np.max(image_bgr, axis=2)            # (H, W)
    mask = (max_channel > BG_THRESHOLD).astype(np.uint8) * 255

    skin_px = cv2.countNonZero(mask)
    if skin_px == 0:
        # Fallback: if the entire image is black, return zeros
        return np.zeros(6 * bins, dtype=np.float32)

    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_ycc = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    img_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)

    hists = []

    # 3 × RGB
    for i in range(3):
        h = cv2.calcHist([img_rgb], [i], mask, [bins], [0, 256]).flatten()
        hists.append(h)

    # Y
    hists.append(
        cv2.calcHist([img_ycc], [0], mask, [bins], [0, 256]).flatten()
    )

    # V
    hists.append(
        cv2.calcHist([img_hsv], [2], mask, [bins], [0, 256]).flatten()
    )

    # L
    hists.append(
        cv2.calcHist([img_lab], [0], mask, [bins], [0, 256]).flatten()
    )

    feat = np.concatenate(hists).astype(np.float32)
    feat /= (feat.sum() + 1e-8)

    return feat


##############################################################
# 3. BUILD FULL DATASET (TRAIN OR VAL)
##############################################################

def build_dataset(df, image_dir, bins=256):
    """
    Expects df with columns: [filename, label, person_id]
    Only uses filename + label for feature/target; person_id is used separately.
    """
    X, y = [], []

    # Use explicit iteration to avoid confusion with extra columns
    for row in tqdm(df.itertuples(index=False), desc="Extracting features"):
        filename = row[0]  # first column: filename
        label    = row[1]  # second column: label

        path = Path(image_dir) / filename
        img = cv2.imread(str(path))

        if img is None:
            print(f"[WARN] Cannot read image: {path}")
            continue

        feat = extract_hist_features(img, bins=bins)
        X.append(feat)
        y.append(float(label))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    print(f"[INFO] Built dataset: X.shape={X.shape}, y.shape={y.shape}")
    return X, y


##############################################################
# 4. METRICS
##############################################################

def off_by_one_accuracy(y_true, y_pred, tol=1.0):
    """
    General off-by-one accuracy:
      - For raw MST: |y_true - y_pred| <= 1.0
      - For bins:    same logic on integer bins.
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return np.mean(np.abs(y_true - y_pred) <= tol)


##############################################################
# 5. HYPERPARAMETER SEARCH (WITH GROUPKFOLD)
##############################################################

def hyperparam_search_clf(X_train, y_train, train_groups,
                          n_jobs=-1, n_iter=30, random_state=42):
    """
    Randomized hyperparameter search for RandomForestClassifier.
    Scoring: accuracy (on training CV folds).
    Uses GroupKFold with person_id groups to avoid identity leakage.
    """
    param_dist = {
        "n_estimators":      [200, 300, 400, 600, 800],
        "max_depth":         [None, 10, 20, 30, 40],
        "min_samples_split": [2, 5, 10, 20, 50],
        "min_samples_leaf":  [1, 2, 5, 10, 20],
        "max_features":      ["sqrt", "log2", 0.2, 0.4, 0.6],
        "class_weight":      [None, "balanced", "balanced_subsample"],
    }

    base = RandomForestClassifier(
        n_jobs=n_jobs,
        random_state=random_state
    )

    cv = GroupKFold(n_splits=3)
    print("[INFO] Using GroupKFold(n_splits=3) for CLASSIFIER hyperparameter tuning.")
    print(f"[INFO] Number of unique persons in training: {len(np.unique(train_groups))}")

    search = HalvingRandomSearchCV(
        estimator=base,
        param_distributions=param_dist,
        cv=cv,
        factor=3,  # Keep top 1/3 each round
        resource='n_samples',  # Increase training data each round
        max_resources='auto',  # Use all data in final round
        aggressive_elimination=False,  # More conservative
        scoring="accuracy",
        min_resources='smallest',
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=3
    )

    print("[INFO] Starting hyperparameter search for CLASSIFIER...")
    search.fit(X_train, y_train, groups=train_groups)
    print("[INFO] Hyperparameter search completed.")
    print("[INFO] Best params:", search.best_params_)
    print("[INFO] Best CV accuracy:", search.best_score_)

    return search.best_estimator_


def hyperparam_search_reg(X_train, y_train, train_groups,
                          n_jobs=-1, n_iter=30, random_state=42):
    """
    Randomized hyperparameter search for RandomForestRegressor.
    Scoring: negative MSE.
    Uses GroupKFold with person_id groups to avoid identity leakage.
    """
    param_dist = {
        "n_estimators":      [200, 300, 400, 600, 800],
        "max_depth":         [None, 10, 20, 30, 40],
        "min_samples_split": [2, 5, 10, 20, 50],
        "min_samples_leaf":  [1, 2, 5, 10, 20],
        "max_features":      ["sqrt", "log2", 0.2, 0.4, 0.6],
    }

    base = RandomForestRegressor(
        n_jobs=n_jobs,
        random_state=random_state
    )

    cv = GroupKFold(n_splits=3)
    print("[INFO] Using GroupKFold(n_splits=3) for REGRESSOR hyperparameter tuning.")
    print(f"[INFO] Number of unique persons in training: {len(np.unique(train_groups))}")

    search = HalvingRandomSearchCV(
        estimator=base,
        param_distributions=param_dist,
        cv=cv,
        factor=3,  # Keep top 1/3 each round
        resource='n_samples',  # Increase training data each round
        max_resources='auto',  # Use all data in final round
        aggressive_elimination=False,  # More conservative
        scoring="neg_mean_squared_error",
        min_resources='smallest',
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=3
    )

    print("[INFO] Starting hyperparameter search for REGRESSOR...")
    search.fit(X_train, y_train, groups=train_groups)
    print("[INFO] Hyperparameter search completed.")
    print("[INFO] Best params:", search.best_params_)
    print("[INFO] Best CV neg-MSE:", search.best_score_)

    return search.best_estimator_


##############################################################
# 6. TRAINING LOGIC (CLF + REG)
##############################################################

def train_rf(csv_path,
             image_dir,
             model_out_dir,
             val_ratio,
             mode="clf",
             n_estimators=300,
             max_depth=None,
             min_samples_split=2,
             min_samples_leaf=1,
             max_features="sqrt",
             n_jobs=-1,
             class_weight="none",
             bins=256,
             hparam_search=False,
             search_iter=30,
             random_state=42):

    model_out_dir = Path(model_out_dir) if model_out_dir else Path(".")
    model_out_dir.mkdir(parents=True, exist_ok=True)

    split_json_path = model_out_dir / "train_val_split.json"

    print("[INFO] Performing subject-level stratified split (same as VGG16)...")
    train_df, val_df = subject_stratified_split(csv_path, val_ratio, split_json_path)

    # Extract group labels (person_id) for training set
    train_groups = train_df["person_id"].values

    print("[INFO] Building TRAIN dataset...")
    X_train, y_train = build_dataset(train_df, image_dir, bins=bins)

    print("[INFO] Building VAL dataset...")
    X_val, y_val = build_dataset(val_df, image_dir, bins=bins)

    # ----------------------------------------------------------
    # 6.1 Model selection (with or without hyperparam tuning)
    # ----------------------------------------------------------
    if mode == "clf":
        # Convert class_weight flag
        cw = None if class_weight.lower() == "none" else class_weight

        if hparam_search:
            rf = hyperparam_search_clf(
                X_train, y_train, train_groups=train_groups,
                n_jobs=n_jobs, n_iter=search_iter, random_state=random_state
            )
        else:
            print("[INFO] Training RandomForestClassifier...")
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                n_jobs=n_jobs,
                random_state=random_state,
                class_weight=cw,
                verbose=1
            )
            rf.fit(X_train, y_train)

        # ------------------------------------------------------
        # 6.2 Evaluation (Classification)
        # ------------------------------------------------------
        y_pred = rf.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        ooacc = off_by_one_accuracy(y_val, y_pred, tol=1.0)
        mse = mean_squared_error(y_val, y_pred)

        print("\n================== CLASSIFIER RESULTS ==================")
        print(f"Accuracy (exact label):          {acc:.4f}")
        print(f"Off-by-one ACC (|Δ|<=1):         {ooacc:.4f}")
        print(f"MSE:                             {mse:.4f}")
        print("========================================================\n")

        # Auto-name model file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        acc_str = f"{acc:.3f}"
        ooacc_str = f"{ooacc:.3f}"
        mse_str = f"{mse:.3f}"

        model_name = (
            f"rf_clf_bins{bins}_ACC{acc_str}_OO{ooacc_str}_MSE{mse_str}_{timestamp}.joblib"
        )

    else:
        # mode == "reg"
        if class_weight.lower() != "none":
            print("[WARN] class_weight is ignored in regression mode.")

        if hparam_search:
            rf = hyperparam_search_reg(
                X_train, y_train, train_groups=train_groups,
                n_jobs=n_jobs, n_iter=search_iter, random_state=random_state
            )
        else:
            print("[INFO] Training RandomForestRegressor...")
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=1
            )
            rf.fit(X_train, y_train)

        # ------------------------------------------------------
        # 6.3 Evaluation (Regression)
        # ------------------------------------------------------
        y_pred_float = rf.predict(X_val)

        # RMSE on continuous MST labels
        rmse = np.sqrt(mean_squared_error(y_val, y_pred_float))

        # Accuracy with ±0.5 tolerance on continuous MST
        acc_half = np.mean(np.abs(y_val - y_pred_float) <= 0.5)

        # Bin-level evaluation (rounded MST bins)
        y_true_bin = np.clip(np.round(y_val),        1, 10)
        y_pred_bin = np.clip(np.round(y_pred_float), 1, 10)

        acc_bin = accuracy_score(y_true_bin, y_pred_bin)
        ooacc_bin = off_by_one_accuracy(y_true_bin, y_pred_bin, tol=1.0)

        print("\n================== REGRESSOR RESULTS ==================")
        print(f"RMSE (continuous MST):           {rmse:.4f}")
        print(f"Accuracy (|Δ|<=0.5):             {acc_half:.4f}")
        print(f"Rounded bin ACC (exact bin):     {acc_bin:.4f}")
        print(f"Off-by-one bin ACC (|Δbin|<=1):  {ooacc_bin:.4f}")
        print("========================================================\n")

        # Auto-name model file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rmse_str = f"{rmse:.3f}"
        acc_half_str = f"{acc_half:.3f}"
        model_name = (
            f"rf_reg_bins{bins}_RMSE{rmse_str}_ACC05{acc_half_str}_{timestamp}.joblib"
        )

    # ----------------------------------------------------------
    # 6.4 Save model
    # ----------------------------------------------------------
    model_path = model_out_dir / model_name
    joblib.dump(rf, model_path)
    print(f"[INFO] Saved model → {model_path}")


##############################################################
# 7. ARGPARSE HELPERS
##############################################################

def parse_max_depth(val):
    if val is None:
        return None
    s = str(val)
    if s.lower() == "none":
        return None
    return int(s)


def parse_max_features(val):
    if val is None:
        return "sqrt"
    s = str(val).lower()
    if s in ["sqrt", "log2", "auto"]:
        return s
    try:
        f = float(s)
        return f
    except ValueError:
        raise argparse.ArgumentTypeError(
            "max_features must be 'sqrt', 'log2', 'auto', or a float in (0,1]."
        )


##############################################################
# 8. MAIN
##############################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True,
                        help="CSV with columns: filename,label,person_id")
    parser.add_argument("--image-dir", required=True,
                        help="Directory with segmented face images")
    parser.add_argument("--model-out", default=".",
                        help="Directory to save the trained model(s)")
    parser.add_argument("--mode", type=str, default="clf",
                        choices=["clf", "reg"],
                        help="Training mode: clf=classifier, reg=regressor")

    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=parse_max_depth, default=None)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--max-features", type=parse_max_features, default="sqrt")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--val-ratio", type=float, default=0.35)
    parser.add_argument("--class-weight", type=str, default="none",
                        choices=["none", "balanced", "balanced_subsample"],
                        help="Only used in classifier mode")
    parser.add_argument("--bins", type=int, default=256,
                        help="Number of histogram bins per channel (default 256 = paper)")
    
    parser.add_argument("--hparam-search", action="store_true",
                        help="Enable hyperparameter search with RandomizedSearchCV")
    parser.add_argument("--search-iter", type=int, default=30,
                        help="Number of RandomizedSearch iterations")
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    train_rf(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        model_out_dir=args.model_out,
        val_ratio=args.val_ratio,
        mode=args.mode,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        n_jobs=args.n_jobs,
        class_weight=args.class_weight,
        bins=args.bins,
        hparam_search=args.hparam_search,
        search_iter=args.search_iter,
        random_state=args.random_state
    )


if __name__ == "__main__":
    main()

# python RandomForest_SkinTone_Training.py --model-out "" --csv-path "" --image-dir "" --val-ratio 0.35 --n_estimators=300 --n_jobs=-1 --class-weight=none