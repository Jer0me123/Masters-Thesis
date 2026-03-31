"""
Sentence-T5 caption probe (100% paper-aligned) for demographic prediction.

Implements the SentenceTransformer + Linear head logic used in:
boyazeng/understand_bias -> models/sentence_embed.py

NEW FEATURES (aligned with paper's main__3_.py):
- Cosine LR scheduler with linear warmup (2 epochs default)
- Per-step LR scheduling (not per-epoch)
- Matches paper's training recipe exactly

EXISTING FEATURES:
- Train / Val / Test protocol
- Early stopping with patience
- Best model per seed saved
- tqdm progress bars (epochs + batches)
- Live validation AUC reporting
- Final test metrics saved to CSV
"""

"""
Sentence-T5 caption probe (paper-aligned, task-agnostic).

Automatically infers:
- Binary classification if 2 unique labels
- Multiclass classification if >2 unique labels

No task flag required.
"""

import argparse
import json
import random
from pathlib import Path
import csv
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize

# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ============================================================
# Utils: infer classes from JSON
# ============================================================

def infer_num_classes(samples):
    """
    Infers number of unique labels and builds label mapping.
    Supports:
    - numeric labels
    - string labels
    - 'label' or 'dataset' keys
    """
    raw_labels = []

    for s in samples:
        if "label" in s:
            raw_labels.append(s["label"])
        elif "dataset" in s:
            raw_labels.append(s["dataset"])
        else:
            raise KeyError("Each sample must contain 'label' or 'dataset'")

    unique = sorted(set(raw_labels))
    label_to_id = {v: i for i, v in enumerate(unique)}

    return len(unique), label_to_id

# ============================================================
# Dataset
# ============================================================

class CaptionLabelDataset(Dataset):
    def __init__(self, samples, label_to_id):
        self.samples = samples
        self.label_to_id = label_to_id

        for i, s in enumerate(samples):
            if "caption" not in s:
                raise KeyError(f"Missing 'caption' in sample {i}")
            if "label" not in s and "dataset" not in s:
                raise KeyError(f"Missing 'label' or 'dataset' in sample {i}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        caption = s["caption"]
        raw_label = s.get("label", s.get("dataset"))
        label = self.label_to_id[raw_label]
        return caption, label

def load_splits(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["train"], data["val"], data["test"]

def collate_batch(batch):
    captions = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return captions, labels

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
    return np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

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
    return np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

# ============================================================
# LR Scheduler (paper-aligned)
# ============================================================

def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    min_lr_ratio=0.0,
):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)

        progress = (step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return LambdaLR(optimizer, lr_lambda)

# ============================================================
# Model
# ============================================================

class Sentence_Embed(nn.Module):
    def __init__(self, model_name, output_dim):
        super().__init__()
        self.backbone = SentenceTransformer(f"sentence-transformers/{model_name}")
        self.head = nn.Linear(768, output_dim)

    def forward(self, x):
        features = self.backbone.tokenize(x)
        for k in features:
            if isinstance(features[k], torch.Tensor):
                features[k] = features[k].to(self.backbone.device)
        out = self.backbone.forward(features)
        return self.head(out["sentence_embedding"])

# ============================================================
# Evaluation
# ============================================================

def evaluate(model, loader, num_classes):
    model.eval()
    logits_all, labels_all = [], []

    with torch.no_grad():
        for captions, labels in loader:
            logits_all.append(model(captions).cpu())
            labels_all.append(labels)

    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all).numpy()

    is_binary = num_classes == 2

    if is_binary:
        probs = torch.sigmoid(logits).squeeze(1).numpy()
        auc, acc, f1 = binary_metrics(labels, probs)
        return auc, acc, f1, probs, None, labels
    else:
        probs = torch.softmax(logits, dim=1).numpy()
        auc, acc, f1, per_class = multiclass_metrics(labels, probs, num_classes)
        return auc, acc, f1, probs, per_class, labels

# ============================================================
# Training
# ============================================================

def train_with_early_stopping(
    train_loader,
    val_loader,
    num_classes,
    lr,
    weight_decay,
    epochs,
    patience,
    device,
    model_name,
    warmup_epochs=2,
    min_lr_ratio=0.0,
):
    is_binary = num_classes == 2
    output_dim = 1 if is_binary else num_classes

    model = Sentence_Embed(model_name, output_dim).to(device)
    model.backbone.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()

    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader) * warmup_epochs

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps,
        total_steps,
        min_lr_ratio,
    )

    best_val_auc = -1.0
    best_state = None
    patience_ctr = 0

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs"):
        model.train()
        for captions, labels in tqdm(train_loader, leave=False):
            labels = labels.to(device)
            optimizer.zero_grad()

            logits = model(captions)
            loss = (
                criterion(logits.squeeze(1), labels.float())
                if is_binary
                else criterion(logits, labels)
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

        val_auc, _, _, _, _, _ = evaluate(model, val_loader, num_classes)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= patience:
            break

    model.load_state_dict(best_state)
    return model

# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--splits_json", required=True)
    p.add_argument("--sentence_model", default="sentence-t5-base")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--weight_decay", type=float, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--min_lr_ratio", type=float, default=0.0)
    return p.parse_args()

# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.out_dir).mkdir(exist_ok=True)

    train_s, val_s, test_s = load_splits(args.splits_json)
    num_classes, label_to_id = infer_num_classes(train_s)

    tqdm.write(f"🧠 Detected {num_classes} classes")
    tqdm.write(f"🔖 Label mapping: {label_to_id}")

    train_loader = DataLoader(
        CaptionLabelDataset(train_s, label_to_id),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        CaptionLabelDataset(val_s, label_to_id),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        CaptionLabelDataset(test_s, label_to_id),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    csv_path = Path(args.out_dir) / "sentence_t5_results.csv"

    fieldnames = [
        "seed", "lr", "weight_decay", "warmup_epochs",
        "test_auc", "test_accuracy", "test_f1",
        "ci_low", "ci_high", "mst3_auc",
    ]
    if num_classes > 2:
        fieldnames += [f"class_{i}_auc" for i in range(num_classes)]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for seed in args.seeds:
            set_seed(seed)

            model = train_with_early_stopping(
                train_loader,
                val_loader,
                num_classes,
                args.lr,
                args.weight_decay,
                args.epochs,
                args.patience,
                device,
                args.sentence_model,
                args.warmup_epochs,
                args.min_lr_ratio,
            )

            auc, acc, f1, scores, per_class, labels = evaluate(
                model, test_loader, num_classes
            )

            ci_low = ci_high = None
            if args.bootstrap:
                if num_classes == 2:
                    _, ci_low, ci_high = bootstrap_auc_binary(y_true=labels, scores=scores, seed=seed)
                else:
                    _, ci_low, ci_high = bootstrap_auc_multiclass(y_true=labels, probs=scores, seed=seed)

            row = {
                "seed": seed,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "warmup_epochs": args.warmup_epochs,
                "test_auc": auc,
                "test_accuracy": acc,
                "test_f1": f1,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }

            if per_class:
                row.update(per_class)

            writer.writerow(row)
            f.flush()

            torch.save(
                model.state_dict(),
                Path(args.out_dir) / f"sentence_t5_seed_{seed}.pt",
            )

    tqdm.write(f"✓ Results saved to {csv_path}")

if __name__ == "__main__":
    main()

# Example CLI:
# python SentenceT5_Probe_PaperAligned.py.py --splits_json test_sentenceT5_gender.json --task gender --seeds 0 1 2 3 4 --bootstrap --out_dir out_sentence_t5_gender --lr 2e-5 --weight_decay 0.01
# python SentenceT5_Probe_PaperAligned.py.py --splits_json splits_skintone_captions.json --task skintone --seeds 0 1 2 3 4 --bootstrap --out_dir out_sentence_t5_skintone
#
# Optional probe-only ablation:
# python SentenceT5_Probe_PaperAligned.py.py --splits_json splits_gender_captions.json --task gender --freeze_backbone --out_dir out_sentence_t5_gender_frozen


# python SentenceT5_Probe_PaperAligned.py ^
#     --splits_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\7_DatasetPreparation\UniversalSplits\DatasetClassificationCoco\coco_splits_face_combined_stratified_captions.json" ^
#     --seeds 0 ^
#     --bootstrap ^
#     --out_dir outputs_dataset_classification\\CocoReLaionSDCaptions ^
#     --lr 2e-5 ^
#     --weight_decay 0.01



# ============================================================
# Example CLI (updated with new args)
# ============================================================

# Basic usage (with paper's default warmup):
# python SentenceT5_Probe_PaperAligned.py ^
#     --splits_json test_sentenceT5_gender.json ^
#     --seeds 0 ^
#     --bootstrap ^
#     --out_dir out_sentence_t5_gender ^
#     --lr 2e-5 ^
#     --weight_decay 0.01


# python SentenceT5_Probe_PaperAligned.py ^
#     --splits_json test_sentenceT5_skintone.json ^
#     --seeds 0 ^
#     --out_dir out_sentence_t5_skintone ^
#     --lr 2e-5 ^
#     --weight_decay 0.01

# Custom warmup (if you want to experiment):
# python SentenceT5_Probe_Aligned.py \
#     --splits_json splits_skintone_captions.json \
#     --task skintone \
#     --seeds 0 1 2 3 4 \
#     --bootstrap \
#     --out_dir out_sentence_t5_skintone \
#     --lr 1e-4 \
#     --weight_decay 0.01 \
#     --warmup_epochs 3 \
#     --min_lr_ratio 0.001

# No warmup (to compare):
# python SentenceT5_Probe_Aligned.py \
#     --splits_json test_sentenceT5_gender.json \
#     --task gender \
#     --lr 2e-5 \
#     --weight_decay 0.01 \
#     --warmup_epochs 0