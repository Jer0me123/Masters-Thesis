"""
Sentence-T5 caption probe (paper- and repo-aligned) for demographic prediction.

Implements the SentenceTransformer + Linear head logic used in:
boyazeng/understand_bias -> models/sentence_embed.py

Features:
- Train / Val / Test protocol
- Early stopping with patience
- Best model per seed saved
- tqdm progress bars (epochs + batches)
- Live validation AUC reporting
- Final test metrics saved to CSV
"""

import argparse
import json
import random
from pathlib import Path
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
# Dataset
# ============================================================

class CaptionLabelDataset(Dataset):
    """
    Supports samples of the form:
    {
        "image": "img_001.jpg",   # optional / ignored
        "caption": "...",         # required
        "label": 0                # required
    }
    """
    def __init__(self, samples):
        self.samples = samples

        # Basic validation (fail fast if JSON is malformed)
        for i, s in enumerate(samples):
            if "caption" not in s:
                raise KeyError(f"Missing 'caption' in sample {i}")
            if "label" not in s:
                raise KeyError(f"Missing 'label' in sample {i}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        caption = s["caption"]
        label = int(s["label"])
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
# Model (repo-aligned)
# ============================================================

class Sentence_Embed(nn.Module):
    def __init__(self, model: str, num_classes: int):
        super().__init__()
        self.backbone = SentenceTransformer(f"sentence-transformers/{model}")
        self.head = nn.Linear(768, num_classes)

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

    if num_classes == 1:
        probs = torch.sigmoid(logits).numpy().reshape(-1)
        auc, acc, f1 = binary_metrics(labels, probs)
        return auc, acc, f1, probs, None, labels
    else:
        probs = torch.softmax(logits, dim=1).numpy()
        auc, acc, f1, per_class = multiclass_metrics(labels, probs, num_classes)
        return auc, acc, f1, probs, per_class, labels

# ============================================================
# Training with early stopping + progress bars
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
):
    model = Sentence_Embed(model_name, num_classes).to(device)
    model.backbone.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()

    best_val_auc = -1.0
    best_state = None
    patience_ctr = 0

    epoch_bar = tqdm(range(1, epochs + 1), desc="Epochs")

    for epoch in epoch_bar:
        model.train()
        batch_bar = tqdm(
            train_loader,
            desc=f"Train (epoch {epoch})",
            leave=False,
        )

        for captions, labels in batch_bar:
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(captions)

            loss = (
                criterion(logits.squeeze(1), labels.float())
                if num_classes == 1
                else criterion(logits, labels)
            )

            loss.backward()
            optimizer.step()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        val_auc, _, _, _, _, _ = evaluate(model, val_loader, num_classes)
        epoch_bar.set_postfix(val_auc=f"{val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
            tqdm.write(f"✓ New best model (val AUC = {val_auc:.4f})")
        else:
            patience_ctr += 1
            tqdm.write(f"✗ No improvement (patience {patience_ctr}/{patience})")

        if patience_ctr >= patience:
            tqdm.write("⏹ Early stopping triggered")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        tqdm.write("✓ Restored best validation model")

    return model

# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--splits_json", required=True)
    p.add_argument("--task", choices=["gender", "skintone"], required=True)
    p.add_argument("--sentence_model", default="sentence-t5-base")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--weight_decay", type=float, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--out_dir", default="outputs")
    return p.parse_args()

# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.out_dir).mkdir(exist_ok=True)

    num_classes = 1 if args.task == "gender" else 3
    train_s, val_s, test_s = load_splits(args.splits_json)

    train_loader = DataLoader(
        CaptionLabelDataset(train_s),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        CaptionLabelDataset(val_s),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        CaptionLabelDataset(test_s),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    csv_path = Path(args.out_dir) / f"sentence_t5_results_{args.task}.csv"
    fieldnames = [
        "task", "sentence_model", "seed",
        "lr", "weight_decay",
        "test_auc", "test_accuracy", "test_f1",
        "ci_low", "ci_high",
        "class_0_auc", "class_1_auc", "class_2_auc",
    ]

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for seed in args.seeds:
            tqdm.write(f"\n===== Seed {seed} =====")
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
            )

            tqdm.write("▶ Evaluating on test set")
            auc, acc, f1, scores, per_class, labels = evaluate(
                model, test_loader, num_classes
            )

            ci_low = ci_high = None
            if args.bootstrap:
                if num_classes == 1:
                    _, ci_low, ci_high = bootstrap_auc_binary(labels, scores, seed=seed)
                else:
                    _, ci_low, ci_high = bootstrap_auc_multiclass(labels, scores, seed=seed)

            row = {
                "task": args.task,
                "sentence_model": args.sentence_model,
                "seed": seed,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "test_auc": auc,
                "test_accuracy": acc,
                "test_f1": f1,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "class_0_auc": None,
                "class_1_auc": None,
                "class_2_auc": None,
            }

            if per_class is not None:
                for k, v in per_class.items():
                    row[k] = v

            writer.writerow(row)
            csv_file.flush()

            ckpt = Path(args.out_dir) / f"sentence_t5_{args.task}_seed_{seed}.pt"
            torch.save(model.state_dict(), ckpt)
            tqdm.write(f"✓ Saved model → {ckpt}")

    tqdm.write(f"\n✓ Results saved to {csv_path}")

if __name__ == "__main__":
    main()



# Example CLI:
# python SentenceT5_Classification.py --splits_json splits_gender_captions.json --task gender --seeds 0 1 2 3 4 --bootstrap --out_dir out_sentence_t5_gender
# python SentenceT5_Classification.py --splits_json splits_skintone_captions.json --task skintone --seeds 0 1 2 3 4 --bootstrap --out_dir out_sentence_t5_skintone
#
# Optional probe-only ablation:
# python SentenceT5_Classification.py --splits_json splits_gender_captions.json --task gender --freeze_backbone --out_dir out_sentence_t5_gender_frozen
