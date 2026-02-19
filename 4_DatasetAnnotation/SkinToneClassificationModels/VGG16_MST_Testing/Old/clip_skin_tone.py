# """
# CLIP Skin Tone Classification - HuggingFace Version
# ====================================================

# UPDATED: Now supports single CSV with person-level stratified splitting
# (same logic as VGG16 training script)

# Usage:
#     # Single CSV (will auto-split)
#     python clip_hf_skin_tone_v2.py \
#         --mode linear_probe \
#         --csv_path "annotations_with_dark.csv" \
#         --image_dir "F:/Thesis/CasualConversationv2_Dataset/Segmented_CCV2" \
#         --label_mapping "label_mapping_4class_optimal.json" \
#         --val_ratio 0.2 \
#         --epochs 20 \
#         --save_dir "Models/CLIP_4Class"
# """
# import os
# os.environ["TRANSFORMERS_NO_TF"] = "1"

# import argparse
# import json
# from pathlib import Path
# from collections import Counter
# from typing import Optional

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# from tqdm import tqdm

# # HuggingFace imports
# from transformers import CLIPProcessor, CLIPModel
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     accuracy_score, 
#     precision_recall_fscore_support,
#     confusion_matrix,
#     classification_report
# )


# ##############################################################
# # REPRODUCIBILITY
# ##############################################################

# def set_seed(seed=42):
#     import random
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# set_seed()


# ##############################################################
# # LABEL MAPPER
# ##############################################################

# class LabelMapper:
#     """Handles flexible label mapping from MST values to class indices"""
#     def __init__(self, mapping_config=None):
#         if mapping_config is None:
#             self.mapping = {str(i): i-1 for i in range(1, 11)}
#             self.num_classes = 10
#         elif isinstance(mapping_config, (str, Path)):
#             with open(mapping_config, 'r') as f:
#                 config = json.load(f)
#             self._parse_config(config)
#         elif isinstance(mapping_config, dict):
#             self._parse_config(mapping_config)
    
#     def _parse_config(self, config):
#         if "label_mapping" in config:
#             self.mapping = {str(k): int(v) for k, v in config["label_mapping"].items()}
#             self.num_classes = config.get("num_classes", max(self.mapping.values()) + 1)
#         elif "bins" in config:
#             self.mapping = {}
#             for bin_def in config["bins"]:
#                 for mst in range(bin_def["range"][0], bin_def["range"][1] + 1):
#                     self.mapping[str(mst)] = bin_def["class"]
#             self.num_classes = config.get("num_classes", max(self.mapping.values()) + 1)
    
#     def map_mst_to_class(self, mst_value):
#         return self.mapping[str(int(mst_value))]
    
#     def get_class_name(self, class_idx):
#         mst_values = [int(k) for k, v in self.mapping.items() if v == class_idx]
#         mst_values.sort()
        
#         if len(mst_values) == 1:
#             return f"MST{mst_values[0]}"
#         elif mst_values == list(range(mst_values[0], mst_values[-1] + 1)):
#             return f"MST{mst_values[0]}-{mst_values[-1]}"
#         else:
#             return f"MST{','.join(map(str, mst_values))}"


# ##############################################################
# # DATA LOADING AND SPLITTING
# ##############################################################

# def load_and_split_data(csv_path, label_mapper, val_ratio=0.2, 
#                         max_samples_per_class=None, save_split_path=None):
#     """
#     Load CSV and perform person-level stratified split.
    
#     Same logic as VGG16 script.
#     """
#     df = pd.read_csv(csv_path).dropna()
#     print(f"[INFO] Loaded CSV with columns: {list(df.columns)}")
    
#     # Auto-detect column names
#     filename_col = None
#     label_col = None
#     person_col = None
    
#     for col in df.columns:
#         col_lower = col.lower()
#         if filename_col is None and any(x in col_lower for x in ['filename', 'cropped', 'image', 'file']):
#             if 'cropped' in col_lower:
#                 filename_col = col
#             elif filename_col is None or 'filename' in col_lower:
#                 filename_col = col
#         if label_col is None and any(x in col_lower for x in ['label', 'mst', 'score']):
#             label_col = col
#         if person_col is None and any(x in col_lower for x in ['person', 'subject', 'user', 'id']):
#             person_col = col
    
#     if filename_col is None:
#         filename_col = df.columns[0]
#         print(f"[WARN] Using first column as filename: {filename_col}")
#     if label_col is None:
#         label_col = df.columns[1] if len(df.columns) > 1 else None
#     if person_col is None:
#         person_col = df.columns[2] if len(df.columns) > 2 else None
    
#     if label_col is None or person_col is None:
#         raise ValueError(f"CSV must have at least filename, label, and person_id columns")
    
#     print(f"[INFO] Using columns: filename='{filename_col}', label='{label_col}', person_id='{person_col}'")
    
#     df = df[[filename_col, label_col, person_col]].copy()
#     df.columns = ["filename", "label", "person_id"]
#     df["person_id"] = df["person_id"].astype(str)
    
#     # Apply per-class sampling if requested
#     if max_samples_per_class is not None:
#         print(f"\n[Balancing] Limiting to {max_samples_per_class} samples per MST class...")
        
#         df['mst_int'] = df['label'].astype(float).round().astype(int)
#         df['mst_int'] = np.clip(df['mst_int'], 1, 10)
        
#         balanced_dfs = []
#         for mst in range(1, 11):
#             class_df = df[df['mst_int'] == mst].copy()
#             n_original = len(class_df)
            
#             if n_original == 0:
#                 continue
            
#             if n_original <= max_samples_per_class:
#                 balanced_dfs.append(class_df)
#                 print(f"  MST {mst:2d}: {n_original:6d} (kept all)")
#             else:
#                 # Sample persons (not images!)
#                 persons = class_df['person_id'].unique()
#                 n_persons = len(persons)
                
#                 images_per_person = n_original / n_persons
#                 target_persons = int(max_samples_per_class / images_per_person)
#                 target_persons = max(1, min(target_persons, n_persons))
                
#                 selected_persons = np.random.choice(persons, size=target_persons, replace=False)
#                 sampled_df = class_df[class_df['person_id'].isin(selected_persons)]
                
#                 balanced_dfs.append(sampled_df)
#                 print(f"  MST {mst:2d}: {n_original:6d} → {len(sampled_df):6d} "
#                       f"({target_persons}/{n_persons} persons)")
        
#         df = pd.concat(balanced_dfs, ignore_index=True)
#         df = df.drop(columns=['mst_int'])
        
#         print(f"\n[INFO] Balanced dataset: {len(df):,} images\n")
    
#     # Person-level stratified split
#     person_labels = df.groupby("person_id").agg({"label": "first"}).reset_index()
#     person_labels["label_int"] = person_labels["label"].astype(float).round().astype(int)
#     person_labels["label_int"] = np.clip(person_labels["label_int"], 1, 10)
    
#     labels_per_person = person_labels["label_int"].apply(
#         lambda x: label_mapper.map_mst_to_class(x)
#     ).values
    
#     print(f"[INFO] Found {len(person_labels)} unique persons")
    
#     # Person distribution
#     person_counts = Counter(labels_per_person)
#     print(f"\n[INFO] Person distribution across {label_mapper.num_classes} classes:")
#     for class_idx in sorted(person_counts.keys()):
#         class_name = label_mapper.get_class_name(class_idx)
#         print(f"  Class {class_idx} ({class_name}): {person_counts[class_idx]} persons")
#     print()
    
#     # Stratified split by person
#     train_persons, val_persons = train_test_split(
#         person_labels["person_id"].values,
#         test_size=val_ratio,
#         shuffle=True,
#         stratify=labels_per_person,
#         random_state=42
#     )
    
#     # Save split info
#     if save_split_path:
#         split_info = {
#             "train_persons": train_persons.tolist(),
#             "val_persons": val_persons.tolist(),
#             "random_state": 42,
#             "val_ratio": val_ratio,
#             "max_samples_per_class": max_samples_per_class,
#         }
        
#         split_path = Path(save_split_path)
#         split_path.parent.mkdir(parents=True, exist_ok=True)
#         with open(split_path, 'w') as f:
#             json.dump(split_info, f, indent=2)
#         print(f"[INFO] Saved train/val split to {split_path}")
    
#     # Create train and val DataFrames
#     train_df = df[df["person_id"].isin(train_persons)].reset_index(drop=True)
#     val_df = df[df["person_id"].isin(val_persons)].reset_index(drop=True)
    
#     print(f"[INFO] Train: {len(train_persons)} persons ({len(train_df)} images)")
#     print(f"[INFO] Val:   {len(val_persons)} persons ({len(val_df)} images)\n")
    
#     return train_df, val_df


# ##############################################################
# # DATASET
# ##############################################################

# class SkinToneDataset(Dataset):
#     def __init__(self, df, image_dir, label_mapper, processor, mode='train'):
#         """
#         Args:
#             df: DataFrame with columns [filename, label, person_id]
#             image_dir: Path to image directory
#             label_mapper: LabelMapper instance
#             processor: CLIP processor
#             mode: 'train' or 'val'
#         """
#         self.image_dir = Path(image_dir)
#         self.label_mapper = label_mapper
#         self.processor = processor
#         self.mode = mode
        
#         # Filter valid samples
#         self.samples = []
#         for _, row in df.iterrows():
#             img_path = self.image_dir / row['filename']
            
#             # Convert label to MST
#             try:
#                 mst = int(float(row['label']))
#             except:
#                 continue
            
#             if img_path.exists() and 1 <= mst <= 10:
#                 class_idx = label_mapper.map_mst_to_class(mst)
#                 self.samples.append((str(img_path), class_idx))
        
#         print(f"[{mode}] Loaded {len(self.samples)} samples")
        
#         # Print distribution
#         class_counts = Counter([c for _, c in self.samples])
#         for class_idx in range(label_mapper.num_classes):
#             count = class_counts.get(class_idx, 0)
#             class_name = label_mapper.get_class_name(class_idx)
#             print(f"  {class_name}: {count} images")
    
#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, idx):
#         img_path, class_idx = self.samples[idx]
        
#         try:
#             image = Image.open(img_path).convert("RGB")
#             inputs = self.processor(images=image, return_tensors="pt")
#             pixel_values = inputs['pixel_values'].squeeze(0)
#             return pixel_values, class_idx
#         except Exception as e:
#             print(f"[WARN] Failed to load {img_path}: {e}")
#             return self.__getitem__((idx + 1) % len(self))


# ##############################################################
# # CLIP CLASSIFIER
# ##############################################################

# class CLIPSkinToneClassifier(nn.Module):
#     """CLIP-based skin tone classifier"""
#     def __init__(self, model_name, num_classes, mode='linear_probe'):
#         super().__init__()
#         self.model_name = model_name
#         self.num_classes = num_classes
#         self.mode = mode
        
#         # Load CLIP
#         self.clip_model = CLIPModel.from_pretrained(model_name)
#         self.embed_dim = self.clip_model.config.projection_dim
        
#         # Classification head
#         self.classifier = nn.Linear(self.embed_dim, num_classes)
        
#         # Freeze CLIP if linear probe
#         if mode == 'linear_probe':
#             for param in self.clip_model.parameters():
#                 param.requires_grad = False
#             print(f"[CLIP] Frozen CLIP weights (linear probe mode)")
#         else:
#             print(f"[CLIP] All weights trainable (fine-tune mode)")
    
#     def forward(self, pixel_values):
#         # Extract image features
#         vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
#         image_embeds = vision_outputs.pooler_output
#         image_embeds = self.clip_model.visual_projection(image_embeds)
#         image_embeds = F.normalize(image_embeds, dim=-1)
        
#         # Classify
#         logits = self.classifier(image_embeds)
#         return logits


# ##############################################################
# # TRAINING
# ##############################################################

# # def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
# #     model.train()
    
# #     total_loss = 0
# #     correct = 0
# #     total = 0
    
# #     pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
# #     for pixel_values, targets in pbar:
# #         pixel_values = pixel_values.to(device)
# #         targets = targets.to(device)
        
# #         logits = model(pixel_values)
# #         loss = criterion(logits, targets)
        
# #         optimizer.zero_grad()
# #         loss.backward()
# #         optimizer.step()
        
# #         total_loss += loss.item()
# #         preds = logits.argmax(dim=-1)
# #         correct += (preds == targets).sum().item()
# #         total += targets.size(0)
        
# #         pbar.set_postfix({
# #             'loss': f'{total_loss / (pbar.n + 1):.4f}',
# #             'acc': f'{100.0 * correct / total:.2f}%'
# #         })
    
# #     return total_loss / len(train_loader), 100.0 * correct / total


# # def evaluate(model, val_loader, criterion, device):
# #     model.eval()
    
# #     total_loss = 0
# #     correct = 0
# #     total = 0
    
# #     with torch.no_grad():
# #         for pixel_values, targets in tqdm(val_loader, desc="Evaluating"):
# #             pixel_values = pixel_values.to(device)
# #             targets = targets.to(device)
            
# #             logits = model(pixel_values)
# #             loss = criterion(logits, targets)
            
# #             total_loss += loss.item()
# #             preds = logits.argmax(dim=-1)
# #             correct += (preds == targets).sum().item()
# #             total += targets.size(0)
    
# #     return total_loss / len(val_loader), 100.0 * correct / total

# def evaluate(model, val_loader, criterion, device, label_mapper=None, show_detailed=True):
#     """
#     Evaluate model with detailed per-class metrics.
    
#     Args:
#         model: Model to evaluate
#         val_loader: Validation DataLoader
#         criterion: Loss function
#         device: torch device
#         label_mapper: LabelMapper instance for class names
#         show_detailed: Whether to print detailed per-class metrics
    
#     Returns:
#         avg_loss: Average loss
#         overall_acc: Overall accuracy (0-100)
#         metrics_dict: Dict with detailed metrics
#     """
#     model.eval()
    
#     total_loss = 0
#     all_preds = []
#     all_targets = []
    
#     with torch.no_grad():
#         for pixel_values, targets in tqdm(val_loader, desc="Evaluating", leave=False):
#             pixel_values = pixel_values.to(device)
#             targets = targets.to(device)
            
#             logits = model(pixel_values)
#             loss = criterion(logits, targets)
            
#             total_loss += loss.item()
#             preds = logits.argmax(dim=-1)
            
#             all_preds.extend(preds.cpu().numpy())
#             all_targets.extend(targets.cpu().numpy())
    
#     all_preds = np.array(all_preds)
#     all_targets = np.array(all_targets)
    
#     # Compute metrics
#     avg_loss = total_loss / len(val_loader)
#     overall_acc = 100.0 * accuracy_score(all_targets, all_preds)
    
#     # Per-class metrics
#     precision, recall, f1, support = precision_recall_fscore_support(
#         all_targets, all_preds, average=None, zero_division=0
#     )
    
#     # Macro/weighted averages
#     macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
#         all_targets, all_preds, average='macro', zero_division=0
#     )
#     weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
#         all_targets, all_preds, average='weighted', zero_division=0
#     )
    
#     # Confusion matrix
#     cm = confusion_matrix(all_targets, all_preds)
    
#     # Store metrics
#     metrics_dict = {
#         'overall_acc': overall_acc,
#         'macro_f1': macro_f1 * 100,
#         'weighted_f1': weighted_f1 * 100,
#         'per_class': {},
#         'confusion_matrix': cm
#     }
    
#     num_classes = len(precision)
#     for class_idx in range(num_classes):
#         class_name = label_mapper.get_class_name(class_idx) if label_mapper else f"Class {class_idx}"
#         metrics_dict['per_class'][class_idx] = {
#             'name': class_name,
#             'accuracy': 100.0 * (cm[class_idx, class_idx] / support[class_idx]) if support[class_idx] > 0 else 0,
#             'precision': precision[class_idx] * 100,
#             'recall': recall[class_idx] * 100,
#             'f1': f1[class_idx] * 100,
#             'support': int(support[class_idx])
#         }
    
#     # Print detailed metrics if requested
#     if show_detailed:
#         print("\n" + "="*70)
#         print("DETAILED VALIDATION METRICS")
#         print("="*70)
        
#         print(f"\nOverall Accuracy: {overall_acc:.2f}%")
#         print(f"Macro F1:         {macro_f1*100:.2f}%")
#         print(f"Weighted F1:      {weighted_f1*100:.2f}%")
        
#         print("\nPer-Class Metrics:")
#         print(f"{'Class':<12} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Support':>9}")
#         print("-" * 70)
        
#         for class_idx in range(num_classes):
#             m = metrics_dict['per_class'][class_idx]
#             print(f"{m['name']:<12} {m['accuracy']:>6.1f}% {m['precision']:>6.1f}% "
#                   f"{m['recall']:>6.1f}% {m['f1']:>6.1f}% {m['support']:>9d}")
        
#         # Confusion matrix
#         print("\nConfusion Matrix:")
#         print("(Rows = True, Cols = Predicted)")
        
#         # Header
#         header = "True \\ Pred  "
#         for i in range(num_classes):
#             name = label_mapper.get_class_name(i) if label_mapper else f"C{i}"
#             header += f"{name:>10s} "
#         print(header)
#         print("-" * (15 + num_classes * 11))
        
#         # Rows
#         for i in range(num_classes):
#             name = label_mapper.get_class_name(i) if label_mapper else f"Class {i}"
#             row = f"{name:<13}"
#             for j in range(num_classes):
#                 if i == j:
#                     # Diagonal (correct predictions) - bold with ✓
#                     row += f"{cm[i,j]:>9d}✓ "
#                 else:
#                     # Off-diagonal (errors)
#                     if cm[i,j] > 0:
#                         row += f"{cm[i,j]:>10d} "
#                     else:
#                         row += f"{'':>10s} "
#             print(row)
        
#         print("="*70 + "\n")
    
#     return avg_loss, overall_acc, metrics_dict


# def train_epoch(model, train_loader, optimizer, criterion, device, epoch, 
#                 show_batch_metrics=False):
#     """
#     Train for one epoch.
    
#     Args:
#         show_batch_metrics: If True, show metrics every 100 batches
#     """
#     model.train()
    
#     total_loss = 0
#     correct = 0
#     total = 0
    
#     pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
#     for batch_idx, (pixel_values, targets) in enumerate(pbar):
#         pixel_values = pixel_values.to(device)
#         targets = targets.to(device)
        
#         logits = model(pixel_values)
#         loss = criterion(logits, targets)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         total_loss += loss.item()
#         preds = logits.argmax(dim=-1)
#         correct += (preds == targets).sum().item()
#         total += targets.size(0)
        
#         # Update progress bar
#         pbar.set_postfix({
#             'loss': f'{total_loss / (batch_idx + 1):.4f}',
#             'acc': f'{100.0 * correct / total:.2f}%'
#         })
        
#         # Optional: Show metrics every 100 batches
#         if show_batch_metrics and (batch_idx + 1) % 100 == 0:
#             print(f"\n  Batch {batch_idx+1}/{len(train_loader)}: "
#                   f"Loss={total_loss/(batch_idx+1):.4f}, Acc={100.0*correct/total:.2f}%")
    
#     return total_loss / len(train_loader), 100.0 * correct / total

# ##############################################################
# # MAIN
# ##############################################################

# def main(args):
#     device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
#     print(f"[Device] {device}\n")
    
#     # Load CLIP processor
#     print(f"[CLIP] Loading {args.clip_model}...")
#     processor = CLIPProcessor.from_pretrained(args.clip_model)
#     print(f"[CLIP] Loaded processor\n")
    
#     # Load label mapper
#     label_mapper = LabelMapper(args.label_mapping)
#     print(f"[Label Mapper] {label_mapper.num_classes} classes\n")
    
#     # Load and split data
#     print("="*70)
#     print("DATA LOADING AND SPLITTING")
#     print("="*70 + "\n")
    
#     train_df, val_df = load_and_split_data(
#         csv_path=args.csv_path,
#         label_mapper=label_mapper,
#         val_ratio=args.val_ratio,
#         max_samples_per_class=args.max_samples_per_class,
#         save_split_path=args.save_split_path
#     )
    
#     # Create datasets
#     train_dataset = SkinToneDataset(train_df, args.image_dir, label_mapper, processor, mode='train')
#     val_dataset = SkinToneDataset(val_df, args.image_dir, label_mapper, processor, mode='val')
    
#     train_loader = DataLoader(
#         train_dataset, batch_size=args.batch_size, shuffle=True,
#         num_workers=4, pin_memory=True
#     )
#     val_loader = DataLoader(
#         val_dataset, batch_size=args.batch_size, shuffle=False,
#         num_workers=4, pin_memory=True
#     )
    
#     print("\n" + "="*70)
#     print("MODEL TRAINING")
#     print("="*70 + "\n")
    
#     # Create model
#     model = CLIPSkinToneClassifier(
#         args.clip_model, label_mapper.num_classes, mode=args.mode
#     ).to(device)
    
#     # Loss and optimizer
#     criterion = nn.CrossEntropyLoss()
    
#     if args.mode == 'linear_probe':
#         optimizer = torch.optim.AdamW(
#             model.classifier.parameters(),
#             lr=args.lr,
#             weight_decay=args.weight_decay
#         )
#     else:
#         optimizer = torch.optim.AdamW(
#             model.parameters(),
#             lr=args.lr,
#             weight_decay=args.weight_decay
#         )
    
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         optimizer, T_max=args.epochs
#     )
    
#     # Training loop
#     best_val_acc = 0.0
#     patience_counter = 0
        
#     for epoch in range(1, args.epochs + 1):
#         train_loss, train_acc = train_epoch(
#             model, train_loader, optimizer, criterion, device, epoch
#         )
        
#         # Detailed validation every epoch
#         val_loss, val_acc, val_metrics = evaluate(
#             model, val_loader, criterion, device, 
#             label_mapper=label_mapper,
#             show_detailed=True  # Show detailed metrics every epoch
#         )
        
#         scheduler.step()
        
#         print(f"\nEpoch {epoch}/{args.epochs}")
#         print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
#         print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
#         # Save best model
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             patience_counter = 0
            
#             if args.save_dir:
#                 save_path = Path(args.save_dir) / f"clip_hf_best.pth"
#                 save_path.parent.mkdir(parents=True, exist_ok=True)
#                 torch.save({
#                     'epoch': epoch,
#                     'model_state_dict': model.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'best_val_acc': best_val_acc,
#                     'clip_model': args.clip_model,
#                     'label_mapping': args.label_mapping,
#                 }, save_path)
#                 print(f"  [Saved] {save_path}")
#         else:
#             patience_counter += 1
        
#         # Early stopping
#         if patience_counter >= args.patience:
#             print(f"\n[Early Stopping] No improvement for {args.patience} epochs")
#             break
    
#     print(f"\n[Training Complete] Best Val Acc: {best_val_acc:.2f}%")
    
#     # Final evaluation with detailed metrics
#     print("\n" + "="*70)
#     print("FINAL EVALUATION ON VALIDATION SET")
#     print("="*70)
    
#     final_loss, final_acc, final_metrics = evaluate(
#         model, val_loader, criterion, device,
#         label_mapper=label_mapper,
#         show_detailed=True
#     )


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="CLIP skin tone classification (single CSV)")
    
#     # Mode
#     parser.add_argument("--mode", choices=['linear_probe', 'finetune'],
#                        default='linear_probe', help="Training mode")
    
#     # Model
#     parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32",
#                        help="HuggingFace CLIP model name")
    
#     # Data
#     parser.add_argument("--csv_path", required=True,
#                        help="Path to CSV with [filename, label/mst, person_id]")
#     parser.add_argument("--image_dir", required=True,
#                        help="Path to image directory")
#     parser.add_argument("--label_mapping", required=True,
#                        help="Label mapping JSON")
#     parser.add_argument("--val_ratio", type=float, default=0.2,
#                        help="Validation split ratio")
#     parser.add_argument("--max_samples_per_class", type=int, default=None,
#                        help="Max samples per MST class (for balancing)")
#     parser.add_argument("--save_split_path", default="train_val_split.json",
#                        help="Path to save train/val split info")
    
#     # Training
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--epochs", type=int, default=20)
#     parser.add_argument("--lr", type=float, default=1e-4)
#     parser.add_argument("--weight_decay", type=float, default=0.01)
#     parser.add_argument("--patience", type=int, default=5)
    
#     # Output
#     parser.add_argument("--save_dir", required=True,
#                        help="Directory to save model")
    
#     # Hardware
#     parser.add_argument("--gpu", type=int, default=0)
    
#     args = parser.parse_args()

#     main(args)

"""
CLIP Skin Tone Classification - HuggingFace Version
====================================================

UPDATED: Now supports single CSV with person-level stratified splitting
(same logic as VGG16 training script)

Usage:
    # Single CSV (will auto-split)
    python clip_hf_skin_tone_v2.py \
        --mode linear_probe \
        --csv_path "annotations_with_dark.csv" \
        --image_dir "F:/Thesis/CasualConversationv2_Dataset/Segmented_CCV2" \
        --label_mapping "label_mapping_4class_optimal.json" \
        --val_ratio 0.2 \
        --epochs 20 \
        --save_dir "Models/CLIP_4Class"
"""

import argparse
import json
from pathlib import Path
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# HuggingFace imports
from transformers import CLIPProcessor, CLIPModel
from sklearn.model_selection import train_test_split


##############################################################
# REPRODUCIBILITY
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
# LABEL MAPPER
##############################################################

class LabelMapper:
    """Handles flexible label mapping from MST values to class indices"""
    def __init__(self, mapping_config=None):
        if mapping_config is None:
            self.mapping = {str(i): i-1 for i in range(1, 11)}
            self.num_classes = 10
        elif isinstance(mapping_config, (str, Path)):
            with open(mapping_config, 'r') as f:
                config = json.load(f)
            self._parse_config(config)
        elif isinstance(mapping_config, dict):
            self._parse_config(mapping_config)
    
    def _parse_config(self, config):
        if "label_mapping" in config:
            self.mapping = {str(k): int(v) for k, v in config["label_mapping"].items()}
            self.num_classes = config.get("num_classes", max(self.mapping.values()) + 1)
        elif "bins" in config:
            self.mapping = {}
            for bin_def in config["bins"]:
                for mst in range(bin_def["range"][0], bin_def["range"][1] + 1):
                    self.mapping[str(mst)] = bin_def["class"]
            self.num_classes = config.get("num_classes", max(self.mapping.values()) + 1)
    
    def map_mst_to_class(self, mst_value):
        return self.mapping[str(int(mst_value))]
    
    def get_class_name(self, class_idx):
        mst_values = [int(k) for k, v in self.mapping.items() if v == class_idx]
        mst_values.sort()
        
        if len(mst_values) == 1:
            return f"MST{mst_values[0]}"
        elif mst_values == list(range(mst_values[0], mst_values[-1] + 1)):
            return f"MST{mst_values[0]}-{mst_values[-1]}"
        else:
            return f"MST{','.join(map(str, mst_values))}"


##############################################################
# DATA LOADING AND SPLITTING
##############################################################

def load_and_split_data(csv_path, label_mapper, val_ratio=0.2, 
                        max_samples_per_class=None, save_split_path=None):
    """
    Load CSV and perform person-level stratified split.
    
    Same logic as VGG16 script.
    """
    df = pd.read_csv(csv_path).dropna()
    print(f"[INFO] Loaded CSV with columns: {list(df.columns)}")
    
    # Auto-detect column names
    filename_col = None
    label_col = None
    person_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if filename_col is None and any(x in col_lower for x in ['filename', 'cropped', 'image', 'file']):
            if 'cropped' in col_lower:
                filename_col = col
            elif filename_col is None or 'filename' in col_lower:
                filename_col = col
        if label_col is None and any(x in col_lower for x in ['label', 'mst', 'score']):
            label_col = col
        if person_col is None and any(x in col_lower for x in ['person', 'subject', 'user', 'id']):
            person_col = col
    
    if filename_col is None:
        filename_col = df.columns[0]
        print(f"[WARN] Using first column as filename: {filename_col}")
    if label_col is None:
        label_col = df.columns[1] if len(df.columns) > 1 else None
    if person_col is None:
        person_col = df.columns[2] if len(df.columns) > 2 else None
    
    if label_col is None or person_col is None:
        raise ValueError(f"CSV must have at least filename, label, and person_id columns")
    
    print(f"[INFO] Using columns: filename='{filename_col}', label='{label_col}', person_id='{person_col}'")
    
    df = df[[filename_col, label_col, person_col]].copy()
    df.columns = ["filename", "label", "person_id"]
    df["person_id"] = df["person_id"].astype(str)
    
    # Apply per-class sampling if requested
    if max_samples_per_class is not None:
        print(f"\n[Balancing] Limiting to {max_samples_per_class} samples per MST class...")
        
        df['mst_int'] = df['label'].astype(float).round().astype(int)
        df['mst_int'] = np.clip(df['mst_int'], 1, 10)
        
        balanced_dfs = []
        for mst in range(1, 11):
            class_df = df[df['mst_int'] == mst].copy()
            n_original = len(class_df)
            
            if n_original == 0:
                continue
            
            if n_original <= max_samples_per_class:
                balanced_dfs.append(class_df)
                print(f"  MST {mst:2d}: {n_original:6d} (kept all)")
            else:
                # Sample persons (not images!)
                persons = class_df['person_id'].unique()
                n_persons = len(persons)
                
                images_per_person = n_original / n_persons
                target_persons = int(max_samples_per_class / images_per_person)
                target_persons = max(1, min(target_persons, n_persons))
                
                selected_persons = np.random.choice(persons, size=target_persons, replace=False)
                sampled_df = class_df[class_df['person_id'].isin(selected_persons)]
                
                balanced_dfs.append(sampled_df)
                print(f"  MST {mst:2d}: {n_original:6d} → {len(sampled_df):6d} "
                      f"({target_persons}/{n_persons} persons)")
        
        df = pd.concat(balanced_dfs, ignore_index=True)
        df = df.drop(columns=['mst_int'])
        
        print(f"\n[INFO] Balanced dataset: {len(df):,} images\n")
    
    # Person-level stratified split
    person_labels = df.groupby("person_id").agg({"label": "first"}).reset_index()
    person_labels["label_int"] = person_labels["label"].astype(float).round().astype(int)
    person_labels["label_int"] = np.clip(person_labels["label_int"], 1, 10)
    
    labels_per_person = person_labels["label_int"].apply(
        lambda x: label_mapper.map_mst_to_class(x)
    ).values
    
    print(f"[INFO] Found {len(person_labels)} unique persons")
    
    # Person distribution
    person_counts = Counter(labels_per_person)
    print(f"\n[INFO] Person distribution across {label_mapper.num_classes} classes:")
    for class_idx in sorted(person_counts.keys()):
        class_name = label_mapper.get_class_name(class_idx)
        print(f"  Class {class_idx} ({class_name}): {person_counts[class_idx]} persons")
    print()
    
    # Stratified split by person
    train_persons, val_persons = train_test_split(
        person_labels["person_id"].values,
        test_size=val_ratio,
        shuffle=True,
        stratify=labels_per_person,
        random_state=42
    )
    
    # Save split info
    if save_split_path:
        split_info = {
            "train_persons": train_persons.tolist(),
            "val_persons": val_persons.tolist(),
            "random_state": 42,
            "val_ratio": val_ratio,
            "max_samples_per_class": max_samples_per_class,
        }
        
        split_path = Path(save_split_path)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        with open(split_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        print(f"[INFO] Saved train/val split to {split_path}")
    
    # Create train and val DataFrames
    train_df = df[df["person_id"].isin(train_persons)].reset_index(drop=True)
    val_df = df[df["person_id"].isin(val_persons)].reset_index(drop=True)
    
    print(f"[INFO] Train: {len(train_persons)} persons ({len(train_df)} images)")
    print(f"[INFO] Val:   {len(val_persons)} persons ({len(val_df)} images)\n")
    
    return train_df, val_df


##############################################################
# DATASET
##############################################################

class SkinToneDataset(Dataset):
    def __init__(self, df, image_dir, label_mapper, processor, mode='train'):
        """
        Args:
            df: DataFrame with columns [filename, label, person_id]
            image_dir: Path to image directory
            label_mapper: LabelMapper instance
            processor: CLIP processor
            mode: 'train' or 'val'
        """
        self.image_dir = Path(image_dir)
        self.label_mapper = label_mapper
        self.processor = processor
        self.mode = mode
        
        # Filter valid samples
        self.samples = []
        for _, row in df.iterrows():
            img_path = self.image_dir / row['filename']
            
            # Convert label to MST
            try:
                mst = int(float(row['label']))
            except:
                continue
            
            if img_path.exists() and 1 <= mst <= 10:
                class_idx = label_mapper.map_mst_to_class(mst)
                self.samples.append((str(img_path), class_idx))
        
        print(f"[{mode}] Loaded {len(self.samples)} samples")
        
        # Print distribution
        class_counts = Counter([c for _, c in self.samples])
        for class_idx in range(label_mapper.num_classes):
            count = class_counts.get(class_idx, 0)
            class_name = label_mapper.get_class_name(class_idx)
            print(f"  {class_name}: {count} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)
            return pixel_values, class_idx
        except Exception as e:
            print(f"[WARN] Failed to load {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))


##############################################################
# CLIP CLASSIFIER
##############################################################

class CLIPSkinToneClassifier(nn.Module):
    """CLIP-based skin tone classifier"""
    def __init__(self, model_name, num_classes, mode='linear_probe'):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.mode = mode
        
        # Load CLIP
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.embed_dim = self.clip_model.config.projection_dim
        
        # Classification head
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        
        # Freeze CLIP if linear probe
        if mode == 'linear_probe':
            for param in self.clip_model.parameters():
                param.requires_grad = False
            print(f"[CLIP] Frozen CLIP weights (linear probe mode)")
        else:
            print(f"[CLIP] All weights trainable (fine-tune mode)")
    
    def forward(self, pixel_values):
        # Extract image features
        vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs.pooler_output
        image_embeds = self.clip_model.visual_projection(image_embeds)
        image_embeds = F.normalize(image_embeds, dim=-1)
        
        # Classify
        logits = self.classifier(image_embeds)
        return logits


##############################################################
# TRAINING
##############################################################

def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for pixel_values, targets in pbar:
        pixel_values = pixel_values.to(device)
        targets = targets.to(device)
        
        logits = model(pixel_values)
        loss = criterion(logits, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        
        pbar.set_postfix({
            'loss': f'{total_loss / (pbar.n + 1):.4f}',
            'acc': f'{100.0 * correct / total:.2f}%'
        })
    
    return total_loss / len(train_loader), 100.0 * correct / total


def evaluate(model, val_loader, criterion, device):
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for pixel_values, targets in tqdm(val_loader, desc="Evaluating"):
            pixel_values = pixel_values.to(device)
            targets = targets.to(device)
            
            logits = model(pixel_values)
            loss = criterion(logits, targets)
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    
    return total_loss / len(val_loader), 100.0 * correct / total


##############################################################
# MAIN
##############################################################

def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}\n")
    
    # Load CLIP processor
    print(f"[CLIP] Loading {args.clip_model}...")
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    print(f"[CLIP] Loaded processor\n")
    
    # Load label mapper
    label_mapper = LabelMapper(args.label_mapping)
    print(f"[Label Mapper] {label_mapper.num_classes} classes\n")
    
    # Load and split data
    print("="*70)
    print("DATA LOADING AND SPLITTING")
    print("="*70 + "\n")
    
    train_df, val_df = load_and_split_data(
        csv_path=args.csv_path,
        label_mapper=label_mapper,
        val_ratio=args.val_ratio,
        max_samples_per_class=args.max_samples_per_class,
        save_split_path=args.save_split_path
    )
    
    # Create datasets
    train_dataset = SkinToneDataset(train_df, args.image_dir, label_mapper, processor, mode='train')
    val_dataset = SkinToneDataset(val_df, args.image_dir, label_mapper, processor, mode='val')
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    print("\n" + "="*70)
    print("MODEL TRAINING")
    print("="*70 + "\n")
    
    # Create model
    model = CLIPSkinToneClassifier(
        args.clip_model, label_mapper.num_classes, mode=args.mode
    ).to(device)
    
    # Compute class weights from training data to prevent mode collapse
    train_classes = [target for _, target in train_dataset]
    class_counts = Counter(train_classes)
    
    # Inverse frequency weights
    total = len(train_classes)
    class_weights = []
    for i in range(label_mapper.num_classes):
        count = class_counts.get(i, 1)
        weight = total / (label_mapper.num_classes * count)
        class_weights.append(weight)
    
    # Smooth weights to prevent extreme values
    class_weights = np.array(class_weights)
    class_weights = np.sqrt(class_weights)  # Square root dampening
    class_weights = np.clip(class_weights, 0.5, 2.0)  # Clip between 0.5 and 2.0
    
    class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    print("\n[Class Weights]")
    for i in range(label_mapper.num_classes):
        print(f"  {label_mapper.get_class_name(i)}: {class_weights[i]:.3f}")
    print()
    
    # Initialize classifier bias to match class frequencies (prevents mode collapse)
    with torch.no_grad():
        for i in range(label_mapper.num_classes):
            count = class_counts.get(i, 1)
            # Set bias = log(frequency) for balanced initial predictions
            model.classifier.bias[i] = np.log(count / total)
    
    print("[Classifier Bias Initialized]\n")
    
    # Weighted loss (critical for preventing mode collapse!)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    
    if args.mode == 'linear_probe':
        optimizer = torch.optim.AdamW(
            model.classifier.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )
        
        scheduler.step()
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            if args.save_dir:
                save_path = Path(args.save_dir) / f"clip_hf_best.pth"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'clip_model': args.clip_model,
                    'label_mapping': args.label_mapping,
                }, save_path)
                print(f"  [Saved] {save_path}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n[Early Stopping] No improvement for {args.patience} epochs")
            break
    
    print(f"\n[Training Complete] Best Val Acc: {best_val_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP skin tone classification (single CSV)")
    
    # Mode
    parser.add_argument("--mode", choices=['linear_probe', 'finetune'],
                       default='linear_probe', help="Training mode")
    
    # Model
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32",
                       help="HuggingFace CLIP model name")
    
    # Data
    parser.add_argument("--csv_path", required=True,
                       help="Path to CSV with [filename, label/mst, person_id]")
    parser.add_argument("--image_dir", required=True,
                       help="Path to image directory")
    parser.add_argument("--label_mapping", required=True,
                       help="Label mapping JSON")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                       help="Validation split ratio")
    parser.add_argument("--max_samples_per_class", type=int, default=None,
                       help="Max samples per MST class (for balancing)")
    parser.add_argument("--save_split_path", default="train_val_split.json",
                       help="Path to save train/val split info")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=5)
    
    # Output
    parser.add_argument("--save_dir", required=True,
                       help="Directory to save model")
    
    # Hardware
    parser.add_argument("--gpu", type=int, default=0)
    
    args = parser.parse_args()
    main(args)

# python clip_skin_tone.py ^
#   --mode zero-shot ^
#   --test_csv "G:\Thesis\MonkSkinTone_Dataset\Segmented_MSTE\annotations.csv" ^
#   --test_image_dir "G:\Thesis\MonkSkinTone_Dataset\Segmented_MSTE" ^
#   --label_mapping "label_mapping_5class.json" ^
#   --clip_model "openai/clip-vit-base-patch32"


# Just do this (4 hours, 80-85% accuracy):
# python clip_skin_tone.py ^
#   --mode linear_probe ^
#   --csv_path "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\\annotations_with_dark.csv" ^
#   --image_dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2" ^
#   --label_mapping "label_mapping_5class.json" ^
#   --clip_model "openai/clip-vit-base-patch32" ^
#   --val_ratio 0.2 ^
#   --batch_size 64 ^
#   --epochs 20 ^
#   --save_dir "Models/CLIP_5Class" ^
#   --gpu 0


# python clip_skin_tone.py ^
#   --mode linear_probe ^
#   --csv_path "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\\annotations_with_dark.csv" ^
#   --image_dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2" ^
#   --label_mapping "label_mapping_4class.json" ^
#   --clip_model "openai/clip-vit-base-patch32" ^
#   --val_ratio 0.2 ^
#   --batch_size 64 ^
#   --epochs 20 ^
#   --save_dir "Models/CLIP_4Class" ^
#   --gpu 0


# python clip_skin_tone.py ^
#   --mode linear_probe ^
#   --csv_path "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\\annotations_with_dark.csv" ^
#   --image_dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2" ^
#   --label_mapping "label_mapping_4class.json" ^
#   --clip_model "openai/clip-vit-base-patch32" ^
#   --val_ratio 0.2 ^
#   --batch_size 64 ^
#   --epochs 20 ^
#   --max_samples_per_class 1500 ^
#   --save_dir "Models/CLIP_4Class_Limited_Weighted" ^
#   --gpu 0