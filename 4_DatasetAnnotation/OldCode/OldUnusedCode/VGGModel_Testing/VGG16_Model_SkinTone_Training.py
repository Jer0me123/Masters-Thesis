# ##############################################################
# #  VGG16-LAB TRAINING PIPELINE (OPTIMIZED FOR I/O)
# #  - Fixed dataloader stalling issues
# #  - Persistent workers & prefetch factor
# #  - Pin memory for faster GPU transfer
# ##############################################################

# import argparse
# from pathlib import Path
# from collections import Counter

# from altair import param
# import numpy as np
# import pandas as pd
# from PIL import Image
# from tqdm import tqdm

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from torchvision import models, transforms

# from skimage.color import rgb2lab
# from sklearn.model_selection import train_test_split


# ##############################################################
# # 0. REPRODUCIBILITY
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
# # 1. MONK -> LAB utilities
# ##############################################################

# MONK_COLORS_RGB = np.array([
#     (246, 237, 228),
#     (243, 231, 219),
#     (247, 234, 208),
#     (243, 218, 186),
#     (215, 189, 150),
#     (160, 126, 86),
#     (130, 92, 67),
#     (96, 65, 52),
#     (58, 49, 42),
#     (41, 36, 32),
# ], dtype=np.float32)


# def _rgb_to_lab(rgb_vec):
#     rgb = rgb_vec.reshape(-1,1,1,3) / 255.0
#     lab = rgb2lab(rgb)
#     return lab.reshape(-1,3).astype(np.float32)


# MONK_COLORS_LAB = _rgb_to_lab(MONK_COLORS_RGB)


# def monk_scalar_to_lab(values):
#     labs = []
#     for v in values:
#         v = float(max(1, min(10, v)))
#         base = int(v)
#         if base >= 10:
#             labs.append(MONK_COLORS_LAB[-1])
#             continue
#         lab0 = MONK_COLORS_LAB[base-1]
#         lab1 = MONK_COLORS_LAB[base]
#         labs.append(lab0 + (v-base)*(lab1-lab0))
#     return np.stack(labs)


# def calc_l2_lab_distance(pred, target):
#     p = pred.detach().cpu().numpy() * 9 + 1
#     t = target.detach().cpu().numpy() * 9 + 1
#     return np.sqrt(np.sum((monk_scalar_to_lab(p)-monk_scalar_to_lab(t))**2, axis=1))


# ##############################################################
# #  LAB DATASET STATISTICS
# ##############################################################

# def compute_dataset_lab_stats(image_dir, csv_path):
#     df = pd.read_csv(csv_path)
#     img_names = df.iloc[:, 0].tolist()

#     print(f"[INFO] Computing LAB dataset statistics on {len(img_names)} images...")

#     resize_op = transforms.Resize((224, 224))

#     sum_lab = np.zeros(3, dtype=np.float64)
#     sum_sq_lab = np.zeros(3, dtype=np.float64)
#     total_pixels = 0

#     for name in tqdm(img_names, desc="LAB stats"):
#         fpath = Path(image_dir) / name

#         try:
#             img = Image.open(fpath).convert("RGB")
#             img = resize_op(img)

#             rgb = np.asarray(img).astype(np.float32) / 255.0
#             lab = rgb2lab(rgb).astype(np.float64) 

#             lab_flat = lab.reshape(-1, 3)
#             sum_lab += lab_flat.sum(axis=0)
#             sum_sq_lab += (lab_flat ** 2).sum(axis=0)
#             total_pixels += lab_flat.shape[0]

#         except Exception as e:
#             print(f"[WARN] Could not read {fpath}: {e}")

#     mean = sum_lab / total_pixels
#     var = (sum_sq_lab / total_pixels) - (mean ** 2)
#     std = np.sqrt(var)

#     print("\n========== LAB DATASET STATISTICS ==========")
#     print("LAB_MEAN =", mean.tolist())
#     print("LAB_STD  =", std.tolist())
#     print("============================================\n")

#     return mean.astype(np.float32), std.astype(np.float32)


# ##############################################################
# # 2. LAB Transform
# ##############################################################

# class RGBToLABTensorTransform:
#     first_call = True

#     def __init__(self, is_train=True, lab_mean=None, lab_std=None):
#         self.lab_mean = lab_mean
#         self.lab_std = lab_std

#         # Original Code
#         # if is_train:
#         #     self.geom = transforms.Compose([
#         #         transforms.Resize((224,224)),
#         #         transforms.RandomHorizontalFlip(),
#         #     ])

#         # Testing out new code
#         # if is_train:
#         #     self.geom = transforms.Compose([
#         #         transforms.Resize((224,224)),
#         #         transforms.RandomHorizontalFlip(),
#         #         transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
#         #         transforms.RandomRotation(15),
#         #         transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
#         #     ])

#         # Testing out new code 2
#         if is_train:
#             self.geom = transforms.Compose([
#                 transforms.Resize((224,224)),
#                 transforms.RandomResizedCrop(224, scale=(0.95,1.0)),
#                 transforms.RandomHorizontalFlip(p=0.5),
#                 transforms.RandomRotation(8),
#             ])
#         else:
#             self.geom = transforms.Compose([
#                 transforms.Resize((224,224)),
#             ])

#     def __call__(self, img_pil):
#         img_pil = self.geom(img_pil)

#         rgb = np.asarray(img_pil).astype(np.float32)/255.0
#         lab = rgb2lab(rgb).astype(np.float32)
#         lab_norm = (lab - self.lab_mean) / self.lab_std

#         if RGBToLABTensorTransform.first_call:
#             RGBToLABTensorTransform.first_call = False
#             print("\n[DIAG] LAB BEFORE NORMALIZATION:")
#             print("  L range:", lab[...,0].min(), lab[...,0].max())
#             print("  a range:", lab[...,1].min(), lab[...,1].max())
#             print("  b range:", lab[...,2].min(), lab[...,2].max())
#             print("[DIAG] LAB AFTER NORMALIZATION:")
#             print("  min:", lab_norm.min(), "max:", lab_norm.max())
#             print("  mean:", lab_norm.mean(), "std:", lab_norm.std(), "\n")

#         return torch.from_numpy(lab_norm.transpose(2,0,1)).float()


# ##############################################################
# # 3. Dataset + Dataloaders (OPTIMIZED)
# ##############################################################

# class SkinToneRegressionDataset(Dataset):
#     def __init__(self, df, img_dir, transform, weights=None):
#         self.df = df.reset_index(drop=True)
#         self.img_dir = Path(img_dir)
#         self.transform = transform
        
#         # Pre-compute weights if not provided
#         if weights is None:
#             print("[INFO] Calculating dynamic sample weights...")

#             raw_labels = self.df.iloc[:, 1].astype(float).values
#             binned_labels = np.round(raw_labels).astype(int)
            
#             counts = Counter(binned_labels)
#             total_samples = len(binned_labels)
            
#             weights_map = {k: total_samples / v for k, v in counts.items()}
#             weights = [weights_map[int(round(x))] for x in raw_labels]
#             weights = np.array(weights)
#             weights = weights / weights.mean()
            
#             print(f"[INFO] Weights calculated. Min: {weights.min():.2f}, Max: {weights.max():.2f}, Mean: {weights.mean():.2f}")
        
#         self.weights = torch.FloatTensor(weights)

#     def __len__(self): 
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         img_path = self.img_dir / str(row.iloc[0])
        
#         # Add error handling for corrupt images
#         try:
#             img = Image.open(img_path).convert("RGB")
#         except Exception as e:
#             print(f"[WARN] Failed to load {img_path}: {e}")
#             # Return a black image as fallback
#             img = Image.new('RGB', (224, 224), color='black')

#         label = float(row.iloc[1])
#         label = (label - 1.0) / 9.0
#         weight = self.weights[idx]

#         return self.transform(img), torch.tensor(label, dtype=torch.float32), weight


# def build_dataloaders(csv_path, image_dir, batch_size, val_ratio, lab_mean, lab_std, num_workers=4):
#     """
#     Build dataloaders with subject-level stratified splitting.
    
#     OPTIMIZATIONS:
#     - Reduced num_workers (4 instead of 8)
#     - Added persistent_workers=True
#     - Added pin_memory=True
#     - Added prefetch_factor
#     """
#     df = pd.read_csv(csv_path).dropna()
    
#     # Ensure we have person_id column
#     if df.shape[1] < 3:
#         raise ValueError("CSV must have 3 columns: filename, label, person_id")
    
#     print(f"[INFO] Loaded CSV with columns: {list(df.columns)}")
    
#     # Extract person_id from column 2
#     df['person_id'] = df.iloc[:, 2].astype(str)
    
#     # Get one label per person (aggregate)
#     person_labels = df.groupby('person_id').agg({
#         df.columns[1]: 'first'
#     }).reset_index()
    
#     person_labels.columns = ['person_id', 'label']
#     labels_per_person = person_labels['label'].astype(float).values
    
#     print(f"[INFO] Found {len(person_labels)} unique persons")
#     print(f"[INFO] Label range: {labels_per_person.min():.2f} to {labels_per_person.max():.2f}")
    
#     # Round labels for stratification
#     rounded_labels = np.round(labels_per_person).astype(int)
#     rounded_labels = np.clip(rounded_labels, 1, 10)
    
#     print(f"[INFO] Rounded MST labels: {np.unique(rounded_labels)}")
    
#     # Count distribution
#     label_counts = Counter(rounded_labels)
#     print("\n[INFO] Person distribution across MST labels:")
#     for mst in sorted(label_counts.keys()):
#         print(f"  MST {mst}: {label_counts[mst]} persons")
#     print()
    
#     # Split persons with stratification
#     train_persons, val_persons = train_test_split(
#         person_labels['person_id'].values,
#         test_size=val_ratio,
#         shuffle=True,
#         stratify=rounded_labels,
#         random_state=42,
#     )
    
#     # Create train/val dataframes
#     train_df = df[df['person_id'].isin(train_persons)].reset_index(drop=True)
#     val_df = df[df['person_id'].isin(val_persons)].reset_index(drop=True)
    
#     # Keep only filename and label columns
#     train_df = train_df[[df.columns[0], df.columns[1]]]
#     val_df = val_df[[df.columns[0], df.columns[1]]]
    
#     print(f"[INFO] Train: {len(train_persons)} persons ({len(train_df)} images)")
#     print(f"[INFO] Val: {len(val_persons)} persons ({len(val_df)} images)")
    
#     # Print distributions
#     train_labels = train_df.iloc[:, 1].astype(float).values
#     val_labels = val_df.iloc[:, 1].astype(float).values
    
#     train_rounded = np.round(train_labels).astype(int)
#     val_rounded = np.round(val_labels).astype(int)
    
#     train_counts = Counter(train_rounded)
#     val_counts = Counter(val_rounded)
    
#     print("\n[INFO] Train MST label distribution (images):")
#     for mst in sorted(train_counts.keys()):
#         print(f"  MST {mst}: {train_counts[mst]} images")
    
#     print("\n[INFO] Val MST label distribution (images):")
#     for mst in sorted(val_counts.keys()):
#         print(f"  MST {mst}: {val_counts[mst]} images")
#     print()

#     train_tf = RGBToLABTensorTransform(is_train=True, lab_mean=lab_mean, lab_std=lab_std)
#     val_tf = RGBToLABTensorTransform(is_train=False, lab_mean=lab_mean, lab_std=lab_std)

#     # OPTIMIZED DATALOADER SETTINGS
#     train_loader = DataLoader(
#         SkinToneRegressionDataset(train_df, image_dir, train_tf),
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=True,              # Faster GPU transfer
#         persistent_workers=True,       # Keep workers alive between epochs
#         prefetch_factor=2,             # Prefetch 2 batches per worker
#     )
    
#     val_loader = DataLoader(
#         SkinToneRegressionDataset(val_df, image_dir, val_tf),
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True,
#         persistent_workers=True,
#         prefetch_factor=2,
#     )

#     return train_loader, val_loader


# ##############################################################
# # 4. VGG Model
# ##############################################################

# class VGG16LabRegressor(nn.Module):
#     def __init__(self, use_bn=True, dropout_p=0.5):
#         super().__init__()

#         base = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1) if use_bn else models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

#         self.features = base.features

#         # Original code
#         # for param in self.features.parameters():
#             # param.requires_grad = True

#         # Testing out new code
#         # for param in self.features.parameters():
#         #     param.requires_grad = False

#         # Testing out new code 2
#         for p in self.features.parameters():
#             p.requires_grad = False

#         for p in self.features[-5:].parameters():
#             p.requires_grad = True


#         with torch.no_grad():
#             dummy = torch.zeros(1, 3, 224, 224)
#             feat = self.features(dummy)
#             flat_dim = feat.view(1,-1).shape[1]
#             print(f"[INFO] VGG feature dimension: {flat_dim}")

#         # Original code
#         # self.classifier = nn.Sequential(
#         #     nn.Linear(flat_dim, 1024),
#         #     nn.ReLU(inplace=True),
#         #     nn.Dropout(dropout_p),
#         #     nn.Linear(1024, 512),
#         #     nn.ReLU(inplace=True),
#         #     nn.Dropout(dropout_p),
#         #     nn.Linear(512, 1),
#         #     nn.Sigmoid(),
#         # )

#         # Testing out new code
#         # self.classifier = nn.Sequential(
#         #     nn.Linear(flat_dim, 512),
#         #     nn.ReLU(inplace=True),
#         #     nn.Dropout(dropout_p),
#         #     nn.Linear(512, 1),
#         #     nn.Sigmoid(),
#         # )

#         # Testing out new code 2
#         self.classifier = nn.Sequential(
#             nn.Linear(flat_dim, 1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout_p),
#             nn.Linear(1024, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout_p),
#             nn.Linear(512, 1),
#             nn.Sigmoid(),
#         )


#         self._init_weights()

#     def _init_weights(self):
#         for m in self.classifier.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.zeros_(m.bias)

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         return self.classifier(x).squeeze(-1)


# ##############################################################
# # 5. Metrics
# ##############################################################

# def compute_metrics(pred, tgt, threshold=0.5):
#     pred_np = pred.detach().cpu().numpy() * 9 + 1
#     tgt_np  = tgt.detach().cpu().numpy() * 9 + 1

#     abs_diff = np.abs(pred_np - tgt_np)
#     acc1 = (abs_diff <= threshold).mean() * 100
#     l2 = calc_l2_lab_distance(pred, tgt)
#     return acc1, l2


# ##############################################################
# # 6. Diagnostics
# ##############################################################

# def print_diagnostics(imgs, labels, preds, tag):
#     print(f"\n=== [DIAG] {tag} ===")
#     print("IMG mean/std:", imgs.mean().item(), imgs.std().item())
#     print("IMG min/max:", imgs.min().item(), imgs.max().item())
#     print("Labels min/max:", labels.min().item(), labels.max().item())
#     print("Preds min/max:", preds.min().item(), preds.max().item())
#     if preds.std().item() < 1e-4:
#         print("WARNING: predictions collapsed to near-constant.")
#     print("===========================================\n")


# ##############################################################
# # 7. Train / Validate
# ##############################################################

# def train_one_epoch(model, loader, optim, device, epoch, threshold=0.5):
#     model.train()
#     total_loss = 0
#     total_acc = 0

#     pbar = tqdm(loader, desc=f"Train Epoch {epoch}", ncols=100)

#     for i, (imgs, labels, weights) in enumerate(pbar):
#         imgs, labels, weights = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True), weights.to(device, non_blocking=True)

#         preds = model(imgs)
        
#         # Original Code
#         # loss_unreduced = F.mse_loss(preds, labels, reduction='none')

#         # Testing out new code
#         # loss_unreduced = F.smooth_l1_loss(preds, labels, reduction='none')

#         # Testing out new code 2
#         loss_unreduced = F.mse_loss(preds, labels, reduction='none')
#         weights = torch.sqrt(weights)
#         loss_weighted = loss_unreduced * weights
#         loss = loss_weighted.mean()

#         optim.zero_grad()
#         loss.backward()
#         optim.step()

#         acc, _ = compute_metrics(preds, labels, threshold=threshold)

#         total_loss += loss.item()*imgs.size(0)
#         total_acc += acc*imgs.size(0)

#         pbar.set_postfix({
#             "loss": f"{loss.item():.4f}",
#             "acc": f"{acc:.4f}"
#         })

#         if i == 0:
#             print_diagnostics(imgs, labels, preds, f"TRAIN EPOCH {epoch}")

#     n = len(loader.dataset)
#     return total_loss/n, total_acc/n


# def validate(model, loader, device, epoch, threshold=0.5):
#     model.eval()
#     mse = nn.MSELoss()
#     total_loss = 0
#     total_acc = 0
#     all_l2 = []

#     with torch.no_grad():
#         pbar = tqdm(loader, desc=f"Validate Epoch {epoch}", ncols=100)

#         for i, batch in enumerate(pbar):
#             if len(batch) == 3:
#                 imgs, labels, _ = batch
#             else:
#                 imgs, labels = batch
            
#             imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
#             preds = model(imgs)

#             loss = mse(preds, labels)
#             acc, l2 = compute_metrics(preds, labels, threshold=threshold)

#             total_loss += loss.item()*imgs.size(0)
#             total_acc += acc*imgs.size(0)
#             all_l2.extend(l2)

#             pbar.set_postfix({
#                 "loss": f"{loss.item():.4f}",
#                 "acc": f"{acc:.4f}"
#             })

#             if i == 0:
#                 print_diagnostics(imgs, labels, preds, f"VALID EPOCH {epoch}")

#     n = len(loader.dataset)
#     return total_loss/n, total_acc/n, float(np.mean(all_l2)), float(np.std(all_l2))


# ##############################################################
# # 8. Checkpoint Management
# ##############################################################

# def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, val_loss, path):
#     """Save training checkpoint with all state"""
#     torch.save({
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'scheduler_state_dict': scheduler.state_dict(),
#         'val_acc': val_acc,
#         'val_loss': val_loss,
#     }, path)
#     print(f"[INFO] Checkpoint saved -> {path}")


# def load_checkpoint(model, optimizer, scheduler, path, device):
#     """Load training checkpoint and return start epoch"""
#     if not Path(path).exists():
#         print(f"[INFO] No checkpoint found at {path}, starting from scratch.")
#         return 1, float('inf'), 0.0
    
#     checkpoint = torch.load(path, map_location=device, weights_only=False)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
#     start_epoch = checkpoint['epoch'] + 1
#     best_val_loss = checkpoint['val_loss']
#     best_val_acc = checkpoint['val_acc']
    
#     print(f"[INFO] Resuming from epoch {checkpoint['epoch']}")
#     print(f"[INFO] Previous best - ValLoss: {best_val_loss:.4f}, ValAcc: {best_val_acc:.2f}%")
    
#     return start_epoch, best_val_loss, best_val_acc


# ##############################################################
# # 9. Early Stopping
# ##############################################################

# class EarlyStopping:
#     """Early stopping based on validation accuracy"""
#     def __init__(self, patience=10, min_delta=0.0):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.best_acc = 0.0
#         self.should_stop = False
    
#     def __call__(self, val_acc):
#         if val_acc > self.best_acc + self.min_delta:
#             self.best_acc = val_acc
#             self.counter = 0
#         else:
#             self.counter += 1
#             print(f"[EARLY STOP] No improvement for {self.counter}/{self.patience} epochs")
            
#             if self.counter >= self.patience:
#                 self.should_stop = True
#                 print(f"[EARLY STOP] Triggered! Best ValAcc: {self.best_acc:.2f}%")
        
#         return self.should_stop


# ##############################################################
# # 10. MAIN
# ##############################################################

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--csv-path", required=True)
#     parser.add_argument("--image-dir", required=True)
#     parser.add_argument("--save-dir", default="./checkpoints")
#     parser.add_argument("--epochs", type=int, default=32)
#     parser.add_argument("--additional-epochs", type=int, default=None)
#     parser.add_argument("--batch-size", type=int, default=32)
#     parser.add_argument("--val-ratio", type=float, default=0.35)
#     parser.add_argument("--lr", type=float, default=1e-4)
#     parser.add_argument("--weight-decay", type=float, default=1e-4)
#     parser.add_argument("--threshold", type=float, default=0.5)
#     parser.add_argument("--gpu", type=int, default=0)
#     parser.add_argument("--use-bn", action="store_true")
#     parser.add_argument("--compute-lab-stats", action="store_true")
#     parser.add_argument("--num-workers", type=int, default=4, 
#                         help="Number of data loading workers (default: 4)")
    
#     # Resume & Early Stopping
#     parser.add_argument("--resume", action="store_true")
#     parser.add_argument("--early-stop-patience", type=int, default=10)
#     parser.add_argument("--early-stop-delta", type=float, default=0.1)
    
#     args = parser.parse_args()

#     device = torch.device(f"cuda:{args.gpu}")

#     # LAB Stats
#     if args.compute_lab_stats:
#         print("[INFO] Computing LAB statistics from dataset...")
#         lab_mean, lab_std = compute_dataset_lab_stats(args.image_dir, args.csv_path)
#         print(f"[INFO] Computed LAB_MEAN: {lab_mean.tolist()}")
#         print(f"[INFO] Computed LAB_STD: {lab_std.tolist()}")
#     else:
#         print("[INFO] Using pre-computed LAB normalization values.")
#         # lab_mean = np.array([27.715821371226003, 10.521480987873188, 8.514460146640673], dtype=np.float32)
#         # lab_std  = np.array([24.70048803837073, 8.827357389195186, 8.660419910058293], dtype=np.float32)

#         lab_mean = np.array([33.618656158447266, 8.958210945129395, 8.925719261169434], dtype=np.float32)
#         lab_std = np.array([26.940208435058594, 8.05940055847168, 9.126977920532227], dtype=np.float32)

#     print(f"[INFO] Active LAB_MEAN: {lab_mean.tolist()}")
#     print(f"[INFO] Active LAB_STD: {lab_std.tolist()}")

#     # Paths
#     save_dir = Path(args.save_dir)
#     save_dir.mkdir(parents=True, exist_ok=True)

#     best_path = save_dir / "vgg16_lab_best.pth"
#     final_path = save_dir / "vgg16_lab_final.pth"
#     checkpoint_path = save_dir / "vgg16_lab_checkpoint.pth"

#     # Data (with optimized num_workers)
#     train_loader, val_loader = build_dataloaders(
#         args.csv_path, args.image_dir, args.batch_size, args.val_ratio, 
#         lab_mean, lab_std, num_workers=args.num_workers
#     )

#     # Model
#     model = VGG16LabRegressor(use_bn=args.use_bn).to(device)

#     # Optimizer & Scheduler
#     optimizer = torch.optim.Adam(
#         model.parameters(),
#         lr=args.lr,
#         weight_decay=args.weight_decay
#     )

#     # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     #     optimizer, mode='min', factor=0.5, patience=5, verbose=True
#     # )

#     # New Code 2
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.3, patience=3, min_lr=1e-6, verbose=True
#     )

#     # Resume or start fresh
#     if args.resume:
#         start_epoch, best_val_loss, best_val_acc = load_checkpoint(
#             model, optimizer, scheduler, checkpoint_path, device
#         )
        
#         if args.additional_epochs is not None:
#             max_epoch = start_epoch - 1 + args.additional_epochs
#             print(f"[INFO] Will train for {args.additional_epochs} additional epochs (until epoch {max_epoch})")
#         else:
#             max_epoch = args.epochs
#             if start_epoch > max_epoch:
#                 print(f"[WARNING] Checkpoint is at epoch {start_epoch-1}, already exceeds --epochs {max_epoch}")
#                 return
#             print(f"[INFO] Will train until epoch {max_epoch}")
#     else:
#         start_epoch = 1
#         max_epoch = args.epochs
#         best_val_loss = float("inf")
#         best_val_acc = 0.0
#         print(f"[INFO] Starting fresh training for {max_epoch} epochs")

#     # Early Stopping
#     early_stopper = EarlyStopping(
#         patience=args.early_stop_patience,
#         min_delta=args.early_stop_delta
#     )

#     # Training Loop
#     for epoch in range(start_epoch, max_epoch + 1):
#         print(f"\n===== EPOCH {epoch}/{max_epoch} =====")
#         tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, epoch, args.threshold)
#         va_loss, va_acc, l2_mean, l2_std = validate(model, val_loader, device, epoch, args.threshold)

#         print(
#             f"[EPOCH {epoch}] "
#             f"TrainLoss={tr_loss:.4f} TrainAcc={tr_acc:.2f}% | "
#             f"ValLoss={va_loss:.4f} ValAcc={va_acc:.2f}% | "
#             f"L2 mean={l2_mean:.3f}, std={l2_std:.3f}"
#         )

#         scheduler.step(va_loss)

#         if va_loss < best_val_loss:
#             best_val_loss = va_loss
#             torch.save(model.state_dict(), best_path)
#             print(f"[INFO] Saved BEST model -> {best_path}")

#         if va_acc > best_val_acc:
#             best_val_acc = va_acc

#         save_checkpoint(model, optimizer, scheduler, epoch, va_acc, va_loss, checkpoint_path)

#         if early_stopper(va_acc):
#             print(f"[INFO] Early stopping triggered at epoch {epoch}")
#             break

#     torch.save(model.state_dict(), final_path)
#     print(f"[INFO] Saved FINAL model -> {final_path}")
#     print(f"[INFO] Training complete. Best ValAcc: {best_val_acc:.2f}%, Best ValLoss: {best_val_loss:.4f}")


# if __name__ == "__main__":
#     main()

##############################################################
#  VGG16-LAB TRAINING PIPELINE (OPTIMIZED FOR I/O)
#  - LAB input + dataset stats
#  - Subject-level stratified split
#  - Inverse-frequency MST weighting (train only)
#  - Partial fine-tuning of last conv blocks
#  - ReduceLROnPlateau + EarlyStopping
##############################################################

import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from skimage.color import rgb2lab
from sklearn.model_selection import train_test_split


##############################################################
# 0. REPRODUCIBILITY
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
# 1. MONK -> LAB utilities
##############################################################

MONK_COLORS_RGB = np.array([
    (246, 237, 228),
    (243, 231, 219),
    (247, 234, 208),
    (243, 218, 186),
    (215, 189, 150),
    (160, 126, 86),
    (130, 92, 67),
    (96, 65, 52),
    (58, 49, 42),
    (41, 36, 32),
], dtype=np.float32)


def _rgb_to_lab(rgb_vec):
    rgb = rgb_vec.reshape(-1, 1, 1, 3) / 255.0
    lab = rgb2lab(rgb)
    return lab.reshape(-1, 3).astype(np.float32)


MONK_COLORS_LAB = _rgb_to_lab(MONK_COLORS_RGB)


def monk_scalar_to_lab(values):
    """
    values: array-like of MST scores in [1,10] (possibly fractional).
    Returns LAB triplets interpolated between canonical Monk colors.
    """
    labs = []
    for v in values:
        v = float(max(1, min(10, v)))
        base = int(v)
        if base >= 10:
            labs.append(MONK_COLORS_LAB[-1])
            continue
        lab0 = MONK_COLORS_LAB[base - 1]
        lab1 = MONK_COLORS_LAB[base]
        labs.append(lab0 + (v - base) * (lab1 - lab0))
    return np.stack(labs)


def calc_l2_lab_distance(pred, target):
    """
    pred, target in [0,1] (normalized MST space).
    Convert back to [1,10] and compute L2 distance in Monk-LAB.
    """
    p = pred.detach().cpu().numpy() * 9 + 1
    t = target.detach().cpu().numpy() * 9 + 1
    return np.sqrt(np.sum((monk_scalar_to_lab(p) - monk_scalar_to_lab(t)) ** 2, axis=1))


##############################################################
#  LAB DATASET STATISTICS
##############################################################

def compute_dataset_lab_stats(image_dir, csv_path):
    df = pd.read_csv(csv_path)
    img_names = df.iloc[:, 0].tolist()

    print(f"[INFO] Computing LAB dataset statistics on {len(img_names)} images...")

    resize_op = transforms.Resize((224, 224))

    sum_lab = np.zeros(3, dtype=np.float64)
    sum_sq_lab = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for name in tqdm(img_names, desc="LAB stats"):
        fpath = Path(image_dir) / name

        try:
            img = Image.open(fpath).convert("RGB")
            img = resize_op(img)

            rgb = np.asarray(img).astype(np.float32) / 255.0
            lab = rgb2lab(rgb).astype(np.float64)

            lab_flat = lab.reshape(-1, 3)
            sum_lab += lab_flat.sum(axis=0)
            sum_sq_lab += (lab_flat ** 2).sum(axis=0)
            total_pixels += lab_flat.shape[0]

        except Exception as e:
            print(f"[WARN] Could not read {fpath}: {e}")

    mean = sum_lab / total_pixels
    var = (sum_sq_lab / total_pixels) - (mean ** 2)
    std = np.sqrt(var)

    print("\n========== LAB DATASET STATISTICS ==========")
    print("LAB_MEAN =", mean.tolist())
    print("LAB_STD  =", std.tolist())
    print("============================================\n")

    return mean.astype(np.float32), std.astype(np.float32)


##############################################################
# 2. LAB Transform
##############################################################

class RGBToLABTensorTransform:
    first_call = True

    def __init__(self, is_train=True, lab_mean=None, lab_std=None):
        self.lab_mean = lab_mean
        self.lab_std = lab_std

        # Moderate geometric augmentation for train
        if is_train:
            self.geom = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomResizedCrop(224, scale=(0.95, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(8),
            ])
        else:
            self.geom = transforms.Compose([
                transforms.Resize((224, 224)),
            ])

    def __call__(self, img_pil):
        img_pil = self.geom(img_pil)

        rgb = np.asarray(img_pil).astype(np.float32) / 255.0
        lab = rgb2lab(rgb).astype(np.float32)
        lab_norm = (lab - self.lab_mean) / self.lab_std

        if RGBToLABTensorTransform.first_call:
            RGBToLABTensorTransform.first_call = False
            print("\n[DIAG] LAB BEFORE NORMALIZATION:")
            print("  L range:", lab[..., 0].min(), lab[..., 0].max())
            print("  a range:", lab[..., 1].min(), lab[..., 1].max())
            print("  b range:", lab[..., 2].min(), lab[..., 2].max())
            print("[DIAG] LAB AFTER NORMALIZATION:")
            print("  min:", lab_norm.min(), "max:", lab_norm.max())
            print("  mean:", lab_norm.mean(), "std:", lab_norm.std(), "\n")

        return torch.from_numpy(lab_norm.transpose(2, 0, 1)).float()


##############################################################
# 3. Dataset + Dataloaders (OPTIMIZED)
##############################################################

class SkinToneRegressionDataset(Dataset):
    """
    Expects:
      df: 2 columns -> [filename, label]
      weights: 1D array-like of per-image sample weights (same length as df)
    """
    def __init__(self, df, img_dir, transform, weights=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform

        if weights is None:
            # Fallback: uniform weights
            print("[WARN] No weights provided, defaulting to 1.0 for all samples.")
            weights = np.ones(len(self.df), dtype=np.float32)
        else:
            weights = np.asarray(weights, dtype=np.float32)
            if len(weights) != len(self.df):
                raise ValueError(
                    f"weights length {len(weights)} does not match dataset length {len(self.df)}"
                )

        self.weights = torch.from_numpy(weights.astype(np.float32))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / str(row.iloc[0])

        # Add error handling for corrupt images
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to load {img_path}: {e}")
            img = Image.new("RGB", (224, 224), color="black")

        label = float(row.iloc[1])
        # Normalize MST from [1,10] → [0,1]
        label = (label - 1.0) / 9.0
        weight = self.weights[idx]

        return self.transform(img), torch.tensor(label, dtype=torch.float32), weight


def build_dataloaders(csv_path, image_dir, batch_size, val_ratio, lab_mean, lab_std, num_workers=4):
    """
    Build dataloaders with subject-level stratified splitting and
    inverse-frequency MST weighting on the training set.
    """
    df = pd.read_csv(csv_path).dropna()

    # Ensure we have person_id column
    if df.shape[1] < 3:
        raise ValueError("CSV must have 3 columns: filename, label, person_id")

    print(f"[INFO] Loaded CSV with columns: {list(df.columns)}")

    # Extract person_id from column 2
    df["person_id"] = df.iloc[:, 2].astype(str)

    # Get one label per person (aggregate)
    person_labels = df.groupby("person_id").agg({
        df.columns[1]: "first"
    }).reset_index()

    person_labels.columns = ["person_id", "label"]
    labels_per_person = person_labels["label"].astype(float).values

    print(f"[INFO] Found {len(person_labels)} unique persons")
    print(f"[INFO] Label range: {labels_per_person.min():.2f} to {labels_per_person.max():.2f}")

    # Round labels for stratification (MST 1–10)
    rounded_labels = np.round(labels_per_person).astype(int)
    rounded_labels = np.clip(rounded_labels, 1, 10)

    print(f"[INFO] Rounded MST labels: {np.unique(rounded_labels)}")

    # Count distribution
    label_counts = Counter(rounded_labels)
    print("\n[INFO] Person distribution across MST labels:")
    for mst in sorted(label_counts.keys()):
        print(f"  MST {mst}: {label_counts[mst]} persons")
    print()

    # Split persons with stratification
    train_persons, val_persons = train_test_split(
        person_labels["person_id"].values,
        test_size=val_ratio,
        shuffle=True,
        # stratify=rounded_labels,
        random_state=42,
    )

    # Create train/val dataframes
    train_df = df[df["person_id"].isin(train_persons)].reset_index(drop=True)
    val_df = df[df["person_id"].isin(val_persons)].reset_index(drop=True)

    # Keep only filename and label columns
    train_df = train_df[[df.columns[0], df.columns[1]]]
    val_df = val_df[[df.columns[0], df.columns[1]]]

    print(f"[INFO] Train: {len(train_persons)} persons ({len(train_df)} images)")
    print(f"[INFO] Val: {len(val_persons)} persons ({len(val_df)} images)")

    # Print distributions
    train_labels = train_df.iloc[:, 1].astype(float).values
    val_labels = val_df.iloc[:, 1].astype(float).values

    train_rounded = np.clip(np.round(train_labels).astype(int), 1, 10)
    val_rounded = np.clip(np.round(val_labels).astype(int), 1, 10)

    train_counts = Counter(train_rounded)
    val_counts = Counter(val_rounded)

    print("\n[INFO] Train MST label distribution (images):")
    for mst in sorted(train_counts.keys()):
        print(f"  MST {mst}: {train_counts[mst]} images")

    print("\n[INFO] Val MST label distribution (images):")
    for mst in sorted(val_counts.keys()):
        print(f"  MST {mst}: {val_counts[mst]} images")
    print()

    # ---------- CLASS WEIGHTS (TRAIN ONLY) ----------
    total_train = len(train_rounded)
    class_counts = Counter(train_rounded)

    # inverse-frequency per MST label
    class_weights = {k: total_train / v for k, v in class_counts.items()}

    # map each image to its class weight
    image_weights_train = np.array(
        [class_weights[int(round(y))] for y in train_labels],
        dtype=np.float32,
    )

    # normalize by mean to keep weights numerically reasonable
    image_weights_train /= image_weights_train.mean()

    print("\n[INFO] Inverse-frequency weights (per MST):")
    for k in sorted(class_weights.keys()):
        w = class_weights[k]
        print(f"  MST {k}: raw={w:.3f}")
    print(f"[INFO] Train image weights: min={image_weights_train.min():.3f}, "
          f"max={image_weights_train.max():.3f}, mean={image_weights_train.mean():.3f}\n")

    # Validation: uniform weights
    image_weights_val = np.ones(len(val_df), dtype=np.float32)

    # Transforms
    train_tf = RGBToLABTensorTransform(is_train=True, lab_mean=lab_mean, lab_std=lab_std)
    val_tf = RGBToLABTensorTransform(is_train=False, lab_mean=lab_mean, lab_std=lab_std)

    # Datasets
    train_dataset = SkinToneRegressionDataset(train_df, image_dir, train_tf, weights=image_weights_train)
    val_dataset = SkinToneRegressionDataset(val_df, image_dir, val_tf, weights=image_weights_val)

    # OPTIMIZED DATALOADER SETTINGS
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    return train_loader, val_loader


##############################################################
# 4. VGG Model
##############################################################

class VGG16LabRegressor(nn.Module):
    def __init__(self, use_bn=True, dropout_p=0.5):
        super().__init__()

        base = (
            models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
            if use_bn
            else models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        )

        self.features = base.features

        # Freeze all conv layers first
        for p in self.features.parameters():
            p.requires_grad = False

        # Unfreeze last few conv layers for mild fine-tuning
        for p in self.features[-5:].parameters():
            p.requires_grad = True

        # Infer flattened feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat = self.features(dummy)
            flat_dim = feat.view(1, -1).shape[1]
            print(f"[INFO] VGG feature dimension: {flat_dim}")

        # MLP head for regression in [0,1] (then scaled back to [1,10] in metrics)
        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x).squeeze(-1)


##############################################################
# 5. Metrics
##############################################################

def compute_metrics(pred, tgt, threshold=0.5):
    """
    pred, tgt in [0,1] (MST normalized). We convert back to [1,10] and
    compute '±threshold' accuracy, where threshold is in MST units.
    """
    pred_np = pred.detach().cpu().numpy() * 9 + 1
    tgt_np = tgt.detach().cpu().numpy() * 9 + 1

    abs_diff = np.abs(pred_np - tgt_np)
    acc1 = (abs_diff <= threshold).mean() * 100.0
    l2 = calc_l2_lab_distance(pred, tgt)
    return acc1, l2


##############################################################
# 6. Diagnostics
##############################################################

def print_diagnostics(imgs, labels, preds, tag):
    print(f"\n=== [DIAG] {tag} ===")
    print("IMG mean/std:", imgs.mean().item(), imgs.std().item())
    print("IMG min/max:", imgs.min().item(), imgs.max().item())
    print("Labels min/max:", labels.min().item(), labels.max().item())
    print("Preds min/max:", preds.min().item(), preds.max().item())
    if preds.std().item() < 1e-4:
        print("WARNING: predictions collapsed to near-constant.")
    print("===========================================\n")


##############################################################
# 7. Train / Validate
##############################################################

def train_one_epoch(model, loader, optim, device, epoch, threshold=0.5):
    model.train()
    total_loss = 0.0
    total_acc = 0.0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", ncols=100)

    for i, (imgs, labels, weights) in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)

        preds = model(imgs)

        # Per-sample MSE
        loss_unreduced = F.mse_loss(preds, labels, reduction="none")

        # Temper class weights: use sqrt to avoid extreme gradients on rare classes
        weights_t = torch.sqrt(weights)
        loss_weighted = loss_unreduced * weights_t
        loss = loss_weighted.mean()

        optim.zero_grad()
        loss.backward()
        optim.step()

        acc, _ = compute_metrics(preds, labels, threshold=threshold)

        total_loss += loss.item() * imgs.size(0)
        total_acc += acc * imgs.size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{acc:.2f}"
        })

        if i == 0:
            print_diagnostics(imgs, labels, preds, f"TRAIN EPOCH {epoch}")

    n = len(loader.dataset)
    return total_loss / n, total_acc / n


def validate(model, loader, device, epoch, threshold=0.5):
    model.eval()
    mse = nn.MSELoss()
    total_loss = 0.0
    total_acc = 0.0
    all_l2 = []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Validate Epoch {epoch}", ncols=100)

        for i, batch in enumerate(pbar):
            if len(batch) == 3:
                imgs, labels, _ = batch
            else:
                imgs, labels = batch

            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            preds = model(imgs)

            loss = mse(preds, labels)
            acc, l2 = compute_metrics(preds, labels, threshold=threshold)

            total_loss += loss.item() * imgs.size(0)
            total_acc += acc * imgs.size(0)
            all_l2.extend(l2)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{acc:.2f}"
            })

            if i == 0:
                print_diagnostics(imgs, labels, preds, f"VALID EPOCH {epoch}")

    n = len(loader.dataset)
    return total_loss / n, total_acc / n, float(np.mean(all_l2)), float(np.std(all_l2))


##############################################################
# 8. Checkpoint Management
##############################################################

def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, val_loss, path):
    """Save training checkpoint with all state"""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_acc": val_acc,
        "val_loss": val_loss,
    }, path)
    print(f"[INFO] Checkpoint saved -> {path}")


def load_checkpoint(model, optimizer, scheduler, path, device):
    """Load training checkpoint and return start epoch"""
    if not Path(path).exists():
        print(f"[INFO] No checkpoint found at {path}, starting from scratch.")
        return 1, float("inf"), 0.0

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint["val_loss"]
    best_val_acc = checkpoint["val_acc"]

    print(f"[INFO] Resuming from epoch {checkpoint['epoch']}")
    print(f"[INFO] Previous best - ValLoss: {best_val_loss:.4f}, ValAcc: {best_val_acc:.2f}%")

    return start_epoch, best_val_loss, best_val_acc


##############################################################
# 9. Early Stopping
##############################################################

class EarlyStopping:
    """Early stopping based on validation accuracy"""
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = 0.0
        self.should_stop = False

    def __call__(self, val_acc):
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.counter = 0
        else:
            self.counter += 1
            print(f"[EARLY STOP] No improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                self.should_stop = True
                print(f"[EARLY STOP] Triggered! Best ValAcc: {self.best_acc:.2f}%")

        return self.should_stop


##############################################################
# 10. MAIN
##############################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--save-dir", default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--additional-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use-bn", action="store_true")
    parser.add_argument("--compute-lab-stats", action="store_true")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )

    # Resume & Early Stopping
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-delta", type=float, default=0.1)

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")

    print("\n================= NEW CODE =================")

    # LAB Stats
    if args.compute_lab_stats:
        print("[INFO] Computing LAB statistics from dataset...")
        lab_mean, lab_std = compute_dataset_lab_stats(args.image_dir, args.csv_path)
        print(f"[INFO] Computed LAB_MEAN: {lab_mean.tolist()}")
        print(f"[INFO] Computed LAB_STD: {lab_std.tolist()}")
    else:
        print("[INFO] Using pre-computed LAB normalization values.")
        # These are your precomputed stats for this dataset
        lab_mean = np.array([33.618656158447266, 8.958210945129395, 8.925719261169434], dtype=np.float32)
        lab_std = np.array([26.940208435058594, 8.05940055847168, 9.126977920532227], dtype=np.float32)

    print(f"[INFO] Active LAB_MEAN: {lab_mean.tolist()}")
    print(f"[INFO] Active LAB_STD: {lab_std.tolist()}")

    # Paths
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_path = save_dir / "vgg16_lab_best.pth"
    final_path = save_dir / "vgg16_lab_final.pth"
    checkpoint_path = save_dir / "vgg16_lab_checkpoint.pth"

    # Data
    train_loader, val_loader = build_dataloaders(
        args.csv_path,
        args.image_dir,
        args.batch_size,
        args.val_ratio,
        lab_mean,
        lab_std,
        num_workers=args.num_workers,
    )

    # Model
    model = VGG16LabRegressor(use_bn=args.use_bn).to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.3,
        patience=3,
        min_lr=1e-6,
        verbose=True,
    )

    # Resume or start fresh
    if args.resume:
        start_epoch, best_val_loss, best_val_acc = load_checkpoint(
            model, optimizer, scheduler, checkpoint_path, device
        )

        if args.additional_epochs is not None:
            max_epoch = start_epoch - 1 + args.additional_epochs
            print(f"[INFO] Will train for {args.additional_epochs} additional epochs (until epoch {max_epoch})")
        else:
            max_epoch = args.epochs
            if start_epoch > max_epoch:
                print(f"[WARNING] Checkpoint is at epoch {start_epoch - 1}, already exceeds --epochs {max_epoch}")
                return
            print(f"[INFO] Will train until epoch {max_epoch}")
    else:
        start_epoch = 1
        max_epoch = args.epochs
        best_val_loss = float("inf")
        best_val_acc = 0.0
        print(f"[INFO] Starting fresh training for {max_epoch} epochs")

    # Early Stopping
    early_stopper = EarlyStopping(
        patience=args.early_stop_patience,
        min_delta=args.early_stop_delta,
    )

    # Training Loop
    for epoch in range(start_epoch, max_epoch + 1):
        print(f"\n===== EPOCH {epoch}/{max_epoch} =====")
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch, args.threshold
        )
        va_loss, va_acc, l2_mean, l2_std = validate(
            model, val_loader, device, epoch, args.threshold
        )

        print(
            f"[EPOCH {epoch}] "
            f"TrainLoss={tr_loss:.4f} TrainAcc={tr_acc:.2f}% | "
            f"ValLoss={va_loss:.4f} ValAcc={va_acc:.2f}% | "
            f"L2 mean={l2_mean:.3f}, std={l2_std:.3f}"
        )

        scheduler.step(va_loss)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(model.state_dict(), best_path)
            print(f"[INFO] Saved BEST model -> {best_path}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc

        save_checkpoint(
            model, optimizer, scheduler, epoch, va_acc, va_loss, checkpoint_path
        )

        if early_stopper(va_acc):
            print(f"[INFO] Early stopping triggered at epoch {epoch}")
            break

    torch.save(model.state_dict(), final_path)
    print(f"[INFO] Saved FINAL model -> {final_path}")
    print(
        f"[INFO] Training complete. Best ValAcc: {best_val_acc:.2f}%, "
        f"Best ValLoss: {best_val_loss:.4f}"
    )


if __name__ == "__main__":
    main()


# python VGG16_Model_SkinTone_Training.py --save-dir "G:\Thesis\Models\VGG16_LAB" --csv-path "G:\Thesis\MonkSkinTone_Dataset\Segmented_MSTE\annotations.csv" --image-dir "G:\Thesis\MonkSkinTone_Dataset\Segmented_MSTE" --epochs 32 --lr 1e-4 --momentum 0.9 --weight-decay 1e-4 --threshold 0.5 --val-ratio 0.35 --use-bn --gpu 0

# python VGG16_Model_SkinTone_Training.py --save-dir "G:\Thesis\Models\VGG16_LAB\VGG16_FACET_0.2_con" --csv-path "G:\Thesis\FACET_Dataset\Segmented_FACET_0.2_continuous\annotations.csv" --image-dir "G:\Thesis\FACET_Dataset\Segmented_FACET_0.2_continuous" --epochs 32 --lr 1e-4 --weight-decay 1e-4 --threshold 0.5 --val-ratio 0.35 --use-bn --gpu 0

# python VGG16_Model_SkinTone_Training.py --save-dir "G:\Thesis\CasualConversationv2_Dataset\Models" --csv-path "G:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\annotations.csv" --image-dir "G:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2" --epochs 32 --lr 1e-4 --weight-decay 1e-4 --threshold 0.5 --val-ratio 0.35 --use-bn --gpu 0 --compute-lab-stats