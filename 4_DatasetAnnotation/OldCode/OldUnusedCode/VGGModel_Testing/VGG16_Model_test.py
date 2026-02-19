# # import argparse
# # import os
# # from pathlib import Path

# # import numpy as np
# # import pandas as pd
# # from PIL import Image
# # from tqdm import tqdm

# # import torch
# # import torch.nn as nn
# # from torch.utils.data import Dataset, DataLoader
# # from torchvision import models, transforms

# # from skimage.color import rgb2lab
# # from sklearn.model_selection import GroupShuffleSplit, train_test_split


# # # ============================================================
# # # 0. UTILS: MONK SCALE APPROX + LAB DISTANCES
# # # ============================================================

# # # Monk Skin Tone base RGB colours (you can adjust if needed)
# # MONK_COLORS_RGB = [
# #     (255, 255, 255),
# #     (246, 237, 228),
# #     (243, 231, 219),
# #     (247, 234, 208),
# #     (243, 218, 186),
# #     (215, 189, 150),
# #     (160, 126, 86),
# #     (130, 92, 67),
# #     (96, 65, 52),
# #     (58, 49, 42),
# #     (41, 36, 32),
# #     (0, 0, 0),
# # ]


# # def rgb_to_lab_approx(rgb):
# #     """
# #     Approximate RGB->LAB mapping, mirroring the original repo's linear transform.
# #     This is only used for evaluation, not for training input.
# #     """
# #     R, G, B = rgb
# #     L = Y1 = 0.2126 * R + 0.7152 * G + 0.0722 * B
# #     A = 1.4749 * (0.2213 * R - 0.3390 * G + 0.1177 * B) + 128
# #     b = 0.6245 * (0.1949 * R + 0.6057 * G - 0.8006 * B) + 128
# #     return np.array([L, A, b], dtype=np.float32)


# # def monk_scalar_to_lab(values):
# #     """
# #     Map continuous Monk index (e.g. 3.2) to LAB via linear interpolation between
# #     MONK_COLORS_RGB (similar to monk_to_lab() in the original code).
# #     """
# #     labs = []
# #     for v in values:
# #         v_float = float(v)
# #         idx = max(0, min(10, int(v_float)))
# #         try:
# #             A = np.array(MONK_COLORS_RGB[idx], dtype=np.float32)
# #             B = np.array(MONK_COLORS_RGB[idx + 1], dtype=np.float32)
# #             frac = v_float - float(idx)
# #             rgb_interp = (B - A) * frac + A
# #             labs.append(rgb_to_lab_approx(rgb_interp))
# #         except Exception:
# #             labs.append(np.array([0.0, 0.0, 0.0], dtype=np.float32))
# #     return np.stack(labs, axis=0)


# # def calc_l2_lab_distance(pred, target):
# #     """
# #     Compute L2 distance in LAB between predicted and target Monk indices.
# #     """
# #     pred_lab = monk_scalar_to_lab(pred)
# #     targ_lab = monk_scalar_to_lab(target)
# #     diff = pred_lab - targ_lab
# #     return np.sqrt(np.sum(diff * diff, axis=1))


# # def compute_metrics(output, target, threshold=0.5, cum_max_steps=20, factor=2.0):
# #     """
# #     Core metrics:
# #     - acc@1: % of predictions within ± threshold of label (Monk units)
# #     - acc@2: % of predictions within ± threshold*factor
# #     - cum_acc: cumulative accuracy curve in steps of 0.1 (0.0 .. 1.9)
# #     - l2_distances: LAB L2 errors for each example
# #     """
# #     with torch.no_grad():
# #         output = output.detach().cpu().numpy()
# #         target = target.detach().cpu().numpy()
# #         batch_size = target.shape[0]

# #         abs_diff = np.abs(output - target)

# #         acc1 = (abs_diff <= threshold).sum() * 100.0 / batch_size
# #         acc2 = (abs_diff <= threshold * factor).sum() * 100.0 / batch_size

# #         cum_acc = []
# #         for delta_idx in range(cum_max_steps):
# #             delta = delta_idx / 10.0
# #             cum_acc.append((abs_diff < delta).sum() * 100.0 / batch_size)

# #         l2_distances = calc_l2_lab_distance(output, target)

# #         return acc1, acc2, cum_acc, l2_distances


# # # ============================================================
# # # 1. DATASET: RGB -> LAB (+optional blur)
# # # ============================================================

# # class SkinToneRegressionDataset(Dataset):
# #     """
# #     Simplified version of the original regression dataset, with:
# #     - Single CSV source (or DataFrame) with columns:
# #         filename,label[,subject_id]
# #     - Relative filename paths, resolved against `img_dir`
# #     - Optional blur and optional conversion to LAB with a/b shift
# #     """

# #     def __init__(
# #         self,
# #         annotations_df: pd.DataFrame = None,
# #         annotations_file: str = None,
# #         img_dir: str = ".",
# #         transform=None,
# #         blur: bool = False,
# #         conv_to_lab_space: bool = True,
# #     ):
# #         if annotations_df is None and annotations_file is None:
# #             raise ValueError("Either annotations_df or annotations_file must be provided.")

# #         if annotations_df is None:
# #             df = pd.read_csv(annotations_file)
# #         else:
# #             df = annotations_df.copy()

# #         # Assume first col = filename, second col = label
# #         df = df.dropna(subset=[df.columns[0], df.columns[1]])
# #         self.img_labels = df.reset_index(drop=True)
# #         self.img_dir = Path(img_dir)
# #         self.transform = transform
# #         self.blur = blur
# #         self.conv_to_lab_space = conv_to_lab_space

# #     def __len__(self):
# #         return len(self.img_labels)

# #     @staticmethod
# #     def _rgb_pil_to_shifted_lab_pil(img: Image.Image) -> Image.Image:
# #         """
# #         Mirrors the original LAB conversion:
# #         - RGB PIL -> float RGB [0,1] (H,W,3)
# #         - rgb2lab -> LAB (L in [0,100], a,b approx [-128,127])
# #         - Shift a,b by +127 (mod_lab_range behaviour)
# #         - Clip to [0,255], cast to uint8, wrap back to PIL
# #         """
# #         rgb = np.asarray(img).astype(np.float32) / 255.0
# #         lab = rgb2lab(rgb)  # (H, W, 3)

# #         lab[..., 1] += 127.0
# #         lab[..., 2] += 127.0

# #         lab = np.clip(lab, 0, 255).astype(np.uint8)
# #         return Image.fromarray(lab)

# #     @staticmethod
# #     def _apply_blur(img: Image.Image) -> Image.Image:
# #         """
# #         Very light Gaussian blur using PIL; enough to simulate the
# #         original optional blur flag if you use it.
# #         """
# #         from PIL import ImageFilter

# #         return img.filter(ImageFilter.GaussianBlur(radius=1.0))

# #     def __getitem__(self, idx):
# #         row = self.img_labels.iloc[idx]
# #         filename = row.iloc[0]
# #         label = float(row.iloc[1])

# #         img_path = self.img_dir / filename
# #         img = Image.open(img_path).convert("RGB")

# #         if self.blur:
# #             img = self._apply_blur(img)

# #         if self.conv_to_lab_space:
# #             img = self._rgb_pil_to_shifted_lab_pil(img)

# #         if self.transform is not None:
# #             img = self.transform(img)

# #         label_tensor = torch.tensor(label, dtype=torch.float32)

# #         if idx == 0:
# #             print("LAB stats:", img.mean().item(), img.min().item(), img.max().item())

# #         return img, label_tensor


# # # ============================================================
# # # 2. TRANSFORMS
# # # ============================================================

# # def build_transforms(is_train: bool):
# #     """
# #     Basic augmentations & normalisation.
# #     Mean/std loosely match the original repo.
# #     """
# #     normalize = transforms.Normalize(
# #         mean=[0.229, 0.5, 0.5],
# #         std=[0.200, 0.224, 0.225],
# #     )

# #     if is_train:
# #         return transforms.Compose(
# #             [
# #                 transforms.RandomResizedCrop(224),
# #                 transforms.RandomHorizontalFlip(),
# #                 transforms.ToTensor(),
# #                 normalize,
# #             ]
# #         )
# #     else:
# #         return transforms.Compose(
# #             [
# #                 transforms.Resize(256),
# #                 transforms.CenterCrop(224),
# #                 transforms.ToTensor(),
# #                 normalize,
# #             ]
# #         )


# # # ============================================================
# # # 3. DATALOADER BUILDER: SINGLE CSV → TRAIN/VAL SPLITS
# # # ============================================================

# # def build_dataloaders_from_single_csv(
# #     csv_path: str,
# #     image_dir: str,
# #     batch_size: int,
# #     val_ratio: float,
# #     blur: bool,
# # ):
# #     """
# #     Builds train/validation DataLoaders from a single CSV.

# #     Behaviour:
# #     -------------------------------------------------------
# #     • If CSV contains 'subject_id' → Identity-aware split
# #         - Ensures all images from one subject go to the same split
# #     • Otherwise → Stratified split on MST labels
# #         - Preserves label distribution for general images

# #     CSV expected structure:
# #         filename,label[,subject_id]

# #     'filename' is a relative path from `image_dir`.
# #     """

# #     df = pd.read_csv(csv_path)
# #     df = df.dropna(subset=[df.columns[0], df.columns[1]])

# #     filenames = df.iloc[:, 0].astype(str).values
# #     labels = df.iloc[:, 1].astype(float).values

# #     # ======================================================
# #     # Detect subject_id column or infer from folder names
# #     # ======================================================
# #     if "subject_id" in df.columns:
# #         print("[INFO] Found 'subject_id' column — using identity-aware grouped split.")
# #         groups = df["subject_id"].astype(str).values

# #     else:
# #         print("[INFO] No 'subject_id' column — checking folder-based identity.")
# #         inferred_groups = []

# #         for fname in filenames:
# #             parts = fname.split("/")  # e.g. "0002/img_003.jpg"
# #             if len(parts) > 1:
# #                 inferred_groups.append(parts[0])
# #             else:
# #                 inferred_groups.append("no_group")

# #         unique_groups = set(inferred_groups)

# #         if "no_group" not in unique_groups and len(unique_groups) > 1:
# #             print(f"[INFO] Inferred subject IDs from directory structure ({len(unique_groups)} subjects).")
# #             groups = np.array(inferred_groups)
# #         else:
# #             print("[INFO] No identity structure detected — using stratified split instead.")
# #             groups = None

# #     # ======================================================
# #     # SPLITTING LOGIC
# #     # ======================================================
# #     if groups is not None:
# #         splitter = GroupShuffleSplit(
# #             n_splits=1,
# #             test_size=val_ratio,
# #             random_state=42
# #         )
# #         train_idx, val_idx = next(splitter.split(df, labels, groups))
# #         print(f"[INFO] Using GROUPED split.")
# #     else:
# #         train_idx, val_idx = train_test_split(
# #             np.arange(len(df)),
# #             test_size=val_ratio,
# #             shuffle=True,
# #             stratify=labels,
# #             random_state=42,
# #         )
# #         print("[INFO] Using STRATIFIED split (label balanced).")

# #     train_df = df.iloc[train_idx].reset_index(drop=True)
# #     val_df = df.iloc[val_idx].reset_index(drop=True)

# #     print(
# #         f"[INFO] Total={len(df)} | Train={len(train_df)} | Val={len(val_df)} "
# #         f"(ratio={val_ratio})"
# #     )

# #     # ======================================================
# #     # TorchVision transforms
# #     # ======================================================
# #     train_transform = build_transforms(is_train=True)
# #     val_transform = build_transforms(is_train=False)

# #     # ======================================================
# #     # Datasets
# #     # ======================================================
# #     train_ds = SkinToneRegressionDataset(
# #         annotations_df=train_df,
# #         img_dir=image_dir,
# #         transform=train_transform,
# #         blur=blur,
# #         conv_to_lab_space=True,
# #     )
# #     val_ds = SkinToneRegressionDataset(
# #         annotations_df=val_df,
# #         img_dir=image_dir,
# #         transform=val_transform,
# #         blur=blur,
# #         conv_to_lab_space=True,
# #     )

# #     # ======================================================
# #     # DataLoaders
# #     # ======================================================
# #     train_loader = DataLoader(
# #         train_ds,
# #         batch_size=batch_size,
# #         shuffle=True,
# #         num_workers=4,
# #         pin_memory=False, #True,
# #         # persistent_workers=True,
# #     )
# #     val_loader = DataLoader(
# #         val_ds,
# #         batch_size=batch_size,
# #         shuffle=False,
# #         num_workers=4,
# #         pin_memory=False, #True,
# #         # persistent_workers=True,
# #     )

# #     return train_loader, val_loader


# # # ============================================================
# # # 4. MODEL: VGG-16 LAB REGRESSOR
# # # ============================================================

# # class VGG16LabRegressor(nn.Module):
# #     """
# #     Simplified regression head on top of VGG-16 or VGG-16-BN:
# #     - features: standard VGG conv + pooling stack
# #     - classifier:
# #         512*7*7 → 1024 → 512 → 1
# #         with Sigmoid + Dropout between FC layers
# #     """

# #     def __init__(self, use_bn: bool = True, pretrained = None):
# #         super().__init__()

# #         if use_bn:
# #             base = models.vgg16_bn(weights=pretrained)
# #         else:
# #             base = models.vgg16(weights=pretrained)

# #         self.features = base.features

# #         self.classifier = nn.Sequential(
# #             nn.Linear(512 * 7 * 7, 1024),
# #             nn.Sigmoid(),
# #             nn.Dropout(p=0.5),
# #             nn.Linear(1024, 512),
# #             nn.Sigmoid(),
# #             nn.Dropout(p=0.5),
# #             nn.Linear(512, 1),
# #         )

# #         self._init_weights()

# #     def _init_weights(self):
# #         for m in self.modules():
# #             if isinstance(m, nn.Conv2d):
# #                 nn.init.kaiming_normal_(
# #                     m.weight, mode="fan_out", nonlinearity="relu"
# #                 )
# #                 if m.bias is not None:
# #                     nn.init.constant_(m.bias, 0.0)
# #             elif isinstance(m, nn.BatchNorm2d):
# #                 nn.init.constant_(m.weight, 1.0)
# #                 nn.init.constant_(m.bias, 0.0)
# #             elif isinstance(m, nn.Linear):
# #                 nn.init.normal_(m.weight, 0.0, 0.01)
# #                 nn.init.constant_(m.bias, 0.0)

# #     def forward(self, x):
# #         x = self.features(x)
# #         x = x.view(x.size(0), -1)
# #         x = self.classifier(x)
# #         x = torch.squeeze(x, dim=-1)
# #         return x


# # # ============================================================
# # # 5. TRAIN / VALIDATE LOOPS
# # # ============================================================

# # def train_one_epoch(model, loader, optimizer, device, threshold):
# #     model.train()
# #     mse = nn.MSELoss(reduction="mean")
# #     running_loss = 0.0
# #     running_acc = 0.0

# #     pbar = tqdm(loader, desc="Training", leave=False)

# #     for imgs, labels in pbar:
# #         imgs = imgs.to(device, non_blocking=True)
# #         labels = labels.to(device, non_blocking=True)

# #         preds = model(imgs)
# #         loss = mse(preds, labels)

# #         optimizer.zero_grad()
# #         loss.backward()
# #         optimizer.step()

# #         acc1, _, _, _ = compute_metrics(preds, labels, threshold=threshold)

# #         running_loss += loss.item() * imgs.size(0)
# #         running_acc += acc1 * imgs.size(0)

# #         # Update progress bar display
# #         pbar.set_postfix({
# #             "loss": f"{loss.item():.4f}",
# #             "acc": f"{acc1:.2f}%"
# #         })

# #     n = len(loader.dataset)
# #     return running_loss / n, running_acc / n



# # def validate(model, loader, device, threshold):
# #     model.eval()
# #     mse = nn.MSELoss(reduction="mean")
# #     running_loss = 0.0
# #     running_acc = 0.0

# #     all_l2 = []

# #     pbar = tqdm(loader, desc="Validating", leave=False)

# #     with torch.no_grad():
# #         for imgs, labels in pbar:
# #             imgs = imgs.to(device, non_blocking=True)
# #             labels = labels.to(device, non_blocking=True)

# #             preds = model(imgs)
# #             loss = mse(preds, labels)

# #             acc1, _, _, l2_dist = compute_metrics(preds, labels, threshold=threshold)

# #             running_loss += loss.item() * imgs.size(0)
# #             running_acc += acc1 * imgs.size(0)
# #             all_l2.extend(list(l2_dist))

# #             pbar.set_postfix({
# #                 "loss": f"{loss.item():.4f}",
# #                 "acc": f"{acc1:.2f}%"
# #             })

# #     n = len(loader.dataset)
# #     mean_l2 = float(np.mean(all_l2)) if all_l2 else 0.0
# #     std_l2 = float(np.std(all_l2)) if all_l2 else 0.0

# #     return running_loss / n, running_acc / n, mean_l2, std_l2



# # # ============================================================
# # # 6. MAIN
# # # ============================================================

# # def main():
# #     parser = argparse.ArgumentParser(
# #         description="VGG-16 LAB Regression (Simplified, Single CSV Splits)"
# #     )
# #     parser.add_argument(
# #         "--csv-path",
# #         type=str,
# #         required=True,
# #         help="Path to annotations CSV (single file with filename,label[,subject_id])",
# #     )
# #     parser.add_argument(
# #         "--image-dir",
# #         type=str,
# #         required=True,
# #         help="Root directory containing images; CSV filenames are relative to this.",
# #     )
# #     parser.add_argument("--epochs", type=int, default=90)
# #     parser.add_argument("--batch-size", type=int, default=32)
# #     parser.add_argument("--val-ratio", type=float, default=0.2)
# #     parser.add_argument("--lr", type=float, default=1e-5)
# #     parser.add_argument("--momentum", type=float, default=0.9)
# #     parser.add_argument("--weight-decay", type=float, default=1e-4)
# #     parser.add_argument("--threshold", type=float, default=0.5)
# #     parser.add_argument("--use-bn", action="store_true", help="Use VGG16-BN backbone")
# #     parser.add_argument("--pretrained", action="store_true", help="Use ImageNet-pretrained weights")
# #     parser.add_argument("--blur", action="store_true", help="Apply a light blur to inputs")
# #     parser.add_argument("--gpu", type=int, default=0)
# #     parser.add_argument("--output", type=str, default="best_vgg16_lab_regressor.pth")

# #     args = parser.parse_args()

# #     device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
# #     print(f"[INFO] Using device: {device}")

# #     # --------------------------------------------------------
# #     # Data
# #     # --------------------------------------------------------
# #     train_loader, val_loader = build_dataloaders_from_single_csv(
# #         csv_path=args.csv_path,
# #         image_dir=args.image_dir,
# #         batch_size=args.batch_size,
# #         val_ratio=args.val_ratio,
# #         blur=args.blur,
# #     )

# #     # --------------------------------------------------------
# #     # Model & Optimiser
# #     # --------------------------------------------------------
# #     model = VGG16LabRegressor(use_bn=args.use_bn, pretrained=args.pretrained)
# #     model.to(device)
# #     print(f"[INFO] Model initialized. Using BN: {args.use_bn}, Pretrained: {args.pretrained}")

# #     optimizer = torch.optim.SGD(
# #         model.parameters(),
# #         lr=args.lr,
# #         momentum=args.momentum,
# #         weight_decay=args.weight_decay,
# #     )
# #     print(f"[INFO] Optimizer: SGD | LR={args.lr} | Momentum={args.momentum} | Weight Decay={args.weight_decay}")

# #     lr_scheduler = torch.optim.lr_scheduler.StepLR(
# #         optimizer, step_size=30, gamma=0.1
# #     )
# #     print(f"[INFO] LR Scheduler: StepLR | Step Size=30 | Gamma=0.1")

# #     # --------------------------------------------------------
# #     # Training loop
# #     # --------------------------------------------------------
# #     best_val_acc = 0.0

# #     for epoch in range(args.epochs):
# #         print(f"\n[INFO] Starting epoch {epoch + 1}/{args.epochs}...")
# #         train_loss, train_acc = train_one_epoch(
# #             model, train_loader, optimizer, device, args.threshold
# #         )
# #         print(f"[INFO] Epoch {epoch + 1} training complete.")
# #         val_loss, val_acc, val_l2_mean, val_l2_std = validate(
# #             model, val_loader, device, args.threshold
# #         )

# #         lr_scheduler.step()

# #         print(
# #             f"Epoch {epoch + 1:03d}/{args.epochs:03d} "
# #             f"| Train Loss: {train_loss:.4f} Acc@0.5: {train_acc:.2f}% "
# #             f"| Val Loss: {val_loss:.4f} Acc@0.5: {val_acc:.2f}% "
# #             f"| Val LAB L2: mean={val_l2_mean:.2f}, std={val_l2_std:.2f}"
# #         )

# #         if val_acc > best_val_acc:
# #             best_val_acc = val_acc
# #             torch.save(
# #                 {
# #                     "epoch": epoch + 1,
# #                     "model_state": model.state_dict(),
# #                     "optimizer_state": optimizer.state_dict(),
# #                     "best_val_acc": best_val_acc,
# #                 },
# #                 args.output,
# #             )
# #             print(f"[INFO] New best model saved to '{args.output}' (Acc@0.5 = {best_val_acc:.2f}%)")


# # if __name__ == "__main__":
# #     main()


# import argparse
# import os
# from pathlib import Path

# import numpy as np
# import pandas as pd
# from PIL import Image
# from tqdm import tqdm

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import models, transforms

# from skimage.color import rgb2lab
# from sklearn.model_selection import GroupShuffleSplit, train_test_split


# # ============================================================
# # 0. REPRODUCIBILITY & DEVICE
# # ============================================================

# def set_seed(seed: int = 42):
#     import random
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


# set_seed(42)


# # ============================================================
# # 1. MONK SCALE UTILS & LAB-SPACE METRICS
# #    (for evaluation only; training target is scalar MST index)
# # ============================================================

# # Monk Skin Tone representative RGB colours (approximate).
# # These can be refined, but this structure matches the paper’s
# # “linear interpolation between defined Monk colours”. :contentReference[oaicite:2]{index=2}
# MONK_COLORS_RGB = [
#     (246, 237, 228),  # MST 1
#     (243, 231, 219),  # MST 2
#     (247, 234, 208),  # MST 3
#     (243, 218, 186),  # MST 4
#     (215, 189, 150),  # MST 5
#     (160, 126, 86),   # MST 6
#     (130, 92, 67),    # MST 7
#     (96, 65, 52),     # MST 8
#     (58, 49, 42),     # MST 9
#     (41, 36, 32),     # MST 10
# ]

# MONK_COLORS_RGB = np.array(MONK_COLORS_RGB, dtype=np.float32)


# def _rgb_vec_to_lab(rgb_vec: np.ndarray) -> np.ndarray:
#     """
#     Convert an array of RGB vectors in [0,255] to LAB using skimage.
#     Shape: (N, 3) -> (N, 3)
#     """
#     rgb_norm = rgb_vec.reshape(-1, 1, 1, 3) / 255.0  # (N,1,1,3)
#     lab = rgb2lab(rgb_norm)  # (N,1,1,3), L in [0,100], a,b in [-128,127]
#     return lab.reshape(-1, 3).astype(np.float32)


# # Precompute LAB positions of the discrete Monk colours for speed
# MONK_COLORS_LAB = _rgb_vec_to_lab(MONK_COLORS_RGB)


# def monk_scalar_to_lab(values: np.ndarray) -> np.ndarray:
#     """
#     Map continuous Monk index (1–10, possibly fractional) to LAB via
#     linear interpolation between the 10 discrete Monk LAB colours,
#     following the description in Section 2.2.6. :contentReference[oaicite:3]{index=3}

#     values: shape (N,)
#     returns: shape (N,3) in LAB.
#     """
#     labs = []
#     for v in values:
#         v_float = float(v)
#         # Clamp to [1, 10] then convert to 0-based index
#         v_clamped = max(1.0, min(10.0, v_float))
#         base = int(np.floor(v_clamped))  # 1..10
#         if base >= 10:
#             labs.append(MONK_COLORS_LAB[-1])
#             continue

#         idx0 = base - 1           # 0..8
#         idx1 = base               # 1..9
#         frac = v_clamped - base   # in [0,1)

#         lab0 = MONK_COLORS_LAB[idx0]
#         lab1 = MONK_COLORS_LAB[idx1]
#         lab_interp = lab0 + (lab1 - lab0) * frac
#         labs.append(lab_interp.astype(np.float32))
#     return np.stack(labs, axis=0)


# def calc_l2_lab_distance(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
#     """
#     Compute L2 distance in LAB between predicted and target Monk indices.
#     This mirrors the paper’s “Average LAB L2 distance error”. :contentReference[oaicite:4]{index=4}
#     """
#     pred_np = pred.detach().cpu().numpy()
#     targ_np = target.detach().cpu().numpy()

#     pred_lab = monk_scalar_to_lab(pred_np)
#     targ_lab = monk_scalar_to_lab(targ_np)

#     diff = pred_lab - targ_lab
#     return np.sqrt(np.sum(diff * diff, axis=1))


# def compute_metrics(output, target, threshold=0.5, cum_max_steps=20, factor=2.0):
#     """
#     Core metrics:
#     - acc@1: % of predictions within ± threshold of label (Monk units)
#     - acc@2: % of predictions within ± threshold*factor
#     - cum_acc: cumulative accuracy curve in steps of 0.1 (0.0 .. 1.9)
#     - l2_distances: LAB L2 errors for each example
#     """
#     with torch.no_grad():
#         out = output.detach().cpu().numpy()
#         tgt = target.detach().cpu().numpy()
#         batch_size = tgt.shape[0]

#         abs_diff = np.abs(out - tgt)

#         acc1 = (abs_diff <= threshold).sum() * 100.0 / batch_size
#         acc2 = (abs_diff <= threshold * factor).sum() * 100.0 / batch_size

#         cum_acc = []
#         for delta_idx in range(cum_max_steps):
#             delta = delta_idx / 10.0
#             cum_acc.append((abs_diff < delta).sum() * 100.0 / batch_size)

#         l2_distances = calc_l2_lab_distance(output, target)

#         return acc1, acc2, cum_acc, l2_distances


# # ============================================================
# # 2. LAB TRANSFORMS (RGB -> LAB, GEOMETRIC AUGS IN RGB)
# # ============================================================

# # LAB normalisation chosen to respect LAB domain and provide
# # roughly zero-mean, unit-variance channels.
# LAB_MEAN = np.array([50.0, 0.0, 0.0], dtype=np.float32)
# LAB_STD = np.array([50.0, 128.0, 128.0], dtype=np.float32)


# class RGBToLABTensorTransform:
#     """
#     Composite transform:
#       1) apply geometric transforms in RGB (PIL)
#       2) convert to numpy RGB [0,1]
#       3) convert to LAB
#       4) normalise using LAB_MEAN / LAB_STD
#       5) return torch.FloatTensor (3,H,W)
#     """

#     def __init__(self, is_train: bool):
#         if is_train:
#             self.geom = transforms.Compose(
#                 [
#                     transforms.RandomResizedCrop(224),
#                     transforms.RandomHorizontalFlip(),
#                 ]
#             )
#         else:
#             self.geom = transforms.Compose(
#                 [
#                     transforms.Resize(256),
#                     transforms.CenterCrop(224),
#                 ]
#             )

#     def __call__(self, img_pil: Image.Image) -> torch.Tensor:
#         # Step 1: geometric aug in RGB domain
#         img_pil = self.geom(img_pil)

#         # Step 2: RGB -> np.float32 in [0,1]
#         rgb = np.asarray(img_pil).astype(np.float32) / 255.0  # (H,W,3)

#         # Step 3: RGB -> LAB
#         lab = rgb2lab(rgb).astype(np.float32)  # L[0,100], a,b≈[-128,127]

#         # Step 4: normalise
#         lab_norm = (lab - LAB_MEAN) / LAB_STD  # (H,W,3)

#         # Step 5: to tensor (3,H,W)
#         lab_tensor = torch.from_numpy(lab_norm.transpose(2, 0, 1))  # CHW
#         return lab_tensor.float()


# # ============================================================
# # 3. DATASET & DATALOADER BUILDER
# # ============================================================

# class SkinToneRegressionDataset(Dataset):
#     """
#     Dataset directly aligned with the paper:
#       - input = LAB image (3x224x224) normalised per LAB_MEAN/STD
#       - target = scalar Monk index (float between 1 and 10)
#     """

#     def __init__(self, annotations_df: pd.DataFrame, img_dir: str, transform):
#         self.df = annotations_df.reset_index(drop=True)
#         self.img_dir = Path(img_dir)
#         self.transform = transform

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         filename = str(row.iloc[0])
#         label = float(row.iloc[1])

#         img_path = self.img_dir / filename
#         img = Image.open(img_path).convert("RGB")

#         img_lab_tensor = self.transform(img)

#         label_tensor = torch.tensor(label, dtype=torch.float32)
#         return img_lab_tensor, label_tensor


# def build_dataloaders_from_single_csv(
#     csv_path: str,
#     image_dir: str,
#     batch_size: int,
#     val_ratio: float,
# ):
#     """
#     Builds train/validation DataLoaders from a single CSV.

#     CSV expected structure:
#         filename,label[,subject_id]

#     Behaviour mirrors the paper:
#       - If subject_id exists -> non-mixed identities split.
#       - Else, we attempt folder-based grouping.
#       - Else, fallback to stratified split on label. :contentReference[oaicite:5]{index=5}
#     """

#     df = pd.read_csv(csv_path)
#     df = df.dropna(subset=[df.columns[0], df.columns[1]])

#     filenames = df.iloc[:, 0].astype(str).values
#     labels = df.iloc[:, 1].astype(float).values

#     # Identity/group handling
#     if "subject_id" in df.columns:
#         print("[INFO] Found 'subject_id' column — using identity-aware grouped split.")
#         groups = df["subject_id"].astype(str).values
#     else:
#         print("[INFO] No 'subject_id' column — inferring IDs from directory structure.")
#         inferred_groups = []
#         for fname in filenames:
#             parts = fname.replace("\\", "/").split("/")
#             if len(parts) > 1:
#                 inferred_groups.append(parts[0])
#             else:
#                 inferred_groups.append("no_group")

#         unique_groups = set(inferred_groups)
#         if "no_group" not in unique_groups and len(unique_groups) > 1:
#             print(
#                 f"[INFO] Inferred {len(unique_groups)} identity groups from folder structure."
#             )
#             groups = np.array(inferred_groups)
#         else:
#             print("[INFO] No usable identity grouping found — using stratified split.")
#             groups = None

#     # Split
#     if groups is not None:
#         splitter = GroupShuffleSplit(
#             n_splits=1, test_size=val_ratio, random_state=42
#         )
#         train_idx, val_idx = next(splitter.split(df, labels, groups))
#         print("[INFO] Using GROUPED (non-mixed identities) split.")
#     else:
#         train_idx, val_idx = train_test_split(
#             np.arange(len(df)),
#             test_size=val_ratio,
#             shuffle=True,
#             stratify=labels,
#             random_state=42,
#         )
#         print("[INFO] Using STRATIFIED split (label balanced).")

#     train_df = df.iloc[train_idx].reset_index(drop=True)
#     val_df = df.iloc[val_idx].reset_index(drop=True)

#     print(
#         f"[INFO] Total={len(df)} | Train={len(train_df)} | Val={len(val_df)} "
#         f"(val_ratio={val_ratio})"
#     )

#     # Transforms
#     train_transform = RGBToLABTensorTransform(is_train=True)
#     val_transform = RGBToLABTensorTransform(is_train=False)

#     # Datasets
#     train_ds = SkinToneRegressionDataset(
#         annotations_df=train_df,
#         img_dir=image_dir,
#         transform=train_transform,
#     )
#     val_ds = SkinToneRegressionDataset(
#         annotations_df=val_df,
#         img_dir=image_dir,
#         transform=val_transform,
#     )

#     # DataLoaders
#     train_loader = DataLoader(
#         train_ds,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True,
#     )
#     val_loader = DataLoader(
#         val_ds,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=4,
#         pin_memory=True,
#     )

#     return train_loader, val_loader


# # ============================================================
# # 4. MODEL: VGG-16 LAB REGRESSOR (PAPER-ALIGNED)
# # ============================================================

# class VGG16LabRegressor(nn.Module):
#     """
#     VGG-16 regression model trained on LAB images as per the paper:

#       - VGG-16 backbone (with BN recommended by the authors). :contentReference[oaicite:6]{index=6}
#       - Fully connected head:
#           512*7*7 -> 1024 (Sigmoid + Dropout 0.5)
#                   -> 512  (Sigmoid + Dropout 0.5)
#                   -> 1    (linear scalar MST index)
#     """

#     def __init__(self, use_bn: bool = True, pretrained: bool = False, dropout_p: float = 0.5):
#         super().__init__()

#         if use_bn:
#             base = models.vgg16_bn(weights=None if not pretrained else models.VGG16_BN_Weights.IMAGENET1K_V1)
#         else:
#             base = models.vgg16(weights=None if not pretrained else models.VGG16_Weights.IMAGENET1K_V1)

#         self.features = base.features  # convolutional part

#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 1024),
#             nn.Sigmoid(),
#             nn.Dropout(p=dropout_p),
#             nn.Linear(1024, 512),
#             nn.Sigmoid(),
#             nn.Dropout(p=dropout_p),
#             nn.Linear(512, 1),
#         )

#         self._init_weights()

#     def _init_weights(self):
#         # He/Kaiming for conv, small normal for linear – common VGG init.
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1.0)
#                 nn.init.constant_(m.bias, 0.0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0.0, 0.01)
#                 nn.init.constant_(m.bias, 0.0)

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         # Output scalar MST index
#         return x.squeeze(-1)


# # ============================================================
# # 5. TRAIN / VALIDATE LOOPS
# # ============================================================

# def train_one_epoch(model, loader, optimizer, device, threshold):
#     model.train()
#     mse = nn.MSELoss(reduction="mean")
#     running_loss = 0.0
#     running_acc = 0.0

#     pbar = tqdm(loader, desc="Training", leave=False)

#     for imgs, labels in pbar:
#         imgs = imgs.to(device, non_blocking=True)
#         labels = labels.to(device, non_blocking=True)

#         preds = model(imgs)
#         loss = mse(preds, labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         acc1, _, _, _ = compute_metrics(preds, labels, threshold=threshold)

#         running_loss += loss.item() * imgs.size(0)
#         running_acc += acc1 * imgs.size(0)

#         pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc@{:.1f}".format(threshold): f"{acc1:.2f}%"})

#     n = len(loader.dataset)
#     return running_loss / n, running_acc / n


# def validate(model, loader, device, threshold):
#     model.eval()
#     mse = nn.MSELoss(reduction="mean")
#     running_loss = 0.0
#     running_acc = 0.0

#     all_l2 = []

#     pbar = tqdm(loader, desc="Validating", leave=False)

#     with torch.no_grad():
#         for imgs, labels in pbar:
#             imgs = imgs.to(device, non_blocking=True)
#             labels = labels.to(device, non_blocking=True)

#             preds = model(imgs)
#             loss = mse(preds, labels)

#             acc1, _, _, l2_dist = compute_metrics(preds, labels, threshold=threshold)

#             running_loss += loss.item() * imgs.size(0)
#             running_acc += acc1 * imgs.size(0)
#             all_l2.extend(list(l2_dist))

#             pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc@{:.1f}".format(threshold): f"{acc1:.2f}%"})

#     n = len(loader.dataset)
#     mean_l2 = float(np.mean(all_l2)) if all_l2 else 0.0
#     std_l2 = float(np.std(all_l2)) if all_l2 else 0.0

#     return running_loss / n, running_acc / n, mean_l2, std_l2


# # ============================================================
# # 6. MAIN
# # ============================================================

# def main():
#     parser = argparse.ArgumentParser(
#         description="Paper-aligned VGG-16 LAB Regression (Mbatha et al. 2024)"
#     )
#     parser.add_argument(
#         "--csv-path",
#         type=str,
#         required=True,
#         help="Path to annotations CSV (filename,label[,subject_id]).",
#     )
#     parser.add_argument(
#         "--image-dir",
#         type=str,
#         required=True,
#         help="Root directory containing images; CSV filenames are relative to this.",
#     )
#     parser.add_argument("--epochs", type=int, default=90)
#     parser.add_argument("--batch-size", type=int, default=32)  # Table 1 :contentReference[oaicite:7]{index=7}
#     parser.add_argument("--val-ratio", type=float, default=0.35)  # paper uses ~65/35 :contentReference[oaicite:8]{index=8}
#     parser.add_argument("--lr", type=float, default=1e-5)  # tuned final LR :contentReference[oaicite:9]{index=9}
#     parser.add_argument("--momentum", type=float, default=0.9)
#     parser.add_argument("--weight-decay", type=float, default=1e-4)
#     parser.add_argument("--threshold", type=float, default=0.5)
#     parser.add_argument("--use-bn", action="store_true", help="Use VGG16-BN backbone (recommended).")
#     parser.add_argument(
#         "--pretrained",
#         action="store_true",
#         help="Use ImageNet-pretrained weights (NOT the paper’s best model; for ablations).",
#     )
#     parser.add_argument("--gpu", type=int, default=0)
#     parser.add_argument("--output", type=str, default="best_vgg16_lab_regressor.pth")

#     args = parser.parse_args()

#     device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
#     print(f"[INFO] Using device: {device}")

#     # --------------------------------------------------------
#     # Data
#     # --------------------------------------------------------
#     train_loader, val_loader = build_dataloaders_from_single_csv(
#         csv_path=args.csv_path,
#         image_dir=args.image_dir,
#         batch_size=args.batch_size,
#         val_ratio=args.val_ratio,
#     )

#     # --------------------------------------------------------
#     # Model & Optimiser
#     # --------------------------------------------------------
#     model = VGG16LabRegressor(
#         use_bn=args.use_bn,
#         pretrained=args.pretrained,
#         dropout_p=0.5,
#     )
#     model.to(device)
#     print(
#         f"[INFO] Model initialized. "
#         f"Backbone={'VGG16-BN' if args.use_bn else 'VGG16'}, "
#         f"Pretrained={args.pretrained}"
#     )

#     optimizer = torch.optim.SGD(
#         model.parameters(),
#         lr=args.lr,
#         momentum=args.momentum,
#         weight_decay=args.weight_decay,
#     )
#     print(
#         f"[INFO] Optimizer: SGD | LR={args.lr} | Momentum={args.momentum} "
#         f"| Weight Decay={args.weight_decay}"
#     )

#     lr_scheduler = torch.optim.lr_scheduler.StepLR(
#         optimizer, step_size=30, gamma=0.1
#     )
#     print(f"[INFO] LR Scheduler: StepLR | Step Size=30 | Gamma=0.1")

#     # --------------------------------------------------------
#     # Training loop
#     # --------------------------------------------------------
#     best_val_acc = 0.0

#     for epoch in range(args.epochs):
#         print(f"\n[INFO] Starting epoch {epoch + 1}/{args.epochs}...")
#         train_loss, train_acc = train_one_epoch(
#             model, train_loader, optimizer, device, args.threshold
#         )
#         val_loss, val_acc, val_l2_mean, val_l2_std = validate(
#             model, val_loader, device, args.threshold
#         )

#         lr_scheduler.step()

#         print(
#             f"Epoch {epoch + 1:03d}/{args.epochs:03d} "
#             f"| Train Loss: {train_loss:.4f} Acc@{args.threshold:.1f}: {train_acc:.2f}% "
#             f"| Val Loss: {val_loss:.4f} Acc@{args.threshold:.1f}: {val_acc:.2f}% "
#             f"| Val LAB L2: mean={val_l2_mean:.2f}, std={val_l2_std:.2f}"
#         )

#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save(
#                 {
#                     "epoch": epoch + 1,
#                     "model_state": model.state_dict(),
#                     "optimizer_state": optimizer.state_dict(),
#                     "best_val_acc": best_val_acc,
#                 },
#                 args.output,
#             )
#             print(
#                 f"[INFO] New best model saved to '{args.output}' "
#                 f"(Acc@{args.threshold:.1f} = {best_val_acc:.2f}%)"
#             )


# if __name__ == "__main__":
#     main()


# # # ##############################################################
# # # #  VGG16-LAB TRAINING PIPELINE (FIXED + DIAGNOSTICS ENABLED)
# # # #  - Enforces EXACT 224x224 input size
# # # #  - Adds dynamic feature-size inference (no more 7x7 hardcode)
# # # #  - Keeps all debugging and LAB diagnostics
# # # ##############################################################

# # # import argparse
# # # from pathlib import Path

# # # from altair import param
# # # import numpy as np
# # # import pandas as pd
# # # from PIL import Image
# # # from tqdm import tqdm

# # # import torch
# # # import torch.nn as nn
# # # from torch.utils.data import Dataset, DataLoader
# # # from torchvision import models, transforms

# # # from skimage.color import rgb2lab
# # # from sklearn.model_selection import train_test_split


# # # ##############################################################
# # # # 0. REPRODUCIBILITY
# # # ##############################################################

# # # def set_seed(seed=42):
# # #     import random
# # #     random.seed(seed)
# # #     np.random.seed(seed)
# # #     torch.manual_seed(seed)
# # #     torch.cuda.manual_seed_all(seed)
# # #     torch.backends.cudnn.deterministic = True
# # #     torch.backends.cudnn.benchmark = False

# # # set_seed()


# # # ##############################################################
# # # # 1. MONK → LAB utilities
# # # ##############################################################

# # # MONK_COLORS_RGB = np.array([
# # #     (246, 237, 228),
# # #     (243, 231, 219),
# # #     (247, 234, 208),
# # #     (243, 218, 186),
# # #     (215, 189, 150),
# # #     (160, 126, 86),
# # #     (130, 92, 67),
# # #     (96, 65, 52),
# # #     (58, 49, 42),
# # #     (41, 36, 32),
# # # ], dtype=np.float32)

# # # def _rgb_to_lab(rgb_vec):
# # #     rgb = rgb_vec.reshape(-1,1,1,3) / 255.0
# # #     lab = rgb2lab(rgb)
# # #     return lab.reshape(-1,3).astype(np.float32)

# # # MONK_COLORS_LAB = _rgb_to_lab(MONK_COLORS_RGB)

# # # def monk_scalar_to_lab(values):
# # #     labs = []
# # #     for v in values:
# # #         v = float(max(1, min(10, v)))
# # #         base = int(v)
# # #         if base >= 10:
# # #             labs.append(MONK_COLORS_LAB[-1])
# # #             continue
# # #         lab0 = MONK_COLORS_LAB[base-1]
# # #         lab1 = MONK_COLORS_LAB[base]
# # #         labs.append(lab0 + (v-base)*(lab1-lab0))
# # #     return np.stack(labs)

# # # def calc_l2_lab_distance(pred, target):
# # #     # p = pred.detach().cpu().numpy()
# # #     # t = target.detach().cpu().numpy()
# # #     p = pred.detach().cpu().numpy() * 9 + 1
# # #     t = target.detach().cpu().numpy() * 9 + 1

# # #     return np.sqrt(np.sum((monk_scalar_to_lab(p)-monk_scalar_to_lab(t))**2, axis=1))


# # # ##############################################################
# # # # 2. LAB Transform + Diagnostics
# # # ##############################################################

# # # LAB_MEAN = np.array([50.0, 0.0, 0.0], dtype=np.float32)
# # # LAB_STD = np.array([50.0, 128.0, 128.0], dtype=np.float32)

# # # class RGBToLABTensorTransform:
# # #     first_call = True

# # #     def __init__(self, is_train=True):
# # #         if is_train:
# # #             self.geom = transforms.Compose([
# # #                 transforms.Resize((224,224)),
# # #                 transforms.RandomHorizontalFlip(),
# # #             ])
# # #         else:
# # #             self.geom = transforms.Compose([
# # #                 transforms.Resize((224,224)),
# # #             ])

# # #     def __call__(self, img_pil):
# # #         img_pil = self.geom(img_pil)

# # #         rgb = np.asarray(img_pil).astype(np.float32)/255.0
# # #         lab = rgb2lab(rgb).astype(np.float32)

# # #         lab_norm = (lab - LAB_MEAN) / LAB_STD

# # #         if RGBToLABTensorTransform.first_call:
# # #             RGBToLABTensorTransform.first_call = False
# # #             print("\n[DIAG] LAB BEFORE NORMALIZATION:")
# # #             print("  L range:", lab[...,0].min(), lab[...,0].max())
# # #             print("  a range:", lab[...,1].min(), lab[...,1].max())
# # #             print("  b range:", lab[...,2].min(), lab[...,2].max())
# # #             print("[DIAG] LAB AFTER NORMALIZATION:")
# # #             print("  min:", lab_norm.min(), "max:", lab_norm.max())
# # #             print("  mean:", lab_norm.mean(), "std:", lab_norm.std(), "\n")

# # #         return torch.from_numpy(lab_norm.transpose(2,0,1)).float()


# # # ##############################################################
# # # # 3. Dataset + Dataloaders
# # # ##############################################################

# # # class SkinToneRegressionDataset(Dataset):
# # #     def __init__(self, df, img_dir, transform):
# # #         self.df = df.reset_index(drop=True)
# # #         self.img_dir = Path(img_dir)
# # #         self.transform = transform

# # #     def __len__(_): return len(_.df)

# # #     def __getitem__(self, idx):
# # #         row = self.df.iloc[idx]
# # #         img = Image.open(self.img_dir / str(row.iloc[0])).convert("RGB")
# # #         label = float(row.iloc[1])
# # #         label = (label - 1.0) / 9.0      # MST→0–1 scaling
# # #         return self.transform(img), torch.tensor(label, dtype=torch.float32)

# # # # def build_dataloaders(csv_path, image_dir, batch_size, val_ratio):
# # # #     df = pd.read_csv(csv_path).dropna()
# # # #     labels = df.iloc[:,1].astype(float)

# # # #     train_idx, val_idx = train_test_split(
# # # #         np.arange(len(df)), test_size=val_ratio, stratify=labels, random_state=42
# # # #     )

# # # #     train_df = df.iloc[train_idx].reset_index(drop=True)
# # # #     val_df = df.iloc[val_idx].reset_index(drop=True)

# # # #     print(f"[INFO] Train={len(train_df)} | Val={len(val_df)}")

# # # #     train_tf = RGBToLABTensorTransform(is_train=True)
# # # #     val_tf = RGBToLABTensorTransform(is_train=False)

# # # #     return (
# # # #         DataLoader(SkinToneRegressionDataset(train_df, image_dir, train_tf),
# # # #                    batch_size=batch_size, shuffle=True, num_workers=4),
# # # #         DataLoader(SkinToneRegressionDataset(val_df, image_dir, val_tf),
# # # #                    batch_size=batch_size, shuffle=False, num_workers=4)
# # # #     )

# # # def build_dataloaders(csv_path, image_dir, batch_size, val_ratio):
# # #     df = pd.read_csv(csv_path).dropna()
# # #     labels = df.iloc[:,1].astype(float)

# # #     # ------------------------------------------------------------
# # #     # AUTO-DETECT CONTINUOUS LABELS (regression) vs DISCRETE LABELS
# # #     # ------------------------------------------------------------
# # #     unique_labels = np.unique(labels)

# # #     continuous_labels = (
# # #         labels.dtype == float and 
# # #         (len(unique_labels) > 20 or np.any(labels % 1 != 0))
# # #     )

# # #     if continuous_labels:
# # #         print("[INFO] Continuous labels detected → RANDOM split (no stratification).")
# # #         stratify_arg = None
# # #     else:
# # #         print("[INFO] Discrete labels detected → STRATIFIED split.")
# # #         stratify_arg = labels

# # #     # ------------------------------------------------------------
# # #     # Perform split
# # #     # ------------------------------------------------------------
# # #     train_idx, val_idx = train_test_split(
# # #         np.arange(len(df)),
# # #         test_size=val_ratio,
# # #         shuffle=True,
# # #         stratify=stratify_arg,
# # #         random_state=42,
# # #     )

# # #     train_df = df.iloc[train_idx].reset_index(drop=True)
# # #     val_df = df.iloc[val_idx].reset_index(drop=True)

# # #     print(f"[INFO] Train={len(train_df)} | Val={len(val_df)}")

# # #     train_tf = RGBToLABTensorTransform(is_train=True)
# # #     val_tf = RGBToLABTensorTransform(is_train=False)

# # #     return (
# # #         DataLoader(SkinToneRegressionDataset(train_df, image_dir, train_tf),
# # #                    batch_size=batch_size, shuffle=True, num_workers=4),
# # #         DataLoader(SkinToneRegressionDataset(val_df, image_dir, val_tf),
# # #                    batch_size=batch_size, shuffle=False, num_workers=4)
# # #     )



# # # ##############################################################
# # # # 4. VGG Model (Dynamic Flatten Size)
# # # ##############################################################

# # # class VGG16LabRegressor(nn.Module):
# # #     def __init__(self, use_bn=True, dropout_p=0.5):
# # #         super().__init__()

# # #         base = models.vgg16_bn(weights=None) if use_bn else models.vgg16(weights=None)
# # #         self.features = base.features

# # #         # for param in self.features[:20].parameters():
# # #         #     param.requires_grad = False

# # #         # === Dynamic flatten size inference ===
# # #         with torch.no_grad():
# # #             dummy = torch.zeros(1, 3, 224, 224)
# # #             feat = self.features(dummy)
# # #             flat_dim = feat.view(1, -1).shape[1]
# # #             print(f"[INFO] VGG feature dimension: {flat_dim}")

# # #         # self.classifier = nn.Sequential(
# # #         #     nn.Linear(flat_dim, 1024),
# # #         #     nn.Sigmoid(),
# # #         #     nn.Dropout(dropout_p),
# # #         #     nn.Linear(1024, 512),
# # #         #     nn.Sigmoid(),
# # #         #     nn.Dropout(dropout_p),
# # #         #     nn.Linear(512, 1),
# # #         # )

# # #         self.classifier = nn.Sequential(
# # #             nn.Linear(flat_dim, 1024),
# # #             nn.ReLU(inplace=True),
# # #             nn.Dropout(dropout_p),
# # #             nn.Linear(1024, 512),
# # #             nn.ReLU(inplace=True),
# # #             nn.Dropout(dropout_p),
# # #             nn.Linear(512, 1),
# # #         )


# # #         self._init_weights()

# # #     def _init_weights(self):
# # #         for m in self.modules():
# # #             if isinstance(m, nn.Conv2d):
# # #                 nn.init.kaiming_normal_(m.weight)
# # #                 if m.bias is not None: nn.init.zeros_(m.bias)
# # #             elif isinstance(m, nn.Linear):
# # #                 nn.init.normal_(m.weight, 0, 0.01)
# # #                 nn.init.zeros_(m.bias)

# # #     def forward(self, x):
# # #         x = self.features(x)
# # #         x = x.view(x.size(0), -1)
# # #         return self.classifier(x).squeeze(-1)


# # # ##############################################################
# # # # 5. Metrics
# # # ##############################################################

# # # def compute_metrics(pred, tgt, threshold=0.5):
# # #     pred_np = pred.detach().cpu().numpy()
# # #     tgt_np = tgt.detach().cpu().numpy()
    
# # #     pred_np = pred_np * 9 + 1
# # #     tgt_np  = tgt_np  * 9 + 1

# # #     abs_diff = np.abs(pred_np - tgt_np)
# # #     acc1 = (abs_diff <= threshold).mean() * 100
# # #     l2 = calc_l2_lab_distance(pred, tgt)
# # #     return acc1, l2


# # # ##############################################################
# # # # 6. Diagnostics
# # # ##############################################################

# # # def print_diagnostics(imgs, labels, preds, tag):
# # #     print(f"\n=== [DIAG] {tag} ===")
# # #     print("IMG mean/std:", imgs.mean().item(), imgs.std().item())
# # #     print("IMG min/max:", imgs.min().item(), imgs.max().item())
# # #     print("Labels min/max:", labels.min().item(), labels.max().item())
# # #     print("Preds min/max:", preds.min().item(), preds.max().item())
# # #     if preds.std().item() < 1e-4:
# # #         print("WARNING: predictions collapsed to near-constant.")
# # #     print("===========================================\n")


# # # ##############################################################
# # # # 7. Train / Validate
# # # ##############################################################

# # # def train_one_epoch(model, loader, optim, device, epoch):
# # #     model.train()
# # #     mse = nn.MSELoss()
# # #     total_loss = 0
# # #     total_acc = 0

# # #     for i,(imgs,labels) in enumerate(loader):
# # #         imgs,labels = imgs.to(device),labels.to(device)

# # #         preds = model(imgs)
# # #         loss = mse(preds,labels)

# # #         optim.zero_grad()
# # #         loss.backward()
# # #         optim.step()

# # #         acc,_ = compute_metrics(preds,labels)

# # #         total_loss += loss.item()*imgs.size(0)
# # #         total_acc += acc*imgs.size(0)

# # #         if i == 0:
# # #             print_diagnostics(imgs, labels, preds, f"TRAIN EPOCH {epoch}")

# # #     n = len(loader.dataset)
# # #     return total_loss/n, total_acc/n


# # # def validate(model, loader, device, epoch):
# # #     model.eval()
# # #     mse = nn.MSELoss()
# # #     total_loss = 0
# # #     total_acc = 0
# # #     all_l2 = []

# # #     with torch.no_grad():
# # #         for i,(imgs,labels) in enumerate(loader):
# # #             imgs,labels = imgs.to(device),labels.to(device)
# # #             preds = model(imgs)

# # #             loss = mse(preds,labels)
# # #             acc,l2 = compute_metrics(preds,labels)

# # #             total_loss += loss.item()*imgs.size(0)
# # #             total_acc += acc*imgs.size(0)
# # #             all_l2.extend(l2)

# # #             if i == 0:
# # #                 print_diagnostics(imgs, labels, preds, f"VALID EPOCH {epoch}")

# # #     n = len(loader.dataset)
# # #     return total_loss/n, total_acc/n, float(np.mean(all_l2)), float(np.std(all_l2))


# # # ##############################################################
# # # # 8. MAIN
# # # ##############################################################

# # # def main():
# # #     parser = argparse.ArgumentParser()
# # #     parser.add_argument("--csv-path", required=True)
# # #     parser.add_argument("--image-dir", required=True)
# # #     parser.add_argument("--epochs", type=int, default=32)
# # #     parser.add_argument("--batch-size", type=int, default=32)
# # #     parser.add_argument("--val-ratio", type=float, default=0.35)
# # #     parser.add_argument("--lr", type=float, default=1e-5)
# # #     parser.add_argument("--momentum", type=float, default=0.9)
# # #     parser.add_argument("--weight-decay", type=float, default=1e-4)
# # #     parser.add_argument("--threshold", type=float, default=0.5)
# # #     parser.add_argument("--gpu", type=int, default=0)
# # #     parser.add_argument("--use-bn", action="store_true")
# # #     args = parser.parse_args()

# # #     device = torch.device(f"cuda:{args.gpu}")

# # #     train_loader, val_loader = build_dataloaders(
# # #         args.csv_path, args.image_dir, args.batch_size, args.val_ratio
# # #     )

# # #     model = VGG16LabRegressor(use_bn=args.use_bn).to(device)

# # #     optimizer = torch.optim.SGD(
# # #         model.parameters(),
# # #         lr=args.lr,
# # #         momentum=args.momentum,
# # #         weight_decay=args.weight_decay
# # #     )

# # #     for epoch in range(1, args.epochs+1):
# # #         print(f"\n===== EPOCH {epoch}/{args.epochs} =====")
# # #         tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, epoch)
# # #         va_loss, va_acc, l2_mean, l2_std = validate(model, val_loader, device, epoch)

# # #         print(
# # #             f"[EPOCH {epoch}] "
# # #             f"TrainLoss={tr_loss:.4f} TrainAcc={tr_acc:.2f}% | "
# # #             f"ValLoss={va_loss:.4f} ValAcc={va_acc:.2f}% | "
# # #             f"L2 mean={l2_mean:.3f}, std={l2_std:.3f}"
# # #         )

# # # if __name__ == "__main__":
# # #     main()


# # ##############################################################
# # #  VGG16-LAB TRAINING PIPELINE (SAVE-ENABLED + DIAGNOSTICS)
# # #  - Enforces EXACT 224x224 input size
# # #  - Adds dynamic feature-size inference (no more 7x7 hardcode)
# # #  - Saves best model + final model
# # ##############################################################

# # import argparse
# # from pathlib import Path

# # import numpy as np
# # import pandas as pd
# # from PIL import Image
# # from tqdm import tqdm

# # import torch
# # import torch.nn as nn
# # from torch.utils.data import Dataset, DataLoader
# # from torchvision import models, transforms

# # from skimage.color import rgb2lab
# # from sklearn.model_selection import train_test_split


# # ##############################################################
# # # 0. REPRODUCIBILITY
# # ##############################################################

# # def set_seed(seed=42):
# #     import random
# #     random.seed(seed)
# #     np.random.seed(seed)
# #     torch.manual_seed(seed)
# #     torch.cuda.manual_seed_all(seed)
# #     torch.backends.cudnn.deterministic = True
# #     torch.backends.cudnn.benchmark = False

# # set_seed()


# # ##############################################################
# # # 1. MONK → LAB utilities
# # ##############################################################

# # MONK_COLORS_RGB = np.array([
# #     (246, 237, 228),
# #     (243, 231, 219),
# #     (247, 234, 208),
# #     (243, 218, 186),
# #     (215, 189, 150),
# #     (160, 126, 86),
# #     (130, 92, 67),
# #     (96, 65, 52),
# #     (58, 49, 42),
# #     (41, 36, 32),
# # ], dtype=np.float32)


# # def _rgb_to_lab(rgb_vec):
# #     rgb = rgb_vec.reshape(-1,1,1,3) / 255.0
# #     lab = rgb2lab(rgb)
# #     return lab.reshape(-1,3).astype(np.float32)


# # MONK_COLORS_LAB = _rgb_to_lab(MONK_COLORS_RGB)


# # def monk_scalar_to_lab(values):
# #     labs = []
# #     for v in values:
# #         v = float(max(1, min(10, v)))
# #         base = int(v)
# #         if base >= 10:
# #             labs.append(MONK_COLORS_LAB[-1])
# #             continue
# #         lab0 = MONK_COLORS_LAB[base-1]
# #         lab1 = MONK_COLORS_LAB[base]
# #         labs.append(lab0 + (v-base)*(lab1-lab0))
# #     return np.stack(labs)


# # def calc_l2_lab_distance(pred, target):
# #     # convert back from 0–1 regression space to MST 1–10
# #     p = pred.detach().cpu().numpy() * 9 + 1
# #     t = target.detach().cpu().numpy() * 9 + 1

# #     return np.sqrt(np.sum((monk_scalar_to_lab(p)-monk_scalar_to_lab(t))**2, axis=1))


# # ##############################################################
# # #  LAB DATASET STATISTICS (GLOBAL MEAN / STD)
# # ##############################################################

# # def compute_dataset_lab_stats(image_dir, csv_path):
# #     """
# #     Computes LAB mean/std on the dataset AFTER resizing to 224×224,
# #     using ONLY the filenames referenced in the CSV (not the entire folder).
# #     This creates consistent LAB statistics for normalization.
# #     """
# #     df = pd.read_csv(csv_path)
# #     img_names = df.iloc[:, 0].tolist()

# #     print(f"[INFO] Computing LAB dataset statistics on {len(img_names)} images...")

# #     resize_op = transforms.Resize((224, 224))

# #     sum_lab = np.zeros(3, dtype=np.float64)
# #     sum_sq_lab = np.zeros(3, dtype=np.float64)
# #     total_pixels = 0

# #     for name in tqdm(img_names, desc="LAB stats"):
# #         fpath = Path(image_dir) / name

# #         try:
# #             img = Image.open(fpath).convert("RGB")
# #             img = resize_op(img)

# #             rgb = np.asarray(img).astype(np.float32) / 255.0
# #             lab = rgb2lab(rgb).astype(np.float64)  # (224,224,3)

# #             lab_flat = lab.reshape(-1, 3)
# #             sum_lab += lab_flat.sum(axis=0)
# #             sum_sq_lab += (lab_flat ** 2).sum(axis=0)
# #             total_pixels += lab_flat.shape[0]

# #         except Exception as e:
# #             print(f"[WARN] Could not read {fpath}: {e}")

# #     mean = sum_lab / total_pixels
# #     var = (sum_sq_lab / total_pixels) - (mean ** 2)
# #     std = np.sqrt(var)

# #     print("\n========== LAB DATASET STATISTICS ==========")
# #     print("LAB_MEAN =", mean.tolist())
# #     print("LAB_STD  =", std.tolist())
# #     print("============================================\n")

# #     return mean.astype(np.float32), std.astype(np.float32)

# # ##############################################################
# # # 2. LAB Transform + Diagnostics
# # ##############################################################

# # # LAB_MEAN = np.array([50.0, 0.0, 0.0], dtype=np.float32)
# # # LAB_STD = np.array([50.0, 128.0, 128.0], dtype=np.float32)

# # # LAB_MEAN = None
# # # LAB_STD = None

# # LAB_MEAN = np.array([27.715821371226003, 10.521480987873188, 8.514460146640673], dtype=np.float32)
# # LAB_STD  = np.array([24.70048803837073, 8.827357389195186, 8.660419910058293], dtype=np.float32)

# # class RGBToLABTensorTransform:
# #     first_call = True

# #     def __init__(self, is_train=True):
# #         global LAB_MEAN, LAB_STD
# #         assert LAB_MEAN is not None and LAB_STD is not None, \
# #             "LAB_MEAN and LAB_STD must be computed before creating transforms."

# #         if is_train:
# #             self.geom = transforms.Compose([
# #                 transforms.Resize((224,224)),
# #                 transforms.RandomHorizontalFlip(),
# #             ])
# #         else:
# #             self.geom = transforms.Compose([
# #                 transforms.Resize((224,224)),
# #             ])

# #     def __call__(self, img_pil):
# #         img_pil = self.geom(img_pil)

# #         rgb = np.asarray(img_pil).astype(np.float32)/255.0
# #         lab = rgb2lab(rgb).astype(np.float32)
# #         # print(LAB_MEAN, LAB_STD)
# #         lab_norm = (lab - LAB_MEAN) / LAB_STD
# #         # lab_norm = (lab - lab.mean((0,1))) / (lab.std((0,1)) + 1e-6)

# #         if RGBToLABTensorTransform.first_call:
# #             RGBToLABTensorTransform.first_call = False
# #             print("\n[DIAG] LAB BEFORE NORMALIZATION:")
# #             print("  L range:", lab[...,0].min(), lab[...,0].max())
# #             print("  a range:", lab[...,1].min(), lab[...,1].max())
# #             print("  b range:", lab[...,2].min(), lab[...,2].max())
# #             print("[DIAG] LAB AFTER NORMALIZATION:")
# #             print("  min:", lab_norm.min(), "max:", lab_norm.max())
# #             print("  mean:", lab_norm.mean(), "std:", lab_norm.std(), "\n")

# #         return torch.from_numpy(lab_norm.transpose(2,0,1)).float()


# # ##############################################################
# # # 3. Dataset + Dataloaders
# # ##############################################################

# # class SkinToneRegressionDataset(Dataset):
# #     def __init__(self, df, img_dir, transform):
# #         self.df = df.reset_index(drop=True)
# #         self.img_dir = Path(img_dir)
# #         self.transform = transform

# #     def __len__(_): return len(_.df)

# #     def __getitem__(self, idx):
# #         row = self.df.iloc[idx]
# #         img = Image.open(self.img_dir / str(row.iloc[0])).convert("RGB")

# #         label = float(row.iloc[1])
# #         label = (label - 1.0) / 9.0   # MST 1–10 → scaled 0–1 regression target

# #         return self.transform(img), torch.tensor(label, dtype=torch.float32)


# # def build_dataloaders(csv_path, image_dir, batch_size, val_ratio):
# #     df = pd.read_csv(csv_path).dropna()
# #     labels = df.iloc[:,1].astype(float)

# #     unique_labels = np.unique(labels)
# #     continuous_labels = (
# #         labels.dtype == float and (
# #             len(unique_labels) > 20 or np.any(labels % 1 != 0)
# #         )
# #     )

# #     if continuous_labels:
# #         print("[INFO] Continuous labels detected → RANDOM split (no stratification).")
# #         stratify_arg = None
# #     else:
# #         print("[INFO] Discrete labels detected → STRATIFIED split.")
# #         stratify_arg = labels

# #     train_idx, val_idx = train_test_split(
# #         np.arange(len(df)),
# #         test_size=val_ratio,
# #         shuffle=True,
# #         stratify=stratify_arg,
# #         random_state=42,
# #     )

# #     train_df = df.iloc[train_idx].reset_index(drop=True)
# #     val_df = df.iloc[val_idx].reset_index(drop=True)

# #     print(f"[INFO] Train={len(train_df)} | Val={len(val_df)}")

# #     train_tf = RGBToLABTensorTransform(is_train=True)
# #     val_tf = RGBToLABTensorTransform(is_train=False)

# #     return (
# #         DataLoader(SkinToneRegressionDataset(train_df, image_dir, train_tf),
# #                    batch_size=batch_size, shuffle=True, num_workers=4),
# #         DataLoader(SkinToneRegressionDataset(val_df, image_dir, val_tf),
# #                    batch_size=batch_size, shuffle=False, num_workers=4),
# #     )


# # ##############################################################
# # # 4. VGG Model (Dynamic Flatten Size)
# # ##############################################################

# # class VGG16LabRegressor(nn.Module):
# #     def __init__(self, use_bn=True, dropout_p=0.5):
# #         super().__init__()

# #         # base = models.vgg16_bn(weights=None) if use_bn else models.vgg16(weights=None)

# #         base = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1) if use_bn else models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

# #         self.features = base.features

# #         # for param in self.features[:20].parameters():
# #         #     param.requires_grad = False

# #         with torch.no_grad():
# #             dummy = torch.zeros(1, 3, 224, 224)
# #             feat = self.features(dummy)
# #             flat_dim = feat.view(1,-1).shape[1]
# #             print(f"[INFO] VGG feature dimension: {flat_dim}")

# #         self.classifier = nn.Sequential(
# #             nn.Linear(flat_dim, 1024),
# #             nn.ReLU(inplace=True),
# #             nn.Dropout(dropout_p),
# #             nn.Linear(1024, 512),
# #             nn.ReLU(inplace=True),
# #             nn.Dropout(dropout_p),
# #             nn.Linear(512, 1),
# #             nn.Sigmoid(), # Ensure output is in 0–1 range
# #         )

# #         self._init_weights()

# #     def _init_weights(self):
# #         for m in self.modules():
# #             if isinstance(m, nn.Conv2d):
# #                 nn.init.kaiming_normal_(m.weight)
# #                 if m.bias is not None:
# #                     nn.init.zeros_(m.bias)
# #             elif isinstance(m, nn.Linear):
# #                 nn.init.normal_(m.weight, 0, 0.01)
# #                 nn.init.zeros_(m.bias)

# #     def forward(self, x):
# #         x = self.features(x)
# #         x = x.view(x.size(0), -1)
# #         return self.classifier(x).squeeze(-1)


# # ##############################################################
# # # 5. Metrics
# # ##############################################################

# # def compute_metrics(pred, tgt, threshold=0.5):
# #     pred_np = pred.detach().cpu().numpy() * 9 + 1
# #     tgt_np  = tgt.detach().cpu().numpy() * 9 + 1

# #     abs_diff = np.abs(pred_np - tgt_np)
# #     acc1 = (abs_diff <= threshold).mean() * 100
# #     l2 = calc_l2_lab_distance(pred, tgt)
# #     return acc1, l2


# # ##############################################################
# # # 6. Diagnostics
# # ##############################################################

# # def print_diagnostics(imgs, labels, preds, tag):
# #     print(f"\n=== [DIAG] {tag} ===")
# #     print("IMG mean/std:", imgs.mean().item(), imgs.std().item())
# #     print("IMG min/max:", imgs.min().item(), imgs.max().item())
# #     print("Labels min/max:", labels.min().item(), labels.max().item())
# #     print("Preds min/max:", preds.min().item(), preds.max().item())
# #     if preds.std().item() < 1e-4:
# #         print("WARNING: predictions collapsed to near-constant.")
# #     print("===========================================\n")


# # ##############################################################
# # # 7. Train / Validate
# # ##############################################################

# # def train_one_epoch(model, loader, optim, device, epoch):
# #     model.train()
# #     mse = nn.MSELoss()
# #     total_loss = 0
# #     total_acc = 0

# #     for i,(imgs,labels) in enumerate(loader):
# #         imgs,labels = imgs.to(device),labels.to(device)

# #         preds = model(imgs)
# #         loss = mse(preds,labels)

# #         optim.zero_grad()
# #         loss.backward()
# #         optim.step()

# #         acc,_ = compute_metrics(preds,labels)

# #         total_loss += loss.item()*imgs.size(0)
# #         total_acc += acc*imgs.size(0)

# #         if i == 0:
# #             print_diagnostics(imgs, labels, preds, f"TRAIN EPOCH {epoch}")

# #     n = len(loader.dataset)
# #     return total_loss/n, total_acc/n


# # def validate(model, loader, device, epoch):
# #     model.eval()
# #     mse = nn.MSELoss()
# #     total_loss = 0
# #     total_acc = 0
# #     all_l2 = []

# #     with torch.no_grad():
# #         for i,(imgs,labels) in enumerate(loader):
# #             imgs,labels = imgs.to(device),labels.to(device)
# #             preds = model(imgs)

# #             loss = mse(preds,labels)
# #             acc,l2 = compute_metrics(preds,labels)

# #             total_loss += loss.item()*imgs.size(0)
# #             total_acc += acc*imgs.size(0)
# #             all_l2.extend(l2)

# #             if i == 0:
# #                 print_diagnostics(imgs, labels, preds, f"VALID EPOCH {epoch}")

# #     n = len(loader.dataset)
# #     return total_loss/n, total_acc/n, float(np.mean(all_l2)), float(np.std(all_l2))


# # ##############################################################
# # # 8. MAIN (Save-enabled)
# # ##############################################################

# # def main():
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--csv-path", required=True)
# #     parser.add_argument("--image-dir", required=True)
# #     parser.add_argument("--save-dir", default="./checkpoints")
# #     parser.add_argument("--epochs", type=int, default=32)
# #     parser.add_argument("--batch-size", type=int, default=32)
# #     parser.add_argument("--val-ratio", type=float, default=0.35)
# #     parser.add_argument("--lr", type=float, default=1e-4)
# #     parser.add_argument("--momentum", type=float, default=0.9)
# #     parser.add_argument("--weight-decay", type=float, default=1e-4)
# #     parser.add_argument("--threshold", type=float, default=0.5)
# #     parser.add_argument("--gpu", type=int, default=0)
# #     parser.add_argument("--use-bn", action="store_true")
# #     parser.add_argument("--compute-lab-stats", action="store_true", help="Compute global LAB mean/std before training.")
# #     args = parser.parse_args()

# #     device = torch.device(f"cuda:{args.gpu}")

# #     # ---------------------------------------------------------
# #     # 1. COMPUTE LAB STATS IF REQUIRED
# #     # ---------------------------------------------------------
# #     global LAB_MEAN, LAB_STD

# #     if args.compute_lab_stats:
# #         LAB_MEAN, LAB_STD = compute_dataset_lab_stats(args.image_dir, args.csv_path)
# #     else:
# #         print("[INFO] Using existing LAB_MEAN / LAB_STD values.")
# #         # If you want, hardcode fallback values here

# #     # Ensure LAB_MEAN and LAB_STD propagate to the module-level globals
# #     globals()["LAB_MEAN"] = LAB_MEAN
# #     globals()["LAB_STD"] = LAB_STD
# #     print("[INFO] LAB normalization stats committed to global scope.")

# #     print("[INFO] LAB_MEAN:", LAB_MEAN)
# #     print("[INFO] LAB_STD:", LAB_STD)
# #     # ---------------------------------------------------------

# #     save_dir = Path(args.save_dir)
# #     save_dir.mkdir(parents=True, exist_ok=True)

# #     best_path = save_dir / "vgg16_lab_best.pth"
# #     final_path = save_dir / "vgg16_lab_final.pth"

# #     best_val_loss = float("inf")

# #     train_loader, val_loader = build_dataloaders(
# #         args.csv_path, args.image_dir, args.batch_size, args.val_ratio
# #     )

# #     model = VGG16LabRegressor(use_bn=args.use_bn).to(device)

# #     # optimizer = torch.optim.SGD(
# #     #     model.parameters(),
# #     #     lr=args.lr,
# #     #     momentum=args.momentum,
# #     #     weight_decay=args.weight_decay
# #     # )

# #     # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

# #     # Replace SGD with Adam
# #     optimizer = torch.optim.Adam(
# #         model.parameters(),
# #         lr=args.lr,
# #         weight_decay=args.weight_decay
# #     )
# #     # You can keep the scheduler, but Adam usually converges faster without aggressive scheduling initially.

# #     for epoch in range(1, args.epochs+1):
# #         print(f"\n===== EPOCH {epoch}/{args.epochs} =====")
# #         tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, epoch)
# #         va_loss, va_acc, l2_mean, l2_std = validate(model, val_loader, device, epoch)

# #         print(
# #             f"[EPOCH {epoch}] "
# #             f"TrainLoss={tr_loss:.4f} TrainAcc={tr_acc:.2f}% | "
# #             f"ValLoss={va_loss:.4f} ValAcc={va_acc:.2f}% | "
# #             f"L2 mean={l2_mean:.3f}, std={l2_std:.3f}"
# #         )

# #         # scheduler.step()

# #         # SAVE BEST CHECKPOINT
# #         if va_loss < best_val_loss:
# #             best_val_loss = va_loss
# #             torch.save(model.state_dict(), best_path)
# #             print(f"[INFO] Saved BEST model → {best_path} (ValLoss={va_loss:.4f})")

# #     # SAVE FINAL MODEL
# #     torch.save(model.state_dict(), final_path)
# #     print(f"[INFO] Saved FINAL model → {final_path}")


# # if __name__ == "__main__":
# #     main()


# ##############################################################
# #  VGG16-LAB TRAINING PIPELINE (DYNAMIC WEIGHTING ENABLED)
# #  - Enforces EXACT 224x224 input size
# #  - Adds dynamic feature-size inference
# #  - Saves best model + final model
# #  - Uses Inverse Frequency Weighting to fix Regression to Mean
# ##############################################################

# import argparse
# from pathlib import Path
# from collections import Counter

# import numpy as np
# import pandas as pd
# from PIL import Image
# from tqdm import tqdm

# import torch
# import torch.nn as nn
# import torch.nn.functional as F  # <--- Added for functional loss
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
#     # convert back from 0-1 regression space to MST 1-10
#     p = pred.detach().cpu().numpy() * 9 + 1
#     t = target.detach().cpu().numpy() * 9 + 1

#     return np.sqrt(np.sum((monk_scalar_to_lab(p)-monk_scalar_to_lab(t))**2, axis=1))


# ##############################################################
# #  LAB DATASET STATISTICS (GLOBAL MEAN / STD)
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
# # 2. LAB Transform + Diagnostics
# ##############################################################

# LAB_MEAN = np.array([27.715821371226003, 10.521480987873188, 8.514460146640673], dtype=np.float32)
# LAB_STD  = np.array([24.70048803837073, 8.827357389195186, 8.660419910058293], dtype=np.float32)

# class RGBToLABTensorTransform:
#     first_call = True

#     def __init__(self, is_train=True):
#         global LAB_MEAN, LAB_STD
#         assert LAB_MEAN is not None and LAB_STD is not None, \
#             "LAB_MEAN and LAB_STD must be computed before creating transforms."

#         if is_train:
#             self.geom = transforms.Compose([
#                 transforms.Resize((224,224)),
#                 transforms.RandomHorizontalFlip(),
#             ])
#         else:
#             self.geom = transforms.Compose([
#                 transforms.Resize((224,224)),
#             ])

#     def __call__(self, img_pil):
#         img_pil = self.geom(img_pil)

#         rgb = np.asarray(img_pil).astype(np.float32)/255.0
#         lab = rgb2lab(rgb).astype(np.float32)
#         lab_norm = (lab - LAB_MEAN) / LAB_STD

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
# # 3. Dataset + Dataloaders (UPDATED WITH WEIGHTING)
# ##############################################################

# class SkinToneRegressionDataset(Dataset):
#     def __init__(self, df, img_dir, transform):
#         self.df = df.reset_index(drop=True)
#         self.img_dir = Path(img_dir)
#         self.transform = transform

#         # --- DYNAMIC INVERSE FREQUENCY WEIGHTING ---
#         print("[INFO] Calculating dynamic sample weights...")
        
#         # 1. Get raw float labels
#         raw_labels = self.df.iloc[:, 1].astype(float).values
        
#         # 2. Bin them to nearest integer to count frequency (1, 2, ... 10)
#         binned_labels = np.round(raw_labels).astype(int)
        
#         # 3. Count frequencies
#         counts = Counter(binned_labels)
#         total_samples = len(binned_labels)
        
#         # 4. Calculate Weight = Total / Count (Inverse Frequency)
#         #    This makes rare classes have HIGH weights, common classes LOW weights
#         weights_map = {k: total_samples / v for k, v in counts.items()}
        
#         # 5. Map back to every sample
#         weights = [weights_map[int(round(x))] for x in raw_labels]
#         weights = np.array(weights)
        
#         # 6. Normalize so mean weight is 1.0 (keeps LR stable)
#         weights = weights / weights.mean()
        
#         self.weights = torch.FloatTensor(weights)
#         print(f"[INFO] Weights calculated. Min: {weights.min():.2f}, Max: {weights.max():.2f}, Mean: {weights.mean():.2f}")
#         # -------------------------------------------

#     def __len__(_): return len(_.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         img = Image.open(self.img_dir / str(row.iloc[0])).convert("RGB")

#         label = float(row.iloc[1])
#         label = (label - 1.0) / 9.0   # MST 1-10 -> scaled 0-1 regression target
        
#         # Get the pre-calculated weight for this sample
#         weight = self.weights[idx]

#         return self.transform(img), torch.tensor(label, dtype=torch.float32), weight


# def build_dataloaders(csv_path, image_dir, batch_size, val_ratio):
#     df = pd.read_csv(csv_path).dropna()
#     labels = df.iloc[:,1].astype(float)

#     unique_labels = np.unique(labels)
#     continuous_labels = (
#         labels.dtype == float and (
#             len(unique_labels) > 20 or np.any(labels % 1 != 0)
#         )
#     )

#     if continuous_labels:
#         print("[INFO] Continuous labels detected -> RANDOM split (no stratification).")
#         stratify_arg = None
#     else:
#         print("[INFO] Discrete labels detected -> STRATIFIED split.")
#         stratify_arg = labels

#     train_idx, val_idx = train_test_split(
#         np.arange(len(df)),
#         test_size=val_ratio,
#         shuffle=True,
#         stratify=stratify_arg,
#         random_state=42,
#     )

#     train_df = df.iloc[train_idx].reset_index(drop=True)
#     val_df = df.iloc[val_idx].reset_index(drop=True)

#     print(f"[INFO] Train={len(train_df)} | Val={len(val_df)}")

#     train_tf = RGBToLABTensorTransform(is_train=True)
#     val_tf = RGBToLABTensorTransform(is_train=False)

#     return (
#         DataLoader(SkinToneRegressionDataset(train_df, image_dir, train_tf),
#                    batch_size=batch_size, shuffle=True, num_workers=4),
#         DataLoader(SkinToneRegressionDataset(val_df, image_dir, val_tf),
#                    batch_size=batch_size, shuffle=False, num_workers=4),
#     )


# ##############################################################
# # 4. VGG Model (Dynamic Flatten Size)
# ##############################################################

# class VGG16LabRegressor(nn.Module):
#     def __init__(self, use_bn=True, dropout_p=0.5):
#         super().__init__()

#         # Use Pre-trained ImageNet Weights
#         base = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1) if use_bn else models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

#         self.features = base.features

#         # UNFREEZE all layers to allow adaptation to LAB color space --
#         for param in self.features.parameters():
#             param.requires_grad = True

#         with torch.no_grad():
#             dummy = torch.zeros(1, 3, 224, 224)
#             feat = self.features(dummy)
#             flat_dim = feat.view(1,-1).shape[1]
#             print(f"[INFO] VGG feature dimension: {flat_dim}")

#         self.classifier = nn.Sequential(
#             nn.Linear(flat_dim, 1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout_p),
#             nn.Linear(1024, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout_p),
#             nn.Linear(512, 1),
#             nn.Sigmoid(), # Ensure output is in 0-1 range
#         )

#         self._init_weights()

#     def _init_weights(self): #--
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
# # 7. Train / Validate (UPDATED FOR WEIGHTS)
# ##############################################################

# def train_one_epoch(model, loader, optim, device, epoch):
#     model.train()
#     total_loss = 0
#     total_acc = 0

#     # Unpack 3 items now: images, labels, weights
#     for i, (imgs, labels, weights) in enumerate(loader):
#         imgs, labels, weights = imgs.to(device), labels.to(device), weights.to(device)

#         preds = model(imgs)
        
#         # --- WEIGHTED LOSS CALCULATION ---
#         # 1. Calculate raw MSE for each sample (no average yet)
#         loss_unreduced = F.mse_loss(preds, labels, reduction='none')
        
#         # 2. Multiply by importance weight
#         loss_weighted = loss_unreduced * weights
        
#         # 3. Take the mean
#         loss = loss_weighted.mean()
#         # ---------------------------------

#         optim.zero_grad()
#         loss.backward()
#         optim.step()

#         acc, _ = compute_metrics(preds, labels)

#         total_loss += loss.item()*imgs.size(0)
#         total_acc += acc*imgs.size(0)

#         if i == 0:
#             print_diagnostics(imgs, labels, preds, f"TRAIN EPOCH {epoch}")

#     n = len(loader.dataset)
#     return total_loss/n, total_acc/n


# def validate(model, loader, device, epoch):
#     model.eval()
#     mse = nn.MSELoss()
#     total_loss = 0
#     total_acc = 0
#     all_l2 = []

#     with torch.no_grad():
#         # Unpack weights but ignore them (_) for validation
#         for i, (imgs, labels, _) in enumerate(loader):
#             imgs, labels = imgs.to(device), labels.to(device)
#             preds = model(imgs)

#             loss = mse(preds, labels)
#             acc, l2 = compute_metrics(preds, labels)

#             total_loss += loss.item()*imgs.size(0)
#             total_acc += acc*imgs.size(0)
#             all_l2.extend(l2)

#             if i == 0:
#                 print_diagnostics(imgs, labels, preds, f"VALID EPOCH {epoch}")

#     n = len(loader.dataset)
#     return total_loss/n, total_acc/n, float(np.mean(all_l2)), float(np.std(all_l2))


# ##############################################################
# # 8. MAIN (Save-enabled)
# ##############################################################

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--csv-path", required=True)
#     parser.add_argument("--image-dir", required=True)
#     parser.add_argument("--save-dir", default="./checkpoints")
#     parser.add_argument("--epochs", type=int, default=32)
#     parser.add_argument("--batch-size", type=int, default=32)
#     parser.add_argument("--val-ratio", type=float, default=0.35)
#     parser.add_argument("--lr", type=float, default=1e-4)
#     # momentum argument removed because we use Adam
#     parser.add_argument("--weight-decay", type=float, default=1e-4)
#     parser.add_argument("--threshold", type=float, default=0.5)
#     parser.add_argument("--gpu", type=int, default=0)
#     parser.add_argument("--use-bn", action="store_true")
#     parser.add_argument("--compute-lab-stats", action="store_true", help="Compute global LAB mean/std before training.")
#     args = parser.parse_args()

#     device = torch.device(f"cuda:{args.gpu}")

#     # ---------------------------------------------------------
#     # 1. COMPUTE LAB STATS IF REQUIRED
#     # ---------------------------------------------------------
#     global LAB_MEAN, LAB_STD

#     if args.compute_lab_stats:
#         LAB_MEAN, LAB_STD = compute_dataset_lab_stats(args.image_dir, args.csv_path)
#     else:
#         print("[INFO] Using existing LAB_MEAN / LAB_STD values.")

#     # Ensure LAB_MEAN and LAB_STD propagate to the module-level globals
#     globals()["LAB_MEAN"] = LAB_MEAN
#     globals()["LAB_STD"] = LAB_STD
#     print("[INFO] LAB normalization stats committed to global scope.")

#     print("[INFO] LAB_MEAN:", LAB_MEAN)
#     print("[INFO] LAB_STD:", LAB_STD)
#     # ---------------------------------------------------------

#     save_dir = Path(args.save_dir)
#     save_dir.mkdir(parents=True, exist_ok=True)

#     best_path = save_dir / "vgg16_lab_best.pth"
#     final_path = save_dir / "vgg16_lab_final.pth"

#     best_val_loss = float("inf")

#     train_loader, val_loader = build_dataloaders(
#         args.csv_path, args.image_dir, args.batch_size, args.val_ratio
#     )

#     model = VGG16LabRegressor(use_bn=args.use_bn).to(device)

#     # Use Adam without momentum argument
#     optimizer = torch.optim.Adam(
#         model.parameters(),
#         lr=args.lr,
#         weight_decay=args.weight_decay
#     )

#     # Optional: Scheduler
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=5, verbose=True
#     )

#     for epoch in range(1, args.epochs+1):
#         print(f"\n===== EPOCH {epoch}/{args.epochs} =====")
#         tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, epoch)
#         va_loss, va_acc, l2_mean, l2_std = validate(model, val_loader, device, epoch)

#         print(
#             f"[EPOCH {epoch}] "
#             f"TrainLoss={tr_loss:.4f} TrainAcc={tr_acc:.2f}% | "
#             f"ValLoss={va_loss:.4f} ValAcc={va_acc:.2f}% | "
#             f"L2 mean={l2_mean:.3f}, std={l2_std:.3f}"
#         )

#         # Update scheduler based on Validation Loss
#         scheduler.step(va_loss)

#         # SAVE BEST CHECKPOINT
#         if va_loss < best_val_loss:
#             best_val_loss = va_loss
#             torch.save(model.state_dict(), best_path)
#             print(f"[INFO] Saved BEST model -> {best_path} (ValLoss={va_loss:.4f})")

#     # SAVE FINAL MODEL
#     torch.save(model.state_dict(), final_path)
#     print(f"[INFO] Saved FINAL model -> {final_path}")


# if __name__ == "__main__":
#     main()