# Model needs to be fixed to align with the corect approch LAB + HSV + RGB read through the paper to know exactly what was done its better than using ChatGPT only

"""
Modernised reproduction of the 'Skin Tone Estimation using VGG-16' system
===========================================================================

This script faithfully replicates the behaviour of the original repository:

- Model definition:        vgg.py (class VGG, vgg16_bn)
- Training loop & metrics: main.py
- Dataset & LAB pipeline:  data_loader.py
- LAB range shifting:      process_image.py + image_utils/main.c

Modernisations:
- Single-file, single-GPU training script
- Pure-Python LAB range modification (no .so / ctypes)
- Type hints, clearer structure, docstrings
"""

import argparse
import os
import math
import cv2
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from skimage.color import rgb2lab


# =============================================================================
# 1. CONFIG
# =============================================================================
@dataclass
class TrainConfig:
    data_dir: str
    arch: str = "vgg16_bn"          # matches main.py choices
    epochs: int = 90                # --epochs 90
    batch_size: int = 256           # --batch-size 256 (default)
    lr: float = 0.1                 # --lr 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    threshold: float = 0.5          # --threshold 0.5
    threshold_factor: float = 1.5   # factor used in accuracy()
    blur: bool = False              # --blur flag
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    layers_freezed: int = 27        # --layers-freezed 27
    vggface_weights: str = ""       # path to vgg_face_dag.pth, if you want that behaviour


# =============================================================================
# 2. LAB PREPROCESSING UTILITIES
#    (Derived from data_loader.py, process_image.py, image_utils/main.c)
# =============================================================================

def mod_lab_range_python(image_lab: np.ndarray) -> np.ndarray:
    """
    Python re-implementation of C function mod_lab_range in image_utils/main.c.

    Original C code:

        typedef struct { float L; float a; float b; } Pixel;

        void mod_lab_range(float *flatted_img,int size){
            Pixel *p;
            float *IMG=flatted_img;
            for (int i=0;i<size;i++) {
                p=(Pixel*)((IMG+i*3));
                p->a+=127;
                p->b+=127;
            }
        }

    Behaviour:
    - For each pixel, adds +127 to 'a' and +127 to 'b'.
    - L channel is unchanged.

    In data_loader.py, after this, the image is cast to uint8 and converted
    back to PIL:

        image = Image.fromarray(np.uint8(image))

    So we exactly mirror that here in Python.
    """
    assert image_lab.shape[2] == 3, "Expected LAB image with 3 channels."
    out = image_lab.copy()
    out[:, :, 1] = out[:, :, 1] + 127.0  # a channel
    out[:, :, 2] = out[:, :, 2] + 127.0  # b channel
    return out


def pil_rgb_to_shifted_lab_uint8(img: Image.Image) -> Image.Image:
    """
    Full LAB preprocessing pipeline as in RegrDataset.__getitem__:

    1) Convert RGB PIL image to numpy.
    2) Convert to LAB using skimage.color.rgb2lab.
    3) Apply mod_lab_range (a+=127, b+=127).
    4) Cast to uint8 and return as PIL image.

    Derived from:
    - data_loader.py (rgb2lab + mod_lab_range + Image.fromarray(np.uint8))
    - process_image.py (ImageUtils.mod_lab_range)
    - main.c (mod_lab_range implementation).
    """
    # RGB -> LAB (skimage)
    lab = rgb2lab(np.array(img))  # float64, L in [0,100], a,b approx [-128,127]
    # Apply the same shift as C code
    lab_shifted = mod_lab_range_python(lab)
    # Cast to uint8 as in original code
    lab_uint8 = np.uint8(lab_shifted)
    # Back to PIL
    return Image.fromarray(lab_uint8)


# =============================================================================
# 3. DATASET CLASS
#    (Faithful modern version of RegrDataset in data_loader.py)
# =============================================================================

class SkinToneRegressionDataset(Dataset):
    """
    Modernised version of RegrDataset with identical behaviour.

    Derived from data_loader.py:
    - Reads annotations CSV (filename, label).
    - Loads RGB image from img_dir.
    - Optional blur (cv.blur with kernel (224,224)).
    - RGB -> LAB -> +127 to a,b -> uint8 -> PIL.
    - Apply torchvision transforms.
    - Return tensor image and float label.
    """

    def __init__(
        self,
        annotations_file: str,
        img_dir: str,
        transform: T.Compose,
        blur: bool = False,
        conv_to_lab_space: bool = True,
        color_correct: bool = False,  # original code has this disabled
    ):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.blur = blur
        self.conv_to_lab_space = conv_to_lab_space
        self.color_correct = color_correct  # kept for completeness (grey_world commented out)

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, idx: int):
        # label = MST score (float) from CSV
        label = float(self.img_labels.iloc[idx, 1])

        # load image (RGB) as PIL
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")

        # Colour correction (grey world) is commented out in original => we keep it disabled

        # Optional blur -> in original code: cv.blur(from_pil(image), (224,224))
        if self.blur:
            img_np = np.array(image)
            img_blur = cv2.blur(img_np, (224, 224))
            image = Image.fromarray(img_blur)

        # Convert to LAB and shift a,b if requested (this is the default in original code)
        if self.conv_to_lab_space:
            image = pil_rgb_to_shifted_lab_uint8(image)

        # Apply transforms (RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize)
        if self.transform:
            image = self.transform(image)

        # Return label as float tensor
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return image, label_tensor

    def data_frame(self) -> pd.DataFrame:
        """Utility equivalent to __data__() in RegrDataset."""
        return self.img_labels.copy()


# =============================================================================
# 4. VGG-16 REGRESSION MODEL
#    (Modernised version of vgg.py::VGG + vgg16_bn)
# =============================================================================

# Configuration dictionary from vgg.py
VGG_CFG = {
    'F': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_layers(cfg: List, batch_norm: bool = False) -> nn.Sequential:
    """
    Direct port of make_layers() from vgg.py.

    Builds the VGG feature extractor with optional BatchNorm.
    """
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG16Regression(nn.Module):
    """
    Faithful port of class VGG in vgg.py, configured for VGG-16-BN regression.

    Differences vs original:
    - Cleaner constructor
    - Type hints
    Behaviour is identical:
    - features: VGG-16 with BatchNorm
    - classifier:
        512*7*7 -> 1024 -> 512 -> 1
      with Sigmoid + Dropout in hidden layers.
    """

    def __init__(self, init_weights: bool = True, num_classes: int = 1):
        super().__init__()
        # features: VGG-16 configuration "D" with BatchNorm (vgg16_bn)
        # self.features = make_layers(VGG_CFG["D"], batch_norm=True)
        self.features = make_layers(VGG_CFG["D"], batch_norm=False)

        # classifier: identical to original VGG class in vgg.py
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.Sigmoid(),
            nn.Dropout(),           # default p=0.5
            nn.Linear(1024, 512),
            nn.Sigmoid(),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)  # (N,512,7,7) -> (N,25088)
        x = self.classifier(x)
        x = torch.clamp(x, 1.0, 10.0)   # <---- ADD THIS to stop errors
        x = torch.squeeze(x, dim=-1)  # (N,1) -> (N,)
        return x

    def _initialize_weights(self) -> None:
        """
        Same initialization logic as in vgg.py::_initialize_weights:
        - Conv: Kaiming normal
        - BN: weight=1, bias=0
        - Linear: N(0,0.01), bias=0
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.constant_(m.bias, 0.0)


def load_vggface_weights_if_available(
    model: VGG16Regression,
    weights_path: str,
    layers_freezed: int,
):
    """
    Modernised version of the VGG-Face weight loading in main.py.

    In the original main.py:
    - torch.load("VGG-face weights/vgg_face_dag.pth")
    - manually maps conv1_1, conv1_2, ..., conv5_3 to the Conv2d layers
      in model.features.features
    - then freezes the first 'layers_freezed' parameters.

    Here we:
    - Only apply this if weights_path is not empty and exists.
    """
    if not weights_path or not os.path.isfile(weights_path):
        print("[INFO] No VGG-Face weights found or path empty, skipping pre-init.")
        return

    print(f"[INFO] Loading VGG-Face weights from {weights_path}")
    prt_w = torch.load(weights_path, map_location="cpu")

    # In original code, they rewrap as Parameters, but that's not strictly needed.
    conv_name_order = [
        "conv1_1", "conv1_2",
        "conv2_1", "conv2_2",
        "conv3_1", "conv3_2", "conv3_3",
        "conv4_1", "conv4_2", "conv4_3",
        "conv5_1", "conv5_2", "conv5_3",
    ]

    conv_layers = [m for m in model.features.modules() if isinstance(m, nn.Conv2d)]
    assert len(conv_layers) >= len(conv_name_order), "Not enough conv layers."

    for i, name in enumerate(conv_name_order):
        w_key = f"{name}.weight"
        b_key = f"{name}.bias"
        if w_key in prt_w and b_key in prt_w:
            conv_layers[i].weight.data.copy_(prt_w[w_key])
            conv_layers[i].bias.data.copy_(prt_w[b_key])
        else:
            print(f"[WARN] Missing keys {w_key}/{b_key} in VGG-Face weights; skipping.")

    # Freeze first 'layers_freezed' parameters (as in main.py)
    counter = 0
    for param in model.parameters():
        counter += 1
        if counter < layers_freezed:
            param.requires_grad = False


# =============================================================================
# 5. METRICS AND UTILS
#    (Derived from main.py: AverageMeter, accuracy, monk_to_lab, rgb_to_lab, calc_l2_distances)
# =============================================================================

class AverageMeter:
    """Tracks average, sum, count — from main.py::AverageMeter."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0


def rgb_to_lab_manual(val: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Direct port of rgb_to_lab() in main.py."""
    R, G, B = val
    L = Y1 = 0.2126 * R + 0.7152 * G + 0.0722 * B
    A = 1.4749 * (0.2213 * R - 0.3390 * G + 0.1177 * B) + 128
    b = 0.6245 * (0.1949 * R + 0.6057 * G - 0.8006 * B) + 128
    return (L, A, b)


# def monk_to_lab(values: List[float]) -> List[Tuple[float, float, float]]:
#     """
#     Port of monk_to_lab() from main.py.

#     Uses the same "Monk palette" defined in that file and linearly interpolates between
#     neighbouring colours based on fractional MST values.
#     """
#     result = []
#     colors = [
#         (255, 255, 255),
#         (246, 237, 228),
#         (243, 231, 219),
#         (247, 234, 208),
#         (243, 218, 186),
#         (215, 189, 150),
#         (160, 126, 86),
#         (130, 92, 67),
#         (96, 65, 52),
#         (58, 49, 42),
#         (41, 36, 32),
#         (0, 0, 0),
#     ]

#     for v in values:
#         v_float = float(v)
#         idx = 10 if int(v_float) > 10 else int(v_float)
#         try:
#             B = np.array(colors[idx + 1], dtype=float)
#             A = np.array(colors[idx], dtype=float)
#             rgb = (B - A) * (v_float - int(v_float)) + A
#             result.append(rgb_to_lab_manual(tuple(rgb)))
#         except Exception as e:
#             print(f"Error with idx: {idx} / value: {v_float} -> {e}")
#             return [(0.0, 0.0, 0.0)] * len(values)
#     return result

def monk_to_lab(values: List[float]) -> List[Tuple[float, float, float]]:
    """
    Port of monk_to_lab() from main.py, with added numerical safety.

    Original behaviour (main.py::monk_to_lab):
    -----------------------------------------
    - Uses a fixed Monk palette (12 RGB anchors).
    - Interpolates between neighbouring anchors based on fractional MST values.

    Modifications vs original:
    --------------------------
    - We *explicitly clamp* v into [1, 10] before converting.
      In the original pipeline MST labels are always in [1..10], so this has
      no effect for valid predictions but protects against early regression
      explosions (e.g. huge negatives) that would cause out-of-range indices.
    - Removed the try/except that returned all zeros on any error; instead we
      prevent the error by construction.
    """
    result: List[Tuple[float, float, float]] = []

    # Same Monk palette as in main.py
    colors = [
        (255, 255, 255),
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
        (0, 0, 0),
    ]

    for v in values:
        # Clamp MST prediction into the valid range [1, 10]
        # (labels in the original system are always within this range)
        v_float = float(v)
        v_clamped = max(1.0, min(10.0, v_float))

        # Integer "bin index" (1..10)
        idx = int(math.floor(v_clamped))
        if idx < 1:
            idx = 1
        if idx > 10:
            idx = 10

        # Interpolation factor between colour[idx] and colour[idx+1]
        if idx == 10:
            # At the last bin, stay at colors[10] (no interpolation with 11)
            t = 0.0
        else:
            t = v_clamped - idx  # in [0,1)

        A = np.array(colors[idx], dtype=float)
        B = np.array(colors[idx + 1], dtype=float)
        rgb = (B - A) * t + A

        result.append(rgb_to_lab_manual(tuple(rgb)))

    return result


def calc_l2_distances(data_1, data_2) -> List[float]:
    """Port of calc_l2_distances() from main.py."""
    diff = np.array(data_1) - np.array(data_2)
    return list(np.sqrt(np.sum(diff * diff, axis=1)))


# def regression_accuracy(
#     output: torch.Tensor,
#     target: torch.Tensor,
#     threshold: float,
#     threshold_factor: float,
# ) -> Tuple[float, float, List[float], List[float]]:
#     """
#     Exact port of accuracy() in main.py for regression:

#     - Acc@1: % predictions within ±threshold of target.
#     - Acc@5: % predictions within ±threshold*factor of target.
#     - cum_d: cumulative accuracies for deltas in [0.0, 0.1, ..., 1.9]
#     - acc_l2: list of L2 distances in LAB space between predicted and target skin tones.
#     """
#     with torch.no_grad():
#         output = output.view(-1).cpu().numpy()
#         target = target.view(-1).cpu().numpy()
#         batch_size = target.shape[0]

#         # Acc@1 and Acc@5
#         num_acc_1 = 0
#         num_acc_2 = 0
#         for pred, targ in zip(output, target):
#             if (targ - threshold) <= pred <= (targ + threshold):
#                 num_acc_1 += 1
#             if (targ - threshold * threshold_factor) <= pred <= (targ + threshold * threshold_factor):
#                 num_acc_2 += 1

#         # Cumulative accuracy vector for deltas (0.0..1.9)
#         MAX = 20
#         cum_d = [0] * MAX
#         for pred, targ in zip(output, target):
#             distance = abs(targ - pred)
#             for delta in range(MAX):
#                 if distance < (delta / 10.0):
#                     cum_d[delta] += 1
#         cum_d = list((np.array(cum_d) * 100.0) / batch_size)

#         # LAB L2 distances
#         pred_lab = monk_to_lab(list(output))
#         targ_lab = monk_to_lab(list(target))
#         acc_l2 = calc_l2_distances(pred_lab, targ_lab)

#         acc1 = 100.0 * num_acc_1 / batch_size
#         acc5 = 100.0 * num_acc_2 / batch_size
#         return acc1, acc5, cum_d, acc_l2

def regression_accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    threshold_factor: float,
) -> Tuple[float, float, List[float], List[float]]:
    """
    Modernised, numerically safe port of accuracy() in main.py for regression.

    Original behaviour (main.py::accuracy):
    --------------------------------------
    - Acc@1: % predictions within ±threshold of target.
    - Acc@5: % predictions within ±threshold*factor of target.
    - cum_d: cumulative accuracies for deltas in [0.0, 0.1, ..., 1.9].
    - acc_l2: list of L2 distances in LAB space between predicted and target
      skin tones, obtained via monk_to_lab() and calc_l2_distances().

    Modifications vs original:
    --------------------------
    - Keep Acc@1, Acc@5 and cum_d *unchanged* (use raw predictions).
    - For LAB L2 (acc_l2), we *clip* both predictions and targets into [1,10]
      before converting to Monk LAB space. This matches the intended MST range
      in the original system and prevents invalid colour index access when the
      regression output temporarily explodes early in training.
    """
    with torch.no_grad():
        # Convert to numpy for metric computation
        output_np = output.view(-1).cpu().numpy().astype(np.float32)
        target_np = target.view(-1).cpu().numpy().astype(np.float32)
        batch_size = target_np.shape[0]

        # --------------------------------------------------
        # Acc@1 and Acc@5  (same logic as main.py::accuracy)
        # --------------------------------------------------
        num_acc_1 = 0
        num_acc_2 = 0
        for pred, targ in zip(output_np, target_np):
            if (targ - threshold) <= pred <= (targ + threshold):
                num_acc_1 += 1
            if (targ - threshold * threshold_factor) <= pred <= (targ + threshold * threshold_factor):
                num_acc_2 += 1

        # --------------------------------------------------
        # Cumulative accuracy vector for deltas (0.0..1.9)
        # main.py::accuracy (unchanged)
        # --------------------------------------------------
        MAX = 20
        cum_d = [0] * MAX
        for pred, targ in zip(output_np, target_np):
            distance = abs(targ - pred)
            for delta in range(MAX):
                if distance < (delta / 10.0):
                    cum_d[delta] += 1
        cum_d = list((np.array(cum_d, dtype=np.float32) * 100.0) / batch_size)

        # --------------------------------------------------
        # LAB L2 distances (main.py::monk_to_lab + calc_l2_distances)
        # with added clipping into valid MST range [1, 10].
        # --------------------------------------------------
        output_clipped = np.clip(output_np, 1.0, 10.0)
        target_clipped = np.clip(target_np, 1.0, 10.0)

        pred_lab = monk_to_lab(list(output_clipped))
        targ_lab = monk_to_lab(list(target_clipped))
        acc_l2 = calc_l2_distances(pred_lab, targ_lab)

        acc1 = 100.0 * num_acc_1 / batch_size
        acc5 = 100.0 * num_acc_2 / batch_size

        return acc1, acc5, cum_d, acc_l2

# =============================================================================
# 6. TRAIN & VALIDATE LOOPS
#    (Modernised single-GPU port of main.py::train / validate)
# =============================================================================

def adjust_learning_rate(optimizer: optim.Optimizer, epoch: int, cfg: TrainConfig):
    """
    Same schedule as main.py::adjust_learning_rate:
    lr = initial_lr * (0.1 ** (epoch // 30))
    """
    lr = cfg.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    cfg: TrainConfig,
):
    model.train()
    losses = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    for i, (images, targets) in enumerate(loader):
        images = images.to(cfg.device, non_blocking=True)
        targets = targets.to(cfg.device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        acc1, acc5, _, _ = regression_accuracy(
            outputs, targets, cfg.threshold, cfg.threshold_factor
        )

        losses.update(loss.item(), images.size(0))
        acc1_meter.update(acc1, images.size(0))
        acc5_meter.update(acc5, images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(
                f"Epoch [{epoch}] Step [{i}/{len(loader)}] "
                f"Loss {losses.avg:.4f} Acc@1 {acc1_meter.avg:.2f} Acc@5 {acc5_meter.avg:.2f}"
            )


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    cfg: TrainConfig,
):
    model.eval()
    losses = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    all_l2: List[float] = []
    cum_d_agg: np.ndarray = np.zeros(20)

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(cfg.device, non_blocking=True)
            targets = targets.to(cfg.device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            acc1, acc5, cum_d, acc_l2 = regression_accuracy(
                outputs, targets, cfg.threshold, cfg.threshold_factor
            )

            losses.update(loss.item(), images.size(0))
            acc1_meter.update(acc1, images.size(0))
            acc5_meter.update(acc5, images.size(0))
            cum_d_agg += np.array(cum_d)
            all_l2.extend(acc_l2)

    cum_d_agg = cum_d_agg / max(1, len(loader))  # average over batches
    mean_l2 = float(np.mean(all_l2)) if all_l2 else 0.0
    std_l2 = float(np.std(all_l2)) if all_l2 else 0.0

    print(
        f"[VAL] Loss {losses.avg:.4f} "
        f"Acc@1 {acc1_meter.avg:.2f} Acc@5 {acc5_meter.avg:.2f} "
        f"L2 mean {mean_l2:.2f} std {std_l2:.2f}"
    )
    print("[VAL] Cumulative accuracy (0.0 .. 1.9):", np.round(cum_d_agg, 2))

    return acc1_meter.avg


# =============================================================================
# 7. MAIN: WIRES EVERYTHING TOGETHER
# =============================================================================

def build_transforms(is_train: bool) -> T.Compose:
    """
    TorchVision transforms matching main.py:
    - Train: RandomResizedCrop(224), RandomHorizontalFlip(), ToTensor(), Normalize(...)
    - Val:   Resize(256), CenterCrop(224), ToTensor(), Normalize(...)
    """
    normalize = T.Normalize(mean=[0.229, 0.5, 0.5],
                            std=[0.200, 0.224, 0.225])

    if is_train:
        return T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])
    else:
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize,
        ])

from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split, GroupShuffleSplit


def build_dataloaders_from_single_csv(
    csv_path: str,
    image_dir: str,
    batch_size: int,
    val_ratio: float,
    blur: bool,
):
    """
    Builds train/validation DataLoaders from a single CSV.

    Behaviour:
    -------------------------------------------------------
    • If CSV contains 'subject_id' → Identity-aware split
        - Ensures all images from one subject go to the same split
    • Otherwise → Stratified split on MST labels
        - Preserves label distribution for general images

    CSV expected structure:
        filename,label[,subject_id]

    If no subject_id column:
        filename must still be relative paths (e.g. "0001/img003.jpg")
        but folder structure is NOT used unless subject_id is detected.

    Directory structure (if using subject_id):
        image_dir/
            0000/*.jpg
            0001/*.jpg
            0002/*.jpg
            ...
    """

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[df.columns[0], df.columns[1]])

    filenames = df.iloc[:, 0].astype(str).values
    labels = df.iloc[:, 1].astype(float).values

    # ======================================================
    # Detect subject_id column or infer from folder names
    # ======================================================
    if "subject_id" in df.columns:
        print("[INFO] Found 'subject_id' column — using identity-aware grouped split.")
        groups = df["subject_id"].astype(str).values
    else:
        # Try inferring subject ID from folder/subfolder name
        # Example: 0003/img_12.jpg → subject_id = "0003"
        print("[INFO] No 'subject_id' column found — checking folder-based identity.")
        inferred_groups = []

        for fname in filenames:
            # case: "0007/img001.jpg"
            parts = fname.split("/")
            if len(parts) > 1:
                inferred_groups.append(parts[0])
            else:
                inferred_groups.append("no_group")

        unique_groups = set(inferred_groups)

        if "no_group" not in unique_groups and len(unique_groups) > 1:
            print(f"[INFO] Inferred subject IDs from directory structure ({len(unique_groups)} subjects).")
            groups = np.array(inferred_groups)
        else:
            print("[INFO] No identity structure detected — using stratified split instead.")
            groups = None

    # ======================================================
    # SPLITTING LOGIC
    # ======================================================
    if groups is not None:
        # ------------------------------
        # Identity-aware grouped split
        # ------------------------------
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=val_ratio,
            random_state=42
        )
        train_idx, val_idx = next(splitter.split(df, labels, groups))
        print(f"[INFO] Using GROUPED split: {len(np.unique(groups))} subjects.")
    else:
        # ------------------------------
        # Stratified split (labels)
        # ------------------------------
        train_idx, val_idx = train_test_split(
            np.arange(len(df)),
            test_size=val_ratio,
            shuffle=True,
            stratify=labels,
            random_state=42,
        )
        print("[INFO] Using STRATIFIED split (label balanced).")

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    print(
        f"[INFO] Total={len(df)} | Train={len(train_df)} | Val={len(val_df)} "
        f"(ratio={val_ratio})"
    )

    # ======================================================
    # TorchVision transforms
    # ======================================================
    train_transform = build_transforms(is_train=True)
    val_transform = build_transforms(is_train=False)

    # ======================================================
    # Datasets
    # ======================================================
    train_ds = SkinToneRegressionDataset(
        annotations_file=csv_path,
        img_dir=image_dir,
        transform=train_transform,
        blur=blur,
        conv_to_lab_space=True,
    )
    val_ds = SkinToneRegressionDataset(
        annotations_file=csv_path,
        img_dir=image_dir,
        transform=val_transform,
        blur=blur,
        conv_to_lab_space=True,
    )

    # Replace internal labels with split subsets
    train_ds.img_labels = train_df
    val_ds.img_labels = val_df

    # ======================================================
    # DataLoaders
    # ======================================================
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Path to root dataset directory with 'training' and 'validation' subdirs.")
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--layers-freezed", type=int, default=27)
    parser.add_argument("--vggface-weights", type=str, default="")
    parser.add_argument("--blur", action="store_true")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = TrainConfig(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        threshold=args.threshold,
        layers_freezed=args.layers_freezed,
        vggface_weights=args.vggface_weights,
        blur=args.blur,
    )

    print(f"[INFO] Using device: {cfg.device}")

    # Build model
    model = VGG16Regression(init_weights=True)
    load_vggface_weights_if_available(model, cfg.vggface_weights, cfg.layers_freezed)
    model.to(cfg.device)

    # Criterion and optimizer (MSELoss(size_average=False) equivalent)
    # criterion = nn.MSELoss(reduction="sum").to(cfg.device)
    criterion = nn.MSELoss(reduction="mean").to(cfg.device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    # ================================
    # RESUME FUNCTIONALITY
    # ================================
    start_epoch = 0
    best_acc1 = 0.0

    if args.resume and os.path.isfile(args.resume):
        print(f"[INFO] Resuming training from checkpoint: {args.resume}")

        checkpoint = torch.load(args.resume, map_location=cfg.device)

        # Restore epoch
        start_epoch = checkpoint.get("epoch", 0)
        best_acc1 = checkpoint.get("best_acc1", 0.0)

        # Restore model
        model.load_state_dict(checkpoint["state_dict"])

        # Restore optimizer (momentum buffers included)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("[INFO] Optimizer state restored")

        print(f"[INFO] Resume complete -> Starting at epoch {start_epoch}, Best Acc@1={best_acc1:.2f}")
    else:
        print("[INFO] No checkpoint resumed; starting fresh.")


    # Datasets and loaders
    # train_dir = os.path.join(cfg.data_dir, "training")
    # val_dir = os.path.join(cfg.data_dir, "validation")

    # train_ds = SkinToneRegressionDataset(
    #     annotations_file=os.path.join(train_dir, "annotations.csv"),
    #     img_dir=os.path.join(train_dir, "data"),
    #     transform=build_transforms(is_train=True),
    #     blur=cfg.blur,
    #     conv_to_lab_space=True,
    # )
    # val_ds = SkinToneRegressionDataset(
    #     annotations_file=os.path.join(val_dir, "annotations.csv"),
    #     img_dir=os.path.join(val_dir, "data"),
    #     transform=build_transforms(is_train=False),
    #     blur=cfg.blur,
    #     conv_to_lab_space=True,
    # )

    # train_loader = DataLoader(
    #     train_ds,
    #     batch_size=cfg.batch_size,
    #     shuffle=True,
    #     num_workers=4,
    #     pin_memory=True,
    # )
    # val_loader = DataLoader(
    #     val_ds,
    #     batch_size=cfg.batch_size,
    #     shuffle=False,
    #     num_workers=4,
    #     pin_memory=True,
    # )

    # ============================================================
    # SINGLE CSV SPLIT / STRATIFIED LABEL BALANCING
    # ============================================================
    csv_path = os.path.join(cfg.data_dir, "annotations.csv")
    # image_dir = os.path.join(cfg.data_dir, "data")
    image_dir = cfg.data_dir

    train_loader, val_loader = build_dataloaders_from_single_csv(
        csv_path=csv_path,
        image_dir=image_dir,
        batch_size=cfg.batch_size,
        val_ratio=0.2,
        blur=cfg.blur,
    )

    # =====================================================
    # TRAINING LOOP (NOW RESUME-AWARE)
    # =====================================================
    best_acc1 = 0.0
    for epoch in range(start_epoch, cfg.epochs):
        adjust_learning_rate(optimizer, epoch, cfg)
        train_one_epoch(model, train_loader, criterion, optimizer, epoch, cfg)
        acc1 = validate(model, val_loader, criterion, cfg)

        if acc1 > best_acc1:
            best_acc1 = acc1
            torch.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                },
                "model_best.pth.tar",
            )
            print(f"[INFO] New best Acc@1: {best_acc1:.2f}, checkpoint saved.")

    print(f"[INFO] Training finished. Best Acc@1: {best_acc1:.2f}")


if __name__ == "__main__":
    main()
