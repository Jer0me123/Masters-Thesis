"""
Universal Gender & Skin Tone Annotation Pipeline
================================================

Fully compatible with:

- VGG16 / ResNet18
- RGB / LAB
- Classification / CORAL / Regression
- Your new training script

Outputs JSONL manifest for downstream analysis.
"""

import os
import glob
import json
import argparse
from tqdm import tqdm
from threading import Lock
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from skimage.color import rgb2lab


# ============================================================
# MST BINNING
# ============================================================

def mst_to_bin(mst: int) -> int:
    if mst <= 3:
        return 0
    elif mst <= 7:
        return 1
    else:
        return 2


def bin_to_name(bin_id: int) -> str:
    return {
        0: "Light (1-3)",
        1: "Mid (4-7)",
        2: "Dark (8-10)"
    }.get(bin_id, "Unknown")


# ============================================================
# LAB STATS LOADING
# ============================================================

def load_lab_stats(path: str):
    with open(path, "r") as f:
        stats = json.load(f)

    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)

    print(f"[LAB] Mean: {mean.tolist()}")
    print(f"[LAB] Std:  {std.tolist()}")
    return mean, std


# ============================================================
# UNIVERSAL MODEL
# ============================================================

class UniversalSkinModel(nn.Module):
    def __init__(self, arch, num_outputs, hidden_dim=4096, dropout=0.5):
        super().__init__()
        self.arch = arch

        if arch == "vgg16":
            base = models.vgg16(weights=None)
            self.features = base.features
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            in_features = 512 * 7 * 7

            self.classifier = nn.Sequential(
                nn.Linear(in_features, 4096),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(4096, hidden_dim),   # uses detected hidden_dim
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_outputs),
            )

        elif arch == "resnet18":
            base = models.resnet18(weights=None)
            in_features = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(in_features, num_outputs)

        else:
            raise ValueError("Unsupported architecture")

    def forward(self, x):
        if self.arch == "vgg16":
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            logits = self.classifier(x)
        else:
            x = self.backbone(x)
            x = self.dropout(x)
            logits = self.head(x)

        return logits


# ============================================================
# CHECKPOINT INSPECTION
# ============================================================

def inspect_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    keys = list(state_dict.keys())

    if any(k.startswith("backbone.") for k in keys):
        arch = "resnet18"
        num_outputs = state_dict["head.weight"].shape[0]
        hidden_dim = 4096  # not used for resnet18
    else:
        arch = "vgg16"
        last = [k for k in keys if "classifier" in k and "weight" in k][-1]
        num_outputs = state_dict[last].shape[0]
        # detect the intermediate hidden dim from classifier.3.weight
        hidden_dim = state_dict["classifier.3.weight"].shape[0]

    mode = "regression" if num_outputs == 1 else "classification"

    print(f"[Inspector] Arch: {arch}")
    print(f"[Inspector] Outputs: {num_outputs}")
    print(f"[Inspector] Hidden dim: {hidden_dim}")
    print(f"[Inspector] Mode guess: {mode}")

    return arch, num_outputs, hidden_dim, mode


# ============================================================
# TRANSFORMS
# ============================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RGBTransform:
    def __init__(self):
        self.t = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __call__(self, img):
        return self.t(img)


class LABTransform:
    def __init__(self, mean, std):
        self.resize = transforms.Resize((224, 224))
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = self.resize(img)
        rgb = np.asarray(img).astype(np.float32) / 255.0
        lab = rgb2lab(rgb).astype(np.float32)
        lab = (lab - self.mean) / self.std
        return torch.from_numpy(lab.transpose(2, 0, 1)).float()


# ============================================================
# SKIN PREDICTOR
# ============================================================

class SkinTonePredictor:

    def __init__(self, model_path, input_space, lab_mean=None, lab_std=None, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        arch, num_outputs, hidden_dim, mode_guess = inspect_checkpoint(model_path)

        self.arch = arch
        self.num_outputs = num_outputs
        self.mode = mode_guess

        self.model = UniversalSkinModel(
            arch=arch,
            num_outputs=num_outputs,
            hidden_dim=hidden_dim
        )

        ckpt = torch.load(model_path, map_location=self.device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        if input_space == "rgb":
            self.transform = RGBTransform()
        else:
            self.transform = LABTransform(lab_mean, lab_std)

        self.input_space = input_space

    def decode(self, logits):

        # Regression
        if self.num_outputs == 1:
            values = logits.squeeze(1).cpu().numpy()
            return np.clip(np.round(values), 1, 10).astype(int)

        # Possible CORAL
        if self.num_outputs <= 9:
            probs = torch.sigmoid(logits)
            probs = torch.cummin(probs, dim=1)[0]
            preds = torch.sum(probs > 0.5, dim=1).cpu().numpy()
            return preds + 1

        # Standard classification
        preds = logits.argmax(dim=1).cpu().numpy()
        return preds + 1

    def predict_batch(self, imgs):
        tensors = [self.transform(img) for img in imgs]
        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            logits = self.model(batch)

        mst_labels = self.decode(logits)

        results = []
        for mst in mst_labels:
            bin_label = mst_to_bin(int(mst))
            results.append((int(mst), int(bin_label)))

        return results


# ============================================================
# GENDER MODEL
# ============================================================

class GenderPredictor:
    def __init__(self, device="cuda"):
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        model_name = "prithivMLmods/Realistic-Gender-Classification"
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def predict_batch(self, imgs):
        inputs = self.processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            classes = probs.argmax(dim=1).cpu().numpy()
            confs = probs.max(dim=1).values.cpu().numpy()

        results = []
        for c, conf in zip(classes, confs):
            label = self.model.config.id2label[int(c)]
            gender = label.split()[0].capitalize()
            results.append((gender, float(conf)))

        return results


# ============================================================
# MAIN
# ============================================================

def main(args):

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, "annotations.jsonl")

    gender_model = GenderPredictor(args.device)

    lab_mean = lab_std = None
    if args.input_space == "lab":
        lab_mean, lab_std = load_lab_stats(args.lab_stats_path)

    skin_model = SkinTonePredictor(
        model_path=args.skin_model_path,
        input_space=args.input_space,
        lab_mean=lab_mean,
        lab_std=lab_std,
        device=args.device,
    )

    images = glob.glob(os.path.join(args.image_dir, "**", "*.jpg"), recursive=True)

    print(f"[Found] {len(images)} images")

    with open(manifest_path, "w", encoding="utf-8") as f:

        for i in tqdm(range(0, len(images), args.batch_size)):
            batch_paths = images[i:i + args.batch_size]
            batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]

            gender_results = gender_model.predict_batch(batch_imgs)
            skin_results = skin_model.predict_batch(batch_imgs)

            for path, (gender, conf), (mst, bin_label) in zip(batch_paths, gender_results, skin_results):

                record = {
                    "image": os.path.relpath(path, args.image_dir),
                    "gender": gender,
                    "gender_confidence": conf,
                    "mst_label": mst,
                    # "bin_label": bin_label,
                    # "bin_name": bin_to_name(bin_label)
                }

                f.write(json.dumps(record) + "\n")

    print("\n[Complete] Annotations saved.")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--skin_model_path", required=True)
    parser.add_argument("--input_space", choices=["rgb", "lab"], default="rgb")
    parser.add_argument("--lab_stats_path")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    if args.input_space == "lab" and args.lab_stats_path is None:
        parser.error("--lab_stats_path required for LAB mode")

    main(args)

# C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\DatasetAnnotation\.venv\Scripts\activate

# python "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\GenderSkinToneProxy\3_AnnotateDatasetsUsingSkinToneModel.py" ^
# --image_dir "G:\Thesis\MonkSkinTone_Dataset\Segmented_MSTE_BGFixed" ^
# --output_dir "F:\VGG_MST_Testing\Models\ResNet18_4CCoral_LAB_FixedBG\Segmented_MSTE_BGFixed" ^
# --skin_model_path "F:\VGG_MST_Testing\Models\ResNet18_4CCoral_LAB_FixedBG\best_model.pth" ^
# --input_space lab ^
# --lab_stats_path "F:\VGG_MST_Testing\Models\ResNet18_4CCoral_LAB_FixedBG\lab_statistics.json" ^
# --batch_size 32

# python "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\GenderSkinToneProxy\3_AnnotateDatasetsUsingSkinToneModel.py" ^
# --image_dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2_BGFixed" ^
# --output_dir "F:\VGG_MST_Testing\Models\ResNet18_4CCoral_LAB_FixedBG\Segmented_CCV2_BGFixed" ^
# --skin_model_path "F:\VGG_MST_Testing\Models\ResNet18_4CCoral_LAB_FixedBG\best_model.pth" ^
# --input_space lab ^
# --lab_stats_path "F:\VGG_MST_Testing\Models\ResNet18_4CCoral_LAB_FixedBG\lab_statistics.json" ^
# --batch_size 32

# python "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\GenderSkinToneProxy\3_AnnotateDatasetsUsingSkinToneModel.py" ^
# --image_dir "F:\Thesis\Segmented_FairFace_BGFixed" ^
# --output_dir "F:\VGG_MST_Testing\Models\ResNet18_4CCoral_LAB_FixedBG\Segmented_FairFace_BGFixed" ^
# --skin_model_path "F:\VGG_MST_Testing\Models\ResNet18_4CCoral_LAB_FixedBG\best_model.pth" ^
# --input_space lab ^
# --lab_stats_path "F:\VGG_MST_Testing\Models\ResNet18_4CCoral_LAB_FixedBG\lab_statistics.json" ^
# --batch_size 32

# python "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\GenderSkinToneProxy\3_AnnotateDatasetsUsingSkinToneModel.py" ^
#   --image_dir "F:\Thesis\Fairface_Dataset\Segmented_FairFace" ^
#   --output_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\GenderSkinToneProxy\MappingRaceToSkinTone" ^
#   --skin_model_path "F:\VGG_MST_Testing\Models\VGG16_10Regression_RGB\best_model.pth" ^
#   --input_space rgb ^
#   --batch_size 32