#!/usr/bin/env python
"""
ITI-GEN + ControlNet + Chamfer Color Alignment (Aligned with zero-shot.py logic)

What changed vs your current ITIGen-ControlNet-ChamferDebiasing.py
-----------------------------------------------------------------
1) Projection is applied INSIDE a DDPM-like backward step (x_t -> x_{t-1}),
   matching zero-shot.py's `AlignedDDPMScheduler.step_backward` placement.
2) The projection uses the same *iterative matching* loop (my_chamfer_distance + mapping peeling)
   used in zero-shot.py (not a single chamfer_distance call).
3) Timesteps are handled in the same "DDPM 1000-step index space" and decimated to the requested
   --num_inference_steps (exactly like zero-shot.py).

Notes
-----
- This keeps your Diffusers UNet/ControlNet/CLIP/VAE stack. Only the *sampler step* is replaced.
- For Stable Diffusion v1.5, the betas used here default to the canonical SD schedule:
  beta_start=0.00085, beta_end=0.0120, beta_schedule=scaled_linear, num_train_timesteps=1000.

Usage (close to your current CLI)
---------------------------------
python ITIGen-ControlNet-ChamferDebiasing_AlignedLikeZeroShot.py ^
  --outdir "./outputs/zeroshot_pred" ^
  --color-align-enabled ^
  --color-reference-image "./pastel.jpeg" ^
  --with-pred-sample-projection ^
  --projection-threshold 200 ^
  --projection-lr 0.1 ^
  --blur-factor 3 ^
  --attr-list "Male,MSTESkin_tone" ^
  --filters Male_Negative MSTESkin_tone_10 ^
  --prompt "a headshot of a person" ^
  --generate-image-prompt "A picture of a doctor" ^
  --num_inference_steps 50 ^
  --guidance_scale 7.5 ^
  --n_iter 3 ^
  --save-intermediate
"""

import argparse
import os
import gc
import cv2
import math
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
from typing import Tuple, List, Optional, Union

from pytorch_lightning import seed_everything

from diffusers import (
    ControlNetModel,
    AutoencoderKL,
    UNet2DConditionModel,
)
from transformers import CLIPTokenizer, CLIPTextModel

from utils import get_folder_names_and_indexes
from iti_gen.model import ITI_GEN

# ----------------------------
# Optional PyTorch3D
# ----------------------------
try:
    from pytorch3d.ops.knn import knn_gather, knn_points
    from pytorch3d.structures.pointclouds import Pointclouds
    PYTORCH3D_AVAILABLE = True
except Exception:
    knn_gather = None
    knn_points = None
    Pointclouds = None
    PYTORCH3D_AVAILABLE = False

# ----------------------------
# ControlNet model mapping
# ----------------------------
CONTROLNET_MODELS = {
    "pose": "lllyasviel/control_v11p_sd15_openpose",
    "canny": "lllyasviel/control_v11p_sd15_canny",
    "depth": "lllyasviel/control_v11f1p_sd15_depth",
    "seg": "lllyasviel/control_v11p_sd15_seg",
}

# ----------------------------
# zero-shot.py: matching Chamfer implementation pieces
# ----------------------------
def _validate_chamfer_reduction_inputs(batch_reduction: Union[str, None], point_reduction: Union[str, None]) -> None:
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction is not None and point_reduction not in ["mean", "sum", "max"]:
        raise ValueError('point_reduction must be one of ["mean", "sum", "max"] or None')
    if point_reduction is None and batch_reduction is not None:
        raise ValueError("Batch reduction must be None if point_reduction is None")

def _handle_pointcloud_input(points, lengths, normals):
    if PYTORCH3D_AVAILABLE and isinstance(points, Pointclouds):
        X = points.points_padded()
        lengths = points.num_points_per_cloud()
        normals = points.normals_padded()
    elif torch.is_tensor(points):
        if points.ndim != 3:
            raise ValueError("Expected points to be of shape (N, P, D)")
        X = points
        if lengths is not None:
            if lengths.ndim != 1 or lengths.shape[0] != X.shape[0]:
                raise ValueError("Expected lengths to be of shape (N,)")
            if lengths.max() > X.shape[1]:
                raise ValueError("A length value was too long")
        if lengths is None:
            lengths = torch.full((X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device)
        if normals is not None and normals.ndim != 3:
            raise ValueError("Expected normals to be of shape (N, P, 3")
    else:
        raise ValueError("Input pointclouds should be Pointclouds or torch.Tensor (N,P,D).")
    return X, lengths, normals

def _chamfer_distance_single_direction(x, y, x_lengths, y_lengths, x_normals, y_normals, weights,
                                      point_reduction: Union[str, None], norm: int, abs_cosine: bool):
    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    is_x_heterogeneous = (x_lengths != P1).any()
    x_mask = (torch.arange(P1, device=x.device)[None] >= x_lengths[:, None])

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)
    cham_x = x_nn.dists[..., 0]
    idx_x = x_nn.idx[..., 0]

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)

    if return_normals:
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
        cosine_sim = F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        cham_norm_x = 1 - (torch.abs(cosine_sim) if abs_cosine else cosine_sim)
        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0
        if weights is not None:
            cham_norm_x *= weights.view(N, 1)

    if point_reduction == "max":
        assert not return_normals
        cham_x = cham_x.max(1).values
    elif point_reduction is not None:
        cham_x = cham_x.sum(1)
        if return_normals:
            cham_norm_x = cham_norm_x.sum(1)
        if point_reduction == "mean":
            x_lengths_clamped = x_lengths.clamp(min=1)
            cham_x /= x_lengths_clamped
            if return_normals:
                cham_norm_x /= x_lengths_clamped

    cham_dist = cham_x
    cham_normals = cham_norm_x if return_normals else None
    return cham_dist, cham_normals, idx_x

def _apply_batch_reduction(cham_x, cham_norm_x, weights, batch_reduction: Union[str, None]):
    if batch_reduction is None:
        return (cham_x, cham_norm_x)
    N = cham_x.shape[0]
    cham_x = cham_x.sum()
    if cham_norm_x is not None:
        cham_norm_x = cham_norm_x.sum()
    if batch_reduction == "mean":
        if weights is None:
            div = max(N, 1)
        elif weights.sum() == 0.0:
            div = 1
        else:
            div = weights.sum()
        cham_x /= div
        if cham_norm_x is not None:
            cham_norm_x /= div
    return (cham_x, cham_norm_x)

def my_chamfer_distance(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: Union[str, None] = "mean",
    norm: int = 2,
    single_directional: bool = False,
    abs_cosine: bool = True,
):
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)
    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")
    if point_reduction == "max" and (x_normals is not None or y_normals is not None):
        raise ValueError('Normals must be None if point_reduction is "max"')

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    cham_x, cham_norm_x, idx_x = _chamfer_distance_single_direction(
        x, y, x_lengths, y_lengths, x_normals, y_normals, weights, point_reduction, norm, abs_cosine
    )

    if single_directional:
        loss = cham_x
        loss_normals = cham_norm_x
    else:
        cham_y, cham_norm_y, _ = _chamfer_distance_single_direction(
            y, x, y_lengths, x_lengths, y_normals, x_normals, weights, point_reduction, norm, abs_cosine
        )
        if point_reduction == "max":
            loss = torch.maximum(cham_x, cham_y)
            loss_normals = None
        elif point_reduction is not None:
            loss = cham_x + cham_y
            loss_normals = (cham_norm_x + cham_norm_y) if cham_norm_x is not None else None
        else:
            loss = (cham_x, cham_y)
            loss_normals = (cham_norm_x, cham_norm_y) if cham_norm_x is not None else None

    return _apply_batch_reduction(loss, loss_normals, weights, batch_reduction), idx_x

# ----------------------------
# Helpers copied from your script
# ----------------------------
def shuffle_latents(latents_original, raw_latent_length, latent_dim, device):
    """Shuffle latents for color conditioning (same indexing logic as zero-shot.py)."""
    cur_bsz = latents_original.shape[0]
    latents_shuffled = torch.zeros_like(latents_original).to(device=device)

    for bsi in range(cur_bsz):
        div_factor_x = 1
        div_factor_y = 1
        shuffle_idx_sub = torch.randperm(int(raw_latent_length * raw_latent_length / div_factor_x / div_factor_y))
        shuffle_idx_sub = shuffle_idx_sub.reshape(int(raw_latent_length/div_factor_x), int(raw_latent_length/div_factor_y))

        shuffle_idx = torch.zeros(raw_latent_length, raw_latent_length)
        for sub_i in range(div_factor_x):
            for sub_j in range(div_factor_y):
                shuffle_idx[sub_i::div_factor_x, sub_j::div_factor_y] = (
                    (shuffle_idx_sub * div_factor_x) +
                    (((shuffle_idx_sub * div_factor_x) // raw_latent_length) * raw_latent_length * div_factor_y) -
                    (((shuffle_idx_sub * div_factor_x) // raw_latent_length) * raw_latent_length) +
                    sub_j + (sub_i * raw_latent_length)
                )
        shuffle_idx = shuffle_idx.reshape(raw_latent_length * raw_latent_length).to(dtype=torch.int64)

        latents_shuffled[bsi] = latents_original[bsi].reshape(latent_dim, -1)[:, shuffle_idx].reshape(
            latent_dim, raw_latent_length, raw_latent_length
        )

    return latents_shuffled

def filter_labels_by_substring(label_map: dict, query: str) -> dict:
    q = query.lower()
    return {label: idx for label, idx in label_map.items() if q in label.lower()}

def apply_filters(label_map: dict, filters: Optional[List[str]]) -> dict:
    if not filters:
        return label_map
    out = label_map
    for f in filters:
        out = filter_labels_by_substring(out, f)
    return out

def parse_device(device_str: str) -> str:
    s = device_str.strip().lower()
    if s == "cuda":
        return "cuda"
    if s.startswith("cuda:"):
        return s
    return "cpu"

def resolve_control_images(path: str) -> List[str]:
    if os.path.isdir(path):
        images = [
            os.path.join(path, f)
            for f in sorted(os.listdir(path))
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not images:
            raise ValueError(f"No images found in directory: {path}")
        return images
    if os.path.isfile(path):
        return [path]
    raise ValueError(f"Invalid --control-image path: {path}")

def resize_with_padding(
    img: Image.Image,
    target_size: Tuple[int, int],
    fill_color=(0, 0, 0),  # black padding (ControlNet-safe)
):
    target_w, target_h = target_size
    w, h = img.size

    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    img_resized = img.resize((new_w, new_h), resample=Image.BILINEAR)

    canvas = Image.new("RGB", (target_w, target_h), fill_color)

    left = (target_w - new_w) // 2
    top = (target_h - new_h) // 2

    canvas.paste(img_resized, (left, top))
    return canvas


def load_control_map(image_path: str, control_type: str, target_size: Tuple[int, int], is_map: bool = False) -> Image.Image:
    if control_type not in CONTROLNET_MODELS:
        raise ValueError(f"Unknown control_type: {control_type}. Choose from {list(CONTROLNET_MODELS.keys())}")

    src = Image.open(image_path).convert("RGB")
    src = ImageOps.exif_transpose(src).convert("RGB")
    src_np = np.array(src)

    if is_map:
        control = src
    elif control_type == "pose":
        from controlnet_aux import OpenposeDetector
        detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        control = detector(src)
    elif control_type == "canny":
        img = cv2.GaussianBlur(src_np, (5, 5), sigmaX=1.5)
        edges = cv2.Canny(img, 100, 200)
        control_np = np.stack([edges] * 3, axis=-1)
        control = Image.fromarray(control_np)
    elif control_type == "depth":
        from controlnet_aux import MidasDetector
        detector = MidasDetector.from_pretrained("lllyasviel/ControlNet")
        control = detector(src)
    elif control_type == "seg":
        from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
        image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
        segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small").to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        with torch.no_grad():
            pixel_values = image_processor(src, return_tensors="pt").pixel_values.to(segmentor.device)
            outputs = segmentor(pixel_values)
            seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[src.size[::-1]])[0].cpu().numpy()

        color_seg = np.zeros((*seg.shape, 3), dtype=np.uint8)
        unique_labels = np.unique(seg)
        for label in unique_labels:
            color = np.random.randint(0, 255, 3)
            color_seg[seg == label] = color
        control = Image.fromarray(color_seg)
    else:
        raise ValueError(f"Unhandled control_type: {control_type}")

    return resize_with_padding(control.convert("RGB"), target_size)
    # return control.convert("RGB").resize(target_size, resample=Image.BILINEAR)

def load_reference_image_for_alignment(ref_path, size, device, dtype, blur_factor=3):
    import torchvision.transforms as transforms

    ref_im = Image.open(ref_path).convert("RGB")
    ref_im = ref_im.resize(size)

    augmentations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    ref_tensor = augmentations(ref_im).unsqueeze(0).to(device, dtype=dtype)

    # blur in pixel space (as in zero-shot.py)
    ref_blurred = torch.nn.functional.interpolate(
        ref_tensor,
        size=(size[1] // blur_factor, size[0] // blur_factor),
        mode="bilinear",
        align_corners=False,
    )
    ref_blurred = torch.nn.functional.interpolate(
        ref_blurred,
        size=size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    return ref_blurred

# ----------------------------
# Aligned sampler (zero-shot.py style step placement)
# ----------------------------
class AlignedDDPMSampler:
    """
    Inference-only DDPM sampler with zero-shot.py projection placement.

    This class mirrors the *structure* of zero-shot.py's AlignedDDPMScheduler.step_backward(),
    but is adapted to Stable Diffusion latents:
      - x_t is 4x64x64 (for 512x512)
      - UNet predicts epsilon ("noise") as in SD
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 50,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
        beta_schedule: str = "scaled_linear",
        prediction_type: str = "noise",   # SD UNet predicts eps
        projection_threshold: int = 200,
        with_noisy_sample_projection: bool = False,
        with_pred_sample_projection: bool = False,
        projection_lr: float = 0.1,
        latent_length: int = 64,          # 512/8
        latent_dim: int = 4,
    ):
        self.weight_dtype = torch.float32
        self.num_train_steps = int(num_train_timesteps)
        self.num_inference_steps = int(num_inference_steps)

        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, self.num_train_steps, dtype=self.weight_dtype)
        elif beta_schedule == "scaled_linear":
            betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, self.num_train_steps, dtype=self.weight_dtype) ** 2
        else:
            raise NotImplementedError(beta_schedule)

        self.betas = betas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        self.prediction_type = prediction_type
        self.projection_threshold = int(projection_threshold)
        self.with_noisy_sample_projection = bool(with_noisy_sample_projection)
        self.with_pred_sample_projection = bool(with_pred_sample_projection)
        self.projection_lr = float(projection_lr)

        self.latent_dim = int(latent_dim)
        self.raw_latent_length = int(latent_length)

        # build inference timestep indices (descending), like zero-shot.py:
        step = self.num_train_steps // self.num_inference_steps
        self.timesteps = list(range(self.num_train_steps - 1, -1, -step))

    def previous_timestep(self, timestep: int) -> int:
        step = self.num_train_steps // self.num_inference_steps
        return timestep - step

    def _get_variance(self, t: int):
        prev_t = self.previous_timestep(t)
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        return torch.clamp(variance, min=1e-20)

    def step_backward(
        self,
        x_t: torch.FloatTensor,
        eps_pred: torch.FloatTensor,
        timestep: int,
        ref_latents: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Performs x_t -> x_{t-1} and applies projection inside the step,
        matching zero-shot.py's placement.
        """
        device = x_t.device
        dtype = x_t.dtype
        bsz = x_t.shape[0]

        t = int(timestep)
        prev_t = self.previous_timestep(t)

        # move buffers
        alphas_cumprod = self.alphas_cumprod.to(device=device, dtype=dtype)
        one = self.one.to(device=device, dtype=dtype)

        alpha_prod_t = alphas_cumprod[t]
        alpha_prod_t_prev = alphas_cumprod[prev_t] if prev_t >= 0 else one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # variance noise (DDPM)
        variance = 0
        if t > 0:
            variance_noise = torch.randn_like(x_t)
            variance = (self._get_variance(t).to(device=device, dtype=dtype) ** 0.5) * variance_noise

        # pred x0
        if self.prediction_type == "noise":
            pred_x0 = (x_t - (beta_prod_t ** 0.5) * eps_pred) / (alpha_prod_t ** 0.5)
        else:
            raise NotImplementedError("This aligned sampler is configured for SD eps prediction (prediction_type=noise).")

        # -----------------------------
        # ZERO-SHOT projection on pred_x0
        # -----------------------------
        if self.with_pred_sample_projection and t > self.projection_threshold:
            pred_x0_var = pred_x0.detach().clone().requires_grad_(True)

            # shuffle like zero-shot
            pred_x0_shuf = shuffle_latents(pred_x0_var, self.raw_latent_length, self.latent_dim, device)

            pred_points = torch.swapaxes(pred_x0_shuf.reshape(bsz, self.latent_dim, -1), 1, 2)
            ref_points = torch.swapaxes(ref_latents.reshape(bsz, self.latent_dim, -1), 1, 2).clone().detach().requires_grad_(True)

            # tiny jitter as in zero-shot.py (note: their code uses (rand - 1.), keep similar magnitude)
            ref_points = ref_points + 1e-6 * (torch.rand_like(ref_points) - 1.0)

            dist_total = 0.0
            for pred_point, ref_point in zip(pred_points, ref_points):
                cur_pred_point = pred_point[None, ...]
                cur_ref_point = ref_point[None, ...]

                while True:
                    cur_ref_point = cur_ref_point.clone().detach().requires_grad_(True)
                    (dist, _), mapping = my_chamfer_distance(
                        cur_pred_point, cur_ref_point,
                        batch_reduction=None, point_reduction=None,
                        single_directional=True
                    )
                    dist = dist[0]
                    mapping = mapping[0]

                    unique, idx, counts = torch.unique(mapping, dim=0, sorted=True, return_inverse=True, return_counts=True)
                    _, ind_sorted = torch.sort(idx, stable=True)
                    cum_sum = counts.cumsum(0)
                    cum_sum = torch.cat((torch.tensor([0], device=device), cum_sum[:-1]))
                    first_indices, _ = torch.sort(ind_sorted[cum_sum])
                    first_indices_opponent = mapping[first_indices]

                    dist_component = torch.sum(dist[first_indices])
                    dist_total = dist_total + dist_component

                    world = torch.ones_like(mapping)
                    world[first_indices] = 0
                    compl_pred = torch.nonzero(world).squeeze()
                    if compl_pred.nelement() == 0:
                        break

                    world = torch.ones_like(mapping)
                    world[first_indices_opponent] = 0
                    compl_ref = torch.nonzero(world).squeeze()

                    cur_pred_point = cur_pred_point[:, compl_pred, :]
                    if len(cur_pred_point.shape) == 2:
                        cur_pred_point = cur_pred_point[:, None, :]

                    cur_ref_point = cur_ref_point[:, compl_ref, :]
                    if len(cur_ref_point.shape) == 2:
                        cur_ref_point = cur_ref_point[:, None, :]

            dist_total.backward()

            with torch.no_grad():
                if pred_x0_var.grad is not None:
                    pred_x0 = (pred_x0_var - self.projection_lr * pred_x0_var.grad).detach()
                else:
                    pred_x0 = pred_x0_var.detach()

        # -----------------------------
        # DDPM recomposition (matches zero-shot.py coefficients)
        # -----------------------------
        pred_x0_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t) / beta_prod_t
        x_t_coeff = (current_alpha_t ** 0.5) * beta_prod_t_prev / beta_prod_t

        x_prev = pred_x0_coeff * pred_x0 + x_t_coeff * x_t
        x_prev = x_prev + (variance if t > 0 else 0)

        # optional training-time projection (rarely used here, kept for parity)
        if self.with_noisy_sample_projection and t > self.projection_threshold:
            x_prev_var = x_prev.detach().clone().requires_grad_(True)
            next_points = torch.swapaxes(x_prev_var.reshape(bsz, self.latent_dim, -1), 1, 2)
            ref_points2 = torch.swapaxes(ref_latents.reshape(bsz, self.latent_dim, -1), 1, 2).clone().detach().requires_grad_(True)
            (dist2, _), _ = my_chamfer_distance(next_points, ref_points2, batch_reduction=None, point_reduction=None, single_directional=True)
            # dist2 is a tuple (forward loss), use first element
            dist2 = dist2[0].sum()
            dist2.backward()
            with torch.no_grad():
                if x_prev_var.grad is not None:
                    x_prev = (x_prev_var - self.projection_lr * x_prev_var.grad).detach()
                else:
                    x_prev = x_prev_var.detach()

        return x_prev.to(dtype=dtype)

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()

    # Output / runtime
    parser.add_argument("--outdir", type=str, default="outputs/iti-gen-chamfer", help="Directory to write results to")
    parser.add_argument("--n_iter", type=int, default=1, help="Number of iterations per demographic combination")
    parser.add_argument("--n_samples", type=int, default=1, help="Batch size / num images per iteration")
    parser.add_argument("--H", type=int, default=512, help="Image height")
    parser.add_argument("--W", type=int, default=512, help="Image width")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda, cuda:0, cpu")

    # Diffusers models
    parser.add_argument("--sd-model", type=str, default="runwayml/stable-diffusion-v1-5", help="SD v1.5 model id")
    parser.add_argument("--lora-model", type=str, default="latent-consistency/lcm-lora-sdv1-5", help="SD v1.5 lora id")
    parser.add_argument("--use-lora", action="store_true", help="Use lora")

    # ControlNet
    parser.add_argument("--control-type", type=str, default=None, choices=list(CONTROLNET_MODELS.keys()),
                        help="Control type: pose | canny | depth | seg")
    parser.add_argument("--control-image", type=str, default=None, help="Path to control image")
    parser.add_argument("--controlnet-scale", type=float, default=1.0, help="ControlNet conditioning scale")
    parser.add_argument("--control-is-map", action="store_true", help="Control image is already a map")

    # ITI-GEN parameters
    parser.add_argument("--attr-list", type=str, default="Male,MSTESkin_tone", help="Attributes separated by commas")
    parser.add_argument("--ckpt-path", type=str, default="./ckpts", help="Path to ITI-GEN checkpoints")
    parser.add_argument("--prompt", type=str, default="a headshot of a person", help="Training prompt")
    parser.add_argument("--load-model-epoch", type=int, default=29, help="Model epoch to load")
    parser.add_argument("--generate-image-prompt", type=str, default="a headshot of a person", help="Prompt for generation")
    parser.add_argument("--data-path", type=str, default="./data", help="Path to reference images")
    parser.add_argument("--token-length", type=int, default=3, help="Length of learned tokens")
    parser.add_argument("--filters", type=str, nargs="*", default=None, help="Label filters")

    # Color alignment (zero-shot)
    parser.add_argument("--color-align-enabled", action="store_true", help="Enable color alignment using chamfer distance")
    parser.add_argument("--color-reference-image", type=str, default=None, help="Reference image for color alignment")
    parser.add_argument("--with-noisy-sample-projection", action="store_true", help="Project x_{t-1} directly (training-time style)")
    parser.add_argument("--with-pred-sample-projection", action="store_true", help="Project predicted x0 (zero-shot method)")
    parser.add_argument("--projection-threshold", type=int, default=200, help="Apply projection when DDPM t > threshold (0..999)")
    parser.add_argument("--blur-factor", type=int, default=3, help="Blur factor for reference image")
    parser.add_argument("--projection-lr", type=float, default=0.1, help="Projection step size (like 0.1..0.5)")
    parser.add_argument("--save-intermediate", action="store_true", help="Save intermediate steps")

    # SD beta schedule (match zero-shot.py run command)
    parser.add_argument("--ddpm_num_steps", type=int, default=1000, help="DDPM train steps (SD default 1000)")
    parser.add_argument("--bdpm_beta_schedule", type=str, default="scaled_linear", choices=["linear", "scaled_linear"])
    parser.add_argument("--beta_start", type=float, default=0.00085)
    parser.add_argument("--beta_end", type=float, default=0.0120)

    # Negative prompt
    parser.add_argument("--negative_prompt", type=str,
                        default="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
                        help="Negative prompt")

    opt = parser.parse_args()
    opt.device = parse_device(opt.device)

    os.makedirs(opt.outdir, exist_ok=True)
    if opt.save_intermediate:
        os.makedirs(os.path.join(opt.outdir, "steps"), exist_ok=True)

    # Validate color alignment parameters
    if opt.color_align_enabled:
        if not PYTORCH3D_AVAILABLE:
            raise ImportError("PyTorch3D (knn_points/knn_gather) is required for aligned zero-shot projection.")
        if not opt.color_reference_image:
            raise ValueError("When --color-align-enabled is set, you must provide --color-reference-image")

    # Determine if we're using ControlNet
    use_controlnet = opt.control_type is not None
    if use_controlnet and not opt.control_image:
        raise ValueError("When --control-type is set, you must also provide --control-image")

    dtype = torch.float32

    print("Loading Stable Diffusion components...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype).to(opt.device)
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype).to(opt.device)
    unet = UNet2DConditionModel.from_pretrained(opt.sd_model, subfolder="unet", torch_dtype=dtype).to(opt.device)

    if opt.use_lora and opt.lora_model is not None:
        print("Loading LCM-LoRA for speed-up...")

        unet.load_lora_adapter(
            opt.lora_model,
            weight_name="pytorch_lora_weights.safetensors",
            adapter_name="lcm"
        )

    # Load ControlNet if specified
    controlnet = None
    if use_controlnet:
        controlnet_id = CONTROLNET_MODELS[opt.control_type]
        print(f"Loading ControlNet ({opt.control_type})...")
        controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype).to(opt.device)
        print("✓ ControlNet loaded")

    # Build the aligned sampler (zero-shot placement)
    sampler = AlignedDDPMSampler(
        num_train_timesteps=opt.ddpm_num_steps,
        num_inference_steps=opt.num_inference_steps,
        beta_start=opt.beta_start,
        beta_end=opt.beta_end,
        beta_schedule=opt.bdpm_beta_schedule,
        prediction_type="noise",
        projection_threshold=opt.projection_threshold,
        with_noisy_sample_projection=opt.with_noisy_sample_projection,
        with_pred_sample_projection=opt.with_pred_sample_projection,
        projection_lr=opt.projection_lr,
        latent_length=opt.H // 8,
        latent_dim=4,
    )

    # Load reference image for color alignment and encode to latents
    ref_blurred_latents = None
    if opt.color_align_enabled:
        print(f"Loading reference for color alignment: {opt.color_reference_image}")
        ref_image = load_reference_image_for_alignment(opt.color_reference_image, (opt.W, opt.H), opt.device, dtype, opt.blur_factor)
        with torch.no_grad():
            ref_blurred_latents = vae.encode(ref_image).latent_dist.sample()
            ref_blurred_latents = ref_blurred_latents * vae.config.scaling_factor
        print("✓ Reference loaded and encoded to latent space")

    # Build ControlNet conditioning image(s)
    control_maps = None
    if use_controlnet:
        control_image_paths = resolve_control_images(opt.control_image)
        print(f"Building {len(control_image_paths)} control maps (type={opt.control_type})")
        control_maps = [load_control_map(img_path, opt.control_type, (opt.W, opt.H), opt.control_is_map) for img_path in control_image_paths]

    # Initialize ITI-GEN model
    print("Loading ITI-GEN embeddings...")

    class ITIArgs:
        def __init__(self, opt_):
            self.attr_list = opt_.attr_list
            self.data_path = opt_.data_path
            self.token_length = opt_.token_length
            if opt_.device == "cuda":
                self.device = 0
            elif opt_.device.startswith("cuda:"):
                self.device = int(opt_.device.split(":")[-1])
            else:
                self.device = -1
            self.prompt = opt_.prompt
            self.refer_size_per_category = 200
            self.steps_per_epoch = 5
            self.lr = 0.01

    iti_args = ITIArgs(opt)
    iti_gen = ITI_GEN(iti_args)

    folder_path = os.path.join(opt.ckpt_path, f"{opt.prompt.replace(' ', '_')}_{'_'.join(iti_gen.attr_list)}")
    for idx, attr in enumerate(iti_gen.attr_list):
        weight_path = os.path.join(folder_path, f"basis_perturbation_embed_{opt.load_model_epoch}_{attr}.pth")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Missing ITI-GEN weight: {weight_path}")
        state = torch.load(weight_path, map_location="cpu")
        iti_gen.fairtoken_model[idx].load_state_dict(state, strict=False)
        iti_gen.fairtoken_model[idx].eval()

    with torch.no_grad():
        prepend_embeddings = iti_gen.prompt_prepend(opt.generate_image_prompt)
        emb = prepend_embeddings.to(opt.device, dtype=dtype)

    print("✓ ITI-GEN embeddings loaded")

    del iti_gen, prepend_embeddings
    gc.collect()
    if "cuda" in opt.device:
        torch.cuda.empty_cache()

    # Folder combinations
    folder_with_indexes = get_folder_names_and_indexes(opt.attr_list.split(","))
    folder_with_indexes = apply_filters(folder_with_indexes, opt.filters)
    if len(folder_with_indexes) == 0:
        raise ValueError("No demographic combinations left after filtering.")

    # Decoder
    def decode_to_image(latent):
        with torch.no_grad():
            image = vae.decode(latent.to(dtype) / vae.config.scaling_factor, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu()
        image = (image * 255).round().to(torch.uint8)
        image = image.permute(0, 2, 3, 1).numpy()
        return Image.fromarray(image[0])

    # Precompute unconditional embeddings for CFG
    max_length = tokenizer.model_max_length
    uncond_input = tokenizer([""] * opt.n_samples, padding="max_length", max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(opt.device))[0].to(dtype)

    print(f"\nGenerating images for {len(folder_with_indexes)} combinations...")
    mode_str = f"ITI-GEN + {'ControlNet=' + opt.control_type if use_controlnet else 'No ControlNet'} + {'Aligned Zero-Shot Projection' if opt.color_align_enabled else 'No Color Alignment'}"
    print(f"Mode: {mode_str}")

    for folder, index in tqdm(folder_with_indexes.items(), desc="Combinations"):
        sample_path = os.path.join(opt.outdir, folder)
        os.makedirs(sample_path, exist_ok=True)

        base_count = len([f for f in os.listdir(sample_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

        combination_emb = emb[index].unsqueeze(0).repeat(opt.n_samples, 1, 1)

        for ci, control_map in enumerate(control_maps or [None]):
            if use_controlnet and control_map is not None:
                control_out = os.path.join(sample_path, f"ctrl_{ci}_{opt.control_type}.png")
                control_map.save(control_out)

            for n in range(opt.n_iter):
                seed = opt.seed + n + ci * 10_000
                seed_everything(seed)
                torch.manual_seed(seed)
                if "cuda" in opt.device:
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)

                # Start from pure noise in latent space (x_T)
                latents = torch.randn((opt.n_samples, 4, opt.H // 8, opt.W // 8), device=opt.device, dtype=dtype)

                # Ensure reference latents repeated per batch
                ref_latents_batch = None
                if opt.color_align_enabled:
                    ref_latents_batch = ref_blurred_latents.repeat(opt.n_samples, 1, 1, 1).to(device=opt.device, dtype=dtype)

                # DDPM-like denoising with aligned step placement
                for step_idx, t in enumerate(tqdm(sampler.timesteps, desc=f"Denoising {folder}", leave=False)):
                    t_int = int(t)

                    # CFG: duplicate latents and embeddings
                    latent_model_input = torch.cat([latents, latents], dim=0)
                    text_emb_input = torch.cat([uncond_embeddings, combination_emb], dim=0)

                    # Predict eps
                    with torch.no_grad():
                        if use_controlnet and control_map is not None:
                            control_tensor = torch.from_numpy(np.array(control_map)).float() / 255.0
                            control_tensor = control_tensor.permute(2, 0, 1).unsqueeze(0).to(opt.device, dtype=dtype)

                            down_blocks, mid_block = controlnet(
                                latent_model_input,
                                torch.tensor([t_int], device=opt.device, dtype=torch.long),
                                encoder_hidden_states=text_emb_input,
                                controlnet_cond=control_tensor,
                                return_dict=False,
                            )

                            down_blocks = [s * opt.controlnet_scale for s in down_blocks]
                            mid_block = mid_block * opt.controlnet_scale

                            noise_pred = unet(
                                latent_model_input,
                                torch.tensor([t_int], device=opt.device, dtype=torch.long),
                                encoder_hidden_states=text_emb_input,
                                down_block_additional_residuals=down_blocks,
                                mid_block_additional_residual=mid_block,
                            ).sample
                        else:
                            noise_pred = unet(
                                latent_model_input,
                                torch.tensor([t_int], device=opt.device, dtype=torch.long),
                                encoder_hidden_states=text_emb_input,
                            ).sample

                    # CFG combine
                    noise_uncond, noise_cond = noise_pred.chunk(2)
                    eps = noise_uncond + opt.guidance_scale * (noise_cond - noise_uncond)

                    # Aligned sampler step (projection lives here)
                    if opt.color_align_enabled:
                        latents = sampler.step_backward(latents, eps, t_int, ref_latents_batch)
                    else:
                        latents = sampler.step_backward(latents, eps, t_int, latents.detach())  # dummy ref, unused

                    # Save intermediate if requested
                    if opt.save_intermediate and (step_idx % 10 == 0 or step_idx == len(sampler.timesteps) - 1):
                        img = decode_to_image(latents)
                        img.save(os.path.join(opt.outdir, "steps", f"{folder}_iter{n}_t{t_int:04d}_s{step_idx:03d}.png"))

                # Final decode and save
                final_img = decode_to_image(latents)
                final_img.save(os.path.join(sample_path, f"{base_count:05d}_ctrl{ci}.png"))
                base_count += 1

                gc.collect()
                if "cuda" in opt.device:
                    torch.cuda.empty_cache()

    print(f"\n✓ Generation complete! Images saved to: {opt.outdir}")


if __name__ == "__main__":
    main()

# conda activate ITIGen_ControlNet_ColourDebias

# python ITIGen-ControlNet-ChamferDebiasing_AlignedLikeZeroShot.py ^
#   --outdir "./outputs/zeroshot_pred" ^
#   --color-align-enabled ^
#   --color-reference-image "./pastel.jpeg" ^
#   --with-pred-sample-projection ^
#   --projection-threshold 200 ^
#   --projection-lr 0.3 ^
#   --blur-factor 3 ^
#   --attr-list "Male,MSTESkin_tone" ^
#   --filters Male_Negative MSTESkin_tone_10 ^
#   --prompt "a headshot of a person" ^
#   --generate-image-prompt "A picture of a doctor" ^
#   --num_inference_steps 50 ^
#   --guidance_scale 7.5 ^
#   --n_iter 3 ^
#   --save-intermediate


# python ITIGen-ControlNet-ChamferDebiasing_AlignedLikeZeroShot.py ^
#   --outdir "./outputs/zeroshot_pred" ^
#   --control-type "pose" ^
#   --control-image "./basic_pose.jpg" ^
#   --controlnet-scale 1.0 ^
#   --color-align-enabled ^
#   --color-reference-image "./pastel.jpeg" ^
#   --with-pred-sample-projection ^
#   --projection-threshold 200 ^
#   --projection-lr 0.3 ^
#   --blur-factor 3 ^
#   --attr-list "Male,MSTESkin_tone" ^
#   --filters Male_Negative MSTESkin_tone_10 ^
#   --prompt "a headshot of a person" ^
#   --generate-image-prompt "A picture of a doctor" ^
#   --num_inference_steps 50 ^
#   --guidance_scale 7.5 ^
#   --n_iter 3 ^
#   --save-intermediate


# python ITIGen-ControlNet-ChamferDebiasing_AlignedLikeZeroShot.py ^
#   --outdir "./outputs/zeroshot_pred" ^
#   --control-type "pose" ^
#   --control-image "./basic_pose.jpg" ^
#   --controlnet-scale 1.0 ^
#   --color-align-enabled ^
#   --color-reference-image "./pastel.jpeg" ^
#   --with-pred-sample-projection ^
#   --projection-threshold 200 ^
#   --projection-lr 0.3 ^
#   --blur-factor 3 ^
#   --attr-list "Male,MSTESkin_tone" ^
#   --filters Male_Negative MSTESkin_tone_10 ^
#   --prompt "a headshot of a person" ^
#   --generate-image-prompt "A picture of a doctor" ^
#   --num_inference_steps 50 ^
#   --guidance_scale 7.5 ^
#   --n_iter 3 ^
#   --save-intermediate

# python ITIGen-ControlNet-ChamferDebiasing_AlignedLikeZeroShot.py ^
#   --outdir "./outputs/zeroshot_pred" ^
#   --control-type "pose" ^
#   --control-image "./basic_pose.jpg" ^
#   --controlnet-scale 1.0 ^
#   --color-align-enabled ^
#   --color-reference-image "./pastel.jpeg" ^
#   --with-pred-sample-projection ^
#   --projection-threshold 200 ^
#   --projection-lr 0.3 ^
#   --blur-factor 3 ^
#   --attr-list "CCv2_Gender,CCv2_MSTE_SkinTone" ^
#   --filters Female MSTESkin_tone_10 ^
#   --prompt "a headshot of a person" ^
#   --generate-image-prompt "A picture of a doctor" ^
#   --num_inference_steps 50 ^
#   --guidance_scale 7.5 ^
#   --n_iter 3 ^
#   --save-intermediate

# python ITIGen-ControlNet-ChamferDebiasing_AlignedLikeZeroShot.py ^
#   --outdir "./outputs/test" ^
#   --control-type "pose" ^
#   --control-image "./basic_pose.jpg" ^
#   --controlnet-scale 1.0 ^
#   --color-align-enabled ^
#   --color-reference-image "./pastel.jpeg" ^
#   --with-pred-sample-projection ^
#   --projection-threshold 200 ^
#   --projection-lr 0.3 ^
#   --blur-factor 3 ^
#   --attr-list "Male,MSTESkin_tone" ^
#   --filters MSTESkin_tone_2 ^
#   --prompt "a headshot of a person" ^
#   --generate-image-prompt "A picture of a doctor in a hospital" ^
#   --num_inference_steps 50 ^
#   --guidance_scale 7.5 ^
#   --n_iter 3 ^
#   --save-intermediate

# python ITIGen-ControlNet-ChamferDebiasing_AlignedLikeZeroShot.py ^
#   --outdir "./outputs/test" ^
#   --use-lora ^
#   --control-type "pose" ^
#   --control-image "./basic_pose.jpg" ^
#   --controlnet-scale 1.0 ^
#   --color-align-enabled ^
#   --color-reference-image "./pastel.jpeg" ^
#   --with-pred-sample-projection ^
#   --projection-threshold 200 ^
#   --projection-lr 0.3 ^
#   --blur-factor 3 ^
#   --attr-list "Male,MSTESkin_tone" ^
#   --filters MSTESkin_tone_2 ^
#   --prompt "a headshot of a person" ^
#   --generate-image-prompt "A picture of a doctor in a hospital" ^
#   --num_inference_steps 8 ^
#   --guidance_scale 1 ^
#   --n_iter 3 ^
#   --save-intermediate


# python ITIGen-ControlNet-ChamferDebiasing_AlignedLikeZeroShot.py ^
#   --outdir "./outputs/Chamfer_CCv2_ITIGEN" ^
#   --n_iter 3 ^
#   --seed 42 ^
#   --guidance_scale 7 ^
#   --num_inference_steps 50 ^
#   --attr-list "CCv2_Gender,CCv2_MSTE_SkinTone" ^
#   --prompt "an image of a person" ^
#   --generate-image-prompt "A picture of a doctor" ^
#   --save-intermediate ^
#   --load-model-epoch 19

# python ITIGen-ControlNet-ChamferDebiasing_AlignedLikeZeroShot.py ^
#   --outdir "./outputs/Chamfer_CCv2_ITIGEN_POSE" ^
#   --n_iter 3 ^
#   --seed 42 ^
#   --guidance_scale 7 ^
#   --num_inference_steps 50 ^
#   --attr-list "CCv2_Gender,CCv2_MSTE_SkinTone" ^
#   --prompt "an image of a person" ^
#   --generate-image-prompt "A picture of a doctor" ^
#   --save-intermediate ^
#   --load-model-epoch 19 ^
#   --control-type pose ^
#   --control-image "./basic_pose.jpg" ^
#   --controlnet-scale 0.8

# python ITIGen-ControlNet-ChamferDebiasing_AlignedLikeZeroShot.py ^
#   --outdir "./outputs/Chamfer_CCv2_ITIGEN_COLOUR" ^
#   --n_iter 3 ^
#   --seed 42 ^
#   --guidance_scale 7 ^
#   --num_inference_steps 50 ^
#   --attr-list "CCv2_Gender,CCv2_MSTE_SkinTone" ^
#   --prompt "an image of a person" ^
#   --generate-image-prompt "A picture of a doctor" ^
#   --save-intermediate ^
#   --color-align-enabled ^
#   --color-reference-image "./pastel.jpeg" ^
#   --with-pred-sample-projection ^
#   --projection-threshold 200 ^
#   --projection-lr 0.3 ^
#   --blur-factor 3
