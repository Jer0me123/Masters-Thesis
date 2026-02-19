import argparse
import logging
import math
import os
import sys
import random
import time
from typing import Union
from datetime import timedelta

import numpy as np
import accelerate
import datasets

import torch
import torch.nn.functional as F
import torchvision
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset, DownloadConfig
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from torchvision.transforms.functional import to_pil_image

import diffusers
from diffusers import UNet2DModel, UNet2DConditionModel, AutoencoderKL
from diffusers.training_utils import cast_training_params
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from transformers import CLIPTextModel, AutoTokenizer

# from pytorch3d.loss import chamfer_distance
# from pytorch3d.ops.knn import knn_gather, knn_points
# from pytorch3d.structures.pointclouds import Pointclouds

try:
    from pytorch3d.loss import chamfer_distance
    from pytorch3d.ops.knn import knn_gather, knn_points
    from pytorch3d.structures.pointclouds import Pointclouds
    PYTORCH3D_AVAILABLE = True
except ImportError:
    chamfer_distance = None
    knn_gather = None
    knn_points = None
    Pointclouds = None
    PYTORCH3D_AVAILABLE = False



negative_prompt = "Low quality,Bad quality,Sketches,Logo,Watermark,Text,Ugly,Morbid,Extra fingers,Poorly drawn hands,Mutation,Blurry,Extra limbs,Gross proportions,Missing arms,Mutated hands,Long neck,Duplicate,Mutilated,Mutilated hands,Poorly drawn face,Deformed,Bad anatomy,Cloned face,Malformed limbs,Missing legs,Too many fingers"
guidance_scale = 5.
do_classifier_free_guidance = True

np.random.seed(10)
torch.manual_seed(10)

test_prompts = ["firefighter", "police officer"]
#     'chair', 'dog', 'car', 'book', 'table', 'house', 'cat',
#     'pen', 'shirt', 'bicycle', 'shoe', 'cup', 'bed', 'clock', 'door',
#     'flower', 'fish', 'camera', 'blanket', 'guitar',
#     'bag', 'bottle', 'lamp', 'desk', 'towel',
#     'suitcase', 'basket', 'helmet', 'skateboard',
#     'umbrella',
#     'soap', 'shampoo', 'ladder', 'painting', 'brush', 'glove', 'hat',
#     'belt', 'wallet', 'ring', 'vase', 'statue', 'map',
#     'ticket', 'kite',
#     'bus', 'airplane', 'rocket', 'boat',
#     'crystal'
# ]

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")
logger = get_logger(__name__, log_level="INFO")

def _validate_chamfer_reduction_inputs(
    batch_reduction: Union[str, None], point_reduction: Union[str, None]
) -> None:
    """Check the requested reductions are valid.

    Args:
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"] or None.
    """
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction is not None and point_reduction not in ["mean", "sum", "max"]:
        raise ValueError(
            'point_reduction must be one of ["mean", "sum", "max"] or None'
        )
    if point_reduction is None and batch_reduction is not None:
        raise ValueError("Batch reduction must be None if point_reduction is None")

def _handle_pointcloud_input(
    points: Union[torch.Tensor, Pointclouds],
    lengths: Union[torch.Tensor, None],
    normals: Union[torch.Tensor, None],
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded normals.
    Otherwise, return the input points (and normals) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    # if isinstance(points, Pointclouds):
    if PYTORCH3D_AVAILABLE and isinstance(points, Pointclouds):
        X = points.points_padded()
        lengths = points.num_points_per_cloud()
        normals = points.normals_padded()  # either a tensor or None
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
            lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
            )
        if normals is not None and normals.ndim != 3:
            raise ValueError("Expected normals to be of shape (N, P, 3")
    else:
        raise ValueError(
            "The input pointclouds should be either "
            + "Pointclouds objects or torch.Tensor of shape "
            + "(minibatch, num_points, 3)."
        )
    return X, lengths, normals

def _chamfer_distance_single_direction(
    x,
    y,
    x_lengths,
    y_lengths,
    x_normals,
    y_normals,
    weights,
    point_reduction: Union[str, None],
    norm: int,
    abs_cosine: bool,
):
    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
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
    cham_x = x_nn.dists[..., 0]  # (N, P1)
    idx_x = x_nn.idx[..., 0]  # (N, P1)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]

        cosine_sim = F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        # If abs_cosine, ignore orientation and take the absolute value of the cosine sim.
        cham_norm_x = 1 - (torch.abs(cosine_sim) if abs_cosine else cosine_sim)

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0

        if weights is not None:
            cham_norm_x *= weights.view(N, 1)

    if point_reduction == "max":
        assert not return_normals
        cham_x = cham_x.max(1).values  # (N,)
    elif point_reduction is not None:
        # Apply point reduction
        cham_x = cham_x.sum(1)  # (N,)
        if return_normals:
            cham_norm_x = cham_norm_x.sum(1)  # (N,)
        if point_reduction == "mean":
            x_lengths_clamped = x_lengths.clamp(min=1)
            cham_x /= x_lengths_clamped
            if return_normals:
                cham_norm_x /= x_lengths_clamped

    cham_dist = cham_x
    cham_normals = cham_norm_x if return_normals else None
    return cham_dist, cham_normals, idx_x

def _apply_batch_reduction(
    cham_x, cham_norm_x, weights, batch_reduction: Union[str, None]
):
    if batch_reduction is None:
        return (cham_x, cham_norm_x)
    # batch_reduction == "sum"
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
    """
    Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum", "max"] or None. Using "max" leads to the
            Hausdorff distance.
        norm: int indicates the norm used for the distance. Supports 1 for L1 and 2 for L2.
        single_directional: If False (default), loss comes from both the distance between
            each point in x and its nearest neighbor in y and each point in y and its nearest
            neighbor in x. If True, loss is the distance between each point in x and its
            nearest neighbor in y.
        abs_cosine: If False, loss_normals is from one minus the cosine similarity.
            If True (default), loss_normals is from one minus the absolute value of the
            cosine similarity, which means that exactly opposite normals are considered
            equivalent to exactly matching normals, i.e. sign does not matter.

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y. If point_reduction is None, a 2-element
          tuple of Tensors containing forward and backward loss terms shaped (N, P1)
          and (N, P2) (if single_directional is False) or a Tensor containing loss
          terms shaped (N, P1) (if single_directional is True) is returned.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None. If point_reduction is None, a 2-element
          tuple of Tensors containing forward and backward loss terms shaped (N, P1)
          and (N, P2) (if single_directional is False) or a Tensor containing loss
          terms shaped (N, P1) (if single_directional is True) is returned.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")

    if point_reduction == "max" and (x_normals is not None or y_normals is not None):
        raise ValueError('Normals must be None if point_reduction is "max"')

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    cham_x, cham_norm_x, idx_x = _chamfer_distance_single_direction(
        x,
        y,
        x_lengths,
        y_lengths,
        x_normals,
        y_normals,
        weights,
        point_reduction,
        norm,
        abs_cosine,
    )
    if single_directional:
        loss = cham_x
        loss_normals = cham_norm_x
    else:
        cham_y, cham_norm_y, _ = _chamfer_distance_single_direction(
            y,
            x,
            y_lengths,
            x_lengths,
            y_normals,
            x_normals,
            weights,
            point_reduction,
            norm,
            abs_cosine,
        )
        if point_reduction == "max":
            loss = torch.maximum(cham_x, cham_y)
            loss_normals = None
        elif point_reduction is not None:
            loss = cham_x + cham_y
            if cham_norm_x is not None:
                loss_normals = cham_norm_x + cham_norm_y
            else:
                loss_normals = None
        else:
            loss = (cham_x, cham_y)
            if cham_norm_x is not None:
                loss_normals = (cham_norm_x, cham_norm_y)
            else:
                loss_normals = None
    return _apply_batch_reduction(loss, loss_normals, weights, batch_reduction), idx_x

class AlignedDDPMScheduler():
    def __init__(
            self,
            num_train_timesteps: int = 1000,
            num_inference_timesteps: int = 1000,
            beta_start: float = 0.0001,
            beta_end: float = 0.004,
            beta_schedule: str = "scaled_linear",
            length: int = 64,
            cur_bsz: int = 16,
            prediction_type: str = "noise",
            projection_threshold: int = 0,
            with_noisy_sample_projection: bool = False,
            with_pred_sample_projection: bool = False,
            with_shuffling: bool = False,
            train_on_rgb: bool = False,
    ):
        self.weight_dtype = torch.float32

        # Schedule
        self.num_train_steps = num_train_timesteps
        self.num_inference_steps = num_inference_timesteps

        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=self.weight_dtype)
            self.betas_cumsum = torch.cumsum(self.betas, dim=0)
        elif beta_schedule == "scaled_linear":
            self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps,
                                        dtype=self.weight_dtype) ** 2
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        # Input/Output
        self.prediction_type = prediction_type
        self.cur_bsz = cur_bsz

        self.raw_length = length
        self.raw_latent_length = int(length / 8) if (not train_on_rgb) else length

        self.train_on_rgb = train_on_rgb
        self.latent_dim = 4 if (not train_on_rgb) else 3

        # Alignment related
        self.with_shuffling = with_shuffling
        self.with_noisy_sample_projection = with_noisy_sample_projection
        self.with_pred_sample_projection = with_pred_sample_projection
        self.projection_threshold = projection_threshold

    # to construct color condition (c in the paper)
    def shuffle_latents(
            self,
            latents_original: torch.FloatTensor,
    ) -> torch.FloatTensor:
        self.cur_bsz = latents_original.shape[0]

        latents_shuffled = torch.zeros_like(latents_original).to(device=latents_original.device)
        for bsi in range(self.cur_bsz):
            div_factor_x = 1
            div_factor_y = 1
            shuffle_idx_sub = torch.randperm(int(self.raw_latent_length * self.raw_latent_length / div_factor_x / div_factor_y))
            shuffle_idx_sub = shuffle_idx_sub.reshape(int(self.raw_latent_length/div_factor_x),
                                                      int(self.raw_latent_length/div_factor_y))

            shuffle_idx = torch.zeros(self.raw_latent_length, self.raw_latent_length)
            for sub_i in range(div_factor_x):
                for sub_j in range(div_factor_y):
                    shuffle_idx[sub_i::div_factor_x, sub_j::div_factor_y] \
                        = (shuffle_idx_sub * div_factor_x) \
                        + (((shuffle_idx_sub * div_factor_x) // self.raw_latent_length) * self.raw_latent_length * div_factor_y) \
                        - (((shuffle_idx_sub * div_factor_x) // self.raw_latent_length) * self.raw_latent_length) \
                        + sub_j \
                        + (sub_i * self.raw_latent_length)
            shuffle_idx = shuffle_idx.reshape(self.raw_latent_length * self.raw_latent_length).to(dtype=torch.int64)

            latents_shuffled[bsi] = latents_original[bsi].reshape(self.latent_dim, -1)[:, shuffle_idx].reshape(self.latent_dim, self.raw_latent_length, self.raw_latent_length)

        latents_shuffled = latents_shuffled.to(device=latents_original.device)

        return {
            'latents_shuffled': latents_shuffled
        }

    def shuffle_img(
            self,
            img_original: torch.FloatTensor,
            seed_count: int,
    ) -> torch.FloatTensor:
        self.cur_bsz = img_original.shape[0]

        img_shuffled = torch.zeros_like(img_original).to(device=img_original.device)
        for bsi in range(self.cur_bsz):
            if self.with_noisy_sample_projection or self.with_pred_sample_projection:
                div_factor_x = 64
                div_factor_y = 64
            else:
                div_factor_x = 1
                div_factor_y = 1

            torch.manual_seed(seed_count + bsi * 10000)
            shuffle_idx_sub = torch.randperm(int(self.raw_length * self.raw_length / div_factor_x / div_factor_y))
            shuffle_idx_sub = shuffle_idx_sub.reshape(int(self.raw_length/div_factor_x),
                                                      int(self.raw_length/div_factor_y))

            shuffle_idx = torch.zeros(self.raw_length, self.raw_length)
            for sub_i in range(div_factor_x):
                for sub_j in range(div_factor_y):
                    shuffle_idx[sub_i::div_factor_x, sub_j::div_factor_y] \
                        = (shuffle_idx_sub * div_factor_x) \
                        + (((shuffle_idx_sub * div_factor_x) // self.raw_length) * self.raw_length * div_factor_y) \
                        - (((shuffle_idx_sub * div_factor_x) // self.raw_length) * self.raw_length) \
                        + sub_j \
                        + (sub_i * self.raw_length)
            shuffle_idx = shuffle_idx.reshape(self.raw_length * self.raw_length).to(dtype=torch.int64)

            img_shuffled[bsi] = img_original[bsi].reshape(3, -1)[:, shuffle_idx].reshape(3, self.raw_length, self.raw_length)

        img_shuffled = img_shuffled.to(device=img_original.device)

        return {
            'img_shuffled': img_shuffled
        }

    def cross_forward(
            self,
            img_original: torch.FloatTensor,
            img_original_blurred: torch.FloatTensor,
            timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # --- cross forward process of diffusion --- #
        self.cur_bsz = img_original.shape[0]

        # prepare diffusion matters
        self.alphas_cumprod = self.alphas_cumprod.to(device=img_original.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=img_original.dtype)
        timesteps = timesteps.to(img_original.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = torch.where(timesteps != self.num_train_steps - 1, sqrt_alpha_prod, 0.)
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(img_original.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = torch.where(timesteps != self.num_train_steps - 1, sqrt_one_minus_alpha_prod, 1.)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(img_original.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        if torch.any(timesteps < 0):
            sqrt_alpha_prod = sqrt_alpha_prod * 0. + 1.
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod * 0.

        # sample noise and do diffusion
        rand_motion = torch.randn(self.cur_bsz, self.latent_dim, self.raw_latent_length, self.raw_latent_length).to(
                device=img_original.device, dtype=img_original.dtype)

        coor_original = img_original.clone().detach().to(device=img_original.device)
        coor_next = sqrt_alpha_prod * coor_original + sqrt_one_minus_alpha_prod * rand_motion

        # color alignment
        if self.with_noisy_sample_projection:
            coor_next_temp = coor_next.clone().detach().requires_grad_(True)

            next_points = torch.swapaxes(coor_next_temp.reshape(self.cur_bsz, self.latent_dim, -1), 1, 2)
            ref_points = torch.swapaxes(img_original_blurred.reshape(self.cur_bsz, self.latent_dim, -1), 1, 2).clone().detach().requires_grad_(True)

            dist, _ = chamfer_distance(next_points, ref_points, batch_reduction="sum", point_reduction="sum", single_directional=True)
            dist.backward()

            img_next = (coor_next_temp - 0.5 * coor_next_temp.grad).clone().detach()

            timesteps_to_project_bool = torch.where(timesteps > self.projection_threshold, 1., 0.)
            while len(timesteps_to_project_bool.shape) < len(img_original.shape):
                timesteps_to_project_bool = timesteps_to_project_bool.unsqueeze(-1)

            img_next = timesteps_to_project_bool * img_next + (1. - timesteps_to_project_bool) * coor_next
        else:
            img_next = coor_next

        if self.prediction_type == "noise":
            target_output = rand_motion
        elif self.prediction_type == "sample":
            target_output = coor_original
        elif self.prediction_type == "adapted_noise":
            target_output = (img_next - sqrt_alpha_prod * coor_original) / sqrt_one_minus_alpha_prod
        else:
            raise NotImplementedError

        # return everything as dict
        return {
            'target_output': target_output,
            'coor_original': coor_original,
            'coor_next': coor_next,
            'img_next': img_next,
        }

    def step_backward(
            self,
            img_original: torch.FloatTensor,
            coor_original: torch.FloatTensor,
            predicted_output: torch.FloatTensor,
            timestep: int,
            img_ref: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # --- step backward process of diffusion --- #
        self.cur_bsz = img_original.shape[0]

        # prepare diffusion matters
        t = timestep
        prev_t = self.previous_timestep(t)

        self.alphas_cumprod = self.alphas_cumprod.to(device=img_original.device)
        self.one = self.one.to(device=img_original.device)

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # ddpm
        variance = 0
        if t > 0:
            variance_noise = torch.randn(self.cur_bsz, self.latent_dim, self.raw_latent_length, self.raw_latent_length).to(
                device=img_original.device, dtype=img_original.dtype)
            variance = ((self._get_variance(t) ** 0.5) * variance_noise).to(device=img_original.device)

        if self.prediction_type == "noise":
            pred_original_sample = (coor_original - beta_prod_t ** (0.5) * predicted_output.to(
                device=img_original.device)) / (alpha_prod_t ** (0.5))
        elif self.prediction_type == "adapted_noise":
            pred_original_sample = (img_original - beta_prod_t ** (0.5) * predicted_output.to(
                device=img_original.device)) / (alpha_prod_t ** (0.5))
        else:
            raise NotImplementedError

        pred_original_sample_raw = pred_original_sample.clone().detach().to(device=img_original.device)

        # color alignment (for zero-shot approximation)
        if self.with_pred_sample_projection and t > self.projection_threshold:
            pred_original_sample = pred_original_sample.clone().detach().requires_grad_(True)
            pred_original_sample_shuffled = self.shuffle_latents(pred_original_sample)['latents_shuffled']

            pred_points = torch.swapaxes(pred_original_sample_shuffled.reshape(self.cur_bsz, self.latent_dim, -1), 1, 2)#.clone().detach().requires_grad_(True)

            ref_points = torch.swapaxes(img_ref.reshape(self.cur_bsz, self.latent_dim, -1), 1, 2).clone().detach().requires_grad_(True)
            ref_points = ref_points + 1e-6 * (torch.rand(*(ref_points.shape)) - 1.).to(device=img_original.device)

            dist_total = 0.
            for pred_point, ref_point in zip(pred_points, ref_points):
                cur_pred_point = pred_point[None, ...]  # (P, D)
                cur_ref_point = ref_point[None, ...]  # (P, D)

                while True:
                    cur_ref_point = cur_ref_point.clone().detach().requires_grad_(True)
                    (dist, _), mapping = my_chamfer_distance(cur_pred_point, cur_ref_point,
                                                             batch_reduction=None, point_reduction=None,
                                                             single_directional=True)
                    dist = dist[0]
                    mapping = mapping[0]

                    unique, idx, counts = torch.unique(mapping, dim=0, sorted=True, return_inverse=True, return_counts=True)
                    _, ind_sorted = torch.sort(idx, stable=True)
                    cum_sum = counts.cumsum(0)
                    cum_sum = torch.cat((torch.tensor([0]).to(device=img_original.device), cum_sum[:-1]))
                    first_indicies, _ = torch.sort(ind_sorted[cum_sum])
                    first_indicies_opponet = mapping[first_indicies]

                    dist_component = torch.sum(dist[first_indicies])
                    dist_total = dist_total + dist_component

                    world = torch.ones_like(mapping)
                    world[first_indicies] = 0
                    compl_pred = torch.nonzero(world).squeeze()
                    if compl_pred.nelement() == 0:
                        break

                    world = torch.ones_like(mapping)
                    world[first_indicies_opponet] = 0
                    compl_ref = torch.nonzero(world).squeeze()

                    cur_pred_point = cur_pred_point[:, compl_pred, :]
                    if len(cur_pred_point.shape) == 2:
                        cur_pred_point = cur_pred_point[:, None, :]

                    cur_ref_point = cur_ref_point[:, compl_ref, :]
                    if len(cur_ref_point.shape) == 2:
                        cur_ref_point = cur_ref_point[:, None, :]

            dist_total.backward()

            pred_original_sample = (pred_original_sample - 0.5 * pred_original_sample.grad).clone().detach().requires_grad_(False)

        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * coor_original
        coor_next = pred_prev_sample + (variance if t > 0 else 0)

        # color alignment (for fine-tuned model)
        if self.with_noisy_sample_projection and t > self.projection_threshold:
            coor_next_temp = coor_next.clone().detach().requires_grad_(True)

            next_points = torch.swapaxes(coor_next_temp.reshape(self.cur_bsz, self.latent_dim, -1), 1, 2)
            ref_points = torch.swapaxes(img_ref.reshape(self.cur_bsz, self.latent_dim, -1), 1, 2).clone().detach().requires_grad_(True)

            dist, _ = chamfer_distance(next_points, ref_points, batch_reduction="sum", point_reduction="sum", single_directional=True)
            dist.backward()

            img_next = (coor_next_temp - 0.5 * coor_next_temp.grad).clone().detach()
        else:
            img_next = coor_next

        # return everything as dict
        return {
            'img_next': img_next,
            'coor_next': coor_next,
            'pred_original_sample': pred_original_sample,
            'pred_original_sample_raw': pred_original_sample_raw,
        }

    def _get_variance(self, t):
        prev_t = self.previous_timestep(t)

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # we always take the log of variance, so clamp it to ensure it's not 0
        variance = torch.clamp(variance, min=1e-20)

        return variance

    def previous_timestep(self, timestep):
        num_inference_steps = self.num_inference_steps
        prev_t = timestep - self.num_train_steps // num_inference_steps

        return prev_t


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that HF Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="The config of the UNet model to train, leave as None to use standard DDPM configuration.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="bdpm-model-64",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=4, help="The number of images to generate for evaluation."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
    )
    parser.add_argument(
        "--logger",
        type=str,
        default=None, #"wandb",
        choices=["wandb"],
        help=(
            "Use [wandb](https://www.wandb.ai)"
            " for experiment tracking and logging of model metrics and model checkpoints"
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "Log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        choices=["fp16"],
        help=(
            "Whether to load mixed precision models."
        ),
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="noise",
        choices=["noise", "adapted_noise"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=20)
    parser.add_argument("--bdpm_beta_schedule", type=str, default="linear")
    parser.add_argument("--beta_start", type=float, default=0.0001)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' "latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--train_split_only", action="store_true", help="True if the dataset have train set only."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default='UNet2DModel',
        choices=['UNet2DModel', 'UNet2DConditionModel'],
        help=(
            "the architecture type of neural network to use"
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts", type=float, default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement)."
    )
    parser.add_argument(
        "--with_noisy_sample_projection", action="store_true", help="Do our training-time projection."
    )
    parser.add_argument(
        "--with_pred_sample_projection", action="store_true", help="Do our zero-shot projection."
    )
    parser.add_argument(
        "--with_shuffling", action="store_true", help="Do our shuffling hint or not."
    )
    parser.add_argument(
        "--projection_threshold", type=float, default=1000, help="Train everything from scratch."
    )
    parser.add_argument(
        "--train_on_rgb", action="store_true", help="Train everything on rgb space."
    )
    parser.add_argument(
        "--test_on_manually_drawn_samples", action="store_true", help="Use manually drawn color samples for testing."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None:
        raise ValueError("You must specify a dataset name from the hub.")

    return args


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


def main(args):
    import torch
    print(torch.__version__)
    print(torch.version.cuda)


    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    if args.model_config_name_or_path is None:
        if args.with_shuffling:
            model = UNet2DConditionModel.from_pretrained(
                "stable-diffusion-v1-5/stable-diffusion-v1-5",
                subfolder="unet",
                in_channels=8,
                ignore_mismatched_sizes=True,
                low_cpu_mem_usage=False,
                use_safetensors=True,
            )
        else:
            model = UNet2DConditionModel.from_pretrained(
                "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="unet", variant=args.variant
            )
        tokenizer = AutoTokenizer.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="tokenizer", use_fast=False,
        )
        text_encoder = CLIPTextModel.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="text_encoder", variant=args.variant
        )
        vae = AutoencoderKL.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="vae", variant=args.variant
        )
    else:
        config = str_to_class(args.model_type).load_config(args.model_config_name_or_path)
        model = str_to_class(args.model_type).from_config(config)

    weight_dtype = torch.float32

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        print("------------> IN HERE 1 <------------")
        # _, 
        test_dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="train", #['train[:90%]', 'train[90%:]'],
            trust_remote_code=True,
            download_config=DownloadConfig(cache_dir=args.cache_dir + "/downloads", resume_download=True)
        )
    else:
        raise Exception('Unimplemented dataset setting.')

    print("--- till load_dataset: %s seconds ---" % (time.time() - start_time))

    # Preprocessing the datasets and DataLoaders creation.
    def tokenize_texts(batch_input_prompt, proportion_empty_prompts):
        captions = []
        for caption in batch_input_prompt:
            if random.random() < proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            else:
                raise NotImplementedError
        tokenized_texts = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        )
        return tokenized_texts.input_ids

    augmentations = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform_images(examples):
        print(f"Transforming images... {examples.keys()}")
        if "image" in examples:
            images = [augmentations(image.convert("RGB")) for image in examples["image"]]
        elif "jpg" in examples:
            images = [augmentations(image.convert("RGB")) for image in examples["jpg"]]
        else:
            raise NotImplementedError

        if args.model_type == 'UNet2DConditionModel':
            if "text" in examples:
                texts = examples["text"]
            elif "TEXT" in examples:
                texts = examples["TEXT"]
            elif "prompt" in examples:
                texts = examples["prompt"]
            elif "caption" in examples:
                texts = examples["caption"]
            elif "json" in examples:
                texts = [json["prompt"] for json in examples["json"]]
            else:
                raise NotImplementedError

            return {"input": images, "input_prompt": texts}
        else:
            return {"input": images}

    print(f"-------------> Dataset size: {len(test_dataset)}")

    test_dataset.set_transform(transform_images)
    # test_dataloader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=args.eval_batch_size, num_workers=args.dataloader_num_workers, shuffle=False
    # )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, num_workers=args.dataloader_num_workers, shuffle=False
    )

    model.to(device)
    if vae:
        vae.to(device, dtype=weight_dtype)
    if text_encoder:
        text_encoder.to(device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(test_dataloader))

    # Test!
    NUMSTEPS = args.ddpm_num_steps

    noise_scheduler = AlignedDDPMScheduler(num_train_timesteps=args.ddpm_num_steps,
                                           num_inference_timesteps=args.ddpm_num_inference_steps,
                                           length=args.resolution, cur_bsz=args.eval_batch_size,
                                           beta_start=args.beta_start, beta_end=args.beta_end,
                                           beta_schedule=args.bdpm_beta_schedule,
                                           prediction_type=args.prediction_type,
                                           with_noisy_sample_projection=args.with_noisy_sample_projection,
                                           with_pred_sample_projection=args.with_pred_sample_projection,
                                           projection_threshold=args.projection_threshold,
                                           with_shuffling=args.with_shuffling,
                                           train_on_rgb=args.train_on_rgb)

    cur_test_samples_nums = 0

    global_step = 0

    if text_encoder:
        text_encoder.requires_grad_(False)
    if vae:
        vae.requires_grad_(False)
    model.requires_grad_(False)
    model.eval()

    with torch.no_grad():
        rgb_list = []
        for r in range(5):
            for g in range(5):
                for b in range(5):
                    rgb_list.append([int(r * 63.75), int(g * 63.75), int(b * 63.75)])

    for prompt in test_prompts:
        np.random.seed(10)
        torch.manual_seed(10)
        print(f"Processing Prompt: {prompt}")
        
        for step, batch_t in enumerate(test_dataloader):
            # Checks if the accelerator has performed an optimization step behind the scenes
            # progress_bar.update(1)
            global_step += 1

            with torch.no_grad():
                clean_images_t = batch_t["input"].to(device)
                blur_factor = 3
                clean_images_t_blurred = torchvision.transforms.functional.resize(clean_images_t, int(args.resolution // blur_factor), interpolation=transforms.InterpolationMode.BILINEAR if blur_factor < 100 else transforms.InterpolationMode.NEAREST)
                clean_images_t_blurred = torchvision.transforms.functional.resize(clean_images_t_blurred, args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)

                if vae:
                    clean_img_latents_t = vae.encode(clean_images_t).latent_dist.sample().to(device=clean_images_t.device)
                    clean_img_latents_t = clean_img_latents_t * vae.config.scaling_factor

                    clean_img_t_blurred_latents = vae.encode(clean_images_t_blurred).latent_dist.sample().to(device=clean_images_t.device)
                    clean_img_t_blurred_latents = clean_img_t_blurred_latents * vae.config.scaling_factor
                else:
                    clean_img_latents_t = clean_images_t.clone().detach().to(device=clean_images_t.device)
                    clean_img_t_blurred_latents = clean_images_t_blurred.clone().detach().to(device=clean_images_t.device)

                ##################################
                # check all-steps sampling
                ##################################
                bsz_t = clean_images_t.shape[0]
                clean_img_t_blurred_latents_shuffled = noise_scheduler.shuffle_latents(clean_img_t_blurred_latents)['latents_shuffled']

            # diffused_output_t = noise_scheduler.cross_forward(clean_img_latents_t,
            #                                                 clean_img_t_blurred_latents,
            #                                                 torch.full((bsz_t,), NUMSTEPS - 1,
            #                                                             device=clean_images_t.device).long())


            total_to_generate = 4#args.images_per_prompt
            gen_bs = 2#args.gen_batch_size

            for gen_start in range(0, total_to_generate, gen_bs):
               
                cur_bs = min(gen_bs, total_to_generate - gen_start)

                ref_blur_latents_shuffled = clean_img_t_blurred_latents_shuffled.repeat(
                    cur_bs, 1, 1, 1
                )

                # --------------------------------------------------
                # Repeat reference latents
                # --------------------------------------------------
                ref_latents = clean_img_latents_t.repeat(cur_bs, 1, 1, 1)
                ref_blur_latents = clean_img_t_blurred_latents.repeat(cur_bs, 1, 1, 1)

                diffused_output_t = noise_scheduler.cross_forward(ref_latents,
                                                ref_blur_latents,
                                                torch.full((cur_bs,), NUMSTEPS - 1,
                                                            device=clean_images_t.device).long())

                t = NUMSTEPS - 1

                while t > -1:
                    print('Doing step', t)

                    with torch.no_grad():
                        print("---------------> Prompt inside loop:", prompt)
                        encoder_hidden_states_t = text_encoder(
                            tokenize_texts([prompt] * cur_bs, args.proportion_empty_prompts).to(
                                device=clean_images_t.device), return_dict=False)[0]

                        if do_classifier_free_guidance:
                            encoder_hidden_states_uncond_t = text_encoder(
                                tokenize_texts([negative_prompt for _ in range(cur_bs)], 0).to(
                                    device=clean_images_t.device), return_dict=False)[0]
                            encoder_hidden_states_t = torch.cat([encoder_hidden_states_uncond_t, encoder_hidden_states_t])

                        # sample = torch.cat((diffused_output_t['img_next'],
                        #                                     clean_img_t_blurred_latents_shuffled
                        #                                     ), dim=1) if args.with_shuffling else diffused_output_t['img_next']

                        sample = torch.cat((diffused_output_t['img_next'],
                                    ref_blur_latents_shuffled
                                    ), dim=1) if args.with_shuffling else diffused_output_t['img_next']
                            
                        if do_classifier_free_guidance:
                            sample = torch.cat([sample] * 2)

                        model_output_t = model(sample=sample,
                            encoder_hidden_states=encoder_hidden_states_t,
                            timestep=t).sample

                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = model_output_t.chunk(2)
                            model_output_t = noise_pred_uncond + \
                                            guidance_scale \
                                            * (noise_pred_text - noise_pred_uncond)

                    # diffused_output_t = noise_scheduler.step_backward(
                    #     diffused_output_t['img_next'],
                    #     diffused_output_t['coor_next'],
                    #     model_output_t,
                    #     t,
                    #     clean_img_t_blurred_latents,
                    # )

                    diffused_output_t = noise_scheduler.step_backward(
                        diffused_output_t['img_next'],
                        diffused_output_t['coor_next'],
                        model_output_t,
                        t,
                        ref_blur_latents,
                    )

                    t = t - int(args.ddpm_num_steps // args.ddpm_num_inference_steps)

                    with torch.no_grad():
                        pred_img_STEP = \
                        vae.decode(diffused_output_t['img_next'].to(weight_dtype) / vae.config.scaling_factor,
                                return_dict=False)[0] if vae else diffused_output_t['img_next']

                        # -----------------------------
                        # SAVE IMAGE TO DISK
                        # -----------------------------
                        save_dir = os.path.join(args.output_dir, "images")
                        os.makedirs(save_dir, exist_ok=True)

                        for i in range(pred_img_STEP.shape[0]):
                            print('Saving image', cur_test_samples_nums + i)
                            img = torch.clamp(pred_img_STEP[i], -1, 1)
                            img = (img + 1) / 2  # [-1,1] → [0,1]

                            save_path = os.path.join(
                                save_dir,
                                # f"sample_{global_step:06d}_{i}.png"
                                f"sample_{global_step:06d}_{gen_start + i}.png"
                            )

                            torchvision.utils.save_image(img, save_path)

                        # cur_test_samples_nums = cur_test_samples_nums + bsz_t
                        cur_test_samples_nums += cur_bs

            # progress_bar.close()

if __name__ == "__main__":
    args = parse_args()
    main(args)


# conda activate py3d_conda  

# python zero-shot.py --dataset_name imagefolder --train_split_only --dataset_config_name local_color_dataset --cache_dir "D:\HuggingFace_Models" --resolution 512 --eval_batch_size 4 --resume_from_checkpoint latest --prediction_type adapted_noise --ddpm_num_steps=100 --ddpm_num_inference_steps=5 --bdpm_beta_schedule="scaled_linear" --beta_start=0.00085 --beta_end=0.0120 --model_type UNet2DConditionModel --proportion_empty_prompts 0 --projection_threshold=20 --with_pred_sample_projection --output_dir "WithProjection-Zeroshot-t2i_2m-512"

# Code Works Seed is a bit wonky try and fix it 