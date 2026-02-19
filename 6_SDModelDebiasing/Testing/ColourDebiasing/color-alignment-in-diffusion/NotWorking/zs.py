# import argparse
# import logging
# import os
# import sys
# import random
# import time
# from typing import Union

# import numpy as np
# import torch
# import torch.nn.functional as F
# import torchvision
# from torchvision import transforms
# from tqdm.auto import tqdm
# from torchvision.transforms.functional import to_pil_image

# from diffusers import UNet2DModel, UNet2DConditionModel, AutoencoderKL
# from diffusers.utils import check_min_version
# from transformers import CLIPTextModel, AutoTokenizer

# try:
#     from pytorch3d.loss import chamfer_distance
#     from pytorch3d.ops.knn import knn_gather, knn_points
#     from pytorch3d.structures.pointclouds import Pointclouds
#     PYTORCH3D_AVAILABLE = True
# except ImportError:
#     chamfer_distance = None
#     knn_gather = None
#     knn_points = None
#     Pointclouds = None
#     PYTORCH3D_AVAILABLE = False

# # Configuration
# negative_prompt = "" #Low quality,Bad quality,Sketches,Logo,Watermark,Text,Ugly,Morbid,Extra fingers,Poorly drawn hands,Mutation,Blurry,Extra limbs,Gross proportions,Missing arms,Mutated hands,Long neck,Duplicate,Mutilated,Mutilated hands,Poorly drawn face,Deformed,Bad anatomy,Cloned face,Malformed limbs,Missing legs,Too many fingers"
# guidance_scale = 5.0
# do_classifier_free_guidance = True

# np.random.seed(10)
# torch.manual_seed(10)

# test_prompts = ["firefighter"]

# check_min_version("0.28.0.dev0")
# logger = logging.getLogger(__name__)


# def _validate_chamfer_reduction_inputs(
#     batch_reduction: Union[str, None], point_reduction: Union[str, None]
# ) -> None:
#     """Check the requested reductions are valid."""
#     if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
#         raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
#     if point_reduction is not None and point_reduction not in ["mean", "sum", "max"]:
#         raise ValueError('point_reduction must be one of ["mean", "sum", "max"] or None')
#     if point_reduction is None and batch_reduction is not None:
#         raise ValueError("Batch reduction must be None if point_reduction is None")


# def _handle_pointcloud_input(
#     points: Union[torch.Tensor, Pointclouds],
#     lengths: Union[torch.Tensor, None],
#     normals: Union[torch.Tensor, None],
# ):
#     """Handle pointcloud input for Chamfer distance computation."""
#     if PYTORCH3D_AVAILABLE and isinstance(points, Pointclouds):
#         X = points.points_padded()
#         lengths = points.num_points_per_cloud()
#         normals = points.normals_padded()
#     elif torch.is_tensor(points):
#         if points.ndim != 3:
#             raise ValueError("Expected points to be of shape (N, P, D)")
#         X = points
#         if lengths is not None:
#             if lengths.ndim != 1 or lengths.shape[0] != X.shape[0]:
#                 raise ValueError("Expected lengths to be of shape (N,)")
#             if lengths.max() > X.shape[1]:
#                 raise ValueError("A length value was too long")
#         if lengths is None:
#             lengths = torch.full(
#                 (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
#             )
#         if normals is not None and normals.ndim != 3:
#             raise ValueError("Expected normals to be of shape (N, P, 3")
#     else:
#         raise ValueError(
#             "The input pointclouds should be either "
#             + "Pointclouds objects or torch.Tensor of shape "
#             + "(minibatch, num_points, 3)."
#         )
#     return X, lengths, normals


# def _chamfer_distance_single_direction(
#     x, y, x_lengths, y_lengths, x_normals, y_normals, weights,
#     point_reduction: Union[str, None], norm: int, abs_cosine: bool,
# ):
#     """Compute Chamfer distance in a single direction."""
#     return_normals = x_normals is not None and y_normals is not None
#     N, P1, D = x.shape

#     is_x_heterogeneous = (x_lengths != P1).any()
#     x_mask = (torch.arange(P1, device=x.device)[None] >= x_lengths[:, None])
    
#     if y.shape[0] != N or y.shape[2] != D:
#         raise ValueError("y does not have the correct shape.")
    
#     if weights is not None:
#         if weights.size(0) != N:
#             raise ValueError("weights must be of shape (N,).")
#         if not (weights >= 0).all():
#             raise ValueError("weights cannot be negative.")
#         if weights.sum() == 0.0:
#             weights = weights.view(N, 1)
#             return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

#     cham_norm_x = x.new_zeros(())

#     x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)
#     cham_x = x_nn.dists[..., 0]
#     idx_x = x_nn.idx[..., 0]

#     if is_x_heterogeneous:
#         cham_x[x_mask] = 0.0

#     if weights is not None:
#         cham_x *= weights.view(N, 1)

#     if return_normals:
#         x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
#         cosine_sim = F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
#         cham_norm_x = 1 - (torch.abs(cosine_sim) if abs_cosine else cosine_sim)

#         if is_x_heterogeneous:
#             cham_norm_x[x_mask] = 0.0
#         if weights is not None:
#             cham_norm_x *= weights.view(N, 1)

#     if point_reduction == "max":
#         assert not return_normals
#         cham_x = cham_x.max(1).values
#     elif point_reduction is not None:
#         cham_x = cham_x.sum(1)
#         if return_normals:
#             cham_norm_x = cham_norm_x.sum(1)
#         if point_reduction == "mean":
#             x_lengths_clamped = x_lengths.clamp(min=1)
#             cham_x /= x_lengths_clamped
#             if return_normals:
#                 cham_norm_x /= x_lengths_clamped

#     cham_dist = cham_x
#     cham_normals = cham_norm_x if return_normals else None
#     return cham_dist, cham_normals, idx_x


# def _apply_batch_reduction(cham_x, cham_norm_x, weights, batch_reduction: Union[str, None]):
#     """Apply batch reduction to Chamfer distance."""
#     if batch_reduction is None:
#         return (cham_x, cham_norm_x)
    
#     N = cham_x.shape[0]
#     cham_x = cham_x.sum()
#     if cham_norm_x is not None:
#         cham_norm_x = cham_norm_x.sum()
    
#     if batch_reduction == "mean":
#         if weights is None:
#             div = max(N, 1)
#         elif weights.sum() == 0.0:
#             div = 1
#         else:
#             div = weights.sum()
#         cham_x /= div
#         if cham_norm_x is not None:
#             cham_norm_x /= div
    
#     return (cham_x, cham_norm_x)


# def my_chamfer_distance(
#     x, y, x_lengths=None, y_lengths=None, x_normals=None, y_normals=None,
#     weights=None, batch_reduction: Union[str, None] = "mean",
#     point_reduction: Union[str, None] = "mean", norm: int = 2,
#     single_directional: bool = False, abs_cosine: bool = True,
# ):
#     """
#     Compute Chamfer distance between two pointclouds.
#     Used for zero-shot color projection.
#     """
#     _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

#     if not ((norm == 1) or (norm == 2)):
#         raise ValueError("Support for 1 or 2 norm.")

#     if point_reduction == "max" and (x_normals is not None or y_normals is not None):
#         raise ValueError('Normals must be None if point_reduction is "max"')

#     x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
#     y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

#     cham_x, cham_norm_x, idx_x = _chamfer_distance_single_direction(
#         x, y, x_lengths, y_lengths, x_normals, y_normals,
#         weights, point_reduction, norm, abs_cosine,
#     )
    
#     if single_directional:
#         loss = cham_x
#         loss_normals = cham_norm_x
#     else:
#         cham_y, cham_norm_y, _ = _chamfer_distance_single_direction(
#             y, x, y_lengths, x_lengths, y_normals, x_normals,
#             weights, point_reduction, norm, abs_cosine,
#         )
#         if point_reduction == "max":
#             loss = torch.maximum(cham_x, cham_y)
#             loss_normals = None
#         elif point_reduction is not None:
#             loss = cham_x + cham_y
#             if cham_norm_x is not None:
#                 loss_normals = cham_norm_x + cham_norm_y
#             else:
#                 loss_normals = None
#         else:
#             loss = (cham_x, cham_y)
#             if cham_norm_x is not None:
#                 loss_normals = (cham_norm_x, cham_norm_y)
#             else:
#                 loss_normals = None
    
#     return _apply_batch_reduction(loss, loss_normals, weights, batch_reduction), idx_x


# class AlignedDDPMScheduler:
#     """
#     DDPM Scheduler with zero-shot color alignment projection.
#     Implements the projection mechanisms from the paper.
#     """
    
#     def __init__(
#         self,
#         num_train_timesteps: int = 1000,
#         num_inference_timesteps: int = 1000,
#         beta_start: float = 0.0001,
#         beta_end: float = 0.004,
#         beta_schedule: str = "scaled_linear",
#         length: int = 64,
#         cur_bsz: int = 16,
#         prediction_type: str = "noise",
#         projection_threshold: int = 0,
#         with_pred_sample_projection: bool = False,
#         with_shuffling: bool = False,
#         train_on_rgb: bool = False,
#     ):
#         self.weight_dtype = torch.float32
#         self.num_train_steps = num_train_timesteps
#         self.num_inference_steps = num_inference_timesteps

#         if beta_schedule == "linear":
#             self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=self.weight_dtype)
#         elif beta_schedule == "scaled_linear":
#             self.betas = torch.linspace(
#                 beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps, dtype=self.weight_dtype
#             ) ** 2
#         else:
#             raise NotImplementedError(f"{beta_schedule} is not implemented")

#         self.alphas = 1.0 - self.betas
#         self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
#         self.one = torch.tensor(1.0)

#         self.prediction_type = prediction_type
#         self.cur_bsz = cur_bsz
#         self.raw_length = length
#         self.raw_latent_length = int(length / 8) if (not train_on_rgb) else length
#         self.train_on_rgb = train_on_rgb
#         self.latent_dim = 4 if (not train_on_rgb) else 3

#         # Zero-shot projection settings
#         self.with_shuffling = with_shuffling
#         self.with_pred_sample_projection = with_pred_sample_projection
#         self.projection_threshold = projection_threshold

#     def shuffle_latents(self, latents_original: torch.FloatTensor) -> torch.FloatTensor:
#         """Shuffle latents for color conditioning (zero-shot hint)."""
#         self.cur_bsz = latents_original.shape[0]
#         latents_shuffled = torch.zeros_like(latents_original).to(device=latents_original.device)
        
#         for bsi in range(self.cur_bsz):
#             div_factor_x = 1
#             div_factor_y = 1
#             shuffle_idx_sub = torch.randperm(
#                 int(self.raw_latent_length * self.raw_latent_length / div_factor_x / div_factor_y)
#             )
#             shuffle_idx_sub = shuffle_idx_sub.reshape(
#                 int(self.raw_latent_length/div_factor_x),
#                 int(self.raw_latent_length/div_factor_y)
#             )

#             shuffle_idx = torch.zeros(self.raw_latent_length, self.raw_latent_length)
#             for sub_i in range(div_factor_x):
#                 for sub_j in range(div_factor_y):
#                     shuffle_idx[sub_i::div_factor_x, sub_j::div_factor_y] = (
#                         (shuffle_idx_sub * div_factor_x) +
#                         (((shuffle_idx_sub * div_factor_x) // self.raw_latent_length) * self.raw_latent_length * div_factor_y) -
#                         (((shuffle_idx_sub * div_factor_x) // self.raw_latent_length) * self.raw_latent_length) +
#                         sub_j + (sub_i * self.raw_latent_length)
#                     )
            
#             shuffle_idx = shuffle_idx.reshape(self.raw_latent_length * self.raw_latent_length).to(dtype=torch.int64)
#             latents_shuffled[bsi] = latents_original[bsi].reshape(
#                 self.latent_dim, -1
#             )[:, shuffle_idx].reshape(self.latent_dim, self.raw_latent_length, self.raw_latent_length)

#         return {'latents_shuffled': latents_shuffled}

#     def step_backward(
#         self,
#         img_original: torch.FloatTensor,
#         coor_original: torch.FloatTensor,
#         predicted_output: torch.FloatTensor,
#         timestep: int,
#         img_ref: torch.FloatTensor,
#     ) -> dict:
#         """
#         Zero-shot denoising step with color projection.
#         This implements the core zero-shot alignment logic.
#         """
#         self.cur_bsz = img_original.shape[0]

#         t = timestep
#         prev_t = self.previous_timestep(t)

#         self.alphas_cumprod = self.alphas_cumprod.to(device=img_original.device)
#         self.one = self.one.to(device=img_original.device)

#         alpha_prod_t = self.alphas_cumprod[t]
#         alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
#         beta_prod_t = 1 - alpha_prod_t
#         beta_prod_t_prev = 1 - alpha_prod_t_prev
#         current_alpha_t = alpha_prod_t / alpha_prod_t_prev
#         current_beta_t = 1 - current_alpha_t

#         # DDPM variance
#         variance = 0
#         if t > 0:
#             variance_noise = torch.randn(
#                 self.cur_bsz, self.latent_dim, self.raw_latent_length, self.raw_latent_length
#             ).to(device=img_original.device, dtype=img_original.dtype)
#             variance = ((self._get_variance(t) ** 0.5) * variance_noise).to(device=img_original.device)

#         # Predict original sample from noise prediction
#         if self.prediction_type == "noise":
#             pred_original_sample = (
#                 coor_original - beta_prod_t ** (0.5) * predicted_output.to(device=img_original.device)
#             ) / (alpha_prod_t ** (0.5))
#         elif self.prediction_type == "adapted_noise":
#             pred_original_sample = (
#                 img_original - beta_prod_t ** (0.5) * predicted_output.to(device=img_original.device)
#             ) / (alpha_prod_t ** (0.5))
#         else:
#             raise NotImplementedError

#         pred_original_sample_raw = pred_original_sample.clone().detach().to(device=img_original.device)

#         # ============================================================
#         # ZERO-SHOT COLOR PROJECTION (Key innovation)
#         # ============================================================
#         if self.with_pred_sample_projection and t > self.projection_threshold:
#             pred_original_sample = pred_original_sample.clone().detach().requires_grad_(True)
#             pred_original_sample_shuffled = self.shuffle_latents(pred_original_sample)['latents_shuffled']

#             pred_points = torch.swapaxes(
#                 pred_original_sample_shuffled.reshape(self.cur_bsz, self.latent_dim, -1), 1, 2
#             )
#             ref_points = torch.swapaxes(
#                 img_ref.reshape(self.cur_bsz, self.latent_dim, -1), 1, 2
#             ).clone().detach().requires_grad_(True)
#             ref_points = ref_points + 1e-6 * (torch.rand(*(ref_points.shape)) - 1.).to(device=img_original.device)

#             dist_total = 0.
#             for pred_point, ref_point in zip(pred_points, ref_points):
#                 cur_pred_point = pred_point[None, ...]
#                 cur_ref_point = ref_point[None, ...]

#                 while True:
#                     cur_ref_point = cur_ref_point.clone().detach().requires_grad_(True)
#                     (dist, _), mapping = my_chamfer_distance(
#                         cur_pred_point, cur_ref_point,
#                         batch_reduction=None, point_reduction=None,
#                         single_directional=True
#                     )
#                     dist = dist[0]
#                     mapping = mapping[0]

#                     unique, idx, counts = torch.unique(
#                         mapping, dim=0, sorted=True, return_inverse=True, return_counts=True
#                     )
#                     _, ind_sorted = torch.sort(idx, stable=True)
#                     cum_sum = counts.cumsum(0)
#                     cum_sum = torch.cat((torch.tensor([0]).to(device=img_original.device), cum_sum[:-1]))
#                     first_indicies, _ = torch.sort(ind_sorted[cum_sum])
#                     first_indicies_opponet = mapping[first_indicies]

#                     dist_component = torch.sum(dist[first_indicies])
#                     dist_total = dist_total + dist_component

#                     world = torch.ones_like(mapping)
#                     world[first_indicies] = 0
#                     compl_pred = torch.nonzero(world).squeeze()
#                     if compl_pred.nelement() == 0:
#                         break

#                     world = torch.ones_like(mapping)
#                     world[first_indicies_opponet] = 0
#                     compl_ref = torch.nonzero(world).squeeze()

#                     cur_pred_point = cur_pred_point[:, compl_pred, :]
#                     if len(cur_pred_point.shape) == 2:
#                         cur_pred_point = cur_pred_point[:, None, :]

#                     cur_ref_point = cur_ref_point[:, compl_ref, :]
#                     if len(cur_ref_point.shape) == 2:
#                         cur_ref_point = cur_ref_point[:, None, :]

#             dist_total.backward()
#             pred_original_sample = (
#                 pred_original_sample - 0.5 * pred_original_sample.grad
#             ).clone().detach().requires_grad_(False)

#         # DDPM step
#         pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
#         current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

#         pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * coor_original
#         coor_next = pred_prev_sample + (variance if t > 0 else 0)
#         img_next = coor_next

#         return {
#             'img_next': img_next,
#             'coor_next': coor_next,
#             'pred_original_sample': pred_original_sample,
#             'pred_original_sample_raw': pred_original_sample_raw,
#         }

#     def _get_variance(self, t):
#         """Compute variance for DDPM sampling."""
#         prev_t = self.previous_timestep(t)
#         alpha_prod_t = self.alphas_cumprod[t]
#         alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
#         current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
#         variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
#         variance = torch.clamp(variance, min=1e-20)
#         return variance

#     def previous_timestep(self, timestep):
#         """Get previous timestep in the inference schedule."""
#         num_inference_steps = self.num_inference_steps
#         prev_t = timestep - self.num_train_steps // num_inference_steps
#         return prev_t


# def parse_args():
#     parser = argparse.ArgumentParser(description="Zero-shot inference with color projection")
#     parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model")
#     parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for generated images")
#     parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
#     parser.add_argument("--eval_batch_size", type=int, default=4, help="Batch size for inference")
#     parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of denoising steps")
#     parser.add_argument("--projection_threshold", type=int, default=200, help="Timestep threshold for projection")
#     parser.add_argument("--with_pred_sample_projection", action="store_true", help="Enable zero-shot projection")
#     parser.add_argument("--with_shuffling", action="store_true", help="Enable shuffling hint")
#     parser.add_argument("--prompt", type=str, default="firefighter", help="Text prompt for generation")
#     parser.add_argument("--ref_img", type=str, default="palette.jpg", help="Reference image for color alignment")
#     parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
#     parser.add_argument("--seed", type=int, default=10, help="Random seed")
    
#     return parser.parse_args()


# def main():
#     from PIL import Image

#     args = parse_args()
    
#     # Setup
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     torch.manual_seed(args.seed)
#     np.random.seed(args.seed)
    
#     os.makedirs(args.output_dir, exist_ok=True)
    
#     # Load models
#     print("Loading models...")
#     unet = UNet2DConditionModel.from_pretrained(args.model_path, subfolder="unet")
#     vae = AutoencoderKL.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="vae")
#     tokenizer = AutoTokenizer.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="tokenizer")
#     text_encoder = CLIPTextModel.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="text_encoder")
    
#     unet.to(device)
#     vae.to(device)
#     text_encoder.to(device)
    
#     unet.eval()
#     vae.eval()
#     text_encoder.eval()
    
#     # Initialize scheduler with zero-shot projection
#     scheduler = AlignedDDPMScheduler(
#         num_train_timesteps=1000,
#         num_inference_timesteps=args.num_inference_steps,
#         length=args.resolution,
#         cur_bsz=args.eval_batch_size,
#         prediction_type="noise",
#         with_pred_sample_projection=args.with_pred_sample_projection,
#         projection_threshold=args.projection_threshold,
#         with_shuffling=args.with_shuffling,
#         train_on_rgb=False,
#     )
    
#     # Prepare reference image (color palette)
#     # In practice, you would load or generate your reference color distribution here
#     from PIL import Image
#     from torchvision import transforms

#     preprocess = transforms.Compose([
#         transforms.Resize((args.resolution, args.resolution)),
#         transforms.ToTensor(),                  # converts PIL → Tensor
#         transforms.Normalize([0.5]*3, [0.5]*3), # [0,1] → [-1,1]
#     ])

#     reference_img = Image.open(args.ref_img).convert("RGB")
#     reference_img = preprocess(reference_img)      # Tensor [3, H, W]
#     reference_img = reference_img.unsqueeze(0)     # [1, 3, H, W]
#     reference_img = reference_img.to(device)
#     reference_latent = vae.encode(reference_img).latent_dist.sample() * vae.config.scaling_factor
    
#     if args.with_shuffling:
#         reference_latent_shuffled = scheduler.shuffle_latents(reference_latent)['latents_shuffled']
    
#     # Generate images
#     print(f"Generating {args.num_samples} samples...")
#     for sample_idx in range(args.num_samples):
#         # Encode text prompt
#         text_input = tokenizer([args.prompt] * args.eval_batch_size, padding="max_length", 
#                               max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
#         text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        
#         # Prepare unconditional embeddings for classifier-free guidance
#         uncond_input = tokenizer([negative_prompt] * args.eval_batch_size, padding="max_length",
#                                 max_length=tokenizer.model_max_length, return_tensors="pt")
#         uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
#         text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
#         # Initialize latents
#         latents = torch.randn(args.eval_batch_size, 4, args.resolution // 8, args.resolution // 8).to(device)
        
#         # Denoising loop with zero-shot projection
#         for t in tqdm(range(scheduler.num_train_steps - 1, -1, -scheduler.num_train_steps // args.num_inference_steps)):
#             # Prepare input
#             latent_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
#             if args.with_shuffling:
#                 latent_input = torch.cat([latent_input, 
#                                          torch.cat([reference_latent_shuffled] * 2)], dim=1)
            
#             # Predict noise
#             with torch.no_grad():
#                 noise_pred = unet(latent_input, t, encoder_hidden_states=text_embeddings).sample
            
#             # Classifier-free guidance
#             if do_classifier_free_guidance:
#                 noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#                 noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
#             # Zero-shot denoising step with projection
#             output = scheduler.step_backward(
#                 img_original=latents,
#                 coor_original=latents,
#                 predicted_output=noise_pred,
#                 timestep=t,
#                 img_ref=reference_latent,
#             )
            
#             latents = output['img_next']
        
#         # Decode latents to images
#         with torch.no_grad():
#             images = vae.decode(latents / vae.config.scaling_factor).sample
        
#         # Save images
#         for i, img in enumerate(images):
#             img = torch.clamp(img, -1, 1)
#             img = (img + 1) / 2
#             save_path = os.path.join(args.output_dir, f"sample_{sample_idx}_{i}.png")
#             torchvision.utils.save_image(img, save_path)
        
#         print(f"Saved sample {sample_idx}")
    
#     print(f"Done! Images saved to {args.output_dir}")


# if __name__ == "__main__":
#     main()

import argparse
import logging
import os
import sys
import random
import time
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from tqdm.auto import tqdm

from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers.utils import check_min_version
from transformers import CLIPTextModel, AutoTokenizer

try:
    from pytorch3d.loss import chamfer_distance
    from pytorch3d.ops.knn import knn_gather, knn_points
    from pytorch3d.structures.pointclouds import Pointclouds
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False

# Configuration - matching test.py
negative_prompt = "Low quality,Bad quality,Sketches,Logo,Watermark,Text,Ugly,Morbid,Extra fingers,Poorly drawn hands,Mutation,Blurry,Extra limbs,Gross proportions,Missing arms,Mutated hands,Long neck,Duplicate,Mutilated,Mutilated hands,Poorly drawn face,Deformed,Bad anatomy,Cloned face,Malformed limbs,Missing legs,Too many fingers"
guidance_scale = 5.0
do_classifier_free_guidance = True

check_min_version("0.28.0.dev0")
logger = logging.getLogger(__name__)


def _validate_chamfer_reduction_inputs(batch_reduction, point_reduction):
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
        raise ValueError("The input pointclouds should be either Pointclouds objects or torch.Tensor")
    return X, lengths, normals


def _chamfer_distance_single_direction(x, y, x_lengths, y_lengths, x_normals, y_normals, weights, point_reduction, norm, abs_cosine):
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

    return cham_x, (cham_norm_x if return_normals else None), idx_x


def _apply_batch_reduction(cham_x, cham_norm_x, weights, batch_reduction):
    if batch_reduction is None:
        return (cham_x, cham_norm_x)
    N = cham_x.shape[0]
    cham_x = cham_x.sum()
    if cham_norm_x is not None:
        cham_norm_x = cham_norm_x.sum()
    if batch_reduction == "mean":
        div = max(N, 1) if weights is None else (1 if weights.sum() == 0.0 else weights.sum())
        cham_x /= div
        if cham_norm_x is not None:
            cham_norm_x /= div
    return (cham_x, cham_norm_x)


def my_chamfer_distance(x, y, x_lengths=None, y_lengths=None, x_normals=None, y_normals=None, weights=None, 
                       batch_reduction="mean", point_reduction="mean", norm=2, single_directional=False, abs_cosine=True):
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)
    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")
    if point_reduction == "max" and (x_normals is not None or y_normals is not None):
        raise ValueError('Normals must be None if point_reduction is "max"')

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    cham_x, cham_norm_x, idx_x = _chamfer_distance_single_direction(
        x, y, x_lengths, y_lengths, x_normals, y_normals, weights, point_reduction, norm, abs_cosine)
    
    if single_directional:
        loss, loss_normals = cham_x, cham_norm_x
    else:
        cham_y, cham_norm_y, _ = _chamfer_distance_single_direction(
            y, x, y_lengths, x_lengths, y_normals, x_normals, weights, point_reduction, norm, abs_cosine)
        if point_reduction == "max":
            loss, loss_normals = torch.maximum(cham_x, cham_y), None
        elif point_reduction is not None:
            loss = cham_x + cham_y
            loss_normals = (cham_norm_x + cham_norm_y) if cham_norm_x is not None else None
        else:
            loss = (cham_x, cham_y)
            loss_normals = (cham_norm_x, cham_norm_y) if cham_norm_x is not None else None
    
    return _apply_batch_reduction(loss, loss_normals, weights, batch_reduction), idx_x


# class AlignedDDPMScheduler:
#     def __init__(self, num_train_timesteps=1000, num_inference_timesteps=1000, beta_start=0.0001, beta_end=0.004,
#                  beta_schedule="scaled_linear", length=64, cur_bsz=16, prediction_type="noise", 
#                  projection_threshold=0, with_pred_sample_projection=False, with_shuffling=False, train_on_rgb=False):
#         self.weight_dtype = torch.float32
#         self.num_train_steps = num_train_timesteps
#         self.num_inference_steps = num_inference_timesteps

#         if beta_schedule == "linear":
#             self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=self.weight_dtype)
#         elif beta_schedule == "scaled_linear":
#             self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps, dtype=self.weight_dtype) ** 2
#         else:
#             raise NotImplementedError(f"{beta_schedule} is not implemented")

#         self.alphas = 1.0 - self.betas
#         self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
#         self.one = torch.tensor(1.0)
#         self.prediction_type = prediction_type
#         self.cur_bsz = cur_bsz
#         self.raw_length = length
#         self.raw_latent_length = int(length / 8) if (not train_on_rgb) else length
#         self.train_on_rgb = train_on_rgb
#         self.latent_dim = 4 if (not train_on_rgb) else 3
#         self.with_shuffling = with_shuffling
#         self.with_pred_sample_projection = with_pred_sample_projection
#         self.projection_threshold = projection_threshold

#     def shuffle_latents(self, latents_original):
#         self.cur_bsz = latents_original.shape[0]
#         latents_shuffled = torch.zeros_like(latents_original).to(device=latents_original.device)
#         for bsi in range(self.cur_bsz):
#             shuffle_idx_sub = torch.randperm(self.raw_latent_length * self.raw_latent_length)
#             shuffle_idx_sub = shuffle_idx_sub.reshape(self.raw_latent_length, self.raw_latent_length)
#             shuffle_idx = shuffle_idx_sub.reshape(self.raw_latent_length * self.raw_latent_length).to(dtype=torch.int64)
#             latents_shuffled[bsi] = latents_original[bsi].reshape(self.latent_dim, -1)[:, shuffle_idx].reshape(
#                 self.latent_dim, self.raw_latent_length, self.raw_latent_length)
#         return {'latents_shuffled': latents_shuffled}

#     def step_backward(self, img_original, coor_original, predicted_output, timestep, img_ref):
#         self.cur_bsz = img_original.shape[0]
#         t = timestep
#         prev_t = self.previous_timestep(t)
#         self.alphas_cumprod = self.alphas_cumprod.to(device=img_original.device)
#         self.one = self.one.to(device=img_original.device)

#         alpha_prod_t = self.alphas_cumprod[t]
#         alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
#         beta_prod_t = 1 - alpha_prod_t
#         beta_prod_t_prev = 1 - alpha_prod_t_prev
#         current_alpha_t = alpha_prod_t / alpha_prod_t_prev
#         current_beta_t = 1 - current_alpha_t

#         variance = 0
#         if t > 0:
#             variance_noise = torch.randn(self.cur_bsz, self.latent_dim, self.raw_latent_length, self.raw_latent_length
#                                         ).to(device=img_original.device, dtype=img_original.dtype)
#             variance = ((self._get_variance(t) ** 0.5) * variance_noise).to(device=img_original.device)

#         if self.prediction_type == "noise":
#             pred_original_sample = (coor_original - beta_prod_t ** (0.5) * predicted_output.to(device=img_original.device)) / (alpha_prod_t ** (0.5))
#         elif self.prediction_type == "adapted_noise":
#             pred_original_sample = (img_original - beta_prod_t ** (0.5) * predicted_output.to(device=img_original.device)) / (alpha_prod_t ** (0.5))
#         else:
#             raise NotImplementedError

#         pred_original_sample_raw = pred_original_sample.clone().detach()

#         # ZERO-SHOT PROJECTION
#         if self.with_pred_sample_projection and t > self.projection_threshold:
#             pred_original_sample = pred_original_sample.clone().detach().requires_grad_(True)
#             pred_original_sample_shuffled = self.shuffle_latents(pred_original_sample)['latents_shuffled']
#             pred_points = torch.swapaxes(pred_original_sample_shuffled.reshape(self.cur_bsz, self.latent_dim, -1), 1, 2)
#             ref_points = torch.swapaxes(img_ref.reshape(self.cur_bsz, self.latent_dim, -1), 1, 2).clone().detach().requires_grad_(True)
#             ref_points = ref_points + 1e-6 * (torch.rand(*(ref_points.shape)) - 1.).to(device=img_original.device)

#             dist_total = 0.
#             for pred_point, ref_point in zip(pred_points, ref_points):
#                 cur_pred_point = pred_point[None, ...]
#                 cur_ref_point = ref_point[None, ...]
#                 while True:
#                     cur_ref_point = cur_ref_point.clone().detach().requires_grad_(True)
#                     (dist, _), mapping = my_chamfer_distance(cur_pred_point, cur_ref_point, batch_reduction=None, 
#                                                             point_reduction=None, single_directional=True)
#                     dist, mapping = dist[0], mapping[0]
#                     unique, idx, counts = torch.unique(mapping, dim=0, sorted=True, return_inverse=True, return_counts=True)
#                     _, ind_sorted = torch.sort(idx, stable=True)
#                     cum_sum = counts.cumsum(0)
#                     cum_sum = torch.cat((torch.tensor([0]).to(device=img_original.device), cum_sum[:-1]))
#                     first_indicies, _ = torch.sort(ind_sorted[cum_sum])
#                     first_indicies_opponet = mapping[first_indicies]
#                     dist_component = torch.sum(dist[first_indicies])
#                     dist_total = dist_total + dist_component
#                     world_pred = torch.ones(cur_pred_point.shape[1], device=img_original.device, dtype=torch.long)
#                     world_pred[first_indicies] = 0
#                     compl_pred = torch.nonzero(world_pred, as_tuple=False).squeeze(-1)
#                     if compl_pred.nelement() == 0:
#                         break
#                     # mask is over REF points, not over mapping (pred points)
#                     world_ref = torch.ones(cur_ref_point.shape[1], device=img_original.device, dtype=torch.long)
#                     world_ref[first_indicies_opponet] = 0
#                     compl_ref = torch.nonzero(world_ref, as_tuple=False).squeeze(-1)
#                     cur_pred_point = cur_pred_point[:, compl_pred, :]
#                     if len(cur_pred_point.shape) == 2:
#                         cur_pred_point = cur_pred_point[:, None, :]
#                     cur_ref_point = cur_ref_point[:, compl_ref, :]
#                     if len(cur_ref_point.shape) == 2:
#                         cur_ref_point = cur_ref_point[:, None, :]
#             dist_total.backward()
#             pred_original_sample = (pred_original_sample - 0.5 * pred_original_sample.grad).clone().detach().requires_grad_(False)

#         pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
#         current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
#         pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * coor_original
#         coor_next = pred_prev_sample + (variance if t > 0 else 0)
#         return {'img_next': coor_next, 'coor_next': coor_next, 'pred_original_sample': pred_original_sample, 
#                 'pred_original_sample_raw': pred_original_sample_raw}

#     def _get_variance(self, t):
#         prev_t = self.previous_timestep(t)
#         alpha_prod_t = self.alphas_cumprod[t]
#         alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
#         current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
#         variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
#         return torch.clamp(variance, min=1e-20)

#     def previous_timestep(self, timestep):
#         return timestep - self.num_train_steps // self.num_inference_steps

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
                    # --- termination guard: if either side is empty, stop ---
                    if cur_pred_point.shape[1] == 0 or cur_ref_point.shape[1] == 0:
                        break

                    cur_ref_point = cur_ref_point.clone().detach().requires_grad_(True)

                    (dist, _), mapping = my_chamfer_distance(
                        cur_pred_point, cur_ref_point,
                        batch_reduction=None, point_reduction=None,
                        single_directional=True
                    )
                    dist = dist[0]
                    mapping = mapping[0]  # shape: (P_pred,)

                    unique, idx, counts = torch.unique(
                        mapping, dim=0, sorted=True, return_inverse=True, return_counts=True
                    )
                    _, ind_sorted = torch.sort(idx, stable=True)
                    cum_sum = counts.cumsum(0)
                    cum_sum = torch.cat((torch.tensor([0], device=cur_pred_point.device), cum_sum[:-1]))
                    first_indicies, _ = torch.sort(ind_sorted[cum_sum])
                    first_indicies_opponet = mapping[first_indicies]  # indices into REF points

                    dist_component = torch.sum(dist[first_indicies])
                    dist_total = dist_total + dist_component

                    # mask over PRED points
                    world_pred = torch.ones(
                        cur_pred_point.shape[1],
                        device=cur_pred_point.device,
                        dtype=torch.long,
                    )
                    world_pred[first_indicies] = 0
                    compl_pred = torch.nonzero(world_pred, as_tuple=False).squeeze(-1)
                    if compl_pred.numel() == 0:
                        break

                    # mask over REF points
                    world_ref = torch.ones(
                        cur_ref_point.shape[1],
                        device=cur_ref_point.device,
                        dtype=torch.long,
                    )
                    world_ref[first_indicies_opponet] = 0
                    compl_ref = torch.nonzero(world_ref, as_tuple=False).squeeze(-1)

                    # --- termination guard: if REF is exhausted, stop ---
                    if compl_ref.numel() == 0:
                        break

                    # slice down
                    cur_pred_point = cur_pred_point[:, compl_pred, :]
                    if cur_pred_point.ndim == 2:
                        cur_pred_point = cur_pred_point[:, None, :]

                    cur_ref_point = cur_ref_point[:, compl_ref, :]
                    if cur_ref_point.ndim == 2:
                        cur_ref_point = cur_ref_point[:, None, :]

            dist_total.backward()

            pred_original_sample = (pred_original_sample - 0.5 * pred_original_sample.grad).clone().detach().requires_grad_(False)

        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * coor_original
        coor_next = pred_prev_sample + (variance if t > 0 else 0)

        # color alignment (for fine-tuned model)
        # if self.with_noisy_sample_projection and t > self.projection_threshold:
        #     coor_next_temp = coor_next.clone().detach().requires_grad_(True)

        #     next_points = torch.swapaxes(coor_next_temp.reshape(self.cur_bsz, self.latent_dim, -1), 1, 2)
        #     ref_points = torch.swapaxes(img_ref.reshape(self.cur_bsz, self.latent_dim, -1), 1, 2).clone().detach().requires_grad_(True)

        #     dist, _ = chamfer_distance(next_points, ref_points, batch_reduction="sum", point_reduction="sum", single_directional=True)
        #     dist.backward()

        #     img_next = (coor_next_temp - 0.5 * coor_next_temp.grad).clone().detach()
        # else:
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="", type=str)
    # parser.add_argument(
    #     "--variant",
    #     type=str,
    #     default=None,
    #     choices=["fp16"],
    #     help=(
    #         "Whether to load mixed precision models."
    #     ),
    # )
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=50)
    parser.add_argument("--beta_start", type=float, default=0.00085)
    parser.add_argument("--beta_end", type=float, default=0.0120)
    parser.add_argument("--bdpm_beta_schedule", type=str, default="scaled_linear")
    parser.add_argument("--prediction_type", type=str, default="adapted_noise")
    parser.add_argument("--projection_threshold", type=int, default=200)
    parser.add_argument("--with_pred_sample_projection", action="store_true")
    parser.add_argument("--with_shuffling", action="store_true")
    parser.add_argument("--prompt", type=str, default="firefighter")
    parser.add_argument("--ref_img", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--proportion_empty_prompts", type=float, default=0)
    return parser.parse_args()


def main():
    from PIL import Image
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading models...")
    # if args.with_shuffling:
    #     unet = UNet2DConditionModel.from_pretrained(args.model_path, subfolder="unet" if "checkpoint" in args.model_path else None,
    #                                                 in_channels=8, ignore_mismatched_sizes=True, low_cpu_mem_usage=False)
    # else:
    #     unet = UNet2DConditionModel.from_pretrained(args.model_path, subfolder="unet" if "checkpoint" in args.model_path else None)
    unet = UNet2DConditionModel.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="unet" #, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="vae")
    tokenizer = AutoTokenizer.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="text_encoder")
    unet.to(device).eval()
    vae.to(device).eval()
    text_encoder.to(device).eval()

    scheduler = AlignedDDPMScheduler(num_train_timesteps=args.ddpm_num_steps, num_inference_timesteps=args.ddpm_num_inference_steps,
                                    beta_start=args.beta_start, beta_end=args.beta_end, beta_schedule=args.bdpm_beta_schedule,
                                    length=args.resolution, cur_bsz=args.eval_batch_size, prediction_type=args.prediction_type,
                                    with_pred_sample_projection=args.with_pred_sample_projection, 
                                    projection_threshold=args.projection_threshold, with_shuffling=args.with_shuffling, train_on_rgb=False)

    print(f"Loading reference image: {args.ref_img}")
    blur_factor = 3
    preprocess = transforms.Compose([transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                                     transforms.CenterCrop(args.resolution), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    reference_img = preprocess(Image.open(args.ref_img).convert("RGB")).unsqueeze(0).to(device)
    reference_img_blurred = torchvision.transforms.functional.resize(reference_img, int(args.resolution // blur_factor), 
                                                                     interpolation=transforms.InterpolationMode.BILINEAR)
    reference_img_blurred = torchvision.transforms.functional.resize(reference_img_blurred, args.resolution, 
                                                                     interpolation=transforms.InterpolationMode.BILINEAR)
    with torch.no_grad():
        reference_latent = vae.encode(reference_img_blurred).latent_dist.sample() * vae.config.scaling_factor
    if args.with_shuffling:
        reference_latent_shuffled = scheduler.shuffle_latents(reference_latent)['latents_shuffled']

    def tokenize_texts(prompts, proportion_empty_prompts):
        captions = ["" if random.random() < proportion_empty_prompts else p for p in prompts]
        return tokenizer(captions, max_length=tokenizer.model_max_length, padding="max_length", 
                        truncation=True, return_tensors="pt").input_ids

    print(f"Generating {args.num_samples} batches...")
    global_step = 0
    for sample_idx in range(args.num_samples):
        print(f"\n=== Batch {sample_idx + 1}/{args.num_samples} ===")
        with torch.no_grad():
            prompts = [args.prompt] * args.eval_batch_size
            encoder_hidden_states = text_encoder(tokenize_texts(prompts, args.proportion_empty_prompts).to(device), return_dict=False)[0]
            if do_classifier_free_guidance:
                encoder_hidden_states_uncond = text_encoder(tokenize_texts([negative_prompt] * args.eval_batch_size, 0).to(device), 
                                                           return_dict=False)[0]
                encoder_hidden_states = torch.cat([encoder_hidden_states_uncond, encoder_hidden_states])

        latents = torch.randn(args.eval_batch_size, 4, args.resolution // 8, args.resolution // 8).to(device)
        t = scheduler.num_train_steps - 1
        
        with tqdm(total=args.ddpm_num_inference_steps, desc=f"Sample {sample_idx+1}") as pbar:
            while t > -1:
                pbar.set_postfix({"t": t})
                sample = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                if args.with_shuffling:
                    sample = torch.cat([sample, torch.cat([reference_latent_shuffled] * 2)], dim=1)
                with torch.no_grad():
                    noise_pred = unet(sample=sample, encoder_hidden_states=encoder_hidden_states, timestep=t).sample
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                output = scheduler.step_backward(img_original=latents, coor_original=latents, predicted_output=noise_pred, 
                                                timestep=t, img_ref=reference_latent)
                latents = output['img_next']
                t -= scheduler.num_train_steps // args.ddpm_num_inference_steps
                pbar.update(1)

        with torch.no_grad():
            images = vae.decode(latents / vae.config.scaling_factor).sample
        for i, img in enumerate(images):
            img = torch.clamp((img + 1) / 2, 0, 1)
            save_path = os.path.join(args.output_dir, f"sample_{sample_idx:06d}_{i}.png")
            torchvision.utils.save_image(img, save_path)
            print(f"Saved: {save_path}")
        global_step += 1
    print(f"\nDone! Images saved to {args.output_dir}")

if __name__ == "__main__":
    main()
    
    
# python zs.py --ref_img="C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\SDModelDebiasing\ColourDebiasing\color-alignment-in-diffusion\local_color_dataset\train\pastel.jpeg" --output_dir="zs_outputs" --resolution=512 --eval_batch_size=4 --ddpm_num_steps=1000 --ddpm_num_inference_steps=50 --beta_start=0.00085 --beta_end=0.0120 --bdpm_beta_schedule="scaled_linear" --prediction_type="adapted_noise" --projection_threshold=200 --with_pred_sample_projection --prompt="firefighter" --seed=10 --num_samples=1