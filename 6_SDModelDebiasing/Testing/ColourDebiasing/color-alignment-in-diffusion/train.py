import argparse
import logging
import math
import os
import sys
import shutil
import random

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
from datetime import timedelta

import diffusers
from diffusers import UNet2DModel, UNet2DConditionModel, AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, cast_training_params
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, AutoTokenizer

from pytorch3d.loss import chamfer_distance

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")
logger = get_logger(__name__, log_level="INFO")

# If train on latent, use classifier free guidance and negative prompt for better image quality.
negative_prompt = "Low quality,Bad quality,Sketches,Logo,Watermark,Text,Ugly,Morbid,Extra fingers,Poorly drawn hands,Mutation,Blurry,Extra limbs,Gross proportions,Missing arms,Mutated hands,Long neck,Duplicate,Mutilated,Mutilated hands,Poorly drawn face,Deformed,Bad anatomy,Cloned face,Malformed limbs,Missing legs,Too many fingers"
guidance_scale = 5.
do_classifier_free_guidance = True

# For diffusion usage
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
            with_projection: bool = False,
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
        self.with_projection = with_projection
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
            shuffle_idx_sub = shuffle_idx_sub.reshape(int(self.raw_latent_length / div_factor_x),
                                                      int(self.raw_latent_length / div_factor_y))

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

            latents_shuffled[bsi] = latents_original[bsi].reshape(self.latent_dim, -1)[:, shuffle_idx].reshape(self.latent_dim,
                                                                                                 self.raw_latent_length,
                                                                                                 self.raw_latent_length)
        latents_shuffled = latents_shuffled.to(device=latents_original.device)

        return {
            'latents_shuffled': latents_shuffled
        }

    # for training
    def cross_forward(
            self,
            img_original: torch.FloatTensor,
            img_ref: torch.FloatTensor,
            timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # --- cross forward process of diffusion --- #
        self.cur_bsz = img_original.shape[0]

        # prepare diffusion matters
        self.alphas_cumprod = self.alphas_cumprod.to(device=img_original.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=img_original.dtype)
        timesteps = timesteps.to(img_original.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(img_original.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
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
        if self.with_projection:
            coor_next_temp = coor_next.clone().detach().requires_grad_(True)

            next_points = torch.swapaxes(coor_next_temp.reshape(self.cur_bsz, self.latent_dim, -1), 1, 2)
            ref_points = torch.swapaxes(img_ref.reshape(self.cur_bsz, self.latent_dim, -1), 1, 2).clone().detach().requires_grad_(True)

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

    # for inference
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

        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + \
                           current_sample_coeff * coor_original
        coor_next = pred_prev_sample + (variance if t > 0 else 0)

        # color alignment
        if self.with_projection and t > self.projection_threshold:
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
        default="ddpm-model-64",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
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
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
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
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_images_steps", type=int, default=10000,
                        help="How often to save images during training.")
    parser.add_argument("--save_fully_diffused_images_steps", type=int, default=10000,
                        help="How often to save images during training.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer."
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="The maximum decay magnitude for EMA.")
    parser.add_argument(
        "--logger",
        type=str,
        default="wandb",
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
    parser.add_argument("--ddpm_beta_schedule", type=str, default="scaled_linear")
    parser.add_argument("--beta_start", type=float, default=0.0001)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
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
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--with_projection", action="store_true", help="Do our color alignment or not."
    )
    parser.add_argument(
        "--projection_threshold", type=float, default=1000, help="Do projection if timestep bigger than this threshold."
    )
    parser.add_argument(
        "--with_shuffling", action="store_true", help="Do our shuffling as color condition hint or not."
    )
    parser.add_argument(
        "--train_on_rgb", action="store_true", help="Train everything on rgb space."
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
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # a big number for high resolution or big dataset
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

                subfolders = ["unet"]
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, subfolders[i]))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"),
                                                        str_to_class(args.model_type))
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            subfolders = ["unet"]
            model_types = [args.model_type]
            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = str_to_class(model_types[i]).from_pretrained(input_dir, subfolder=subfolders[i])
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                if args.mixed_precision == "fp16":
                    cast_training_params([model], dtype=torch.float32)

                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the model
    if args.model_config_name_or_path is None:
        if args.model_type == 'UNet2DModel':
            model = UNet2DModel(
                sample_size=args.resolution,
                in_channels=(8 if args.with_shuffling else 4) if (not args.train_on_rgb) else (6 if args.with_shuffling else 3),
                out_channels=4 if (not args.train_on_rgb) else 3,
                act_fn="silu",
                attention_head_dim=8,
                block_out_channels=(
                    224,
                    224,
                    448,
                    448,
                    672,
                    672
                ),
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D",
                    "DownBlock2D",
                ),
                downsample_padding=1,
                flip_sin_to_cos=True,
                freq_shift=0,
                layers_per_block=2,
                mid_block_scale_factor=1,
                norm_eps=1e-05,
                norm_num_groups=32,
                up_block_types=(
                    "UpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
            )
            tokenizer = None
            vae = None
            text_encoder = None
            if not args.train_on_rgb:
                vae = AutoencoderKL.from_pretrained(
                    "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="vae", variant=args.variant
                )

        elif args.model_type == 'UNet2DConditionModel':
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
            raise NotImplementedError

    else:
        config = str_to_class(args.model_type).load_config(args.model_config_name_or_path)
        model = str_to_class(args.model_type).from_config(config)

    # Create EMA for the model.
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=str_to_class(args.model_type),
            model_config=model.config,
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # Initialize the optimizer
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        if args.train_split_only:
            if args.dataset_name == "jackyhate/text-to-image-2M":
                dataset, test_dataset = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    cache_dir=args.cache_dir,
                    split=['train[:60000]+train[80000:]', 'train[60000:80000]'],
                    trust_remote_code=True,
                    download_config=DownloadConfig(cache_dir=args.cache_dir + "/downloads", resume_download=True)
                )
            else:
                dataset, test_dataset = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    cache_dir=args.cache_dir,
                    split=['train[:90%]', 'train[90%:]'],
                    trust_remote_code=True,
                    download_config=DownloadConfig(cache_dir=args.cache_dir + "/downloads", resume_download=True)
                )
        else:
            dataset, test_dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
                split=['train', 'test'],
                trust_remote_code=True,
                download_config=DownloadConfig(cache_dir=args.cache_dir + "/downloads", resume_download=True)
            )
    else:
        raise Exception('Unimplemented dataset setting.')
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    if args.mixed_precision == "fp16":
        cast_training_params([model], dtype=torch.float32)

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
            elif "json" in examples:
                texts = [json["prompt"] for json in examples["json"]]
            else:
                raise NotImplementedError

            return {"input": images, "input_prompt": texts}
        else:
            return {"input": images}

    logger.info(f"Dataset size: {len(dataset)}")

    dataset.set_transform(transform_images)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers, shuffle=True
    )

    test_dataset.set_transform(transform_images)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.eval_batch_size, num_workers=args.dataloader_num_workers, shuffle=True
    )
    it_test_dataloader = iter(test_dataloader)

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    if vae:
        vae.to(accelerator.device, dtype=weight_dtype)
    if text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    if args.use_ema:
        ema_model.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Train!
    NUMSTEPS = args.ddpm_num_steps

    noise_scheduler = AlignedDDPMScheduler(num_train_timesteps=args.ddpm_num_steps,
                                           num_inference_timesteps=args.ddpm_num_inference_steps,
                                           length=args.resolution, cur_bsz=args.train_batch_size,
                                           beta_start=args.beta_start, beta_end=args.beta_end,
                                           beta_schedule=args.ddpm_beta_schedule,
                                           prediction_type=args.prediction_type,
                                           with_projection=args.with_projection,
                                           projection_threshold=args.projection_threshold,
                                           with_shuffling=args.with_shuffling,
                                           train_on_rgb=args.train_on_rgb)

    for epoch in range(first_epoch, args.num_epochs):
        if text_encoder:
            text_encoder.requires_grad_(False)
        if vae:
            vae.requires_grad_(False)
        model.train()

        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        accelerator.wait_for_everyone()

        for step, batch in enumerate(train_dataloader):
            model.train()
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            # Sample input: images; time steps; prompts;
            clean_images = batch["input"].to(weight_dtype)
            bsz = clean_images.shape[0]  # [bs, channels, height, width]

            timesteps = torch.randint(
                0, NUMSTEPS, (bsz,), device=clean_images.device
            ).long()

            # Shuffle the image
            if not args.train_on_rgb:
                blur_factor = 3
                clean_images_blurred = torchvision.transforms.functional.resize(clean_images, int(args.resolution // blur_factor), interpolation=transforms.InterpolationMode.BILINEAR)
                clean_images_blurred = torchvision.transforms.functional.resize(clean_images_blurred, args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
            else:
                clean_images_blurred = clean_images

            if vae:
                clean_img_latents = vae.encode(clean_images).latent_dist.sample().to(device=clean_images.device)
                clean_img_latents = clean_img_latents * vae.config.scaling_factor

                clean_img_blurred_latents = vae.encode(clean_images_blurred).latent_dist.sample().to(device=clean_images.device)
                clean_img_blurred_latents = clean_img_blurred_latents * vae.config.scaling_factor
            else:
                clean_img_latents = clean_images.clone().detach().to(device=clean_images.device)
                clean_img_blurred_latents = clean_images_blurred.clone().detach().to(device=clean_images.device)

            clean_img_blurred_latents_shuffled = noise_scheduler.shuffle_latents(clean_img_blurred_latents)['latents_shuffled']

            diffused_output = noise_scheduler.cross_forward(clean_img_latents, clean_img_blurred_latents, timesteps)

            # Forward pass
            loss_dir = {}
            with accelerator.accumulate(model):
                # Predict the noise residual
                if args.model_type == 'UNet2DModel':
                    temp_input = torch.cat((diffused_output['img_next'],
                                                           clean_img_blurred_latents_shuffled
                                                           ), dim=1) if args.with_shuffling else diffused_output['img_next']
                    model_output = model(sample=temp_input,
                                         timestep=timesteps).sample
                elif args.model_type == 'UNet2DConditionModel':
                    encoder_hidden_states = text_encoder(
                        tokenize_texts(batch["input_prompt"], args.proportion_empty_prompts).to(
                            device=clean_images.device), return_dict=False)[0]

                    temp_input = torch.cat((diffused_output['img_next'],
                                                           clean_img_blurred_latents_shuffled
                                                           ), dim=1) if args.with_shuffling else diffused_output['img_next']

                    model_output = model(sample=temp_input,
                                         encoder_hidden_states=encoder_hidden_states,
                                         timestep=timesteps).sample
                else:
                    raise NotImplementedError

                # Calculate loss
                loss = F.mse_loss(model_output.float(),
                                  diffused_output['target_output'])  # this could have different weights!

                loss_dir["loss"] = loss.detach().item()
                loss_dir["timesteps"] = torch.mean(timesteps.clone().float()).item()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    # save model
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    # testing
                    if global_step % args.save_images_steps == 0:
                        unet = accelerator.unwrap_model(model)

                        if args.use_ema:
                            ema_model.store(unet.parameters())
                            ema_model.copy_to(unet.parameters())

                        with torch.no_grad():
                            batch_t = next(it_test_dataloader)  # next(iter(test_dataloader))
                            batch_t["input"] = batch_t["input"].to(weight_dtype).to(device=clean_images.device)
                            clean_images_t = batch_t["input"]  # .to(weight_dtype).to(device=clean_images.device)

                            if not args.train_on_rgb:
                                blur_factor = 3
                                clean_images_t_blurred = torchvision.transforms.functional.resize(clean_images_t,
                                                                                                 int(args.resolution // blur_factor),
                                                                                                 interpolation=transforms.InterpolationMode.BILINEAR)
                                clean_images_t_blurred = torchvision.transforms.functional.resize(clean_images_t_blurred,
                                                                                                 args.resolution,
                                                                                                 interpolation=transforms.InterpolationMode.BILINEAR)
                            else:
                                clean_images_t_blurred = clean_images_t

                            if vae:
                                clean_img_latents_t = vae.encode(clean_images_t).latent_dist.sample().to(device=clean_images_t.device)
                                clean_img_latents_t = clean_img_latents_t * vae.config.scaling_factor

                                clean_img_t_blurred_latents = vae.encode(clean_images_t_blurred).latent_dist.sample().to(
                                    device=clean_images_t.device)
                                clean_img_t_blurred_latents = clean_img_t_blurred_latents * vae.config.scaling_factor
                            else:
                                clean_img_latents_t = clean_images_t.clone().detach().to(device=clean_images_t.device)
                                clean_img_t_blurred_latents = clean_images_t_blurred.clone().detach().to(device=clean_images_t.device)

                            clean_img_t_blurred_latents_shuffled = noise_scheduler.shuffle_latents(clean_img_t_blurred_latents)['latents_shuffled']

                        ##################################
                        # check all-steps sampling
                        ##################################
                        bsz_t = clean_images_t.shape[0]
                        diffused_output_t = noise_scheduler.cross_forward(clean_img_latents_t,
                                                                               clean_img_t_blurred_latents,
                                                                               torch.full((bsz_t,),
                                                                                          NUMSTEPS - 1,
                                                                                          device=clean_images_t.device).long())
                        with torch.no_grad():
                            if args.with_projection:
                                x_t = diffused_output_t['img_next']
                                c_t = diffused_output_t['coor_next']
                            else:
                                x_t = torch.randn(bsz_t,
                                                  noise_scheduler.latent_dim,
                                                  noise_scheduler.raw_latent_length,
                                                  noise_scheduler.raw_latent_length).to(device=clean_images_t.device, dtype=clean_images_t.dtype)
                                c_t = x_t

                            diffused_images_to_save_STEP = \
                            vae.decode(x_t.to(weight_dtype) / vae.config.scaling_factor, return_dict=False)[0] if vae else x_t

                            STEP_list = diffused_images_to_save_STEP.clone().to(device=clean_images_t.device)
                            FW_list = diffused_images_to_save_STEP.clone().to(device=clean_images_t.device)

                        for t in range(NUMSTEPS - 1, -1,
                                       -int(args.ddpm_num_steps // args.ddpm_num_inference_steps)):
                            print('Doing step:', t)

                            if args.model_type == 'UNet2DModel':
                                with torch.no_grad():
                                    model_output_t = model(torch.cat((x_t,
                                                              clean_img_t_blurred_latents_shuffled
                                                              ), dim=1) if args.with_shuffling else x_t,
                                                           t).sample

                                diffused_output_t = noise_scheduler.step_backward(
                                    x_t,
                                    c_t,
                                    model_output_t,
                                    t,
                                    clean_img_t_blurred_latents)

                            elif args.model_type == 'UNet2DConditionModel':
                                with torch.no_grad():
                                    encoder_hidden_states_t = text_encoder(
                                        tokenize_texts(batch_t["input_prompt"], 1).to(
                                            device=clean_images_t.device), return_dict=False)[0]
                                    if do_classifier_free_guidance:
                                        encoder_hidden_states_uncond_t = text_encoder(
                                            tokenize_texts([negative_prompt for _ in range(bsz_t)],
                                                           args.proportion_empty_prompts).to(
                                                device=clean_images_t.device), return_dict=False)[0]
                                        encoder_hidden_states_t = torch.cat(
                                            [encoder_hidden_states_uncond_t, encoder_hidden_states_t])

                                    sample = torch.cat((x_t,
                                                        clean_img_t_blurred_latents_shuffled
                                                        ), dim=1) if args.with_shuffling else x_t

                                    if do_classifier_free_guidance:
                                        sample = torch.cat([sample] * 2)

                                    model_output_t = model(sample=sample,
                                          encoder_hidden_states=encoder_hidden_states_t,
                                          timestep=t).sample

                                    if do_classifier_free_guidance:
                                        noise_pred_uncond, noise_pred_text = model_output_t.chunk(2)
                                        model_output_t = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                                diffused_output_t = noise_scheduler.step_backward(
                                    x_t,
                                    c_t,
                                    model_output_t,
                                    t,
                                    clean_img_t_blurred_latents)
                            else:
                                raise NotImplementedError

                            with torch.no_grad():
                                x_t = diffused_output_t['img_next']
                                c_t = diffused_output_t['coor_next']

                            saving_step = ((t + 1) / int(args.ddpm_num_steps // args.ddpm_num_inference_steps))
                            if (saving_step - 1) % 10 == 0:
                                with torch.no_grad():
                                    STEP_list = torch.cat((STEP_list, vae.decode(
                                        x_t.to(weight_dtype) / vae.config.scaling_factor,
                                        return_dict=False)[0] if vae else x_t), dim=0)

                                fw_dic = noise_scheduler.cross_forward(
                                    clean_img_latents_t.clone(),
                                    clean_img_t_blurred_latents,
                                    torch.full((bsz_t,),
                                               t - int(args.ddpm_num_steps // args.ddpm_num_inference_steps),
                                               device=clean_images_t.device).long())

                                with torch.no_grad():
                                    FW_list = torch.cat((FW_list, vae.decode(
                                        fw_dic['img_next'].to(weight_dtype) / vae.config.scaling_factor,
                                        return_dict=False)[0] if vae else fw_dic['img_next']), dim=0)

                        with torch.no_grad():
                            pred_img_STEP = \
                            vae.decode(x_t.to(weight_dtype) / vae.config.scaling_factor,
                                       return_dict=False)[0] if vae else x_t

                            clean_images_t_points = torch.swapaxes(
                                clean_images_t.reshape(clean_images_t.shape[0], 3, -1), 1, 2)
                            pred_img_STEP_points = torch.swapaxes(pred_img_STEP.reshape(pred_img_STEP.shape[0], 3, -1),
                                                                  1, 2)

                            rgb_chamfer_dist, _ = chamfer_distance(pred_img_STEP_points, clean_images_t_points, norm=1,
                                                                   batch_reduction=None, point_reduction=None)

                            rgb_chamfer_dist_l1_accuracy = torch.mean(rgb_chamfer_dist[0] / 2. * 255., dim=1).tolist()
                            rgb_cham_acc_str = ['{:.2f}'.format(x) for x in rgb_chamfer_dist_l1_accuracy]
                            rgb_cham_acc_str = " - ".join(rgb_cham_acc_str)

                            rgb_chamfer_dist_l1_complement = torch.mean(rgb_chamfer_dist[1] / 2. * 255., dim=1).tolist()
                            rgb_cham_com_str = ['{:.2f}'.format(x) for x in rgb_chamfer_dist_l1_complement]
                            rgb_cham_com_str = " - ".join(rgb_cham_com_str)

                        if args.use_ema:
                            ema_model.restore(unet.parameters())

                        # denormalize and save the images
                        if args.logger == "wandb":
                            accelerator.get_tracker("wandb").log(
                                {
                                    "1_test_original_img": [wandb.Image(
                                        torchvision.utils.make_grid(clean_images_t),
                                        caption=rgb_cham_com_str)],
                                    "2_test_pred_img_STEP": [wandb.Image(torchvision.utils.make_grid(
                                        torch.clamp(pred_img_STEP, min=-1., max=1.)),
                                        caption=rgb_cham_acc_str)],
                                    "3_FW_list": [wandb.Image(torchvision.utils.make_grid(
                                        torch.clamp(FW_list, min=-1., max=1.), nrow=8))],
                                    "4_STEP_list": [wandb.Image(torchvision.utils.make_grid(
                                        torch.clamp(STEP_list, min=-1., max=1.), nrow=8))],
                                    "epoch": epoch},
                                step=global_step,
                            )

            logs = loss_dir
            logs["lr"] = lr_scheduler.get_last_lr()[0]
            logs["step"] = global_step
            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        progress_bar.close()
        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
