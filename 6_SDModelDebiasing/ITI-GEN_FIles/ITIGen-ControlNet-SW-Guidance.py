"""
ITI-GEN + Unified ControlNet + SW-Guidance (Color Debiasing)

What it does
------------
- Loads SD v1.5 (Diffusers)
- Optionally loads ControlNet based on --control-type
- Optionally fuses an LCM-LoRA for speed
- Loads ITI-GEN learned token embeddings from checkpoints
- Applies SW-guidance for color debiasing using reference images
- Generates images per ITI-GEN attribute combination (with optional substring filters)

Key Features
------------
- Combines demographic control (ITI-GEN) + structural control (ControlNet) + color debiasing (SW-guidance)
- Multiple loss types: mean_cov, SWD, DSWD, GSWD, ISEBSW
- Time travel blocks for improved generation
- Optional SARAH gradient optimization
"""

import argparse
import os
import gc
import cv2
import torch
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
from typing import Tuple, List, Optional
from pytorch_lightning import seed_everything

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
    DDIMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
)
from transformers import CLIPTokenizer, CLIPTextModel
import torch.nn as nn
from torch import optim

from utils import get_folder_names_and_indexes
from iti_gen.model import ITI_GEN
from utils import rand_rotation_matrix, get_cdf, compute_w_dist

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
# SW-Guidance Components
# ----------------------------

def get_schedule_jump(T_sampling, travel_length, travel_repeat):
    """Time travel blocks for DDNM-style generation"""
    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)
    _check_times(ts, -1, T_sampling)
    return ts

def _check_times(times, t_0, T_sampling):
    assert times[0] > times[1], (times[0], times[1])
    assert times[-1] == -1, times[-1]
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)

class TransformNet(nn.Module):
    def __init__(self, size):
        super(TransformNet, self).__init__()
        self.size = size
        self.net = nn.Sequential(nn.Linear(self.size, self.size))

    def forward(self, input):
        out = self.net(input)
        return out / torch.sqrt(torch.sum(out ** 2, dim=1, keepdim=True))

def rand_projections(dim, num_projections=1000):
    projections = torch.randn((num_projections, dim))
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections

def sliced_wasserstein_distance(first_samples, second_samples, num_projections=1000, p=2, device="cuda"):
    dim = second_samples.size(1)
    # Match dtype of input samples
    projections = rand_projections(dim, num_projections).to(device=device, dtype=first_samples.dtype)
    first_projections = first_samples.matmul(projections.transpose(0, 1))
    second_projections = second_samples.matmul(projections.transpose(0, 1))
    
    sort_x = torch.sort(first_projections.T, dim=1)[0]
    sort_y = torch.sort(second_projections.T, dim=1)[0]
    wasserstein_distance = torch.abs(sort_x - sort_y)
    wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1).mean(), 1.0 / p)
    return wasserstein_distance

def distributional_sliced_wasserstein_distance(
    first_samples, second_samples, num_projections, f, f_op, p=2, max_iter=10, lam=1, device="cuda"
):
    embedding_dim = first_samples.size(1)
    # Match dtype of input samples
    pro = rand_projections(embedding_dim, num_projections).to(device=device, dtype=first_samples.dtype)
    first_samples_detach = first_samples.detach()
    second_samples_detach = second_samples.detach()
    
    for _ in range(max_iter):
        projections = f(pro)
        cos = cosine_distance_torch(projections, projections)
        reg = lam * cos
        encoded_projections = first_samples_detach.matmul(projections.transpose(0, 1))
        distribution_projections = second_samples_detach.matmul(projections.transpose(0, 1))
        wasserstein_distance = torch.abs(
            (torch.sort(encoded_projections.transpose(0, 1), dim=1)[0]
             - torch.sort(distribution_projections.transpose(0, 1), dim=1)[0])
        )
        wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1).mean(), 1.0 / p)
        loss = reg - wasserstein_distance
        f_op.zero_grad()
        loss.backward(retain_graph=True)
        f_op.step()

    projections = f(pro)
    encoded_projections = first_samples.matmul(projections.transpose(0, 1))
    distribution_projections = second_samples.matmul(projections.transpose(0, 1))
    wasserstein_distance = torch.abs(
        (torch.sort(encoded_projections.transpose(0, 1), dim=1)[0]
         - torch.sort(distribution_projections.transpose(0, 1), dim=1)[0])
    )
    wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1).mean(), 1.0 / p)
    return wasserstein_distance

def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mean(torch.abs(torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)))

def linear(X, theta):
    if len(theta.shape) == 1:
        return torch.matmul(X, theta)
    else:
        return torch.matmul(X, theta.t())

def poly(X, theta, degree=2, device="cuda"):
    N, d = X.shape
    assert theta.shape[1] == homopoly(d, degree)
    powers = list(get_powers(d, degree))
    HX = torch.ones((N, len(powers))).to(device)
    for k, power in enumerate(powers):
        for i, p in enumerate(power):
            HX[:, k] *= X[:, i] ** p
    if len(theta.shape) == 1:
        return torch.matmul(HX, theta)
    else:
        return torch.matmul(HX, theta.t())

def get_powers(dim, degree):
    if dim == 1:
        yield (degree,)
    else:
        for value in range(degree + 1):
            for permutation in get_powers(dim - 1, degree - value):
                yield (value,) + permutation

def homopoly(dim, degree):
    return len(list(get_powers(dim, degree)))

def gsw(first_samples, second_samples, num_projections=10, ftype="linear", degree=2, device='cuda', p=1):
    dim = second_samples.size(1)
    
    if ftype == 'linear':
        theta = torch.randn((num_projections, dim))
        theta = torch.stack([th / torch.sqrt((th ** 2).sum()) for th in theta]).to(device=device, dtype=first_samples.dtype)
        first_projections = linear(first_samples, theta)
        second_projections = linear(second_samples, theta)
    elif ftype == 'poly':
        dpoly = homopoly(dim, degree)
        theta = torch.randn((num_projections, dpoly))
        theta = torch.stack([th / torch.sqrt((th ** 2).sum()) for th in theta]).to(device=device, dtype=first_samples.dtype)
        first_projections = poly(first_samples, theta, degree, device)
        second_projections = poly(second_samples, theta, degree, device)

    sort_x = torch.sort(first_projections, dim=0)[0]
    sort_y = torch.sort(second_projections, dim=0)[0]
    wasserstein_distance = torch.abs(sort_x - sort_y)
    wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1).mean(), 1.0 / p)
    return wasserstein_distance

def ISEBSW(first_samples, second_samples, num_projections=10, device='cuda', p=1):
    dim = second_samples.size(1)
    theta = torch.randn((num_projections, dim))
    theta = torch.stack([th / torch.sqrt((th ** 2).sum()) for th in theta]).to(device=device, dtype=first_samples.dtype)
    first_projections = linear(first_samples, theta)
    second_projections = linear(second_samples, theta)

    sort_x = torch.sort(first_projections, dim=0)[0]
    sort_y = torch.sort(second_projections, dim=0)[0]
    wasserstein_distance = torch.abs(sort_x - sort_y)
    wasserstein_distance = torch.pow(wasserstein_distance, p)
    weights = torch.softmax(wasserstein_distance, dim=1)
    wasserstein_distance = torch.pow(torch.sum(weights * wasserstein_distance, dim=1).mean(), 1.0 / p)
    
    return wasserstein_distance

# ----------------------------
# Original ITI-GEN Functions
# ----------------------------

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
        segmentor = UperNetForSemanticSegmentation.from_pretrained(
            "openmmlab/upernet-convnext-small"
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            pixel_values = image_processor(src, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(segmentor.device)
            outputs = segmentor(pixel_values)
            seg = image_processor.post_process_semantic_segmentation(
                outputs, target_sizes=[src.size[::-1]]
            )[0].cpu().numpy()

        # Use a simplified palette for segmentation
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

def load_reference_image(ref_path: str, size: Tuple[int, int], device: str, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load and process reference image for SW-guidance"""
    ref_im = Image.open(ref_path).convert('RGB')
    ref_im = ref_im.resize(size)  # Simple resize
    
    # CRITICAL: Use permute(2, 1, 0) not (2, 0, 1) - this matches the original!
    pixels_ref = torch.Tensor(np.array(ref_im) / 255).permute(2, 1, 0).reshape(3, -1).to(device=device, dtype=dtype)
    ref_mean = torch.mean(pixels_ref, dim=1).to(device)
    ref_cov = torch.cov(pixels_ref).to(device)
    
    print(f"Reference loaded: shape={pixels_ref.shape}, mean={ref_mean}, dtype={pixels_ref.dtype}")
    return pixels_ref, ref_mean, ref_cov

def main():
    parser = argparse.ArgumentParser()

    # Output / runtime
    parser.add_argument("--outdir", type=str, default="outputs/iti-gen-sw", help="Directory to write results to")
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
    parser.add_argument("--generate-image-prompt", type=str, default="a headshot of a person",
                        help="Prompt for generation")
    parser.add_argument("--data-path", type=str, default="./data", help="Path to reference images")
    parser.add_argument("--token-length", type=int, default=3, help="Length of learned tokens")
    parser.add_argument("--filters", type=str, nargs="*", default=None, help="Label filters")

    # SW-Guidance parameters
    parser.add_argument("--sw-reference-image", type=str, default=None,
                        help="Reference image for color debiasing (SW-guidance)")
    parser.add_argument("--sw-enabled", action="store_true", help="Enable SW-guidance for color debiasing")
    parser.add_argument("--sw-M", type=int, default=5, help="Iterations per denoising step for SW-guidance")
    parser.add_argument("--sw-u-lr", type=float, default=1/25, help="Learning rate for guidance")
    parser.add_argument("--sw-loss-type", type=int, default=1,
                        help="Loss type: 0=mean_cov, 1=swd, 2=dswd, 3=gswd, 4=isebsw")
    parser.add_argument("--sw-wsd-p", type=int, default=1, help="Sliced Wasserstein power")
    parser.add_argument("--sw-num-projections", type=int, default=100, help="Number of projections")
    parser.add_argument("--sw-ftype", type=str, default="poly", choices=["linear", "poly"],
                        help="Function type for GSWD")
    parser.add_argument("--sw-degree", type=int, default=5, help="Polynomial degree for GSWD")
    parser.add_argument("--sw-stop-guidance", type=float, default=0.99, help="Stop guidance threshold")
    parser.add_argument("--sw-decode-start", type=int, default=0, help="Start decoding step")
    parser.add_argument("--sw-use-sarah", action="store_true", help="Use SARAH gradient optimization")
    parser.add_argument("--sw-travel-length", type=int, default=1, help="Time travel length")
    parser.add_argument("--sw-travel-repeat", type=int, default=1, help="Time travel repeat")
    parser.add_argument("--sw-downscale-min", type=int, default=48, help="Min size for downscaling")
    parser.add_argument("--sw-downscale-max", type=int, default=64, help="Max size for downscaling")
    parser.add_argument("--save-intermediate", action="store_true", help="Save intermediate steps")

    # Negative prompt
    parser.add_argument("--negative_prompt", type=str,
                        default="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
                        help="Negative prompt")

    opt = parser.parse_args()
    opt.device = parse_device(opt.device)

    os.makedirs(opt.outdir, exist_ok=True)
    if opt.save_intermediate:
        os.makedirs(os.path.join(opt.outdir, "steps"), exist_ok=True)

    # Validate SW-guidance parameters
    if opt.sw_enabled and not opt.sw_reference_image:
        raise ValueError("When --sw-enabled is set, you must provide --sw-reference-image")

    # Determine if we're using ControlNet
    use_controlnet = opt.control_type is not None
    if use_controlnet and not opt.control_image:
        raise ValueError("When --control-type is set, you must also provide --control-image")

    # ----- Load Models Manually (for SW-guidance compatibility) -----
    dtype = torch.float16 if "cuda" in opt.device else torch.float32

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

    # NEW - Use the SD v1.5 default scheduler
    from diffusers import PNDMScheduler
    scheduler = PNDMScheduler.from_pretrained(opt.sd_model, subfolder="scheduler")

    # scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    scheduler.set_timesteps(opt.num_inference_steps)

    # Load ControlNet if specified
    controlnet = None
    if use_controlnet:
        controlnet_id = CONTROLNET_MODELS[opt.control_type]
        print(f"Loading ControlNet ({opt.control_type})...")
        controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype).to(opt.device)
        print("ControlNet loaded")

    # SW-guidance specific setup
    transform_net = None
    op_trannet = None
    if opt.sw_enabled and opt.sw_loss_type == 2:  # DSWD
        transform_net = TransformNet(3).to(opt.device)
        op_trannet = optim.Adam(transform_net.parameters(), betas=(0.5, 0.999))

    # Time travel schedule
    times = get_schedule_jump(opt.num_inference_steps, opt.sw_travel_length, opt.sw_travel_repeat)
    time_pairs = list(zip(times[:-1], times[1:]))

    # ----- Load reference image for SW-guidance -----
    pixels_ref = None
    ref_mean = None
    ref_cov = None
    if opt.sw_enabled:
        print(f"Loading reference image for SW-guidance: {opt.sw_reference_image}")
        pixels_ref, ref_mean, ref_cov = load_reference_image(
            opt.sw_reference_image, (opt.W, opt.H), opt.device, dtype
        )
        # Convert reference pixels to the same dtype as the model
        pixels_ref = pixels_ref.to(dtype)
        ref_mean = ref_mean.to(dtype)
        ref_cov = ref_cov.to(dtype)
        print("Reference image loaded")

    # ----- Build ControlNet conditioning image -----
    control_maps = None
    if use_controlnet:
        control_image_paths = resolve_control_images(opt.control_image)
        print(f"Building {len(control_image_paths)} control maps (type={opt.control_type})")
        control_maps = [
            load_control_map(img_path, opt.control_type, (opt.W, opt.H), opt.control_is_map)
            for img_path in control_image_paths
        ]

    # ----- Initialize ITI-GEN model -----
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

    folder_path = os.path.join(
        opt.ckpt_path,
        f"{opt.prompt.replace(' ', '_')}_{'_'.join(iti_gen.attr_list)}"
    )

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

    print("ITI-GEN embeddings loaded")

    del iti_gen, prepend_embeddings
    gc.collect()
    if "cuda" in opt.device:
        torch.cuda.empty_cache()

    # ----- Build folder combinations -----
    folder_with_indexes = get_folder_names_and_indexes(opt.attr_list.split(","))
    folder_with_indexes = apply_filters(folder_with_indexes, opt.filters)

    if len(folder_with_indexes) == 0:
        raise ValueError("No demographic combinations left after filtering.")

    print(f"\nGenerating images for {len(folder_with_indexes)} combinations...")
    print(f"Mode: ITI-GEN + {'ControlNet=' + opt.control_type if use_controlnet else 'No ControlNet'} + {'SW-Guidance' if opt.sw_enabled else 'No SW-Guidance'}")

    # ----- Helper function for decoding -----
    def decode_to_image(latent):
        with torch.no_grad():
            # Ensure latent is in correct dtype
            image = vae.decode(1 / 0.18215 * latent.to(dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        return Image.fromarray(images[0])

    # ----- Generation loop -----
    for folder, index in tqdm(folder_with_indexes.items(), desc="Combinations"):
        sample_path = os.path.join(opt.outdir, folder)
        os.makedirs(sample_path, exist_ok=True)

        base_count = len([
            f for f in os.listdir(sample_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        combination_emb = emb[index].unsqueeze(0).repeat(opt.n_samples, 1, 1)

        # Prepare unconditional embeddings
        max_length = tokenizer.model_max_length
        uncond_input = tokenizer(
            [""] * opt.n_samples, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(opt.device))[0]
            uncond_embeddings = uncond_embeddings.to(dtype)  # Ensure correct dtype

        for ci, control_map in enumerate(control_maps or [None]):
            if use_controlnet and control_map is not None:
                control_out = os.path.join(sample_path, f"ctrl_{ci}_{opt.control_type}.png")
                control_map.save(control_out)

            for n in range(opt.n_iter):
                seed = opt.seed + n + ci * 10_000
                seed_everything(seed)
                
                # Set manual seeds for reproducibility
                torch.manual_seed(seed)
                if 'cuda' in opt.device:
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                
                # Initialize latent (no generator when using device parameter)
                x_t = torch.randn(
                    (opt.n_samples, unet.config.in_channels, opt.H // 8, opt.W // 8),
                    device=opt.device,
                ) * scheduler.init_noise_sigma

                # SARAH setup
                if opt.sw_enabled and opt.sw_use_sarah:
                    v = torch.zeros_like(x_t, requires_grad=False)
                    buf_grad = torch.zeros_like(v)

                x0_pred_last = None

                # COMPLETE FIXED DENOISING LOOP
                # Replace the entire denoising section in your code with this

                # COMPLETE FIXED DENOISING LOOP - Replace in your main() function
                # This version properly handles VAE gradients and memory

                # Before the loop, ensure VAE requires gradients for decoder
                vae.decoder.requires_grad_(True)

                # Denoising loop with time travel
                for i, j in tqdm(time_pairs, desc=f"Denoising {folder}", leave=False):
                    if i > 1:
                        if j < i:
                            # Forward step
                            i = scheduler.num_inference_steps - i
                            j = scheduler.num_inference_steps - j

                            timesteps = scheduler.timesteps[i]
                            at = scheduler.alphas_cumprod[timesteps]

                            u = torch.zeros_like(x_t, requires_grad=True)
                            do_guidance = (opt.sw_enabled and opt.sw_u_lr != 0)
                            # print(f"Step {i}: do_guidance={do_guidance}")

                            # Apply SW-guidance
                            if do_guidance and i < int(scheduler.num_inference_steps * opt.sw_stop_guidance) and i > opt.sw_decode_start:
                                # print(f" Applying SW-guidance at step {i}")
                                # Generate downscale size ONCE per step (outside M loop)
                                size = np.random.randint(opt.sw_downscale_min, opt.sw_downscale_max)
                                
                                for m_iter in range(opt.sw_M):
                                    # Detach and re-enable gradients
                                    u = u.detach()
                                    u.requires_grad = True
                                    x_hat_t = x_t.detach() + u

                                    # Predict noise - use ONLY conditional embeddings (no CFG)
                                    noise_pred = unet(
                                        x_hat_t.to(dtype), 
                                        timesteps, 
                                        encoder_hidden_states=combination_emb.to(dtype)
                                    ).sample

                                    # Apply ControlNet if enabled
                                    if use_controlnet and control_map is not None:
                                        control_tensor = torch.from_numpy(np.array(control_map)).float() / 255.0
                                        control_tensor = control_tensor.permute(2, 0, 1).unsqueeze(0).to(opt.device, dtype=dtype)
                                        
                                        down_block_res_samples, mid_block_res_sample = controlnet(
                                            x_hat_t.to(dtype),
                                            timesteps,
                                            encoder_hidden_states=combination_emb.to(dtype),
                                            controlnet_cond=control_tensor,
                                            return_dict=False,
                                        )
                                        
                                        down_block_res_samples = [
                                            sample * opt.controlnet_scale for sample in down_block_res_samples
                                        ]
                                        mid_block_res_sample *= opt.controlnet_scale
                                        
                                        noise_pred = unet(
                                            x_hat_t.to(dtype),
                                            timesteps,
                                            encoder_hidden_states=combination_emb.to(dtype),
                                            down_block_additional_residuals=down_block_res_samples,
                                            mid_block_additional_residual=mid_block_res_sample,
                                        ).sample

                                    # Predict x_0
                                    x_0 = (x_hat_t - (1 - at) ** 0.5 * noise_pred) / at ** 0.5
                                    
                                    # Downscale with bicubic
                                    x_0_scaled = torch.nn.functional.interpolate(
                                        x_0, (size, size), mode='bicubic'
                                    )

                                    # Decode to image space - CRITICAL: No torch.no_grad() here!
                                    image = vae.decode(1 / 0.18215 * x_0_scaled.to(dtype)).sample
                                    image = (image / 2 + 0.5).clamp(0, 1)
                                    pixels_gen = image.squeeze(0).reshape(3, -1)

                                    # Compute statistics
                                    gen_mean = torch.mean(pixels_gen, dim=1)
                                    gen_cov = torch.cov(pixels_gen)

                                    # Sample reference pixels
                                    rand_idxes = np.random.randint(0, pixels_ref.shape[1], pixels_gen.shape[1])

                                    # Compute loss based on type
                                    if opt.sw_loss_type == 0:  # mean_cov
                                        loss = (torch.mean(torch.square(gen_mean - ref_mean)) + 
                                            torch.mean(torch.square(gen_cov - ref_cov)))
                                    elif opt.sw_loss_type == 1:  # SWD
                                        loss = sliced_wasserstein_distance(
                                            pixels_gen.T, pixels_ref[:, rand_idxes].T,
                                            device=opt.device, 
                                            num_projections=opt.sw_num_projections, 
                                            p=opt.sw_wsd_p
                                        )
                                    elif opt.sw_loss_type == 2:  # DSWD
                                        loss = distributional_sliced_wasserstein_distance(
                                            pixels_gen.T, pixels_ref[:, rand_idxes].T,
                                            f=transform_net, f_op=op_trannet,
                                            device=opt.device, 
                                            num_projections=opt.sw_num_projections, 
                                            p=opt.sw_wsd_p,
                                        )
                                    elif opt.sw_loss_type == 3:  # GSWD
                                        loss = gsw(
                                            pixels_gen.T, pixels_ref[:, rand_idxes].T,
                                            ftype=opt.sw_ftype, degree=opt.sw_degree,
                                            device=opt.device, 
                                            num_projections=opt.sw_num_projections, 
                                            p=opt.sw_wsd_p,
                                        )
                                    elif opt.sw_loss_type == 4:  # ISEBSW
                                        loss = ISEBSW(
                                            pixels_gen.T, pixels_ref[:, rand_idxes].T,
                                            device=opt.device, 
                                            num_projections=opt.sw_num_projections, 
                                            p=opt.sw_wsd_p,
                                        )

                                    # Compute gradients with retain_graph for all but last iteration
                                    u_t_grad = torch.autograd.grad(
                                        loss, u, 
                                        retain_graph=(m_iter < opt.sw_M - 1)
                                    )[0]
                                    
                                    with torch.no_grad():
                                        # Normalize gradient by std
                                        u_t_grad = u_t_grad / (u_t_grad.std() + 1e-8)
                                        
                                        if opt.sw_use_sarah:
                                            if i == 0:
                                                v.data = u_t_grad
                                            else:
                                                v.data = v.data + u_t_grad - buf_grad
                                            buf_grad = u_t_grad.clone()
                                            u.data = u.data - opt.sw_u_lr * v.data
                                        else:
                                            u.data = u.data - opt.sw_u_lr * u_t_grad.data
                                    
                                    # Print loss for debugging
                                    if m_iter == 0 and i % 5 == 0:
                                        print(f"  Step {i}, Loss: {loss.item():.6f}")

                            # Denoising step with CFG (separate from guidance)
                            with torch.no_grad():
                                x_star_t = x_t.detach() + (u.detach() if do_guidance else 0)
                                
                                # Predict noise with CFG
                                noise_pred_cond = unet(
                                    x_star_t.to(dtype), 
                                    timesteps, 
                                    encoder_hidden_states=combination_emb.to(dtype)
                                ).sample
                                noise_pred_uncond = unet(
                                    x_star_t.to(dtype), 
                                    timesteps, 
                                    encoder_hidden_states=uncond_embeddings
                                ).sample
                                
                                # Apply ControlNet to conditional prediction
                                if use_controlnet and control_map is not None:
                                    control_tensor = torch.from_numpy(np.array(control_map)).float() / 255.0
                                    control_tensor = control_tensor.permute(2, 0, 1).unsqueeze(0).to(opt.device, dtype=dtype)
                                    
                                    down_block_res_samples, mid_block_res_sample = controlnet(
                                        x_star_t.to(dtype), timesteps,
                                        encoder_hidden_states=combination_emb.to(dtype),
                                        controlnet_cond=control_tensor,
                                        return_dict=False,
                                    )
                                    
                                    down_block_res_samples = [s * opt.controlnet_scale for s in down_block_res_samples]
                                    mid_block_res_sample *= opt.controlnet_scale
                                    
                                    noise_pred_cond = unet(
                                        x_star_t.to(dtype), timesteps,
                                        encoder_hidden_states=combination_emb.to(dtype),
                                        down_block_additional_residuals=down_block_res_samples,
                                        mid_block_additional_residual=mid_block_res_sample,
                                    ).sample

                                # Apply CFG
                                guided_noise_pred = noise_pred_uncond + opt.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                                timesteps_next = scheduler.timesteps[j]
                                at_next = scheduler.alphas_cumprod[timesteps_next] if timesteps_next >= 0 else scheduler.final_alpha_cumprod

                                x_0 = (x_star_t - (1 - at) ** 0.5 * guided_noise_pred) / at ** 0.5
                                x_t.data = at_next ** 0.5 * x_0 + (1 - at_next) ** 0.5 * guided_noise_pred

                            x0_pred_last = x_0.to('cpu')

                            # Save intermediate
                            if opt.save_intermediate:
                                step_img = decode_to_image(x0_pred_last.to(opt.device))
                                step_img.save(f"{opt.outdir}/steps/{folder}_ctrl{ci}_iter{n}_step{i:02d}.png")

                            # Scale SARAH buffers AFTER the denoising step
                            if opt.sw_use_sarah and do_guidance:
                                buf_grad.data *= opt.sw_u_lr
                                v.data *= opt.sw_u_lr
                            
                            # Clean up
                            del u
                            # if do_guidance:
                            #     del x_hat_t, noise_pred, x_0, x_0_scaled, image, pixels_gen, loss
                            gc.collect()
                            torch.cuda.empty_cache()

                        else:
                            # Backward step (time travel)
                            print("--> Time travel backward step")
                            j = scheduler.num_inference_steps - j
                            timesteps_next = scheduler.timesteps[j]
                            at_next = scheduler.alphas_cumprod[timesteps_next]
                            x0_t = x0_pred_last.to(opt.device)
                            x_t = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                # Save final image
                final_img = decode_to_image(x0_pred_last.to(opt.device))
                final_img.save(os.path.join(sample_path, f"{base_count:05}_ctrl{ci}.png"))
                base_count += 1

                gc.collect()
                torch.cuda.empty_cache()

    print(f"\nGeneration complete! Images saved to: {opt.outdir}")


if __name__ == "__main__":
    main()