import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import os
import uuid
import queue
import threading
import argparse
import numpy as np
import cv2
import torch
import csv
from diffusers import DiffusionPipeline, LCMScheduler
from ultralytics import YOLO
from PIL import Image
import open_clip
import torch.nn as nn
from tqdm import tqdm
import json
import concurrent.futures
from collections import deque

# ============================================================
# GEOMETRIC UTILITIES
# ============================================================

def load_prompt_config(path: str):
    """
    Load prompt configuration from a JSON file.
    
    Args:
        path: Path to the JSON configuration file
        
    Returns:
        tuple: (professions list, prompt_template string, negative_prompt string or None)
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    professions = cfg["professions"]
    prompt_template = cfg["prompt_template"]
    negative_prompt = cfg.get("negative_prompt", None)

    if negative_prompt in ("", "none", "None"):
        negative_prompt = None

    return professions, prompt_template, negative_prompt

def generate_without_validation(
    prompt,
    negative_prompt,
    label,
    total_images,
    batch_size,
    valid_dir,
    csv_path,
    guidance_scale,
    num_inference_steps
):
    """
    Generate images with async validation pipeline and save valid results.
    
    This function generates images in batches, validates them asynchronously,
    computes aesthetic scores, and saves valid images with metadata.
    
    Args:
        prompt: Text prompt for image generation
        negative_prompt: Negative prompt to avoid unwanted features
        label: Class label (e.g., profession name)
        total_images: Target number of valid images to generate
        batch_size: Number of images to generate per batch
        valid_dir: Directory to save valid images
        invalid_dir: Directory to save invalid images (or None)
        validator: ValidationWorker instance
        args: Generation and validation parameters
        csv_path: Path to CSV file for metadata logging
    """

    label_dir = os.path.join(valid_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    init_csv(csv_path)

    existing = count_existing_valid(valid_dir, label)
    if existing >= total_images:
        print(f"[INFO] {label}: already has {existing} images. Skipping.")
        return

    saved_count = existing
    img_idx = existing

    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    io_futures = []

    pbar = tqdm(total=total_images, initial=saved_count, desc=label, dynamic_ncols=True)

    try:
        while saved_count < total_images:
            images = pipe(
                prompt=[prompt] * batch_size,
                negative_prompt=[negative_prompt] * batch_size if negative_prompt else [""] * batch_size,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images

            for img in images:
                if saved_count >= total_images:
                    break
                uid = f"{label}_{img_idx:06d}_{uuid.uuid4().hex[:6]}"
                img_rgb = np.array(img)
                img_path = os.path.join(label_dir, f"{uid}.png")
                future = io_executor.submit(save_image_async, img_rgb, img_path)
                io_futures.append(future)

                csv_writer.writerow([f"{uid}.png", label, None, None, None])
                saved_count += 1
                img_idx += 1
                pbar.update(1)

            if len(io_futures) > 20:
                io_futures = [f for f in io_futures if not f.done()]

    finally:
        concurrent.futures.wait(io_futures)
        io_executor.shutdown(wait=True)
        csv_file.close()
        pbar.close()

    print(f"[INFO] Completed {label}: {saved_count} images")

def save_image_async(img_rgb, path):
    """
    Save image in background thread to avoid blocking the main pipeline.
    
    Args:
        img_rgb: Image array in RGB format
        path: Output file path
    """
    Image.fromarray(img_rgb).save(path)

def count_existing_valid(valid_dir, label):
    """
    Count existing valid images for a given label.
    
    Args:
        valid_dir: Directory containing valid images
        label: Class label subdirectory
        
    Returns:
        int: Number of existing PNG images
    """
    if not valid_dir:
        return 0
    label_dir = os.path.join(valid_dir, label)
    if not os.path.isdir(label_dir):
        return 0
    return len([f for f in os.listdir(label_dir) if f.lower().endswith(".png")])

def init_csv(csv_path):
    """
    Initialize CSV file with headers if it doesn't exist.
    
    Args:
        csv_path: Path to the CSV metadata file
    """
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "image_name",
                "label",
                "aesthetic_score",
                "face_box",
                "person_box"
            ])


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image generation + validation pipeline")

    # Core paths
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Base output directory (valid/, invalid/, CSV will be created inside)")
    
    parser.add_argument("--prompt_config", type=str, required=True,
                        help="Path to JSON file defining professions, prompt template, and negative prompt")

    # Generation controls
    parser.add_argument("--total_images_per_label", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=0.0)

    # Runtime
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Determinism
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Derived paths
    output_dir = os.path.abspath(args.output_dir)
    valid_dir = os.path.join(output_dir, "valid")
    invalid_dir = os.path.join(output_dir, "invalid")
    csv_path = os.path.join(output_dir, "ImageGenMetadata.csv")

    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(invalid_dir, exist_ok=True)

    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Valid images → {valid_dir}")
    print(f"[INFO] Invalid images → {invalid_dir}")
    print(f"[INFO] CSV metadata → {csv_path}")

    # Diffusion pipeline
    pipe = DiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    pipe.fuse_lora()
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.to(args.device)

    pipe.set_progress_bar_config(disable=True)

    # CLIP + Aesthetic model
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        model_name="ViT-L-14",
        pretrained="laion2b_s32b_b82k",
        device=args.device
    )
    clip_model.eval()

    # Prompt configuration
    profession_list, prompt_template, negative_prompts = load_prompt_config(args.prompt_config)

    print(f"[INFO] Loaded {len(profession_list)} professions from {args.prompt_config}")

    # Run generation for all professions
    for index, label in enumerate(profession_list):
        prompt = prompt_template.format(profession=label)
        print(f"\n[INFO] Processing label {index}='{label}'")

        generate_without_validation(
            prompt=prompt,
            negative_prompt=negative_prompts,
            label=label,
            total_images=args.total_images_per_label,
            batch_size=args.batch_size,
            valid_dir=valid_dir,
            csv_path=csv_path,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps
        )

# .\.venv\Scripts\python.exe ImageGeneration_NoValidation.py --output_dir "F:\ImageRetrieval\StableDiffusion_Random10k" --total_images_per_label 10_000 --batch_size 2 --prompt_config "unconditional_prompts.json" --device "cuda" 