# import os
# from pathlib import Path
# from concurrent.futures import ProcessPoolExecutor, Future
# import argparse
# from typing import List, Tuple

# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# from tqdm import tqdm
# from diffusers import AutoencoderKL
# import numpy as np
# import cv2

# # ---------------- Worker Function ----------------
# def save_reconstructed_image_cv2(recon_tensor: torch.Tensor, orig_size: Tuple[int, int], save_path: str):
#     """
#     Convert VAE output tensor to NumPy, resize using OpenCV, and save as PNG with minimal compression.
#     """
#     recon_np = (recon_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
#     recon_resized = cv2.resize(recon_np, dsize=orig_size, interpolation=cv2.INTER_CUBIC)
#     cv2.imwrite(str(save_path), cv2.cvtColor(recon_resized, cv2.COLOR_RGB2BGR))
#     return save_path

# # ---------------- Dataset & Collate ----------------
# class ImageDataset(Dataset):
#     """Dataset for images of arbitrary size, resized for VAE input."""
#     def __init__(self, image_dir, output_dir, fixed_size=(512, 512), extensions=(".jpg", ".jpeg", ".png")):
#         self.fixed_size = fixed_size
#         self.image_dir = Path(image_dir)
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(parents=True, exist_ok=True)

#         self.image_paths = [p for p in sorted(self.image_dir.iterdir()) if p.suffix.lower() in extensions]
#         if not self.image_paths:
#             raise ValueError(f"No images found in {image_dir}")

#         self.transform = transforms.Compose([
#             transforms.Resize(self.fixed_size, interpolation=Image.BICUBIC),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5], [0.5])
#         ])

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         path = self.image_paths[idx]
#         img = Image.open(path).convert("RGB")
#         orig_size = img.size  # (width, height)
#         img_tensor = self.transform(img)
#         save_path = self.output_dir / f"{path.stem}_vae.png"
#         return img_tensor, orig_size, save_path

# def collate_batch(batch):
#     imgs, orig_sizes, save_paths = zip(*batch)
#     imgs_tensor = torch.stack(imgs, dim=0)
#     return imgs_tensor, list(orig_sizes), list(save_paths)

# # ---------------- Main VAE Pipeline ----------------
# def process_images_with_vae(
#     image_dir, output_dir, batch_size=16, fixed_size=(512, 512),
#     checkpoint_batches=100, resume=True, max_workers=None
# ):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device.upper()}")

#     # Load VAE with half precision
#     print("Loading Stable Diffusion VAE...")
#     vae = AutoencoderKL.from_pretrained(
#         "stabilityai/sd-vae-ft-mse", 
#         torch_dtype=torch.float16,
#         load_in_8bit=True).to(device)
#     # vae = AutoencoderKL.from_pretrained(
#     #     "madebyollin/sdxl-vae-fp16-fix", 
#     #     torch_dtype=torch.float16,
#     #     device_map="auto", 
#     #     load_in_8bit=True).to(device)
#     vae.eval()

#     dataset = ImageDataset(image_dir, output_dir, fixed_size)

#     # Filter already processed images
#     if resume:
#         processed = {p.stem.replace("_vae", "") for p in Path(output_dir).glob("*_vae.png")}
#         dataset.image_paths = [p for p in dataset.image_paths if p.stem not in processed]
#         print(f"Resuming: {len(dataset.image_paths)} images left to process")

#     if not dataset.image_paths:
#         print("All images already processed. Exiting.")
#         return

#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
#                         num_workers=max(1, os.cpu_count()//2), pin_memory=True,
#                         collate_fn=collate_batch)

#     all_futures: List[Future] = []
#     all_processed = 0
#     max_workers = max_workers or os.cpu_count()

#     # Use ProcessPoolExecutor for CPU-heavy resizing/saving
#     with ProcessPoolExecutor(max_workers=max_workers) as executor, torch.no_grad():

#         for batch_idx, (imgs_tensor, orig_sizes, save_paths) in enumerate(tqdm(loader, desc="VAE Reconstruction")):

#             imgs_tensor = imgs_tensor.to(device, non_blocking=True, dtype=torch.float16)

#             # ---------------- Mixed Precision ----------------
#             with torch.amp.autocast(device_type=device):
#                 latents = vae.encode(imgs_tensor).latent_dist.sample() * 0.18215
#                 recons = vae.decode(latents / 0.18215).sample
#                 recons = ((recons.clamp(-1, 1) + 1) / 2)  # [0,1]

#             # Submit CPU post-processing tasks asynchronously
#             for recon_tensor, orig_size, save_path in zip(recons, orig_sizes, save_paths):
#                 if save_path.exists() and resume:
#                     continue
#                 future = executor.submit(save_reconstructed_image_cv2, recon_tensor, orig_size, save_path)
#                 all_futures.append(future)
#                 all_processed += 1

#             # Periodic checkpoint: limit pending tasks to avoid memory overload
#             if (batch_idx + 1) % checkpoint_batches == 0 and all_futures:
#                 print(f"Checkpoint: waiting for {len(all_futures)} save tasks to finish...")
#                 for future in tqdm(all_futures, desc=f"Saving checkpoint (Batch {batch_idx+1})"):
#                     future.result()
#                 all_futures = []

#         # Wait for any remaining saves
#         if all_futures:
#             print(f"Final wait: {len(all_futures)} remaining save tasks...")
#             for future in tqdm(all_futures, desc="Saving final results"):
#                 future.result()

#     print(f"✅ Completed processing {all_processed} images. Output saved in {output_dir}")

# # ---------------- CLI ----------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Optimized Batch VAE Reconstruction")
#     parser.add_argument("--image_dir", type=str, required=True)
#     parser.add_argument("--output_dir", type=str, required=True)
#     parser.add_argument("--batch_size", type=int, default=16)
#     parser.add_argument("--checkpoint_batches", type=int, default=100)
#     parser.add_argument("--resume", type=bool, default=True)
#     parser.add_argument("--fixed_size", type=int, nargs=2, default=[256, 256])
#     args = parser.parse_args()

#     process_images_with_vae(
#         image_dir=args.image_dir,
#         output_dir=args.output_dir,
#         batch_size=args.batch_size,
#         checkpoint_batches=args.checkpoint_batches,
#         resume=args.resume,
#         fixed_size=tuple(args.fixed_size)
#     )


# import os
# from pathlib import Path
# from concurrent.futures import ProcessPoolExecutor, Future
# import argparse
# from typing import List, Tuple
# import random

# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image, ImageFile, PngImagePlugin
# from tqdm import tqdm
# from diffusers import AutoencoderKL
# import numpy as np
# import cv2

# # ---------------- Robust PIL ----------------
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# Image.LOAD_TRUNCATED_IMAGES = True
# PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)

# # ---------------- Worker Function ----------------
# def save_reconstructed_image_cv2(recon_tensor: torch.Tensor, orig_size: Tuple[int, int], save_path: str):
#     """
#     Convert VAE output tensor to NumPy, resize using OpenCV, and save as PNG with minimal compression.
#     """
#     recon_np = (recon_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
#     recon_resized = cv2.resize(recon_np, dsize=orig_size, interpolation=cv2.INTER_CUBIC)
#     cv2.imwrite(str(save_path), cv2.cvtColor(recon_resized, cv2.COLOR_RGB2BGR))
#     return save_path

# # ---------------- Dataset & Collate ----------------
# class ImageDataset(Dataset):
#     """Dataset for images of arbitrary size, resized for VAE input with aspect ratio preservation."""
#     def __init__(self, image_dir, output_dir, final_size=(256, 256), preprocess_size=500,
#                  extensions=(".jpg", ".jpeg", ".png")):
#         self.final_size = final_size
#         self.preprocess_size = preprocess_size
#         self.image_dir = Path(image_dir)
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(parents=True, exist_ok=True)

#         self.image_paths = [p for p in sorted(self.image_dir.iterdir()) if p.suffix.lower() in extensions]
#         if not self.image_paths:
#             raise ValueError(f"No images found in {image_dir}")

#         self.preprocess = transforms.Resize(preprocess_size, interpolation=Image.BICUBIC)
#         self.transform = transforms.Compose([
#             transforms.Resize(final_size, interpolation=Image.BICUBIC),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5], [0.5])
#         ])

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         path = self.image_paths[idx]
#         save_path = self.output_dir / f"{path.stem}_vae.png"
#         try:
#             img = Image.open(path).convert("RGB")
#             orig_size = img.size

#             # Conditional preprocessing: resize large images while preserving aspect ratio
#             if min(img.size) > self.preprocess_size:
#                 img = self.preprocess(img)

#             # Final fixed-size transform for VAE input
#             img_tensor = self.transform(img)
#             return img_tensor, orig_size, save_path
#         except:
#             # Return None if image is corrupted
#             return None, None, save_path

# def collate_batch(batch):
#     # Filter out failed images
#     batch = [b for b in batch if b[0] is not None]
#     if not batch:
#         return None, None, None
#     imgs, orig_sizes, save_paths = zip(*batch)
#     imgs_tensor = torch.stack(imgs, dim=0)
#     return imgs_tensor, list(orig_sizes), list(save_paths)

# # ---------------- Main VAE Pipeline ----------------
# def process_images_with_vae(
#     image_dir, output_dir, batch_size=16, final_size=(256, 256), preprocess_size=500,
#     checkpoint_batches=100, resume=True, max_workers=None
# ):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device.upper()}")

#     # ---------------- Load VAE ----------------
#     print("Loading Stable Diffusion VAE (FP16 + 8-bit)...")
#     vae = AutoencoderKL.from_pretrained(
#         "stabilityai/sd-vae-ft-mse",
#         torch_dtype=torch.float16,
#         load_in_8bit=True
#     ).to(device)
#     vae.eval()

#     # ---------------- Dataset ----------------
#     dataset = ImageDataset(image_dir, output_dir, final_size, preprocess_size)

#     # Filter already processed images
#     if resume:
#         processed = {p.stem.replace("_vae", "") for p in Path(output_dir).glob("*_vae.png")}
#         dataset.image_paths = [p for p in dataset.image_paths if p.stem not in processed]
#         print(f"Resuming: {len(dataset.image_paths)} images left to process")

#     if not dataset.image_paths:
#         print("All images already processed. Exiting.")
#         return

#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
#                         num_workers=max(1, os.cpu_count()//2), pin_memory=True,
#                         collate_fn=collate_batch)

#     all_futures: List[Future] = []
#     all_processed = 0
#     max_workers = max_workers or os.cpu_count()

#     # ---------------- ProcessPoolExecutor for CPU-heavy saving ----------------
#     with ProcessPoolExecutor(max_workers=max_workers) as executor, torch.no_grad():

#         for batch_idx, (imgs_tensor, orig_sizes, save_paths) in enumerate(tqdm(loader, desc="VAE Reconstruction")):

#             if imgs_tensor is None:
#                 continue

#             imgs_tensor = imgs_tensor.to(device, non_blocking=True, dtype=torch.float16)

#             # ---------------- Automatic Mixed Precision ----------------
#             with torch.amp.autocast(device_type=device):
#                 latents = vae.encode(imgs_tensor).latent_dist.sample() * 0.18215
#                 recons = vae.decode(latents / 0.18215).sample
#                 recons = ((recons.clamp(-1, 1) + 1) / 2)  # [0,1]

#             # ---------------- CPU-side async saving ----------------
#             for recon_tensor, orig_size, save_path in zip(recons, orig_sizes, save_paths):
#                 if save_path.exists() and resume:
#                     continue
#                 future = executor.submit(save_reconstructed_image_cv2, recon_tensor, orig_size, save_path)
#                 all_futures.append(future)
#                 all_processed += 1

#             # ---------------- Periodic checkpoint ----------------
#             if (batch_idx + 1) % checkpoint_batches == 0 and all_futures:
#                 print(f"Checkpoint: waiting for {len(all_futures)} save tasks to finish...")
#                 for future in tqdm(all_futures, desc=f"Saving checkpoint (Batch {batch_idx+1})"):
#                     future.result()
#                 all_futures = []

#         # ---------------- Final save wait ----------------
#         if all_futures:
#             print(f"Final wait: {len(all_futures)} remaining save tasks...")
#             for future in tqdm(all_futures, desc="Saving final results"):
#                 future.result()

#     print(f"✅ Completed processing {all_processed} images. Output saved in {output_dir}")

# # ---------------- CLI ----------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Highly Optimized VAE Reconstruction Pipeline")
#     parser.add_argument("--image_dir", type=str, required=True)
#     parser.add_argument("--output_dir", type=str, required=True)
#     parser.add_argument("--batch_size", type=int, default=16)
#     parser.add_argument("--checkpoint_batches", type=int, default=100)
#     parser.add_argument("--resume", type=bool, default=True)
#     parser.add_argument("--final_size", type=int, nargs=2, default=[256, 256])
#     parser.add_argument("--preprocess_size", type=int, default=500)
#     parser.add_argument("--max_workers", type=int, default=None)
#     args = parser.parse_args()

#     process_images_with_vae(
#         image_dir=args.image_dir,
#         output_dir=args.output_dir,
#         batch_size=args.batch_size,
#         checkpoint_batches=args.checkpoint_batches,
#         resume=args.resume,
#         final_size=tuple(args.final_size),
#         preprocess_size=args.preprocess_size,
#         max_workers=args.max_workers
#     )


import os

# Required for torch.model.compile
msvc_path = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
python_lib = r"C:\Users\User\AppData\Local\Programs\Python\Python310\libs"
python_include = r"C:\Users\User\AppData\Local\Programs\Python\Python310\Include"
os.environ["PATH"] = msvc_path + os.pathsep + os.environ["PATH"]
os.environ["LIB"] = python_lib + os.pathsep + os.environ.get("LIB", "")
os.environ["INCLUDE"] = python_include + os.pathsep + os.environ.get("INCLUDE", "")


from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, Future
import argparse
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
import sys
import subprocess

# # ---------------- Auto-Clone latent-diffusion if Missing ----------------
# REPO_NAME = "latent_diffusion"
# REPO_URL = "https://github.com/CompVis/latent-diffusion.git"

# # Determine base directory (e.g., "Spurious_Feature/")
# BASE_DIR = Path(__file__).resolve().parent
# REPO_DIR = BASE_DIR.parent / REPO_NAME

# if not REPO_DIR.exists():
#     print(f"⚙️  '{REPO_NAME}' not found — cloning from {REPO_URL} ...")
#     try:
#         subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
#         print("✅ Successfully cloned latent-diffusion repository.")
#     except subprocess.CalledProcessError as e:
#         print("❌ Failed to clone latent-diffusion. Please check your internet or git installation.")
#         raise e

# # # Add the repo to Python path
# # sys.path.append(str(REPO_DIR.parent))

# # Add the *parent directory* to sys.path (so 'latent_diffusion' can be imported)
# if str(REPO_DIR.parent) not in sys.path:
#     sys.path.insert(0, str(REPO_DIR.parent))

# # Add also the latent_diffusion subfolder itself to handle "ldm" imports
# if str(REPO_DIR) not in sys.path:
#     sys.path.insert(0, str(REPO_DIR))


import sys
import subprocess
from pathlib import Path
import torch.nn.functional as F

# --- latent-diffusion repo ---
LD_REPO_NAME = "latent_diffusion"
LD_REPO_URL = "https://github.com/CompVis/latent-diffusion.git"

# --- taming-transformers repo ---
TAMING_REPO_NAME = "taming-transformers"
TAMING_REPO_URL = "https://github.com/CompVis/taming-transformers.git"

# Determine base directory (e.g., project root)
BASE_DIR = Path(__file__).resolve().parent

# --- latent-diffusion ---
LD_REPO_DIR = BASE_DIR.parent / LD_REPO_NAME
if not LD_REPO_DIR.exists():
    print(f"⚙️  '{LD_REPO_NAME}' not found — cloning from {LD_REPO_URL} ...")
    try:
        subprocess.run(["git", "clone", LD_REPO_URL, str(LD_REPO_DIR)], check=True)
        print(f"✅ Successfully cloned {LD_REPO_NAME}.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to clone {LD_REPO_NAME}.")
        raise e

# Add latent-diffusion to sys.path
if str(LD_REPO_DIR.parent) not in sys.path:
    sys.path.insert(0, str(LD_REPO_DIR.parent))
if str(LD_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(LD_REPO_DIR))

# --- taming-transformers ---
TAMING_REPO_DIR = BASE_DIR.parent / TAMING_REPO_NAME
if not TAMING_REPO_DIR.exists():
    print(f"⚙️  '{TAMING_REPO_NAME}' not found — cloning from {TAMING_REPO_URL} ...")
    try:
        subprocess.run(["git", "clone", TAMING_REPO_URL, str(TAMING_REPO_DIR)], check=True)
        print(f"✅ Successfully cloned {TAMING_REPO_NAME}.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to clone {TAMING_REPO_NAME}.")
        raise e

# Add taming-transformers to sys.path
if str(TAMING_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(TAMING_REPO_DIR))

from omegaconf import OmegaConf
from latent_diffusion.ldm.util import instantiate_from_config

# ---------------- Worker Function ----------------
def save_reconstructed_image_cv2(recon_tensor: torch.Tensor, orig_size: Tuple[int, int], save_path: str):
    """
    Convert VAE output tensor to NumPy, resize using OpenCV, and save as PNG with minimal compression.
    """
    # recon_np = (recon_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    # recon_resized = cv2.resize(recon_np, dsize=orig_size, interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite(str(save_path), cv2.cvtColor(recon_resized, cv2.COLOR_RGB2BGR))

    recon_tensor = F.interpolate(recon_tensor.unsqueeze(0), size=orig_size[::-1], mode='bicubic', align_corners=False).squeeze(0)
    recon_np = (recon_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    cv2.imwrite(str(save_path), cv2.cvtColor(recon_np, cv2.COLOR_RGB2BGR))
    return save_path

# ---------------- Dataset & Collate ----------------
class ImageDataset(Dataset):
    """Dataset for images of arbitrary size, resized for VAE input."""
    def __init__(self, image_dir, output_dir, fixed_size=(256, 256), extensions=(".jpg", ".jpeg", ".png")):
        self.fixed_size = fixed_size
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.image_paths = [p for p in sorted(self.image_dir.iterdir()) if p.suffix.lower() in extensions]
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")

        self.transform = transforms.Compose([
            transforms.Resize(self.fixed_size, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        orig_size = img.size  # (width, height)
        img_tensor = self.transform(img)
        img_tensor = (img_tensor - 0.5) * 2  # normalize to [-1, 1]
        save_path = self.output_dir / f"{path.stem}_vae.png"
        return img_tensor, orig_size, save_path


def collate_batch(batch):
    imgs, orig_sizes, save_paths = zip(*batch)
    imgs_tensor = torch.stack(imgs, dim=0)
    return imgs_tensor, list(orig_sizes), list(save_paths)

# ---------------- Main VAE Pipeline ----------------
def process_images_with_ldm_vae(
    image_dir, output_dir, batch_size=16, fixed_size=(256, 256),
    checkpoint_batches=100, resume=True, max_workers=None,
    config_path="autoencoder_kl_64x64x3.yaml", ckpt_path="model.ckpt"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    # ---------------- Load CompVis VAE ----------------
    print(f"Loading VAE from {config_path} and {ckpt_path}...")
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)

    model = model.to(device)
    model = torch.compile(model, mode="max-autotune")
    model.eval()
    print("VAE loaded successfully")

    dataset = ImageDataset(image_dir, output_dir, fixed_size)

    # ---------------- Resume mode ----------------
    if resume:
        processed = {p.stem.replace("_vae", "") for p in Path(output_dir).glob("*_vae.png")}
        dataset.image_paths = [p for p in dataset.image_paths if p.stem not in processed]
        print(f"Resuming: {len(dataset.image_paths)} images left to process")

    if not dataset.image_paths:
        print("All images already processed. Exiting.")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=max(1, os.cpu_count() // 2), pin_memory=True,
                        collate_fn=collate_batch, prefetch_factor=4)

    all_futures: List[Future] = []
    all_processed = 0
    max_workers = max_workers or os.cpu_count()

    # ---------------- Reconstruction Loop ----------------
    # with ProcessPoolExecutor(max_workers=max_workers) as executor, torch.no_grad(), torch.amp.autocast(device_type=device):
    with ProcessPoolExecutor(max_workers=max_workers) as executor, torch.inference_mode(), torch.amp.autocast(device_type=device):

        for batch_idx, (imgs_tensor, orig_sizes, save_paths) in enumerate(tqdm(loader, desc="VAE Reconstruction")):
            imgs_tensor = imgs_tensor.to(device, non_blocking=True)

            # Run encoder-decoder pass
            output, _ = model(imgs_tensor, sample_posterior=False)
            recons = torch.clamp((output / 2) + 0.5, 0, 1)  # map to [0,1]

            # Async CPU saves
            for recon_tensor, orig_size, save_path in zip(recons, orig_sizes, save_paths):
                if save_path.exists() and resume:
                    continue
                future = executor.submit(save_reconstructed_image_cv2, recon_tensor, orig_size, save_path)
                all_futures.append(future)
                all_processed += 1

            # Periodic checkpointing
            if (batch_idx + 1) % checkpoint_batches == 0 and all_futures:
                print(f"Checkpoint: waiting for {len(all_futures)} save tasks to finish...")
                for future in tqdm(all_futures, desc=f"Saving checkpoint (Batch {batch_idx+1})"):
                    future.result()
                all_futures = []

        # Wait for all remaining save tasks
        if all_futures:
            print(f"Final wait: {len(all_futures)} remaining save tasks...")
            for future in tqdm(all_futures, desc="Saving final results"):
                future.result()

    print(f"Completed processing {all_processed} images. Output saved in {output_dir}")


# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized Batch VAE Reconstruction using CompVis Latent Diffusion")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--checkpoint_batches", type=int, default=100)
    parser.add_argument("--resume", type=bool, default=True)
    parser.add_argument("--fixed_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--config_path", type=str, default=r"C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\VAE\autoencoder_kl_64x64x3.yaml")
    parser.add_argument("--ckpt_path", type=str, default=r"C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\VAE\model.ckpt")
    args = parser.parse_args()

    process_images_with_ldm_vae(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        checkpoint_batches=args.checkpoint_batches,
        resume=args.resume,
        fixed_size=tuple(args.fixed_size),
        config_path=args.config_path,
        ckpt_path=args.ckpt_path,
    )

# "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\.venv-Copy-Copy\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\VAE\vae.py" --image_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ImageCaptioningEvaluationDatasets\LAION-5B-10k\LAION-5B-10k-images" --output_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\VAE\LAION-5B-10k-vae" --batch_size 4 --checkpoint_batches 100

# "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\.venv-Copy-Copy\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\VAE\vae.py" --image_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ImageCaptioningEvaluationDatasets\LAION-5B-10k\LAION-5B-10k-images" --output_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\VAE\LAION-5B-10k-vae" --batch_size 4 --checkpoint_batches 100


# "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\.venv-Copy-Copy\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\VAE\vae.py" --image_dir "G:\Thesis\ImageRetrieval\Professions_125k_Cleaned" --output_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\VAE\test" --batch_size 4 --checkpoint_batches 100