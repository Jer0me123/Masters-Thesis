import os
import sys
import glob
import json
import argparse
import subprocess
import time
from pathlib import Path
from typing import Tuple
from threading import Thread, Lock
from queue import Queue

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image

torch.backends.cudnn.benchmark = True

# ============================================================
# REPO BOOTSTRAP (latent-diffusion + taming-transformers)
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

LD_REPO_DIR = BASE_DIR / "latent_diffusion"
TAMING_REPO_DIR = BASE_DIR / "taming-transformers"

for repo, url in [
    (LD_REPO_DIR, "https://github.com/CompVis/latent-diffusion.git"),
    (TAMING_REPO_DIR, "https://github.com/CompVis/taming-transformers.git"),
]:
    if not repo.exists():
        subprocess.run(["git", "clone", url, str(repo)], check=True)

sys.path.insert(0, str(LD_REPO_DIR.parent))
sys.path.insert(0, str(LD_REPO_DIR))
sys.path.insert(0, str(TAMING_REPO_DIR))

# ============================================================
# FAST RGB DECODER
# ============================================================
class FastRGBDecoder:
    def __init__(self):
        try:
            from turbojpeg import TurboJPEG
            self.jpeg = TurboJPEG(r"C:\libjpeg-turbo-gcc64\bin\libturbojpeg.dll")
            self.use_turbo = True
        except Exception:
            self.use_turbo = False

    def load(self, path: str) -> np.ndarray | None:
        try:
            if self.use_turbo and path.lower().endswith((".jpg", ".jpeg")):
                with open(path, "rb") as f:
                    return self.jpeg.decode(f.read())
            arr = cv2.imread(path)
            return None if arr is None else arr[..., ::-1]
        except Exception:
            return None

# ============================================================
# ASYNC IMAGE WRITER + MANIFEST
# ============================================================
class AsyncWriter:
    def __init__(self, output_dir: Path, num_workers=4):
        self.queue = Queue(maxsize=8192)
        self.stop = object()
        self.workers = []
        self.manifest_path = output_dir / "vae_manifest.jsonl"
        self.lock = Lock()

        for _ in range(num_workers):
            t = Thread(target=self._worker, daemon=True)
            t.start()
            self.workers.append(t)

    def _worker(self):
        while True:
            item = self.queue.get()
            if item is self.stop:
                break
            path, arr, rel = item
            os.makedirs(os.path.dirname(path), exist_ok=True)

            if arr.ndim == 3 and arr.shape[2] == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

            cv2.imwrite(path, arr)
            with self.lock:
                with open(self.manifest_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"image": rel}) + "\n")
            self.queue.task_done()

    def submit(self, path: Path, arr: np.ndarray, rel: str):
        self.queue.put((str(path), arr, rel))

    def close(self):
        self.queue.join()
        for _ in self.workers:
            self.queue.put(self.stop)

# ============================================================
# DATASET (JSONL-BASED RESUME)
# ============================================================
class ImageDataset(Dataset):
    def __init__(self, image_dir, output_dir, exclude_dirs, fixed_size):
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.exclude_dirs = {d.lower() for d in exclude_dirs}
        self.fixed_size = fixed_size
        self.decoder = None

        # --- load completed set ---
        manifest = self.output_dir / "vae_manifest.jsonl"
        completed = set()
        if manifest.exists():
            with open(manifest, "r", encoding="utf-8") as f:
                for line in f:
                    completed.add(json.loads(line)["image"])

        self.samples = []
        for p in glob.glob(str(self.image_dir / "**" / "*"), recursive=True):
            if not p.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            rel = Path(p).relative_to(self.image_dir).as_posix()
            if set(rel.lower().split("/")) & self.exclude_dirs:
                continue
            if rel in completed:
                continue
            out = (self.output_dir / rel).with_name(Path(rel).stem + "_vae.png")
            self.samples.append((Path(p), rel, out))

        self.transform = transforms.Compose([
            transforms.Resize(self.fixed_size, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.decoder is None:
            self.decoder = FastRGBDecoder()

        path, rel, out = self.samples[idx]
        arr = self.decoder.load(str(path))
        if arr is None:
            arr = np.zeros((224, 224, 3), dtype=np.uint8)

        img = Image.fromarray(arr)
        x = (self.transform(img) - 0.5) * 2
        return x, out, rel

def collate(batch):
    xs, outs, rels = zip(*batch)
    return torch.stack(xs), outs, rels

# ============================================================
# MAIN PIPELINE
# ============================================================
def process_images_with_ldm_vae(
    image_dir,
    output_dir,
    batch_size,
    fixed_size,
    resize,
    exclude_dirs,
    config_path,
    ckpt_path,
):
    from omegaconf import OmegaConf
    from latent_diffusion.ldm.util import instantiate_from_config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    model = instantiate_from_config(OmegaConf.load(config_path).model)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"]
    model.load_state_dict(state, strict=False)
    model = model.to(device).half().eval()
    model = torch.compile(model, mode="reduce-overhead")

    dataset = ImageDataset(image_dir, output_dir, exclude_dirs, fixed_size)
    print(f"Images remaining: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, os.cpu_count() // 2),
        pin_memory=True,
        collate_fn=collate,
        persistent_workers=True,
    )

    writer = AsyncWriter(Path(output_dir))

    with torch.inference_mode(), torch.amp.autocast(device_type=device):
        for xs, outs, rels in tqdm(loader, desc="VAE Reconstruction"):
            xs = xs.to(device, non_blocking=True)
            y, _ = model(xs, sample_posterior=False)
            y = torch.clamp((y / 2) + 0.5, 0, 1)

            for img, out, rel in zip(y, outs, rels):
                if resize:
                    img = F.interpolate(
                        img.unsqueeze(0),
                        size=(resize[1], resize[0]),
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze(0)
                arr = (img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                writer.submit(out, arr, rel)

    writer.close()
    print("Done.")

# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--fixed_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--resize", type=int, nargs=2, default=None)
    parser.add_argument("--exclude_dirs", nargs="+", default=["facemesh"])
    parser.add_argument("--config_path", type=str, default=r"latent_diffusion/configs/autoencoder/autoencoder_kl_64x64x3.yaml")
    parser.add_argument("--ckpt_path", type=str, default=r"model/model.ckpt")
    args = parser.parse_args()

    process_images_with_ldm_vae(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        fixed_size=tuple(args.fixed_size),
        resize=tuple(args.resize) if args.resize else None,
        exclude_dirs=args.exclude_dirs,
        config_path=args.config_path,
        ckpt_path=args.ckpt_path,
    )

# ===========================================================
# EXAMPLE USAGE
# python VAE.py --image_dir  "path/to/input" --output_dir  "path/to/output"  --batch_size 8 --fixed_size 224 224 --resize 224 224 --exclude_dirs facemesh