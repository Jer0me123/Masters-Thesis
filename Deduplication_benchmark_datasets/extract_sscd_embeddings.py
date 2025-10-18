# extract_sscd_embeddings.py
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# Dataset
class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, extensions=("jpg", "jpeg", "png", "bmp")):
        self.img_paths = [
            os.path.join(img_dir, f) 
            for f in os.listdir(img_dir)
            if os.path.isfile(os.path.join(img_dir, f)) and f.lower().endswith(extensions)
        ]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
        except Exception:
            img = torch.zeros(3, 320, 320)
        return img, path

def main(
    IMAGE_DIR,
    SAVE_DIR,
    MODEL_PATH,
    BATCH_SIZE=128,
    NUM_WORKERS=4,
    DEVICE="cuda",
    CHECKPOINT_FILE="checkpoint.txt"
):
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Transform
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([transforms.Resize([320, 320]), transforms.ToTensor(), normalize])

    dataset = ImageDataset(IMAGE_DIR, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Load model
    model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
    model.eval().to(DEVICE)

    # Resume checkpoint
    start_idx = 0
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            start_idx = int(f.read().strip())

    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total=len(loader), initial=start_idx, desc="Processing batches")
        for batch_idx, (imgs, paths) in pbar:
            if batch_idx < start_idx:
                continue
            imgs = imgs.to(DEVICE)
            embs = model(imgs).cpu().numpy()
            shard_file = os.path.join(SAVE_DIR, f"embeddings_{batch_idx:06d}.npz")
            np.savez(shard_file, embeddings=embs, paths=paths)
            with open(CHECKPOINT_FILE, "w") as f:
                f.write(str(batch_idx + 1))
            pbar.set_postfix({"batch": batch_idx})

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint_file", type=str, required=False, default="checkpoint.txt")
    args = parser.parse_args()

    main(
        IMAGE_DIR=args.image_dir,
        SAVE_DIR=args.save_dir,
        MODEL_PATH=args.model_path,
        BATCH_SIZE=args.batch_size,
        NUM_WORKERS=args.num_workers,
        DEVICE=args.device,
        CHECKPOINT_FILE=args.checkpoint_file
    )
