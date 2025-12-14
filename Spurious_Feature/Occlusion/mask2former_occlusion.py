# # import os
# # from pathlib import Path
# # from concurrent.futures import ProcessPoolExecutor, Future
# # from typing import List, Tuple
# # import argparse
# # import numpy as np
# # import cv2
# # from PIL import Image
# # from tqdm import tqdm
# # import csv
# # import torch
# # from torch.utils.data import Dataset, DataLoader
# # from torchvision.io import read_image
# # from torchvision.transforms.functional import resize
# # from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

# # # ================================
# # # CONFIG
# # # ================================
# # ADE20K_PERSON_CLASS = 12  # ADE20K "person" index

# # MIN_MASK_AREA_RATIO = 0.02
# # MAX_MASK_AREA_RATIO = 0.60

# # MIN_BBOX_AREA_RATIO = 0.03
# # MAX_BBOX_AREA_RATIO = 0.80


# # # ================================
# # # WORKER FUNCTION (CPU)
# # # ================================
# # def process_occlusions_and_save(
# #     seg_map_tensor: torch.Tensor,
# #     original_size: Tuple[int, int],
# #     image_path: str,
# #     output_root: str
# # ):
# #     seg_map_np = seg_map_tensor.numpy().astype(np.uint8)

# #     H, W = original_size
# #     seg_resized = cv2.resize(seg_map_np, dsize=(W, H), interpolation=cv2.INTER_NEAREST)

# #     # Extract binary person mask
# #     person_mask = (seg_resized == ADE20K_PERSON_CLASS).astype(np.uint8)

# #     image = cv2.imread(image_path)
# #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# #     img_area = H * W
# #     mask_area = int(person_mask.sum())
# #     mask_ratio = mask_area / img_area if img_area > 0 else 0.0

# #     # Reject empty / invalid masks
# #     if mask_ratio < MIN_MASK_AREA_RATIO or mask_ratio > MAX_MASK_AREA_RATIO:
# #         return None

# #     ys, xs = np.where(person_mask == 1)
# #     if len(xs) == 0:
# #         return None

# #     x1, x2 = xs.min(), xs.max()
# #     y1, y2 = ys.min(), ys.max()

# #     bbox_area = (x2 - x1) * (y2 - y1)
# #     bbox_ratio = bbox_area / img_area if img_area > 0 else 0.0

# #     if bbox_ratio < MIN_BBOX_AREA_RATIO or bbox_ratio > MAX_BBOX_AREA_RATIO:
# #         return None

# #     stem = Path(image_path).stem

# #     # ---------- Full ----------
# #     full = image.copy()

# #     # ---------- Full NoBg ----------
# #     full_nobg = image * person_mask[..., None]

# #     # ---------- MaskSegm ----------
# #     masksegm = image.copy()
# #     masksegm[person_mask == 1] = 255

# #     # ---------- MaskSegm NoBg ----------
# #     masksegm_nobg = np.zeros_like(image)
# #     masksegm_nobg[person_mask == 1] = 255

# #     # ---------- MaskRect ----------
# #     maskrect = image.copy()
# #     maskrect[y1:y2, x1:x2] = 255

# #     # ---------- MaskRect NoBg ----------
# #     maskrect_nobg = np.zeros_like(image)
# #     maskrect_nobg[y1:y2, x1:x2] = 255

# #     outputs = {
# #         "Full": full,
# #         "Full_NoBg": full_nobg,
# #         "MaskSegm": masksegm,
# #         "MaskSegm_NoBg": masksegm_nobg,
# #         "MaskRect": maskrect,
# #         "MaskRect_NoBg": maskrect_nobg,
# #     }

# #     for k, img in outputs.items():
# #         out_dir = Path(output_root) / k
# #         out_dir.mkdir(parents=True, exist_ok=True)
# #         out_path = out_dir / f"{stem}.png"
# #         Image.fromarray(img).save(out_path, compress_level=1)

# #     return {
# #         "image": stem,
# #         "mask_ratio": round(mask_ratio, 4),
# #         "bbox_ratio": round(bbox_ratio, 4),
# #     }


# # # ================================
# # # DATASET
# # # ================================
# # class ImageDataset(Dataset):
# #     def __init__(self, image_dir, fixed_size=(512, 512)):
# #         self.fixed_size = fixed_size
# #         image_dir = Path(image_dir)
# #         self.image_paths = [str(p) for p in sorted(image_dir.iterdir()) if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
# #         if not self.image_paths:
# #             raise ValueError("No images found.")

# #     def __len__(self):
# #         return len(self.image_paths)

# #     def __getitem__(self, idx):
# #         path = self.image_paths[idx]
# #         img = read_image(path).float() / 255.0
# #         original_size = img.shape[1:]
# #         img = resize(img, self.fixed_size)
# #         return img, path, original_size


# # def collate_fixed(batch):
# #     imgs, paths, sizes = zip(*batch)
# #     return torch.stack(imgs, dim=0), list(paths), list(sizes)


# # # ================================
# # # MAIN PIPELINE
# # # ================================
# # def segment_and_generate_occlusions(
# #     image_dir,
# #     output_dir,
# #     batch_size=8,
# #     fixed_size=(512, 512),
# #     checkpoint_batches=100,
# # ):
# #     device = "cuda" if torch.cuda.is_available() else "cpu"
# #     os.makedirs(output_dir, exist_ok=True)

# #     print("Loading Mask2Former...")
# #     model_name = "facebook/mask2former-swin-large-ade-semantic"
# #     model = Mask2FormerForUniversalSegmentation.from_pretrained(
# #         model_name, torch_dtype=torch.float16
# #     ).to(device).eval()

# #     processor = AutoImageProcessor.from_pretrained(model_name)

# #     mean = torch.tensor(processor.image_mean, device=device).view(1, 3, 1, 1)
# #     std = torch.tensor(processor.image_std, device=device).view(1, 3, 1, 1)

# #     dataset = ImageDataset(image_dir, fixed_size=fixed_size)
# #     loader = DataLoader(
# #         dataset,
# #         batch_size=batch_size,
# #         shuffle=False,
# #         num_workers=max(1, os.cpu_count() // 2),
# #         pin_memory=True,
# #         collate_fn=collate_fixed,
# #     )

# #     csv_path = Path(output_dir) / "occlusion_qc.csv"
# #     csv_file = open(csv_path, "w", newline="")
# #     csv_writer = csv.DictWriter(csv_file, fieldnames=["image", "mask_ratio", "bbox_ratio"])
# #     csv_writer.writeheader()

# #     all_futures: List[Future] = []

# #     with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor, tqdm(total=len(loader)) as pbar:
# #         for batch_idx, (imgs_tensor, paths, original_sizes) in enumerate(loader):
# #             imgs_tensor = imgs_tensor.to(device, dtype=torch.float16)
# #             pixel_values = (imgs_tensor - mean) / std

# #             with torch.no_grad(), torch.cuda.amp.autocast():
# #                 outputs = model(pixel_values=pixel_values)

# #             seg_maps = processor.post_process_semantic_segmentation(
# #                 outputs, target_sizes=[fixed_size for _ in imgs_tensor]
# #             )

# #             for seg_map, orig_size, p in zip(seg_maps, original_sizes, paths):
# #                 future = executor.submit(
# #                     process_occlusions_and_save,
# #                     seg_map.cpu(),
# #                     orig_size,
# #                     p,
# #                     output_dir,
# #                 )
# #                 all_futures.append(future)

# #             pbar.update(1)

# #             if (batch_idx + 1) % checkpoint_batches == 0:
# #                 for f in all_futures:
# #                     res = f.result()
# #                     if res:
# #                         csv_writer.writerow(res)
# #                 all_futures = []

# #         for f in all_futures:
# #             res = f.result()
# #             if res:
# #                 csv_writer.writerow(res)

# #     csv_file.close()
# #     print("✅ Occlusion dataset generated successfully!")


# # # ================================
# # # CLI
# # # ================================
# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--image_dir", required=True)
# #     parser.add_argument("--output_dir", required=True)
# #     parser.add_argument("--batch_size", type=int, default=8)
# #     parser.add_argument("--fixed_size", type=int, nargs=2, default=[512, 512])
# #     parser.add_argument("--checkpoint_batches", type=int, default=100)

# #     args = parser.parse_args()

# #     segment_and_generate_occlusions(
# #         args.image_dir,
# #         args.output_dir,
# #         args.batch_size,
# #         tuple(args.fixed_size),
# #         args.checkpoint_batches,
# #     )

# import os
# os.environ["TRANSFORMERS_NO_TF"] = "1"     # ✅ HARD disable TensorFlow
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# from pathlib import Path
# from typing import Tuple
# import argparse
# import numpy as np
# import cv2
# from PIL import Image
# from tqdm import tqdm
# import csv
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision.io import read_image
# from torchvision.transforms.functional import resize
# from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

# # ================================
# # CONFIG
# # ================================
# ADE20K_PERSON_CLASS = 12

# MIN_MASK_AREA_RATIO = 0.02
# MAX_MASK_AREA_RATIO = 0.60
# MIN_BBOX_AREA_RATIO = 0.03
# MAX_BBOX_AREA_RATIO = 0.80


# # ================================
# # CPU POST-PROCESS (INLINE, NO MULTIPROC)
# # ================================
# def process_occlusions_and_save(seg_map_tensor, original_size, image_path, output_root):
#     H, W = original_size
#     seg_np = seg_map_tensor.numpy().astype(np.uint8)
#     seg_resized = cv2.resize(seg_np, (W, H), cv2.INTER_NEAREST)

#     person_mask = (seg_resized == ADE20K_PERSON_CLASS).astype(np.uint8)

#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     img_area = H * W
#     mask_area = int(person_mask.sum())
#     mask_ratio = mask_area / img_area

#     if mask_ratio < MIN_MASK_AREA_RATIO or mask_ratio > MAX_MASK_AREA_RATIO:
#         return None

#     ys, xs = np.where(person_mask == 1)
#     if len(xs) == 0:
#         return None

#     x1, x2 = xs.min(), xs.max()
#     y1, y2 = ys.min(), ys.max()
#     bbox_area = (x2 - x1) * (y2 - y1)
#     bbox_ratio = bbox_area / img_area

#     if bbox_ratio < MIN_BBOX_AREA_RATIO or bbox_ratio > MAX_BBOX_AREA_RATIO:
#         return None

#     stem = Path(image_path).stem

#     full = image.copy()
#     full_nobg = image * person_mask[..., None]

#     masksegm = image.copy()
#     masksegm[person_mask == 1] = 255

#     masksegm_nobg = np.zeros_like(image)
#     masksegm_nobg[person_mask == 1] = 255

#     maskrect = image.copy()
#     maskrect[y1:y2, x1:x2] = 255

#     maskrect_nobg = np.zeros_like(image)
#     maskrect_nobg[y1:y2, x1:x2] = 255

#     outputs = {
#         "Full": full,
#         "Full_NoBg": full_nobg,
#         "MaskSegm": masksegm,
#         "MaskSegm_NoBg": masksegm_nobg,
#         "MaskRect": maskrect,
#         "MaskRect_NoBg": maskrect_nobg,
#     }

#     for k, img in outputs.items():
#         out_dir = Path(output_root) / k
#         out_dir.mkdir(parents=True, exist_ok=True)
#         out_path = out_dir / f"{stem}.png"
#         Image.fromarray(img).save(out_path, compress_level=1)

#     return {"image": stem, "mask_ratio": round(mask_ratio, 4), "bbox_ratio": round(bbox_ratio, 4)}


# # ================================
# # DATASET
# # ================================
# class ImageDataset(Dataset):
#     def __init__(self, image_dir, fixed_size=(512, 512)):
#         self.fixed_size = fixed_size
#         self.image_paths = [str(p) for p in Path(image_dir).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         path = self.image_paths[idx]
#         img = read_image(path).float() / 255.0
#         original_size = img.shape[1:]
#         img = resize(img, self.fixed_size)
#         return img, path, original_size


# def collate_fixed(batch):
#     imgs, paths, sizes = zip(*batch)
#     return torch.stack(imgs, dim=0), list(paths), list(sizes)


# # ================================
# # MAIN
# # ================================
# def main(image_dir, output_dir, batch_size=4, fixed_size=(512, 512)):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     os.makedirs(output_dir, exist_ok=True)

#     print("Loading Mask2Former...")
#     model_name = "facebook/mask2former-swin-large-ade-semantic"
#     model = Mask2FormerForUniversalSegmentation.from_pretrained(
#         model_name, torch_dtype=torch.float16
#     ).to(device).eval()

#     processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

#     mean = torch.tensor(processor.image_mean, device=device).view(1, 3, 1, 1)
#     std = torch.tensor(processor.image_std, device=device).view(1, 3, 1, 1)

#     dataset = ImageDataset(image_dir, fixed_size)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fixed)

#     csv_path = Path(output_dir) / "occlusion_qc.csv"
#     with open(csv_path, "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=["image", "mask_ratio", "bbox_ratio"])
#         writer.writeheader()

#         for imgs_tensor, paths, sizes in tqdm(loader):
#             imgs_tensor = imgs_tensor.to(device, dtype=torch.float16)
#             pixel_values = (imgs_tensor - mean) / std

#             with torch.no_grad(), torch.amp.autocast("cuda"):
#                 outputs = model(pixel_values=pixel_values)

#             seg_maps = processor.post_process_semantic_segmentation(
#                 outputs, target_sizes=[fixed_size for _ in imgs_tensor]
#             )

#             for seg, size, p in zip(seg_maps, sizes, paths):
#                 res = process_occlusions_and_save(seg.cpu(), size, p, output_dir)
#                 if res:
#                     writer.writerow(res)

#     print("✅ Completed successfully.")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--image_dir", required=True)
#     parser.add_argument("--output_dir", required=True)
#     parser.add_argument("--batch_size", type=int, default=4)
#     parser.add_argument("--fixed_size", type=int, nargs=2, default=[512, 512])
#     args = parser.parse_args()

#     main(args.image_dir, args.output_dir, args.batch_size, tuple(args.fixed_size))

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path
from typing import Tuple, Optional, Set
import argparse
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms.functional import resize
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


# ================================
# CONFIG
# ================================
ADE20K_PERSON_CLASS = 12

MIN_MASK_AREA_RATIO = 0.02
MAX_MASK_AREA_RATIO = 0.60
MIN_BBOX_AREA_RATIO = 0.03
MAX_BBOX_AREA_RATIO = 0.80


# ================================
# CPU POST-PROCESS
# ================================
def process_occlusions_and_save(seg_map_tensor, original_size, image_path, output_root):
    H, W = original_size
    seg_np = seg_map_tensor.numpy().astype(np.uint8)
    seg_resized = cv2.resize(seg_np, (W, H), cv2.INTER_NEAREST)

    person_mask = (seg_resized == ADE20K_PERSON_CLASS).astype(np.uint8)

    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img_area = H * W
    mask_area = int(person_mask.sum())
    mask_ratio = mask_area / img_area

    if mask_ratio < MIN_MASK_AREA_RATIO or mask_ratio > MAX_MASK_AREA_RATIO:
        return None

    ys, xs = np.where(person_mask == 1)
    if len(xs) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    bbox_area = (x2 - x1) * (y2 - y1)
    bbox_ratio = bbox_area / img_area

    if bbox_ratio < MIN_BBOX_AREA_RATIO or bbox_ratio > MAX_BBOX_AREA_RATIO:
        return None

    stem = Path(image_path).stem

    full = image.copy()
    full_nobg = image * person_mask[..., None]

    masksegm = image.copy()
    masksegm[person_mask == 1] = 255

    masksegm_nobg = np.zeros_like(image)
    masksegm_nobg[person_mask == 1] = 255

    maskrect = image.copy()
    maskrect[y1:y2, x1:x2] = 255

    maskrect_nobg = np.zeros_like(image)
    maskrect_nobg[y1:y2, x1:x2] = 255

    outputs = {
        "Full": full,
        "Full_NoBg": full_nobg,
        "MaskSegm": masksegm,
        "MaskSegm_NoBg": masksegm_nobg,
        "MaskRect": maskrect,
        "MaskRect_NoBg": maskrect_nobg,
    }

    for k, img in outputs.items():
        out_dir = Path(output_root) / k
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}.png"
        Image.fromarray(img).save(out_path, compress_level=1)

    return {"image": stem, "mask_ratio": round(mask_ratio, 4), "bbox_ratio": round(bbox_ratio, 4)}


# ================================
# DATASET WITH TRUE RESUME FILTER
# ================================
class ImageDataset(Dataset):
    def __init__(self, image_dir, fixed_size=(512, 512), exclude_set: Optional[Set[str]] = None):
        self.fixed_size = fixed_size
        all_paths = sorted(
            [str(p) for p in Path(image_dir).iterdir()
             if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        )

        if exclude_set:
            self.image_paths = [
                p for p in all_paths if Path(p).stem not in exclude_set
            ]
        else:
            self.image_paths = all_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = read_image(path).float() / 255.0
        original_size = img.shape[1:]
        img = resize(img, self.fixed_size)
        return img, path, original_size


def collate_fixed(batch):
    imgs, paths, sizes = zip(*batch)
    return torch.stack(imgs, dim=0), list(paths), list(sizes)


# ================================
# MAIN
# ================================
def main(image_dir, output_dir, batch_size=4, fixed_size=(512, 512), resume=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    # ================================
    # TRUE RESUME — LOAD PROCESSED SET
    # ================================
    csv_path = Path(output_dir) / "occlusion_qc.csv"
    processed = set()

    if resume and csv_path.exists():
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed.add(row["image"])

    # ================================
    # STARTUP MESSAGE
    # ================================
    all_images = [
        p.stem for p in Path(image_dir).iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]

    if resume:
        print("🔁 TRUE RESUME ENABLED")
        print(f"✅ Already completed: {len(processed)}")
        print(f"⏳ Remaining to process: {len(all_images) - len(processed)}")
    else:
        print("🆕 Fresh run (no resume)")
        print(f"⏳ Total images to process: {len(all_images)}")

    # ================================
    # MODEL
    # ================================
    print("Loading Mask2Former...")
    model_name = "facebook/mask2former-swin-large-ade-semantic"
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device).eval()

    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

    mean = torch.tensor(processor.image_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, device=device).view(1, 3, 1, 1)

    # ================================
    # DATASET — FILTERED FOR TRUE RESUME
    # ================================
    dataset = ImageDataset(image_dir, fixed_size, exclude_set=processed if resume else None)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fixed
    )

    # ================================
    # CSV WRITE MODE
    # ================================
    csv_mode = "a" if resume and csv_path.exists() else "w"

    with open(csv_path, csv_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "mask_ratio", "bbox_ratio"])
        if csv_mode == "w":
            writer.writeheader()

        for imgs_tensor, paths, sizes in tqdm(loader, total=len(loader)):
            imgs_tensor = imgs_tensor.to(device, dtype=torch.float16)
            pixel_values = (imgs_tensor - mean) / std

            with torch.no_grad(), torch.amp.autocast("cuda"):
                outputs = model(pixel_values=pixel_values)

            seg_maps = processor.post_process_semantic_segmentation(
                outputs,
                target_sizes=[fixed_size for _ in imgs_tensor]
            )

            for seg, size, p in zip(seg_maps, sizes, paths):
                res = process_occlusions_and_save(seg.cpu(), size, p, output_dir)
                if res:
                    writer.writerow(res)

    print("✅ Completed successfully.")


# ================================
# CLI
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--fixed_size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    main(args.image_dir, args.output_dir, args.batch_size, tuple(args.fixed_size), args.resume)


# ---------------- Example Usage ----------------
# "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\.venv-Copy-Copy\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\Occlusion\mask2former_occlusion.py" --image_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ImageCaptioningEvaluationDatasets\LAION-5B-10k\LAION-5B-10k-images" --output_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\Occlusion\LAION-5B-10k-occlusion" --batch_size 8 --fixed_size 512 512 --save_results True --checkpoint_batches 100
