# import os
# import torch
# import numpy as np
# import pandas as pd
# from PIL import Image
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from tqdm import tqdm
# import pyiqa


# # --- Dataset definition
# class FlatFolderDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.files = [
#             os.path.join(root_dir, f)
#             for f in os.listdir(root_dir)
#             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
#         ]

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         img_path = self.files[idx]
#         img = Image.open(img_path).convert('RGB')
#         if self.transform:
#             img = self.transform(img)
#         return img, os.path.basename(img_path)


# # --- Dummy watermark function
# def dummy_detect_watermark(batch):
#     return (batch.mean(dim=[1, 2, 3]) > 0.5).cpu().numpy()


# # --- Main function
# def main():
#     # --- transforms
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#     ])

#     # --- Path
#     image_folder_dir = r"C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\clip_embeddings_resumable_symlink\downloaded_images\0000"
#     dataset = FlatFolderDataset(image_folder_dir, transform=transform)

#     # --- DataLoader
#     loader = DataLoader(
#         dataset,
#         batch_size=64,  # IQA models are heavy; smaller batches
#         shuffle=False,
#         num_workers=12,
#         pin_memory=True
#     )

#     # --- Device
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"Using device: {device}")

#     # --- Load IQA models
#     models = {
#         # 'paq2piq': pyiqa.create_metric('paq2piq').to(device)#,
#         # 'brisque': pyiqa.create_metric('brisque').to(device),
#         # 'maniqa': pyiqa.create_metric('maniqa-pipal').to(device), #Not viable take too long to execute 100,000 images and too resource intensive
#         'niqe': pyiqa.create_metric('niqe').to(device) #Slower than paq2piq & brisque but faster than maniqa, due to how the model works envounters issues due to invalid images which would need to be resolved
#     }

#     # --- Storage for results
#     results = []

#     # --- Loop over batches
#     pbar = tqdm(loader, desc="Processing images", total=len(loader))
#     for batch, filenames in pbar:
#         batch = batch.to(device)

#         # --- Watermark detection
#         wm_results = dummy_detect_watermark(batch)

#         # --- Compute IQA scores per model
#         iqa_scores = {}
#         for name, model in models.items():
#             with torch.no_grad():
#                 scores = model(batch).detach().cpu().numpy()
#             iqa_scores[name] = scores

#         # --- Combine results per image
#         for i, fname in enumerate(filenames):
#             results.append({
#                 'filename': fname,
#                 'watermark': bool(wm_results[i]),
#                 # 'paq2piq': float(iqa_scores['paq2piq'][i])
#                 # 'brisque': float(iqa_scores['brisque'][i])
#                 # 'maniqa': float(iqa_scores['maniqa'][i])
#                 'niqe': float(iqa_scores['niqe'][i])
#             })

#     pbar.close()

#     # --- Convert to DataFrame for convenience
#     df = pd.DataFrame(results)
#     print(df.head())


# # --- Entry point (safe for multiprocessing)
# if __name__ == "__main__":
#     torch.multiprocessing.set_start_method('spawn', force=True)
#     main()





# # --------------------------- Watermark Detection Testing Code ---------------------------

# import os
# import time
# from PIL import Image
# from ultralytics import YOLO
# import torchvision.transforms.functional as TVF
# from torch.utils.data import Dataset, DataLoader
# import torch
# import torch.nn.functional as F
# from transformers import Owlv2VisionModel
# from torch import nn
# from tqdm import tqdm

# # --- OWLv2 Classification Head ---
# class DetectorModelOwl(nn.Module):
#     owl: Owlv2VisionModel

#     def __init__(self, model_path: str, dropout: float, n_hidden: int = 768, device: str = "cpu"):
#         super().__init__()

#         owl = Owlv2VisionModel.from_pretrained(model_path).to(device)
#         assert isinstance(owl, Owlv2VisionModel)
#         self.owl = owl
#         self.owl.requires_grad_(False)

#         self.dropout1 = nn.Dropout(dropout)
#         self.ln1 = nn.LayerNorm(n_hidden, eps=1e-5)
#         self.linear1 = nn.Linear(n_hidden, n_hidden * 2)
#         self.act1 = nn.GELU()
#         self.dropout2 = nn.Dropout(dropout)
#         self.ln2 = nn.LayerNorm(n_hidden * 2, eps=1e-5)
#         self.linear2 = nn.Linear(n_hidden * 2, 2)

#     def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor | None = None):
#         with torch.autocast(pixel_values.device.type, dtype=torch.bfloat16):
#             outputs = self.owl(pixel_values=pixel_values, output_hidden_states=True)
#             x = outputs.last_hidden_state  # B, N, C

#             x = self.dropout1(x)
#             x = self.ln1(x)
#             x = self.linear1(x)
#             x = self.act1(x)

#             x = self.dropout2(x)
#             x, _ = x.max(dim=1)
#             x = self.ln2(x)

#             x = self.linear2(x)

#         if labels is not None:
#             loss = F.cross_entropy(x, labels)
#             return (x, loss)

#         return (x,)

# # --- Dataset ---
# class ImageDataset(Dataset):
#     def __init__(self, image_paths):
#         self.image_paths = image_paths

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         path = self.image_paths[idx]
#         try:
#             image = Image.open(path).convert("RGB")

#             # OWLv2 preprocessing
#             big_side = max(image.size)
#             new_image = Image.new("RGB", (big_side, big_side), (128, 128, 128))
#             new_image.paste(image, (0, 0))
#             preped = new_image.resize((960, 960), Image.BICUBIC)

#             preped = TVF.pil_to_tensor(preped) / 255.0
#             owl_ready = TVF.normalize(
#                 preped,
#                 [0.48145466, 0.4578275, 0.40821073],
#                 [0.26862954, 0.26130258, 0.27577711]
#             )

#             return owl_ready, path
#         except Exception as e:
#             print(f"Error loading {path}: {e}")
#             return None, path


# # --- Batch Detector ---
# def watermark_detector_batch(
#     image_paths: list[str],
#     run_yolo: bool,
#     yolo_conf_thresh: float,
#     owl_conf_thresh: float,
#     batch_size: int = 8,
#     num_workers: int = 4,
#     device: str = "cpu"
# ):
#     print(f"Starting batch processing of {len(image_paths)} images on {device} with batch size {batch_size} and {num_workers} workers.")

#     # Load models only in the main process
#     owl_model = DetectorModelOwl("google/owlv2-base-patch16-ensemble", dropout=0.0, device=device).to(device)
#     owl_model.load_state_dict(torch.load("far5y1y5-8000.pt", map_location=device))
#     owl_model.eval()
#     print("OWLv2 model loaded.")

#     yolo_model = None
#     if run_yolo:
#         yolo_model = YOLO("yolo11x-train28-best.pt").to(device)
#         print("YOLO model loaded.")

#     dataset = ImageDataset(image_paths)
#     loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

#     results = []
#     start_time = time.time()

#     # wrap loader with tqdm
#     for owl_inputs, paths in tqdm(loader, desc="Processing batches", unit="batch"):
#         # Filter out bad images
#         valid_idx = [i for i, x in enumerate(owl_inputs) if x is not None]
#         if not valid_idx:
#             continue

#         owl_batch = torch.stack([owl_inputs[i] for i in valid_idx]).to(device)

#         # OWLv2 forward
#         with torch.no_grad():
#             logits, = owl_model(owl_batch, None)
#         probs = F.softmax(logits, dim=1)

#         # Collect results
#         for i, idx in enumerate(valid_idx):
#             watermark_prob = probs[i][1].item()
#             is_watermarked = watermark_prob >= owl_conf_thresh
#             results.append({
#                 "image_path": paths[idx],
#                 "is_watermarked": is_watermarked,
#                 "yolo_boxes": []
#             })

#         # # YOLO part (reload images in main process only)
#         # if run_yolo:
#         #     yolo_images = [Image.open(paths[i]).convert("RGB") for i in valid_idx]
#         #     yolo_results = yolo_model(yolo_images, imgsz=1024, augment=True,
#         #                               iou=0.5, conf=yolo_conf_thresh)

#         #     for j, res in enumerate(yolo_results):
#         #         yolo_boxes = []
#         #         for box in res.boxes:
#         #             coords = box.xyxy[0].tolist()
#         #             yolo_boxes.append({
#         #                 "class_id": int(box.cls.item()),
#         #                 "confidence": float(box.conf.item()),
#         #                 "bbox_coords": [round(c, 2) for c in coords]
#         #             })
#         #         results[j]["yolo_boxes"] = yolo_boxes
#     end_time = time.time()
#     print(f"Processed {len(results)} images in {end_time - start_time:.2f} sec, Avg {((end_time - start_time) / len(results)):.2f} sec/image.")
#     return results

# import warnings
# warnings.simplefilter("ignore", FutureWarning)

# # --- Main ---
# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     image_directory = "C:/MastersRepos/ARI5902-Research-Topics-in-AI/LAION-5B Testing/clip_embeddings_resumable_symlink/downloaded_images/0000"
#     process_limit = 100#0

#     all_files = os.listdir(image_directory)
#     image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
#     all_image_paths = [
#         os.path.join(image_directory, f)
#         for f in all_files
#         if os.path.splitext(f)[1].lower() in image_extensions
#     ][:process_limit]

#     if not all_image_paths:
#         print(f"No images found in {image_directory}")
#     else:
#         results = watermark_detector_batch(
#             image_paths=all_image_paths,
#             run_yolo=False,
#             yolo_conf_thresh=0.7,
#             owl_conf_thresh=0.8,
#             batch_size=4,
#             num_workers=0,#min(4, os.cpu_count() // 2),
#             device=device
#         )
#         # print("\n--- Results ---")
#         # for r in results:
#         #     print(r)


# if __name__ == "__main__":
#     main()



# --------------------------- Watermark Detection Testing Code 2 ---------------------------
# --------------------------- Watermark Detection Testing Code ---------------------------

import os
import time
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from transformers import Owlv2VisionModel
from torch import nn
from torchvision import transforms
from torchvision.transforms import functional as TVF
from tqdm import tqdm
import warnings

# --- Silence warnings ---
warnings.simplefilter("ignore", FutureWarning)

# --- OWLv2 Classification Head ---
class DetectorModelOwl(nn.Module):
    owl: Owlv2VisionModel

    def __init__(self, model_path: str, dropout: float, n_hidden: int = 768, device: str = "cpu"):
        super().__init__()
        self.device = device

        owl = Owlv2VisionModel.from_pretrained(model_path).to(device)
        self.owl = owl
        self.owl.requires_grad_(False)

        self.dropout1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(n_hidden, eps=1e-5)
        self.linear1 = nn.Linear(n_hidden, n_hidden * 2)
        self.act1 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(n_hidden * 2, eps=1e-5)
        self.linear2 = nn.Linear(n_hidden * 2, 2)

    def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor | None = None):
        # Use mixed precision only on GPU
        if pixel_values.device.type == "cuda":
            from torch.cuda.amp import autocast
            with autocast(dtype=torch.float16):
                outputs = self.owl(pixel_values=pixel_values, output_hidden_states=True)
        else:
            outputs = self.owl(pixel_values=pixel_values, output_hidden_states=True)

        x = outputs.last_hidden_state  # B, N, C
        x = self.dropout1(x)
        x = self.ln1(x)
        x = self.linear1(x)
        x = self.act1(x)
        x = self.dropout2(x)
        x, _ = x.max(dim=1)
        x = self.ln2(x)
        x = self.linear2(x)

        if labels is not None:
            loss = F.cross_entropy(x, labels.to(pixel_values.device))
            return x, loss

        return x,

# --- Dataset ---
class ImageDataset(Dataset):
    def __init__(self, image_paths, resize=960):
        self.image_paths = image_paths
        self.resize = resize

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            # Pad to square
            big_side = max(image.size)
            new_image = Image.new("RGB", (big_side, big_side), (128, 128, 128))
            new_image.paste(image, (0, 0))
            # Resize
            preped = new_image.resize((self.resize, self.resize), Image.BICUBIC)
            # Tensor and normalize
            preped = TVF.pil_to_tensor(preped) / 255.0
            owl_ready = TVF.normalize(
                preped,
                [0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711]
            )
            return owl_ready, path
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None, path

# --- Batch Detector ---
def watermark_detector_batch(
    image_paths: list[str],
    batch_size: int = 4,
    num_workers: int = 0,
    owl_conf_thresh: float = 0.8,
    device: str = "cpu"
):
    print(f"Starting batch processing of {len(image_paths)} images on {device} "
          f"with batch size {batch_size} and {num_workers} workers.")

    # Load OWLv2 model once
    owl_model = DetectorModelOwl("google/owlv2-base-patch16-ensemble", dropout=0.0, device=device).to(device)
    owl_model.load_state_dict(torch.load("far5y1y5-8000.pt", map_location=device))
    owl_model.eval()
    print("OWLv2 model loaded.")

    dataset = ImageDataset(image_paths)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    results = []
    start_time = time.time()

    # tqdm progress bar
    for owl_inputs, paths in tqdm(loader, desc="Processing batches", unit="batch"):
        # Filter out failed images
        valid_idx = [i for i, x in enumerate(owl_inputs) if x is not None]
        if not valid_idx:
            continue

        owl_batch = torch.stack([owl_inputs[i] for i in valid_idx]).to(device)

        # Forward pass
        with torch.no_grad():
            logits, = owl_model(owl_batch)
        probs = F.softmax(logits, dim=1)

        # Collect results
        for i, idx in enumerate(valid_idx):
            watermark_prob = probs[i][1].item()
            is_watermarked = watermark_prob >= owl_conf_thresh
            results.append({
                "image_path": paths[idx],
                "is_watermarked": is_watermarked
            })

    end_time = time.time()
    avg_time = (end_time - start_time) / max(1, len(results))
    print(f"Processed {len(results)} images in {end_time - start_time:.2f} sec "
          f"(avg {avg_time:.3f} sec/image).")
    return results

# --- Main ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_directory = "C:/MastersRepos/ARI5902-Research-Topics-in-AI/LAION-5B Testing/clip_embeddings_resumable_symlink/downloaded_images/0000"
    process_limit = 1000

    all_files = os.listdir(image_directory)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    all_image_paths = [
        os.path.join(image_directory, f)
        for f in all_files
        if os.path.splitext(f)[1].lower() in image_extensions
    ][:process_limit]

    if not all_image_paths:
        print(f"No images found in {image_directory}")
        return

    results = watermark_detector_batch(
        image_paths=all_image_paths,
        batch_size=6,             # Start with 2; increase to find optimal
        num_workers=0,            # Use 0 for notebooks; try >0 in .py script
        owl_conf_thresh=0.8,
        device=device
    )

    # Optionally print results
    # for r in results:
    #     print(r)

if __name__ == "__main__":
    main()
