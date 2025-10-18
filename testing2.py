import os
import time
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TVF
from transformers import Owlv2VisionModel
from torch import nn
from tqdm import tqdm

# -------------------- OWLv2 Classifier --------------------
class DetectorModelOwl(nn.Module):
    def __init__(self, model_path: str, dropout: float, device: str = "cuda"):
        super().__init__()
        self.device = device
        owl = Owlv2VisionModel.from_pretrained(model_path).to(device)
        self.owl = owl
        self.owl.requires_grad_(False)
        self.dropout1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(768)
        self.linear1 = nn.Linear(768, 768*2)
        self.act1 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(768*2)
        self.linear2 = nn.Linear(768*2, 2)

    def forward(self, pixel_values: torch.Tensor):
        with torch.amp.autocast(device_type='cuda',dtype=torch.float16):
            outputs = self.owl(pixel_values=pixel_values, output_hidden_states=True)
            x = outputs.last_hidden_state
            x = self.dropout1(x)
            x = self.ln1(x)
            x = self.linear1(x)
            x = self.act1(x)
            x = self.dropout2(x)
            x, _ = x.max(dim=1)
            x = self.ln2(x)
            x = self.linear2(x)
        return x

# -------------------- Preprocessing Function --------------------
def preprocess_image(path: str, target_size=960):
    try:
        image = Image.open(path).convert("RGB")
        big_side = max(image.size)
        new_image = Image.new("RGB", (big_side, big_side), (128, 128, 128))
        new_image.paste(image, (0, 0))
        preped = new_image.resize((target_size, target_size), Image.BICUBIC)
        tensor = TVF.pil_to_tensor(preped) / 255.0
        tensor = TVF.normalize(tensor,
                               [0.48145466, 0.4578275, 0.40821073],
                               [0.26862954, 0.26130258, 0.27577711])
        return tensor
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None

# -------------------- Main Pipeline --------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_directory = "C:/MastersRepos/ARI5902-Research-Topics-in-AI/LAION-5B Testing/clip_embeddings_resumable_symlink/downloaded_images/0000"
    image_files = [os.path.join(image_directory, f)
                   for f in os.listdir(image_directory)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))][:1000]

    # -------------------- Step 1: Load and preprocess all images --------------------
    print(f"Preprocessing {len(image_files)} images on CPU...")
    start_preproc = time.time()
    preprocessed_images = []
    valid_paths = []
    for path in tqdm(image_files, desc="Preprocessing images"):
        tensor = preprocess_image(path, target_size=960)
        if tensor is not None:
            preprocessed_images.append(tensor)
            valid_paths.append(path)
    end_preproc = time.time()
    print(f"Finished preprocessing in {end_preproc - start_preproc:.2f}s")

    # -------------------- Step 2: Move images to DataLoader for batching --------------------
    dataset = torch.stack(preprocessed_images)  # shape: [N, C, H, W]
    print(f"Dataset tensor shape: {dataset.shape}")

    batch_size = 4
    results = []

    # Load OWLv2 model
    owl_model = DetectorModelOwl("google/owlv2-base-patch16-ensemble", dropout=0.0, device=device).to(device)
    owl_model.load_state_dict(torch.load("far5y1y5-8000.pt", map_location=device))
    owl_model.eval()
    print("OWLv2 model loaded.")

    # -------------------- Step 3: Run batches on GPU --------------------
    print(f"Running batches of {batch_size} images on GPU...")
    start_gpu = time.time()
    for i in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
        batch = dataset[i:i+batch_size].to(device)
        with torch.no_grad():
            logits = owl_model(batch)
            probs = F.softmax(logits, dim=1)
        for j, path in enumerate(valid_paths[i:i+batch_size]):
            watermark_prob = probs[j][1].item()
            is_watermarked = watermark_prob >= 0.8
            results.append({
                "image_path": path,
                "is_watermarked": is_watermarked
            })
    end_gpu = time.time()
    print(f"Processed {len(results)} images in {end_gpu - start_gpu:.2f}s, avg {((end_gpu - start_gpu)/len(results)):.2f}s per image.")

if __name__ == "__main__":
    main()
