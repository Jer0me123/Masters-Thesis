import os
import time
import json
from math import ceil
from tqdm import tqdm
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, Kosmos2ForConditionalGeneration, BitsAndBytesConfig

# ---------------- Dataset ----------------
class PILImageDataset(Dataset):
    """Loads images as PIL.Image objects, no conversion to tensors."""
    def __init__(self, image_dir, extensions=(".jpg", ".jpeg", ".png")):
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(extensions)
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path)

            # Handle grayscale or other modes
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Check for invalid dimensions (common in partial/corrupt images)
            w, h = img.size
            if w <= 1 or h <= 1:
                raise ValueError("Invalid image dimensions")

        except (UnidentifiedImageError, OSError, ValueError) as e:
            # Return a black 224x224 placeholder image if corrupted/unreadable
            img = Image.new("RGB", (224, 224), color=(0, 0, 0))
            print(f"⚠️ Replaced corrupted image with black placeholder: {path} ({e})")

        return img, path


# ---------------- Collate Function ----------------
def collate_images(batch):
    imgs, paths = zip(*batch)
    return list(imgs), list(paths)


# ---------------- DataLoader setup ----------------
def get_pil_image_loader(image_dir, batch_size=32, num_workers=4):
    dataset = PILImageDataset(image_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_images,
        pin_memory=True
    )
    return loader, len(dataset)


# ---------------- Main script ----------------
def main(args):
    total_start_time = time.time()

    # ---------------- Configuration ----------------
    image_dir = args.image_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    output_file = args.output_file
    prompt = args.prompt
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_length = 100

    # ---------------- Load model, tokenizer, processor ----------------
    model_name = "microsoft/kosmos-2-patch14-224"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    model = Kosmos2ForConditionalGeneration.from_pretrained(
        model_name,
        device_map=device,
        quantization_config=quantization_config
    ).eval()
    image_processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

    # ---------------- Initialize DataLoader ----------------
    loader, dataset_len = get_pil_image_loader(image_dir, batch_size=batch_size, num_workers=num_workers)
    startup_time = time.time() - total_start_time

    # ---------------- Initialize JSON file ----------------
    json_data = {"dataset": os.path.basename(image_dir), "images": []}
    imgid_counter = 0

    captioning_start_time = time.time()

    # ---------------- Batched inference & streaming ----------------
    for imgs, paths in tqdm(loader, desc="Processing batches", total=ceil(dataset_len / batch_size)):
        inputs = image_processor(text=[prompt] * len(imgs), images=imgs, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_embeds_position_mask=inputs["image_embeds_position_mask"],
                max_new_tokens=max_length,
                use_cache=True
            )

        # Append batch results to JSON
        captions = image_processor.batch_decode(outputs, skip_special_tokens=True)
        for path, text in zip(paths, captions):
            caption, _ = image_processor.post_process_generation(text)
            json_data["images"].append({
                "filename": os.path.basename(path),
                "imgid": imgid_counter,
                "sentences": [{"raw": caption[prompt]}]
            })
            imgid_counter += 1

        # Stream to disk after each batch
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)

        torch.cuda.empty_cache()
        if device == "cuda":
            torch.cuda.synchronize()

    captioning_time = time.time() - captioning_start_time
    print(f"\nCaptions saved to {output_file}")
    print(f"\nTiming Summary:")
    print(f" - Startup time (model + dataloader): {startup_time:.2f}s")
    print(f" - Captioning time (processing only): {captioning_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch image captioning with kosmos-2-patch14-224 model")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the image directory")
    parser.add_argument("--batch_size", type=int, default=25, help="Batch size for DataLoader")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker threads for DataLoader")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save output JSON")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to pass to the model")
    args = parser.parse_args()

    main(args)