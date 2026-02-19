import os
import time
import json
from math import ceil
from tqdm import tqdm
import argparse
import re
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np

np.float_ = np.float64
np.complex_ = np.complex128

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ============================================================
# FAST RGB DECODER (Optional TurboJPEG)
# ============================================================

class FastImageLoader:
    def __init__(self):
        self.jpeg = None
        self.use_turbo = False

    def _init_turbo(self):
        if self.jpeg is None:
            try:
                from turbojpeg import TurboJPEG
                self.jpeg = TurboJPEG(r"C:\libjpeg-turbo-gcc64\bin\libturbojpeg.dll")
                self.use_turbo = True
                print("[Loader] TurboJPEG initialized in worker")
            except Exception as e:
                print(f"[Loader] TurboJPEG unavailable ({e}), using PIL")

    def load(self, path: str) -> Image.Image:
        self._init_turbo()

        try:
            ext = os.path.splitext(path)[1].lower()
            if self.use_turbo and ext in {".jpg", ".jpeg"}:
                with open(path, "rb") as f:
                    rgb_array = self.jpeg.decode(f.read())
                img = Image.fromarray(rgb_array, mode="RGB")
            else:
                img = Image.open(path)
                if img.mode != "RGB":
                    img = img.convert("RGB")

            w, h = img.size
            if w <= 1 or h <= 1:
                raise ValueError("Invalid image dimensions")

            return img

        except (UnidentifiedImageError, OSError, ValueError) as e:
            print(f"Replaced corrupted image with black placeholder: {path} ({e})")
            return Image.new("RGB", (224, 224), color=(0, 0, 0))



# ============================================================
# WORD REMAPPING
# ============================================================
class WordRemapper:
    """
    Handles word remapping with plural support.
    """
    def __init__(self, mapping_file: str = None):
        self.mappings = {}
        
        if mapping_file and os.path.exists(mapping_file):
            with open(mapping_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if ':' in line:
                        source, target = line.split(':', 1)
                        source = source.strip().lower()
                        target = target.strip()
                        self.mappings[source] = target
                        
                        # Add plural forms
                        self.mappings[self._pluralize(source)] = self._pluralize(target)
            
            print(f"[WordRemapper] Loaded {len(self.mappings)} word mappings")
        else:
            print("[WordRemapper] No mapping file provided, remapped captions will be identical to raw")
    
    def _pluralize(self, word: str) -> str:
        """Simple pluralization rules."""
        if word.endswith('y') and len(word) > 1 and word[-2] not in 'aeiou':
            return word[:-1] + 'ies'
        elif word.endswith(('s', 'x', 'z', 'ch', 'sh')):
            return word + 'es'
        else:
            return word + 's'
    
    def remap(self, caption: str) -> str:
        """
        Remap words in caption (case-insensitive matching, preserves original case in output).
        """
        if not self.mappings:
            return caption
        
        # Create regex pattern for whole word matching
        pattern = r'\b(' + '|'.join(re.escape(k) for k in self.mappings.keys()) + r')\b'
        
        def replace_func(match):
            word = match.group(0)
            word_lower = word.lower()
            replacement = self.mappings.get(word_lower, word)
            
            # Preserve original capitalization pattern
            if word[0].isupper():
                replacement = replacement.capitalize()
            
            return replacement
        
        return re.sub(pattern, replace_func, caption, flags=re.IGNORECASE)


# ============================================================
# DATASET (Subdirectory-based)
# ============================================================
class SubdirectoryImageDataset(Dataset):
    """
    Loads images from a single subdirectory as PIL.Image objects.
    """
    def __init__(self, image_paths, image_loader):
        self.image_paths = image_paths
        self.image_loader = image_loader

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = self.image_loader.load(path)
        return img, path


# ============================================================
# COLLATE FUNCTION
# ============================================================
def collate_images(batch):
    imgs, paths = zip(*batch)
    return list(imgs), list(paths)


# ============================================================
# RESUME LOGIC
# ============================================================
def load_completed_images(output_file: str) -> set:
    """
    Load set of completed image filenames from existing JSONL file.
    """
    completed = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    completed.add(entry['filename'])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"[Resume] Found {len(completed)} completed images")
    return completed


# ============================================================
# SUBDIRECTORY ENUMERATION
# ============================================================
def get_subdirectories_and_images(image_dir: str, exclude_dirs: list) -> dict:
    """
    Enumerate all subdirectories and their images.
    Returns dict: {subdirectory_path: [image_paths]}
    """
    exclude_dirs_lower = {d.lower() for d in exclude_dirs}
    subdirs = defaultdict(list)
    
    extensions = (".jpg", ".jpeg", ".png")
    
    for root, dirs, files in os.walk(image_dir):
        # Remove excluded directories from traversal
        dirs[:] = [d for d in dirs if d.lower() not in exclude_dirs_lower]
        
        for file in files:
            if file.lower().endswith(extensions):
                full_path = os.path.join(root, file)
                subdirs[root].append(full_path)
    
    # Sort subdirectories for deterministic processing order
    subdirs = {k: sorted(v) for k, v in sorted(subdirs.items())}
    
    total_images = sum(len(v) for v in subdirs.values())
    print(f"[Discovery] Found {len(subdirs)} subdirectories with {total_images} total images")
    
    return subdirs


# ============================================================
# MAIN SCRIPT
# ============================================================
def main(args):
    total_start_time = time.time()

    # ---------------- Configuration ----------------
    image_dir = args.image_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    output_file = args.output_file
    prompt = args.prompt
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    max_length = args.max_length

    print(f"Using device: {device}")

    # ---------------- Initialize Image Loader ----------------
    image_loader = FastImageLoader()

    # ---------------- Initialize Word Remapper ----------------
    word_remapper = WordRemapper(mapping_file=args.word_mapping_file)

    # ---------------- Load Model ----------------
    model_name = "Salesforce/blip-image-captioning-large"
    print(f"[Model] Loading {model_name}...")
    model = BlipForConditionalGeneration.from_pretrained(
        model_name,
        device_map=device,
        use_safetensors=True
    ).eval()

    model = torch.compile(model, mode="reduce-overhead")

    image_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-large", 
        use_fast=True
    )

    # ---------------- Resume Logic ----------------
    completed_images = load_completed_images(output_file)

    # ---------------- Discover Subdirectories ----------------
    subdirs_images = get_subdirectories_and_images(image_dir, args.exclude_dirs)

    startup_time = time.time() - total_start_time
    print(f"[Startup] Completed in {startup_time:.2f}s\n")

    # ---------------- Process Each Subdirectory ----------------
    imgid_counter = 0
    total_processed = 0
    captioning_start_time = time.time()

    for subdir_path, image_paths in subdirs_images.items():
        subdir_name = os.path.relpath(subdir_path, image_dir)
        
        # Filter out already completed images
        remaining_images = [
            p for p in image_paths
            if os.path.relpath(p, image_dir).replace("\\", "/") not in completed_images
        ]
        
        if not remaining_images:
            print(f"[Skip] {subdir_name}: All {len(image_paths)} images already processed")
            continue
        
        skipped = len(image_paths) - len(remaining_images)
        print(f"\n[Processing] {subdir_name}: {len(remaining_images)} images "
              f"({skipped} skipped)")
        
        # Create dataset and dataloader for this subdirectory
        dataset = SubdirectoryImageDataset(remaining_images, image_loader)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=collate_images,
            pin_memory=True,
            prefetch_factor=1,
            persistent_workers=True 
        )
        
        # Process batches
        for imgs, paths in tqdm(
            loader, 
            desc=f"  Batches ({subdir_name})", 
            total=ceil(len(dataset) / batch_size)
        ):
            inputs = image_processor(
                text=[prompt] * len(imgs), 
                images=imgs, 
                return_tensors="pt"
            ).to(device, torch_dtype)

            # Generate captions
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=max_length, 
                    num_beams=1, 
                    do_sample=False,
                )

            # Decode and remap captions
            captions = image_processor.batch_decode(outputs, skip_special_tokens=True)
            
            # Write results immediately (streaming to disk)
            with open(output_file, "a+", encoding="utf-8") as f:
                for path, caption in zip(paths, captions):
                    relative_path = os.path.relpath(path, image_dir).replace("\\", "/")
                    remapped_caption = word_remapper.remap(caption)
                    
                    entry = {
                        "filename": relative_path,
                        "imgid": imgid_counter,
                        "sentences": [
                            {
                                "raw": caption,
                                "remapped": remapped_caption
                            }
                        ]
                    }
                    f.write(json.dumps(entry) + "\n")
                    imgid_counter += 1
            
            total_processed += len(paths)
            
            # Cleanup
            torch.cuda.empty_cache()
            if device == "cuda":
                torch.cuda.synchronize()

    captioning_time = time.time() - captioning_start_time
    
    print(f"\n{'='*60}")
    print(f"Captions saved to: {output_file}")
    print(f"Total images processed: {total_processed}")
    print(f"\nTiming Summary:")
    print(f" - Startup time (model + discovery): {startup_time:.2f}s")
    print(f" - Captioning time (processing only): {captioning_time:.2f}s")
    print(f" - Total time: {time.time() - total_start_time:.2f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch image captioning with BLIP Large model"
    )
    parser.add_argument(
        "--image_dir", 
        type=str, 
        required=True, 
        help="Path to the root image directory"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=25, 
        help="Batch size for DataLoader"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=4, 
        help="Number of worker threads for DataLoader"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        required=True, 
        help="Path to save output JSONL file"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        required=True,
        default="An image of",
        help="Prompt to pass to the model"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=30, 
        help="Maximum length of generated captions (in tokens)"
    )
    parser.add_argument(
        "--exclude_dirs", 
        nargs="+", 
        default=["facemesh"], 
        help="Subdirectory names to exclude from processing"
    )
    parser.add_argument(
        "--word_mapping_file", 
        type=str, 
        default=None, 
        help="Path to word mapping file (format: source:target, one per line)"
    )
    
    args = parser.parse_args()
    main(args)


# ============================================================
# EXAMPLE USAGE
# ============================================================
# python ImageCaptioning.py --image_dir "path/to/images" --batch_size 8 --num_workers 8 --output_file "output/captions.jsonl" --prompt "An image of" --max_length 30 --exclude_dirs facemesh --word_mapping_file "path/to/mappings.txt"

# --prompt "An image of" -> This is required for the BLIP model to generate a description of the image. Should be left as is.
# --max_lenght 30 -> This dictates the maximum legnth in tokens of the generated caption. 30 tokens (Aprox 30 word sentence) are sufficient, increasing this is not guaranteed to generate longer captions.
# --exclude_dirs facemesh -> This is done to exclude any images in the facemesh directory from processing as these are not actual images but rather facemesh data.
# --word_mapping_file "mappings.txt" -> This is a file that dictates what gendered words should be remapped to ex: woman -> person (The original & updated caption are saved as output)

# python ImageCaptioning.py --image_dir "G:\Thesis\ImageRetrieval\Professions_125k_Cleaned" --batch_size 8 --num_workers 8 --output_file "test_captions.jsonl" --prompt "An image of" --max_length 15 --exclude_dirs facemesh --word_mapping_file "mappings.txt"

# NOTE: The mappings.txt file is not finished but should be mostly sufficient for general use

# ============================================================
# WORD MAPPING FILE FORMAT (mappings.txt)
# ============================================================
# woman:person
# man:person
# boy:child
# girl:child
# he:they
# she:they
# his:their
# her:their
# himself:themselves
# herself:themselves
# ============================================================


# python ImageCaptioning.py --image_dir "E:\ImageRetrieval\StableDiffusionGeneratedImages\valid" --batch_size 8 --num_workers 8 --output_file "E:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\ImageCaptioning\captions.jsonl" --prompt "An image of" --max_length 30 --exclude_dirs face_crops --word_mapping_file "mappings.txt"

# python ImageCaptioning.py --image_dir "F:\ImageRetrieval\Professions_125k_ISCO_Aligned_1k_Subset" --batch_size 8 --num_workers 8 --output_file "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\ImageCaptioning\captions.jsonl" --prompt "An image of" --max_length 30 --exclude_dirs facemesh --word_mapping_file "mappings.txt"

# python ImageCaptioning.py --image_dir "F:\ImageRetrieval\Coco" --batch_size 8 --num_workers 8 --output_file "F:\ImageRetrieval\SpuriousFeatureImages\Coco\ImageCaptioning\captions.jsonl" --prompt "An image of" --max_length 30 --exclude_dirs facemesh --word_mapping_file "mappings.txt"