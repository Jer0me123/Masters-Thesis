from datasets import load_dataset
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Where images will be saved
OUT_DIR = Path("F:\Thesis\\Fairface_Dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
dataset = load_dataset("HuggingFaceM4/FairFace", "0.25")

from pathlib import Path
from PIL import Image

def save_split(split, split_name):
    split_dir = OUT_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    race_names = split.features["race"].names
    gender_names = split.features["gender"].names

    for idx, sample in enumerate(split):
        race = race_names[sample["race"]]
        gender = gender_names[sample["gender"]]

        subdir = split_dir / race / gender
        subdir.mkdir(parents=True, exist_ok=True)

        image = sample["image"]
        image.save(subdir / f"{idx}.jpg")


# Save train & validation
save_split(dataset["train"], "train")
save_split(dataset["validation"], "validation")
