import os

# Silence TensorFlow / MediaPipe C++ logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

# Silence absl logging
from absl import logging
logging.set_verbosity(logging.ERROR)
logging.set_stderrthreshold(logging.ERROR)

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import mediapipe as mp

from datasets import load_dataset
from PIL import Image

# ============================================================
# CONFIG
# ============================================================

FAIRFACE_CONFIG = "0.25"          # or "0.25"
OUTPUT_DIR = Path("F:\Thesis\Segmented_FairFace")

NUM_WORKERS = 4
BATCH_SIZE = 32

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# FACE SEGMENTATION (UNCHANGED)
# ============================================================

def segment_face(image, face_mesh):
    h_img, w_img, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0]
    points = np.array([(int(lm.x * w_img), int(lm.y * h_img)) for lm in landmarks.landmark])

    x, y, w_box, h_box = cv2.boundingRect(points)
    if w_box < 50 or h_box < 50:
        return None

    aspect = h_box / w_box
    if aspect < 0.6 or aspect > 2.5:
        return None

    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)

    segmented = cv2.bitwise_and(image, image, mask=mask)

    pad = 10
    x = max(0, x - pad)
    y = max(0, y - pad)
    w_box = min(w_img, x + w_box + 2 * pad) - x
    h_box = min(h_img, y + h_box + 2 * pad) - y

    return segmented[y:y + h_box, x:x + w_box]

# ============================================================
# BATCH WORKER
# ============================================================

def process_batch(batch, race_names, gender_names, split):
    records = []

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        for idx, image, race_id, gender_id in batch:
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            crop = segment_face(img, face_mesh)
            if crop is None:
                continue

            race = race_names[race_id]
            gender = gender_names[gender_id]

            out_dir = OUTPUT_DIR / split / race / gender
            out_dir.mkdir(parents=True, exist_ok=True)

            fname = f"{idx}.jpg"
            out_path = out_dir / fname
            cv2.imwrite(str(out_path), crop)

            records.append({
                "image_id": idx,
                "split": split,
                "race": race,
                "gender": gender,
                "path": str(out_path).replace("\\", "/")
            })

    return records

# ============================================================
# MAIN
# ============================================================

def process_fairface():
    print("\n--- Processing FairFace Dataset (train + validation) ---")

    all_records = []

    for split in ["train", "validation"]:
        print(f"\n-> Processing split: {split}")

        dataset = load_dataset(
            "HuggingFaceM4/FairFace",
            FAIRFACE_CONFIG,
            split=split
        )

        race_names = dataset.features["race"].names
        gender_names = dataset.features["gender"].names

        tasks = [
            (i, sample["image"], sample["race"], sample["gender"])
            for i, sample in enumerate(dataset)
        ]

        batches = [
            tasks[i:i + BATCH_SIZE]
            for i in range(0, len(tasks), BATCH_SIZE)
        ]

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [
                executor.submit(
                    process_batch,
                    batch,
                    race_names,
                    gender_names,
                    split
                )
                for batch in batches
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Processing {split}"
            ):
                all_records.extend(future.result())

    df = pd.DataFrame(all_records)
    csv_path = OUTPUT_DIR / "annotations.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n Saved {len(df)} face crops total")
    print(f" Combined annotations → {csv_path}")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    process_fairface()