import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import os
import uuid
import queue
import threading
import argparse
import numpy as np
import cv2
import torch
import csv
from diffusers import DiffusionPipeline, LCMScheduler
from ultralytics import YOLO
from PIL import Image
import open_clip
import torch.nn as nn
from tqdm import tqdm
import json
import concurrent.futures
from collections import deque

"""
==============================================================================
PROGRAM OVERVIEW
==============================================================================

This program implements an automated pipeline for generating and validating
synthetic images of people in various professions using Stable Diffusion.

KEY COMPONENTS:

1. IMAGE GENERATION
   - Uses Stable Diffusion v1.5 with LCM LoRA for fast generation (4 steps)
   - Generates images based on profession-specific prompts from JSON config
   - Supports batch generation with configurable batch sizes

2. MULTI-THREADED VALIDATION
   - Asynchronous validation using a worker thread pool
   - Each image is validated against multiple criteria:
     * Person detection (YOLO)
     * Face detection (YOLO) - must detect exactly one face
     * Face segmentation (MediaPipe FaceMesh)
     * Spatial consistency between face and person bounding boxes
     * Aesthetic quality score using CLIP embeddings

3. VALIDATION CRITERIA
   - Person must be detected with sufficient confidence
   - Exactly one face must be present
   - Face must be spatially consistent with person (centroid-based association)
   - Face should be in upper half of person box (anatomical constraint)
   - Face-to-person size ratio must exceed minimum threshold
   - Aesthetic score must meet minimum quality threshold

4. OUTPUT STRUCTURE
   - valid/[profession]/: Valid images organized by profession
   - valid/[profession]/face_crops/: Segmented face crops
   - invalid/[profession]/: Invalid images with rejection reason in filename
   - ImageGenMetadata.csv: Metadata including boxes and aesthetic scores

5. PERFORMANCE OPTIMIZATIONS
   - Pipelined generation and validation (GPU generates while CPU validates)
   - Async I/O using thread pools for non-blocking file saves
   - Image caching to avoid redundant operations
   - Buffered CSV writing

WORKFLOW:
1. Load prompt configuration and model weights
2. For each profession:
   a. Generate images in batches
   b. Submit each image for async validation
   c. Compute aesthetic scores for valid detections
   d. Save valid images and face crops
   e. Log metadata to CSV
3. Continue until target number of valid images reached per profession

USAGE EXAMPLE:
python ImageGeneration.py \
    --output_dir "./output" \
    --yolo_person_path "models/yolo12s.pt" \
    --yolo_face_path "models/yolov12l-face.pt" \
    --prompt_config "prompts.json" \
    --total_images_per_label 1000 \
    --batch_size 2 \
    --device "cuda"
==============================================================================
"""

# ============================================================
# GEOMETRIC UTILITIES
# ============================================================

def load_prompt_config(path: str):
    """
    Load prompt configuration from a JSON file.
    
    Args:
        path: Path to the JSON configuration file
        
    Returns:
        tuple: (professions list, prompt_template string, negative_prompt string or None)
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    professions = cfg["professions"]
    prompt_template = cfg["prompt_template"]
    negative_prompt = cfg.get("negative_prompt", None)

    if negative_prompt in ("", "none", "None"):
        negative_prompt = None

    return professions, prompt_template, negative_prompt


def box_iou(a, b):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        a: First bounding box [x1, y1, x2, y2]
        b: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        float: IoU value between 0 and 1
    """
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))

    if area_a <= 0 or area_b <= 0:
        return 0.0

    return inter_area / float(area_a + area_b - inter_area)


def segment_face(face_roi_bgr, face_mesh):
    """
    Extract and segment a face from a region of interest using MediaPipe FaceMesh.
    
    Args:
        face_roi_bgr: Face region in BGR format
        face_mesh: MediaPipe FaceMesh object
        
    Returns:
        np.ndarray or None: Segmented face with black background, or None if segmentation fails
    """
    h, w, _ = face_roi_bgr.shape
    rgb = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0]
    pts = np.array([
        (int(lm.x * w), int(lm.y * h))
        for lm in landmarks.landmark
    ])

    x, y, w_box, h_box = cv2.boundingRect(pts)
    fx1, fy1, fx2, fy2 = x, y, x + w_box, y + h_box

    if pts.shape[0] < 10:
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)

    segmented = cv2.bitwise_and(face_roi_bgr, face_roi_bgr, mask=mask)

    pad = 10
    fx1 = max(0, fx1 - pad)
    fy1 = max(0, fy1 - pad)
    fx2 = min(w, fx2 + pad)
    fy2 = min(h, fy2 + pad)

    segmented = segmented[fy1:fy2, fx1:fx2]

    return segmented


# ============================================================
# ASYNC VALIDATION WORKER
# ============================================================

class ValidationWorker:
    """
    Multi-threaded worker for validating generated images asynchronously.
    
    Runs YOLO person/face detection and MediaPipe face segmentation on separate
    worker threads to avoid blocking the main generation pipeline.
    """
    
    def __init__(self, yolo_person_path, yolo_face_path, num_workers=2):
        """
        Initialize validation worker pool.
        
        Args:
            yolo_person_path: Path to YOLO person detection model
            yolo_face_path: Path to YOLO face detection model
            num_workers: Number of worker threads
        """
        self.yolo_person_path = yolo_person_path
        self.yolo_face_path = yolo_face_path

        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.workers = []

        for _ in range(num_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self.workers.append(t)

    def _worker_loop(self):
        """
        Main loop for worker threads. Loads models and processes validation tasks.
        """
        yolo_person = YOLO(self.yolo_person_path)
        yolo_face = YOLO(self.yolo_face_path)

        import mediapipe as mp
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        while not self.stop_event.is_set():
            try:
                task = self.task_queue.get(timeout=0.1)
                if task is None:
                    break

                uid, img_bgr, args = task
                valid, reason, face_box, person_box, face_crop = self._validate(
                    img_bgr, args, yolo_person, yolo_face, face_mesh
                )

                self.result_queue.put(
                    (uid, valid, reason, face_box, person_box, face_crop)
                )
                self.task_queue.task_done()

            except queue.Empty:
                continue

    def _validate(self, img_bgr, args, yolo_person, yolo_face, face_mesh):
        """
        Validate a single image for person/face detection and spatial consistency.
        
        Args:
            img_bgr: Image in BGR format
            args: Validation parameters
            yolo_person: YOLO person detection model
            yolo_face: YOLO face detection model
            face_mesh: MediaPipe FaceMesh object
            
        Returns:
            tuple: (valid, reason, face_box, person_box, face_crop)
        """
        # YOLO PERSON DETECTION
        rp = yolo_person(img_bgr, verbose=False, device="cpu")[0]
        if rp.boxes is None:
            return False, "no_person", None, None, None

        person_boxes = [
            b for b in rp.boxes.data.cpu().numpy()
            if int(b[5]) == 0 and float(b[4]) >= args.min_person_conf
        ]
        if not person_boxes:
            return False, "no_person_conf", None, None, None

        # YOLO FACE DETECTION (MANDATORY)
        rf = yolo_face(img_bgr, verbose=False, device="cpu")[0]
        if rf.boxes is None:
            return False, "no_face", None, None, None

        face_boxes = [
            b for b in rf.boxes.data.cpu().numpy()
            if float(b[4]) >= args.min_face_conf
        ]
        if len(face_boxes) != 1:
            return False, f"{len(face_boxes)}_faces", None, None, None

        fx1, fy1, fx2, fy2, _, _ = face_boxes[0]
        fx1, fy1, fx2, fy2 = map(int, [fx1, fy1, fx2, fy2])
        face_box = [fx1, fy1, fx2, fy2]

        face_roi = img_bgr[fy1:fy2, fx1:fx2]
        if face_roi.size == 0:
            return False, "invalid_face_crop", None, None, None

        # FACEMESH REFINEMENT (REQUIRED)
        face_crop = segment_face(face_roi, face_mesh)
        if face_crop is None:
            return False, "facemesh_failed", None, None, None

        # SPATIAL + ANATOMICAL CONSISTENCY
        face_area = max(1, (fx2 - fx1) * (fy2 - fy1))
        fcx, fcy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0

        min_tiny_face_ratio = getattr(args, "min_tiny_face_ratio", 0.002)

        best_candidate = None
        for pb in person_boxes:
            px1, py1, px2, py2, _, _ = pb
            px1, py1, px2, py2 = map(int, [px1, py1, px2, py2])

            person_w = max(1, (px2 - px1))
            person_h = max(1, (py2 - py1))
            person_area = float(person_w * person_h)

            centroid_inside = (px1 <= fcx <= px2 and py1 <= fcy <= py2)

            person_mid_y = (py1 + py2) / 2.0
            upper_half_ok = (fcy <= person_mid_y)

            ratio = face_area / person_area

            score = 0.0
            if centroid_inside:
                score += 10.0
            if upper_half_ok:
                score += 2.0
            score += min(ratio, 0.05) * 20.0

            if best_candidate is None or score > best_candidate[0]:
                best_candidate = (score, [px1, py1, px2, py2], ratio, centroid_inside, upper_half_ok)

        if best_candidate is None:
            return False, "no_person_conf", None, None, None

        _, best_person_box, best_ratio, best_centroid_inside, best_upper_half_ok = best_candidate

        if best_centroid_inside:
            if not best_upper_half_ok:
                if best_ratio < min_tiny_face_ratio:
                    return False, "face_too_small_or_low", None, None, None
                return True, "ok", face_box, best_person_box, face_crop

            if best_ratio < min_tiny_face_ratio:
                return False, "face_too_small", None, None, None

            return True, "ok", face_box, best_person_box, face_crop

        expand_y = getattr(args, "person_box_expand_y", 0.15)
        px1, py1, px2, py2 = best_person_box
        expand_px = int((px2 - px1) * 0.05)
        expand_py = int((py2 - py1) * expand_y)

        exp_px1 = max(0, px1 - expand_px)
        exp_py1 = max(0, py1 - expand_py)
        exp_px2 = px2 + expand_px
        exp_py2 = py2

        expanded_centroid_inside = (exp_px1 <= fcx <= exp_px2 and exp_py1 <= fcy <= exp_py2)

        if expanded_centroid_inside:
            if best_ratio < min_tiny_face_ratio:
                return False, "face_too_small", None, None, None
            return True, "ok_expanded", face_box, best_person_box, face_crop

        return False, "face_person_association_failed", None, None, None

    def submit(self, idx, img_bgr, args):
        """
        Submit an image for validation.
        
        Args:
            idx: Unique identifier for the image
            img_bgr: Image in BGR format
            args: Validation parameters
        """
        self.task_queue.put((idx, img_bgr, args))

    def get_result(self):
        """
        Get the next validation result from the queue.
        
        Returns:
            tuple: (uid, valid, reason, face_box, person_box, face_crop)
        """
        return self.result_queue.get()

    def shutdown(self):
        """
        Stop all worker threads and clean up resources.
        """
        self.stop_event.set()
        for _ in self.workers:
            self.task_queue.put(None)
        for w in self.workers:
            w.join()


def generate_with_validation(
    prompt,
    negative_prompt,
    label,
    total_images,
    batch_size,
    valid_dir,
    invalid_dir,
    validator,
    args,
    csv_path
):
    """
    Generate images with async validation pipeline and save valid results.
    
    This function generates images in batches, validates them asynchronously,
    computes aesthetic scores, and saves valid images with metadata.
    
    Args:
        prompt: Text prompt for image generation
        negative_prompt: Negative prompt to avoid unwanted features
        label: Class label (e.g., profession name)
        total_images: Target number of valid images to generate
        batch_size: Number of images to generate per batch
        valid_dir: Directory to save valid images
        invalid_dir: Directory to save invalid images (or None)
        validator: ValidationWorker instance
        args: Generation and validation parameters
        csv_path: Path to CSV file for metadata logging
    """
    valid_label_dir = os.path.join(valid_dir, label)
    os.makedirs(valid_label_dir, exist_ok=True)

    face_crop_dir = os.path.join(valid_label_dir, "face_crops")
    os.makedirs(face_crop_dir, exist_ok=True)

    if invalid_dir is not None:
        invalid_label_dir = os.path.join(invalid_dir, label)
        os.makedirs(invalid_label_dir, exist_ok=True)
    else:
        invalid_label_dir = None

    init_csv(csv_path)

    existing_valid = count_existing_valid(valid_dir, label)
    if existing_valid >= total_images:
        print(f"[INFO] {label}: already has {existing_valid} valid images. Skipping.")
        return

    valid_count = existing_valid
    img_idx = existing_valid
    
    image_cache = {}
    
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    
    io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    io_futures = []

    pbar = tqdm(total=total_images, initial=valid_count, desc=label, dynamic_ncols=True)

    pending_validations = 0
    generation_buffer = 2

    try:
        while valid_count < total_images:
            while pending_validations < generation_buffer * batch_size and valid_count + pending_validations < total_images:
                images = pipe(
                    prompt=[prompt] * batch_size,
                    negative_prompt=[negative_prompt] * batch_size,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale
                ).images

                for img in images:
                    uid = f"{label}_{img_idx:06d}_{uuid.uuid4().hex[:6]}"
                    img_rgb = np.array(img)
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                    image_cache[uid] = img_rgb
                    validator.submit(uid, img_bgr, args)
                    img_idx += 1
                    pending_validations += 1

            uid, valid, reason, face_box, person_box, face_crop = validator.get_result()

            pending_validations -= 1
            
            img_rgb = image_cache.pop(uid, None)
            if img_rgb is None:
                continue

            if valid:
                aesth_score = compute_aesthetic_score(img_rgb)
                if aesth_score < args.min_aesthetic:
                    valid = False
                    reason = f"low_aesthetic_{aesth_score:.2f}"

            if valid:
                img_path = os.path.join(valid_label_dir, f"{uid}.png")
                future = io_executor.submit(save_image_async, img_rgb, img_path)
                io_futures.append(future)

                if face_crop is not None:
                    face_path = os.path.join(face_crop_dir, f"{uid}_face.png")
                    future = io_executor.submit(save_face_crop_async, face_crop, face_path)
                    io_futures.append(future)

                csv_writer.writerow([
                    f"{uid}.png",
                    label,
                    round(aesth_score, 3),
                    face_box,
                    person_box
                ])

                valid_count += 1
                pbar.update(1)

            elif invalid_label_dir is not None:
                invalid_path = os.path.join(invalid_label_dir, f"{uid}_{reason}.png")
                future = io_executor.submit(save_image_async, img_rgb, invalid_path)
                io_futures.append(future)


            if len(io_futures) > 20:
                io_futures = [f for f in io_futures if not f.done()]

    finally:
        concurrent.futures.wait(io_futures)
        io_executor.shutdown(wait=True)
        csv_file.close()
        pbar.close()
        
    print(f"[INFO] Completed {label}: {valid_count} valid images")


def save_image_async(img_rgb, path):
    """
    Save image in background thread to avoid blocking the main pipeline.
    
    Args:
        img_rgb: Image array in RGB format
        path: Output file path
    """
    Image.fromarray(img_rgb).save(path)


def save_face_crop_async(face_crop_bgr, path):
    """
    Save face crop in background thread.
    
    Args:
        face_crop_bgr: Face crop in BGR format
        path: Output file path
    """
    cv2.imwrite(path, face_crop_bgr)


def count_existing_valid(valid_dir, label):
    """
    Count existing valid images for a given label.
    
    Args:
        valid_dir: Directory containing valid images
        label: Class label subdirectory
        
    Returns:
        int: Number of existing PNG images
    """
    if not valid_dir:
        return 0
    label_dir = os.path.join(valid_dir, label)
    if not os.path.isdir(label_dir):
        return 0
    return len([f for f in os.listdir(label_dir) if f.lower().endswith(".png")])


def init_csv(csv_path):
    """
    Initialize CSV file with headers if it doesn't exist.
    
    Args:
        csv_path: Path to the CSV metadata file
    """
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "image_name",
                "label",
                "aesthetic_score",
                "face_box",
                "person_box"
            ])


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image generation + validation pipeline")

    # Core paths
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Base output directory (valid/, invalid/, CSV will be created inside)")

    parser.add_argument("--yolo_person_path", type=str, required=True,
                        help="Path to YOLO person detection model")

    parser.add_argument("--yolo_face_path", type=str, required=True,
                        help="Path to YOLO face detection model")
    
    parser.add_argument("--prompt_config", type=str, required=True,
                        help="Path to JSON file defining professions, prompt template, and negative prompt")

    # Generation controls
    parser.add_argument("--total_images_per_label", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=1.0)

    # Validation thresholds
    parser.add_argument("--min_person_conf", type=float, default=0.4)
    parser.add_argument("--min_face_conf", type=float, default=0.4)
    parser.add_argument("--min_face_person_ratio", type=float, default=0.03)
    parser.add_argument("--min_tiny_face_ratio", type=float, default=0.002)
    parser.add_argument("--person_box_expand_y", type=float, default=0.15)
    parser.add_argument("--min_aesthetic", type=float, default=3.0)

    # Runtime
    parser.add_argument("--num_validator_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Determinism
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Derived paths
    output_dir = os.path.abspath(args.output_dir)
    valid_dir = os.path.join(output_dir, "valid")
    invalid_dir = os.path.join(output_dir, "invalid")
    csv_path = os.path.join(output_dir, "ImageGenMetadata.csv")

    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(invalid_dir, exist_ok=True)

    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Valid images → {valid_dir}")
    print(f"[INFO] Invalid images → {invalid_dir}")
    print(f"[INFO] CSV metadata → {csv_path}")

    # Diffusion pipeline
    pipe = DiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    pipe.fuse_lora()
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.to(args.device)

    pipe.set_progress_bar_config(disable=True)

    # Validator
    validator = ValidationWorker(
        yolo_person_path=args.yolo_person_path,
        yolo_face_path=args.yolo_face_path,
        num_workers=args.num_validator_workers
    )

    # CLIP + Aesthetic model
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        model_name="ViT-L-14",
        pretrained="laion2b_s32b_b82k",
        device=args.device
    )
    clip_model.eval()

    aesthetic_model = nn.Linear(768, 1)
    aesthetic_model.load_state_dict(
        torch.load("models/sa_0_4_vit_l_14_linear.pth", map_location="cpu")
    )
    aesthetic_model.eval().to(args.device)

    @torch.no_grad()
    def compute_aesthetic_score(img_rgb: np.ndarray) -> float:
        """
        Compute aesthetic quality score for an image using CLIP and a learned linear regressor.
        
        Args:
            img_rgb: Image array in RGB format
            
        Returns:
            float: Aesthetic score (higher is better)
        """
        pil = Image.fromarray(img_rgb)
        clip_img = clip_preprocess(pil).unsqueeze(0).to(args.device)
        img_features = clip_model.encode_image(clip_img)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        return aesthetic_model(img_features).item()

    # Prompt configuration
    profession_list, prompt_template, negative_prompts = load_prompt_config(args.prompt_config)

    print(f"[INFO] Loaded {len(profession_list)} professions from {args.prompt_config}")

    # Run generation for all professions
    for index, label in enumerate(profession_list):
        prompt = prompt_template.format(profession=label)

        print(f"\n[INFO] Processing label {index}='{label}'")

        generate_with_validation(
            prompt=prompt,
            negative_prompt=negative_prompts,
            label=label,
            total_images=args.total_images_per_label,
            batch_size=args.batch_size,
            valid_dir=valid_dir,
            invalid_dir=invalid_dir,
            validator=validator,
            args=args,
            csv_path=csv_path
        )

    validator.shutdown()

# .\.venv\Scripts\python.exe ImageGeneration.py --output_dir "E:\ImageRetrieval\StableDiffusionGeneratedImages" --yolo_person_path "models\yolo12s.pt" --yolo_face_path "models\yolov12l-face.pt" --total_images_per_label 1000 --batch_size 2 --device "cuda" --prompt_config "prompts.json"