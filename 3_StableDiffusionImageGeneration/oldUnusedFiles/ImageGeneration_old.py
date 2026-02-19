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
import time
import json
import concurrent.futures
from collections import deque

# ============================================================
# GEOMETRIC UTILITIES
# ============================================================
def load_prompt_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    professions = cfg["professions"]
    prompt_template = cfg["prompt_template"]
    negative_prompt = cfg.get("negative_prompt", None)

    # Normalize empty or explicit nulls
    if negative_prompt in ("", "none", "None"):
        negative_prompt = None

    return professions, prompt_template, negative_prompt

def box_iou(a, b):
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
    Runs FaceMesh on a YOLO face ROI.
    Returns:
        segmented_face (BGR image with black background) or None
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

    # Crop with padding
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
    def __init__(self, yolo_person_path, yolo_face_path, num_workers=2):
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
        # ---------------- YOLO PERSON DETECTION ----------------
        rp = yolo_person(img_bgr, verbose=False, device="cpu")[0]
        if rp.boxes is None:
            return False, "no_person", None, None, None

        person_boxes = [
            b for b in rp.boxes.data.cpu().numpy()
            if int(b[5]) == 0 and float(b[4]) >= args.min_person_conf
        ]
        if not person_boxes:
            return False, "no_person_conf", None, None, None

        # ---------------- YOLO FACE DETECTION (MANDATORY) ----------------
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

        # ---------------- FACEMESH REFINEMENT (REQUIRED) ----------------
        face_crop = segment_face(face_roi, face_mesh)
        if face_crop is None:
            return False, "facemesh_failed", None, None, None

        # ---------------- SPATIAL + ANATOMICAL CONSISTENCY ----------------
        # Key improvements:
        # 1) Accept when face centroid is inside person box (primary association)
        # 2) Do NOT use IoU as a primary criterion (it is structurally tiny for face vs person)
        # 3) Use face/person ratio ONLY as a "tiny face" safeguard (prevents distant/background faces)
        # 4) Add an anatomical constraint: face centroid should be in upper half of the person box

        face_area = max(1, (fx2 - fx1) * (fy2 - fy1))
        fcx, fcy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0

        # "tiny face" safeguard: only reject if face is implausibly small relative to the best-matching person
        # This is intentionally much lower than the previous 0.03 which biases against full-body shots.
        min_tiny_face_ratio = getattr(args, "min_tiny_face_ratio", 0.002)

        # If you still want to keep the old args.min_face_person_ratio, use it as an optional "soft target"
        # but do NOT hard-reject when centroid is inside.
        soft_min_ratio = getattr(args, "min_face_person_ratio", 0.03)

        best_candidate = None  # (score, person_box, ratio, centroid_inside, upper_half_ok)
        for pb in person_boxes:
            px1, py1, px2, py2, _, _ = pb
            px1, py1, px2, py2 = map(int, [px1, py1, px2, py2])

            person_w = max(1, (px2 - px1))
            person_h = max(1, (py2 - py1))
            person_area = float(person_w * person_h)

            centroid_inside = (px1 <= fcx <= px2 and py1 <= fcy <= py2)

            # Anatomical prior: face should be above the midpoint of the person's box
            person_mid_y = (py1 + py2) / 2.0
            upper_half_ok = (fcy <= person_mid_y)

            ratio = face_area / person_area

            # Scoring: prefer centroid-inside, then upper-half, then larger ratio
            score = 0.0
            if centroid_inside:
                score += 10.0
            if upper_half_ok:
                score += 2.0
            score += min(ratio, 0.05) * 20.0  # cap ratio contribution

            if best_candidate is None or score > best_candidate[0]:
                best_candidate = (score, [px1, py1, px2, py2], ratio, centroid_inside, upper_half_ok)

        # No candidate (shouldn't happen if person_boxes non-empty)
        if best_candidate is None:
            return False, "no_person_conf", None, None, None

        _, best_person_box, best_ratio, best_centroid_inside, best_upper_half_ok = best_candidate

        # Primary acceptance rule:
        # If face centroid is inside the best person box AND face mesh succeeded, accept.
        # This fixes false negatives on full-body images where face is small.
        if best_centroid_inside:
            # Optional: if you want to reject anatomically implausible cases, keep this check.
            # In most full-body images this is true; in weird false positives it helps.
            if not best_upper_half_ok:
                # Still allow if ratio is reasonably sized; otherwise reject
                if best_ratio < min_tiny_face_ratio:
                    return False, "face_too_small_or_low", None, None, None
                return True, "ok", face_box, best_person_box, face_crop

            # Tiny-face safeguard: only reject extremely small faces even if inside
            if best_ratio < min_tiny_face_ratio:
                return False, "face_too_small", None, None, None

            # Soft ratio is NOT a hard constraint; it is informational only.
            return True, "ok", face_box, best_person_box, face_crop

        # Secondary acceptance rule (handles truncated/odd person boxes):
        # If face centroid is just outside but near the top edge, allow an expanded-box check.
        # This protects against person boxes that start below the head.
        expand_y = getattr(args, "person_box_expand_y", 0.15)  # 15% vertical expansion upwards
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

        # Otherwise, association failed
        return False, "face_person_association_failed", None, None, None

    def submit(self, idx, img_bgr, args):
        self.task_queue.put((idx, img_bgr, args))

    def get_result(self):
        return self.result_queue.get()

    def shutdown(self):
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
    # start_time = time.time()
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
    
    # Use deque for faster pop operations
    image_cache = {}
    
    # Pre-open CSV file to avoid repeated file operations
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    
    # Thread pool for I/O operations (saving images)
    io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    io_futures = []

    pbar = tqdm(total=total_images, initial=valid_count, desc=label, dynamic_ncols=True)

    # Pipeline to keep GPU busy: generate next batch while validating current
    pending_validations = 0
    generation_buffer = 2  # Number of batches to keep in validation queue

    try:
        while valid_count < total_images:
            start_time=time.time()
            image_get_start = time.time()

            # Generate images in batches, keeping the pipeline full
            while pending_validations < generation_buffer * batch_size and valid_count + pending_validations < total_images:
                images = pipe(
                    prompt=[prompt] * batch_size,
                    negative_prompt=[negative_prompt] * batch_size,
                    num_inference_steps=4,
                    guidance_scale=1.0
                ).images

                # Submit all images for validation immediately
                for img in images:
                    uid = f"{label}_{img_idx:06d}_{uuid.uuid4().hex[:6]}"
                    img_rgb = np.array(img)
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                    image_cache[uid] = img_rgb
                    validator.submit(uid, img_bgr, args)
                    img_idx += 1
                    pending_validations += 1
            image_get_end = time.time()
            # print(f"Image Gen: {image_get_end-image_get_start:.2f} sec")


            # Process validation results
            validation_start = time.time()
            uid, valid, reason, face_box, person_box, face_crop = validator.get_result()
            validation_end = time.time()
            # print(f"Image Validation: {validation_end-validation_start:.2f} sec")

            pending_validations -= 1
            
            img_rgb = image_cache.pop(uid, None)
            if img_rgb is None:
                continue

            if valid:
                aes_score_start = time.time()
                # Compute aesthetic score (can't avoid this on main thread due to GPU)
                aesth_score = compute_aesthetic_score(img_rgb)
                if aesth_score < args.min_aesthetic:
                    valid = False
                    reason = f"low_aesthetic_{aesth_score:.2f}"
                aes_score_end = time.time()
                # print(f"Aes Score: {aes_score_end-aes_score_start:.2f} sec")


            if valid:
                save_start = time.time()
                # Async I/O: save images in background threads
                img_path = os.path.join(valid_label_dir, f"{uid}.png")
                future = io_executor.submit(save_image_async, img_rgb, img_path)
                io_futures.append(future)

                if face_crop is not None:
                    face_path = os.path.join(face_crop_dir, f"{uid}_face.png")
                    future = io_executor.submit(save_face_crop_async, face_crop, face_path)
                    io_futures.append(future)

                # Write to CSV (buffered, so fast)
                csv_writer.writerow([
                    f"{uid}.png",
                    label,
                    round(aesth_score, 3),
                    face_box,
                    person_box
                ])

                valid_count += 1
                pbar.update(1)
                save_end = time.time()
                # print(f"Saving: {save_end-save_start:.2f} sec")

            elif invalid_label_dir is not None:
                # Async save invalid images too
                invalid_path = os.path.join(invalid_label_dir, f"{uid}_{reason}.png")
                future = io_executor.submit(save_image_async, img_rgb, invalid_path)
                io_futures.append(future)

            end_time=time.time()
            # print(f"Overall Time: {end_time-start_time:.2f} sec\n")

            # Clean up completed I/O futures periodically
            if len(io_futures) > 20:
                io_futures = [f for f in io_futures if not f.done()]

    finally:
        # Wait for all I/O operations to complete
        concurrent.futures.wait(io_futures)
        io_executor.shutdown(wait=True)
        csv_file.close()
        pbar.close()
        
    end_time = time.time()
    print(f"[INFO] {label}: Total time: {end_time - start_time:.2f}s, {valid_count} valid images")

# Helper functions for async I/O
def save_image_async(img_rgb, path):
    """Save image in background thread"""
    Image.fromarray(img_rgb).save(path)

def save_face_crop_async(face_crop_bgr, path):
    """Save face crop in background thread"""
    cv2.imwrite(path, face_crop_bgr)

def count_existing_valid(valid_dir, label):
    if not valid_dir:
        return 0
    label_dir = os.path.join(valid_dir, label)
    if not os.path.isdir(label_dir):
        return 0
    return len([f for f in os.listdir(label_dir) if f.lower().endswith(".png")])

def init_csv(csv_path):
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

    # ---------------- Core paths ----------------
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Base output directory (valid/, invalid/, CSV will be created inside)")

    parser.add_argument("--yolo_person_path", type=str, required=True,
                        help="Path to YOLO person detection model")

    parser.add_argument("--yolo_face_path", type=str, required=True,
                        help="Path to YOLO face detection model")
    
    parser.add_argument("--prompt_config", type=str, required=True,
                        help="Path to JSON file defining professions, prompt template, and negative prompt")

    # ---------------- Generation controls ----------------
    parser.add_argument("--total_images_per_label", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=1.0)

    # ---------------- Validation thresholds ----------------
    parser.add_argument("--min_person_conf", type=float, default=0.4)
    parser.add_argument("--min_face_conf", type=float, default=0.4)
    parser.add_argument("--min_face_person_ratio", type=float, default=0.03)
    parser.add_argument("--min_tiny_face_ratio", type=float, default=0.002)
    parser.add_argument("--person_box_expand_y", type=float, default=0.15)
    parser.add_argument("--min_aesthetic", type=float, default=3.0)

    # ---------------- Runtime ----------------
    parser.add_argument("--num_validator_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # ---------------- Determinism ----------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---------------- Derived paths ----------------
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

    # ---------------- Diffusion pipeline ----------------
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

    # pipe.unet = torch.compile(
    #     pipe.unet,
    #     mode="reduce-overhead",
    #     backend="eager"
    # )

    pipe.set_progress_bar_config(disable=True)

    # ---------------- Validator ----------------
    validator = ValidationWorker(
        yolo_person_path=args.yolo_person_path,
        yolo_face_path=args.yolo_face_path,
        num_workers=args.num_validator_workers
    )

    # ---------------- CLIP + Aesthetic model ----------------
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        model_name="ViT-L-14",
        pretrained="laion2b_s32b_b82k",
        device=args.device
    )
    clip_model.eval()

    aesthetic_model = nn.Linear(768, 1)
    aesthetic_model.load_state_dict(
        torch.load("sa_0_4_vit_l_14_linear.pth", map_location="cpu")
    )
    aesthetic_model.eval().to(args.device)

    @torch.no_grad()
    def compute_aesthetic_score(img_rgb: np.ndarray) -> float:
        pil = Image.fromarray(img_rgb)
        clip_img = clip_preprocess(pil).unsqueeze(0).to(args.device)
        img_features = clip_model.encode_image(clip_img)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        return aesthetic_model(img_features).item()


    # ---------------- Prompt configuration ----------------
    profession_list, prompt_template, negative_prompts = load_prompt_config(args.prompt_config)

    print(f"[INFO] Loaded {len(profession_list)} professions from {args.prompt_config}")

    # "black and white, blurry, low resolution, pixelated, jpeg artifacts, distorted face, deformed face, asymmetrical face, extra face, cropped, cut off, out of frame, partial face, cgi, 3d render, cartoon, illustration, watermark, text, logo",

    # ---------------- Professions ----------------
    # profession_list = [
    #     'Accountant', 'Actor', 'Actuarial Analyst', 'Actuary', 'Administrative Assistant',
    #     'Administrator', 'Air Traffic Controller', 'Airplane Pilot', 'Analyst',
    #     'Animal Trainer', 'Anthropologist', 'Appraiser', 'Archaeologist', 'Architect',
    #     'Archivist', 'Art Director', 'Artist', 'Astronaut', 'Athlete', 'Attorney',
    #     'Audio Technician', 'Auditor', 'Automotive Designer', 'Baker', 'Baker Assistant',
    #     'Banker', 'Bankruptcy Specialist', 'Barber', 'Barista', 'Bartender',
    #     'Bioinformatician', 'Biologist', 'Biomedical Engineer', 'Blacksmith', 'Bodyguard',
    #     'Bounty Hunter', 'Brand Manager', 'Brewer', 'Bricklayer', 'Broker',
    #     'Budget Analyst', 'Builder', 'Butcher', 'Caregiver', 'Carpenter',
    #     'Cartographer', 'Chef', 'Chemical Engineer', 'Chemist', 'Chiropractor',
    #     'Civil Engineer', 'Claims Adjuster', 'Cleaner', 'Clerk',
    #     'Clinical Laboratory Scientist', 'Coach', 'Comedian', 'Compliance Officer',
    #     'Composer', 'Conservation Officer', 'Construction Worker', 'Content Creator',
    #     'Cook', 'Copywriter', 'Court Reporter', 'Crime Scene Investigator',
    #     'Customer Support Specialist', 'DJ', 'Dancer', 'Data Scientist',
    #     'Database Administrator', 'Debt Counselor', 'Delivery Driver', 'Dentist',
    #     'Designer', 'Detective', 'Development Officer', 'Dietitian', 'Director',
    #     'Doctor', 'Dog Walker', 'Driver', 'Economist', 'Editor',
    #     'Electrical Technician', 'Electrician',
    #     'Emergency Management Specialist', 'Engineer', 'Entrepreneur',
    #     'Environmental Engineer', 'Ergonomist', 'Estate Planner',
    #     'Event Coordinator', 'Executive Assistant', 'Facilities Manager', 'Farmer',
    #     'Fashion Designer', 'Financial Analyst', 'Firefighter', 'Fisherman',
    #     'Flight Attendant', 'Flight Dispatcher', 'Florist', 'Forensic Scientist',
    #     'Freight Coordinator', 'Gardener', 'Genetic Counselor', 'Geologist',
    #     'Grant Writer', 'Graphic Designer', 'Hairdresser', 'Handyman',
    #     'Health Inspector', 'Historian', 'Hotel Concierge',
    #     'Human Resources Specialist', 'IT Support Specialist', 'Ice Cream Maker',
    #     'Illustrator', 'Industrial Designer', 'Insurance Underwriter',
    #     'Investment Banker', 'Janitor', 'Journalist', 'Judge', 'Laborer', 'Lawyer',
    #     'Librarian', 'Lifeguard', 'Loan Officer', 'Logger', 'Logistics Manager',
    #     'Magician', 'Makeup Artist', 'Marine Biologist',
    #     'Market Research Analyst', 'Marketing Manager', 'Mathematician', 'Mechanic',
    #     'Medical Researcher', 'Meteorologist', 'Midwife', 'Miner', 'Model', 'Musician',
    #     'News Anchor', 'Nurse', 'Nutritionist', 'Occupational Therapist',
    #     'Oceanographer', 'Office Assistant', 'Operations Manager', 'Optician',
    #     'Painter', 'Paralegal', 'Paramedic', 'Park Ranger', 'Pastry Chef',
    #     'Payroll Specialist', 'Personal Trainer', 'Pharmacist', 'Photographer',
    #     'Physical Therapist', 'Physicist', 'Pilot', 'Plumber', 'Police Officer',
    #     'Politician', 'Procurement Officer', 'Professor', 'Property Manager',
    #     'Psychologist', 'Quality Assurance Inspector', 'Realtor', 'Researcher',
    #     'Roofer', 'Sailor', 'Salesperson', 'Scientist', 'Screenwriter',
    #     'Security Officer', 'Singer', 'Skilled Technician', 'Social Worker',
    #     'Software Engineer', 'Soldier', 'Sound Engineer', 'Statistician', 'Surgeon',
    #     'Tailor', 'Teacher', 'Technician', 'Therapist', 'Translator',
    #     'Urban Planner', 'Veterinarian', 'Videographer', 'Waiter', 'Welder',
    #     'Writer', 'Zoologist'
    # ]

    # NEG_PROMPTS = None

    # ---------------- Run ----------------
    for label in profession_list:
        # prompt = (
        #     f"full-body realistic photo of a {label}, standing, centered, "
        #     f"in a professional setting appropriate for their occupation, "
        #     f"sharp focus, 35mm lens, natural professional lighting"
        # )
        prompt = prompt_template.format(profession=label)

        print(f"\n[INFO] Processing label='{label}'")

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

# python ImageGeneration.py --output_dir "G:\Thesis\ImageGentest" --yolo_person_path "models\yolo12s.pt" --yolo_face_path "models\yolov12l-face.pt" --total_images_per_label 1000 --batch_size 2 --device "cuda" --prompt_config "prompts.json"

# python ImageGeneration.py --output_dir "G:\Thesis\ImageGentest" --yolo_person_path "yolo12s.pt" --yolo_face_path "yolo12l-face.pt" --total_images_per_label 1000 --batch_size 2 --device "cuda" --prompt_config "prompts.json"


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--min_person_conf", type=float, default=0.4)
#     parser.add_argument("--min_face_conf", type=float, default=0.4)
#     parser.add_argument("--iou_thresh", type=float, default=0.3)
#     parser.add_argument("--min_face_person_ratio", type=float, default=0.03)
#     parser.add_argument("--min_aesthetic", type=float, default=3.0) # Was previously set to 5 but good images were being discarded
#     args = parser.parse_args([])

#     # Diffusion setup
#     pipe = DiffusionPipeline.from_pretrained(
#         "runwayml/stable-diffusion-v1-5",
#         torch_dtype=torch.float16,
#         safety_checker=None,
#         requires_safety_checker=False,
#     )
#     pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
#     pipe.fuse_lora()
#     pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
#     pipe.to("cuda")
#     pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", backend="eager")
#     pipe.set_progress_bar_config(disable=True)

#     # Validator
#     validator = ValidationWorker(
#         yolo_person_path="models/yolov8n.pt", #"yolo12s.pt",
#         yolo_face_path="models/yolov8n-face.pt", #"yolov12l-face.pt",
#         num_workers=2
#     )

#     # Load CLIP
#     clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
#         model_name="ViT-L-14",
#         pretrained="laion2b_s32b_b82k",
#         device="cuda"
#     )

#     clip_model.eval()

#     # Load pretrained aesthetic regressor (LAION)
#     # LAION aesthetic predictor is a single linear layer
#     aesthetic_model = nn.Linear(768, 1)
#     aesthetic_model.load_state_dict(
#         torch.load("sa_0_4_vit_l_14_linear.pth", map_location="cpu")
#     )
#     aesthetic_model.eval().to("cuda")

#     @torch.no_grad()
#     def compute_aesthetic_score(img_rgb: np.ndarray) -> float:
#         pil = Image.fromarray(img_rgb)
#         clip_img = clip_preprocess(pil).unsqueeze(0).to("cuda")

#         img_features = clip_model.encode_image(clip_img)
#         img_features = img_features / img_features.norm(dim=-1, keepdim=True)

#         score = aesthetic_model(img_features).item()
#         return score
    
#     TOTAL_IMAGES_PER_LABEL = 1000
#     BATCH_SIZE = 2

#     NEG_PROMPTS = None 

#     # (
#     #     "black and white, blurry, low resolution, low-res, pixelated, noisy, jpeg artifacts, "
#     #     "distorted face, deformed face, asymmetrical face, extra face, "
#     #     "cropped, cut off, out of frame, partial face, bad framing, "
#     #     "cgi, 3d render, cartoon, illustration, plastic skin, uncanny, "
#     #     "watermark, text, logo"
#     # )

#     profession_list = [
#         'Accountant', 'Actor', 'Actuarial Analyst', 'Actuary', 'Administrative Assistant',
#         'Administrator', 'Air Traffic Controller', 'Airplane Pilot', 'Analyst',
#         'Animal Trainer', 'Anthropologist', 'Appraiser', 'Archaeologist', 'Architect',
#         'Archivist', 'Art Director', 'Artist', 'Astronaut', 'Athlete', 'Attorney',
#         'Audio Technician', 'Auditor', 'Automotive Designer', 'Baker', 'Baker Assistant',
#         'Banker', 'Bankruptcy Specialist', 'Barber', 'Barista', 'Bartender',
#         'Bioinformatician', 'Biologist', 'Biomedical Engineer', 'Blacksmith', 'Bodyguard',
#         'Bounty Hunter', 'Brand Manager', 'Brewer', 'Bricklayer', 'Broker',
#         'Budget Analyst', 'Builder', 'Butcher', 'Caregiver', 'Carpenter',
#         'Cartographer', 'Chef', 'Chemical Engineer', 'Chemist', 'Chiropractor',
#         'Civil Engineer', 'Claims Adjuster', 'Cleaner', 'Clerk',
#         'Clinical Laboratory Scientist', 'Coach', 'Comedian', 'Compliance Officer',
#         'Composer', 'Conservation Officer', 'Construction Worker', 'Content Creator',
#         'Cook', 'Copywriter', 'Court Reporter', 'Crime Scene Investigator',
#         'Customer Support Specialist', 'DJ', 'Dancer', 'Data Scientist',
#         'Database Administrator', 'Debt Counselor', 'Delivery Driver', 'Dentist',
#         'Designer', 'Detective', 'Development Officer', 'Dietitian', 'Director',
#         'Doctor', 'Dog Walker', 'Driver', 'Economist', 'Editor',
#         'Electrical Technician', 'Electrician',
#         'Emergency Management Specialist', 'Engineer', 'Entrepreneur',
#         'Environmental Engineer', 'Ergonomist', 'Estate Planner',
#         'Event Coordinator', 'Executive Assistant', 'Facilities Manager', 'Farmer',
#         'Fashion Designer', 'Financial Analyst', 'Firefighter', 'Fisherman',
#         'Flight Attendant', 'Flight Dispatcher', 'Florist', 'Forensic Scientist',
#         'Freight Coordinator', 'Gardener', 'Genetic Counselor', 'Geologist',
#         'Grant Writer', 'Graphic Designer', 'Hairdresser', 'Handyman',
#         'Health Inspector', 'Historian', 'Hotel Concierge',
#         'Human Resources Specialist', 'IT Support Specialist', 'Ice Cream Maker',
#         'Illustrator', 'Industrial Designer', 'Insurance Underwriter',
#         'Investment Banker', 'Janitor', 'Journalist', 'Judge', 'Laborer', 'Lawyer',
#         'Librarian', 'Lifeguard', 'Loan Officer', 'Logger', 'Logistics Manager',
#         'Magician', 'Makeup Artist', 'Marine Biologist',
#         'Market Research Analyst', 'Marketing Manager', 'Mathematician', 'Mechanic',
#         'Medical Researcher', 'Meteorologist', 'Midwife', 'Miner', 'Model', 'Musician',
#         'News Anchor', 'Nurse', 'Nutritionist', 'Occupational Therapist',
#         'Oceanographer', 'Office Assistant', 'Operations Manager', 'Optician',
#         'Painter', 'Paralegal', 'Paramedic', 'Park Ranger', 'Pastry Chef',
#         'Payroll Specialist', 'Personal Trainer', 'Pharmacist', 'Photographer',
#         'Physical Therapist', 'Physicist', 'Pilot', 'Plumber', 'Police Officer',
#         'Politician', 'Procurement Officer', 'Professor', 'Property Manager',
#         'Psychologist', 'Quality Assurance Inspector', 'Realtor', 'Researcher',
#         'Roofer', 'Sailor', 'Salesperson', 'Scientist', 'Screenwriter',
#         'Security Officer', 'Singer', 'Skilled Technician', 'Social Worker',
#         'Software Engineer', 'Soldier', 'Sound Engineer', 'Statistician', 'Surgeon',
#         'Tailor', 'Teacher', 'Technician', 'Therapist', 'Translator',
#         'Urban Planner', 'Veterinarian', 'Videographer', 'Waiter', 'Welder',
#         'Writer', 'Zoologist'
#     ]

#     for label in profession_list:

#         prompt = (
#             f"full-body realistic photo of a {label}, standing, centered, "
#             f"in a professional setting appropriate for their occupation, "
#             f"sharp focus, 35mm lens, natural professional lighting"
#         )

#         print(f"\n[INFO] Starting generation for label='{label}'")

#         generate_with_validation(
#             prompt=prompt,
#             label=label,
#             negative_prompt=NEG_PROMPTS,
#             total_images=TOTAL_IMAGES_PER_LABEL,
#             batch_size=BATCH_SIZE,
#             valid_dir="G:\Thesis\ImageGentest\\valid3", #"G:\Thesis\StableDiffusionGeneratedImages\\valid",
#             invalid_dir="G:\Thesis\ImageGentest\invalid3", #"G:\Thesis\StableDiffusionGeneratedImages\invalid",
#             validator=validator,
#             args=args,
#             csv_path="G:\Thesis\ImageGentest\ImageGenMetadata3.csv" #"G:\Thesis\StableDiffusionGeneratedImages\ImageGenMetadata.csv"
#         )

#         print(f"[INFO] Finished generation for label='{label}'")

#     validator.shutdown()