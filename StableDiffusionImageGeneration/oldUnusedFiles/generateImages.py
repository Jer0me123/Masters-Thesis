
# # # python generateImages.py --json-path test_prompts.json --output-root generated_dataset --images-per-prompt 6 --sd-batch-size 2 --img-size 512 --yolo-person-model yolo12s.pt --yolo-face-model yolov12l-face.pt

# # import json
# # import torch
# # import cv2
# # import numpy as np
# # from pathlib import Path
# # from collections import Counter
# # from diffusers import StableDiffusionPipeline
# # from ultralytics import YOLO
# # from tqdm import tqdm
# # import argparse

# # # ============================================================
# # # GEOMETRIC UTILITIES
# # # ============================================================
# # def box_iou(a, b):
# #     xA = max(a[0], b[0])
# #     yA = max(a[1], b[1])
# #     xB = min(a[2], b[2])
# #     yB = min(a[3], b[3])

# #     inter_w = max(0.0, xB - xA)
# #     inter_h = max(0.0, yB - yA)
# #     inter_area = inter_w * inter_h

# #     if inter_area <= 0:
# #         return 0.0

# #     area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
# #     area_b = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))

# #     if area_a <= 0 or area_b <= 0:
# #         return 0.0

# #     return inter_area / float(area_a + area_b - inter_area)


# # # ============================================================
# # # UPDATED VALIDATION (PERSON-RELATIVE FACE SIZE ONLY)
# # # ============================================================
# # def is_valid(
# #     img_bgr,
# #     yolo_person,
# #     yolo_face,
# #     iou_thresh,
# #     min_face_person_ratio,
# #     min_face_conf,
# #     min_person_conf,
# # ):
# #     # ----------------------------
# #     # PERSON DETECTION
# #     # ----------------------------
# #     rp = yolo_person(img_bgr, verbose=False, device="cpu")[0]
# #     if rp.boxes is None or len(rp.boxes) == 0:
# #         return False, "no_person"

# #     person_boxes = [
# #         b for b in rp.boxes.data.cpu().numpy()
# #         if int(b[5]) == 0 and float(b[4]) >= min_person_conf
# #     ]

# #     if len(person_boxes) == 0:
# #         return False, "no_person_conf"

# #     # ----------------------------
# #     # FACE DETECTION
# #     # ----------------------------
# #     rf = yolo_face(img_bgr, verbose=False, device="cpu")[0]
# #     if rf.boxes is None or len(rf.boxes) == 0:
# #         return False, "no_face"

# #     face_boxes = [
# #         b for b in rf.boxes.data.cpu().numpy()
# #         if float(b[4]) >= min_face_conf
# #     ]

# #     if len(face_boxes) != 1:
# #         return False, f"{len(face_boxes)}_faces"

# #     face_box = face_boxes[0]
# #     fx1, fy1, fx2, fy2, fconf, _ = face_box
# #     face_area = (fx2 - fx1) * (fy2 - fy1)

# #     fcx = (fx1 + fx2) / 2
# #     fcy = (fy1 + fy2) / 2

# #     # ----------------------------
# #     # SPATIAL + SIZE ASSOCIATION
# #     # ----------------------------
# #     for pb in person_boxes:
# #         px1, py1, px2, py2, pconf, _ = pb
# #         person_area = (px2 - px1) * (py2 - py1)

# #         if person_area <= 0:
# #             continue

# #         centroid_inside = (px1 <= fcx <= px2 and py1 <= fcy <= py2)
# #         iou = box_iou([fx1, fy1, fx2, fy2], [px1, py1, px2, py2])

# #         if centroid_inside or iou >= iou_thresh:
# #             fp_ratio = face_area / float(person_area)

# #             if fp_ratio >= min_face_person_ratio:
# #                 return True, "ok"
# #             else:
# #                 return False, "face_not_primary"

# #     return False, "face_not_on_person"


# # # ============================================================
# # # SD BATCH GENERATION
# # # ============================================================
# # def generate_batch(pipe, prompt, batch_size, img_size):
# #     prompts = [prompt] * batch_size
# #     out = pipe(
# #         prompts,
# #         height=img_size,
# #         width=img_size,
# #         num_inference_steps=30,
# #         guidance_scale=7.0,
# #     ).images

# #     imgs_bgr = []
# #     for img in out:
# #         arr = np.array(img)
# #         imgs_bgr.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

# #     return imgs_bgr


# # # ============================================================
# # # GENERATE UNTIL QUOTA MET
# # # ============================================================
# # def generate_valid_set(
# #     pipe,
# #     yolo_person,
# #     yolo_face,
# #     prompt,
# #     base_stem_path,
# #     images_per_prompt,
# #     sd_batch_size,
# #     img_size,
# #     fail_log,
# #     args,
# # ):
# #     saved = 0
# #     job_name = base_stem_path.name

# #     while saved < images_per_prompt:
# #         current_bs = min(sd_batch_size, images_per_prompt - saved)

# #         print(
# #             f"[GEN] {job_name}: "
# #             f"batch_size={current_bs}, saved={saved}/{images_per_prompt}"
# #         )

# #         batch_imgs = generate_batch(pipe, prompt, current_bs, img_size)

# #         for img in batch_imgs:
# #             valid, reason = is_valid(
# #                 img,
# #                 yolo_person,
# #                 yolo_face,
# #                 args.iou_thresh,
# #                 args.min_face_person_ratio,
# #                 args.min_face_conf,
# #                 args.min_person_conf,
# #             )

# #             if valid:
# #                 save_path = base_stem_path.with_name(
# #                     f"{base_stem_path.stem}_{saved:02d}.png"
# #                 )
# #                 cv2.imwrite(str(save_path), img)
# #                 print(f"[OK] {job_name}: saved {save_path.name}")
# #                 saved += 1

# #                 if saved >= images_per_prompt:
# #                     break
# #             else:
# #                 fail_log[reason] += 1

# #     print(f"[DONE] {job_name}: quota reached")


# # # ============================================================
# # # MAIN
# # # ============================================================
# # def main(args):
# #     device = "cuda" if torch.cuda.is_available() else "cpu"
# #     print(f"[INFO] Using device: {device}")

# #     # -----------------------
# #     # Load JSON
# #     # -----------------------
# #     with open(args.json_path, "r", encoding="utf-8") as f:
# #         prompt_data = json.load(f)

# #     def is_named_format(job_block):
# #         return isinstance(job_block, dict)

# #     # -----------------------
# #     # Load SD
# #     # -----------------------
# #     print("[INFO] Loading Stable Diffusion 2.1...")
# #     pipe = StableDiffusionPipeline.from_pretrained(
# #         args.model_id,
# #         torch_dtype=torch.float16 if device == "cuda" else torch.float32,
# #         use_safetensors=True,
# #     )
# #     pipe = pipe.to(device)
# #     pipe.set_progress_bar_config(disable=True)

# #     if device == "cuda":
# #         pipe.enable_xformers_memory_efficient_attention()
# #         pipe.enable_attention_slicing("max")

# #     # -----------------------
# #     # Load YOLO
# #     # -----------------------
# #     print("[INFO] Loading YOLO models on CPU...")
# #     yolo_person = YOLO(args.yolo_person_model).to("cpu")
# #     yolo_face = YOLO(args.yolo_face_model).to("cpu")

# #     fail_log = Counter()
# #     output_root = Path(args.output_root)
# #     output_root.mkdir(exist_ok=True)

# #     # -----------------------
# #     # Main Loop
# #     # -----------------------
# #     for job_title, job_block in prompt_data.items():
# #         job_dir = output_root / job_title.replace(" ", "_")
# #         job_dir.mkdir(exist_ok=True)
# #         print(f"\n[JOB] {job_title}")   
# #         print(job_block[prompt])
# #         if is_named_format(job_block):
# #             for image_name, prompt in tqdm(job_block.items(), unit="img"):
# #                 stem = Path(image_name).stem
# #                 base_stem_path = job_dir / stem
# #                 generate_valid_set(
# #                     pipe,
# #                     yolo_person,
# #                     yolo_face,
# #                     job_block[prompt],
# #                     base_stem_path,
# #                     args.images_per_prompt,
# #                     args.sd_batch_size,
# #                     args.img_size,
# #                     fail_log,
# #                     args,
# #                 )
# #         else:
# #             for idx, prompt in enumerate(tqdm(job_block, unit="prompt")):
# #                 base_stem_path = job_dir / f"{job_title.replace(' ', '_')}_{idx:04d}"
# #                 generate_valid_set(
# #                     pipe,
# #                     yolo_person,
# #                     yolo_face,
# #                     job_block[prompt],
# #                     base_stem_path,
# #                     args.images_per_prompt,
# #                     args.sd_batch_size,
# #                     args.img_size,
# #                     fail_log,
# #                     args,
# #                 )

# #     print("\n[FAILURE STATS]")
# #     for reason, count in fail_log.items():
# #         print(f"  {reason}: {count}")


# # # ============================================================
# # # ARGUMENT PARSER
# # # ============================================================
# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser()

# #     parser.add_argument("--json-path", type=str, required=True)
# #     parser.add_argument("--output-root", type=str, default="generated_dataset")
# #     parser.add_argument("--model-id", type=str, default="stabilityai/stable-diffusion-2-1")

# #     parser.add_argument("--images-per-prompt", type=int, default=6)
# #     parser.add_argument("--sd-batch-size", type=int, default=2)
# #     parser.add_argument("--img-size", type=int, default=512)

# #     parser.add_argument("--yolo-person-model", type=str, default="yolov12s.pt")
# #     parser.add_argument("--yolo-face-model", type=str, default="yolov12l-face.pt")

# #     # Face / Person Detection Parameters
# #     parser.add_argument("--iou-thresh", type=float, default=0.08)
# #     parser.add_argument("--min-face-person-ratio", type=float, default=0.025)
# #     parser.add_argument("--min-face-conf", type=float, default=0.45)
# #     parser.add_argument("--min-person-conf", type=float, default=0.40)

# #     args = parser.parse_args()
# #     main(args)


# import json
# import torch
# import cv2
# import numpy as np
# from pathlib import Path
# from collections import Counter
# from diffusers import DiffusionPipeline
# from ultralytics import YOLO
# from tqdm import tqdm
# import argparse

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

# # ============================================================
# # GEOMETRIC UTILITIES
# # ============================================================
# def box_iou(a, b):
#     xA = max(a[0], b[0])
#     yA = max(a[1], b[1])
#     xB = min(a[2], b[2])
#     yB = min(a[3], b[3])

#     inter_w = max(0.0, xB - xA)
#     inter_h = max(0.0, yB - yA)
#     inter_area = inter_w * inter_h

#     if inter_area <= 0:
#         return 0.0

#     area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
#     area_b = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))

#     if area_a <= 0 or area_b <= 0:
#         return 0.0

#     return inter_area / float(area_a + area_b - inter_area)


# # ============================================================
# # UPDATED VALIDATION
# # ============================================================
# def is_valid(
#     img_bgr,
#     yolo_person,
#     yolo_face,
#     iou_thresh,
#     min_face_person_ratio,
#     min_face_conf,
#     min_person_conf,
# ):
#     rp = yolo_person(img_bgr, verbose=False, device="cpu")[0]
#     if rp.boxes is None or len(rp.boxes) == 0:
#         return False, "no_person"

#     person_boxes = [
#         b for b in rp.boxes.data.cpu().numpy()
#         if int(b[5]) == 0 and float(b[4]) >= min_person_conf
#     ]

#     if len(person_boxes) == 0:
#         return False, "no_person_conf"

#     rf = yolo_face(img_bgr, verbose=False, device="cpu")[0]
#     if rf.boxes is None or len(rf.boxes) == 0:
#         return False, "no_face"

#     face_boxes = [
#         b for b in rf.boxes.data.cpu().numpy()
#         if float(b[4]) >= min_face_conf
#     ]

#     if len(face_boxes) != 1:
#         return False, f"{len(face_boxes)}_faces"

#     fx1, fy1, fx2, fy2, _, _ = face_boxes[0]
#     face_area = (fx2 - fx1) * (fy2 - fy1)

#     fcx = (fx1 + fx2) / 2
#     fcy = (fy1 + fy2) / 2

#     for pb in person_boxes:
#         px1, py1, px2, py2, _, _ = pb
#         person_area = (px2 - px1) * (py2 - py1)

#         centroid_inside = (px1 <= fcx <= px2 and py1 <= fcy <= py2)
#         iou = box_iou([fx1, fy1, fx2, fy2], [px1, py1, px2, py2])

#         if centroid_inside or iou >= iou_thresh:
#             if (face_area / float(person_area)) >= min_face_person_ratio:
#                 return True, "ok"
#             else:
#                 return False, "face_not_primary"

#     return False, "face_not_on_person"


# # ============================================================
# # SDXL BATCH GENERATION
# # ============================================================
# def generate_batch(pipe, prompt, batch_size, img_size):
#     prompts = [prompt] * batch_size
#     out = pipe(
#         prompts,
#         height=img_size,
#         width=img_size,
#         num_inference_steps=30,
#         guidance_scale=7.0,
#     ).images

#     imgs_bgr = []
#     for img in out:
#         arr = np.array(img)
#         imgs_bgr.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

#     return imgs_bgr


# # ============================================================
# # GENERATE UNTIL QUOTA MET
# # ============================================================
# def generate_valid_set(
#     pipe,
#     yolo_person,
#     yolo_face,
#     prompt,
#     base_stem_path,
#     images_per_prompt,
#     sd_batch_size,
#     img_size,
#     fail_log,
#     args,
# ):
#     saved = 0
#     job_name = base_stem_path.name

#     while saved < images_per_prompt:
#         current_bs = min(sd_batch_size, images_per_prompt - saved)

#         print(f"[GEN] {job_name}: batch_size={current_bs}, saved={saved}/{images_per_prompt}")

#         print(f"PROMPT: {prompt}")
#         batch_imgs = generate_batch(pipe, prompt, current_bs, img_size)

#         for img in batch_imgs:
#             valid, reason = is_valid(
#                 img,
#                 yolo_person,
#                 yolo_face,
#                 args.iou_thresh,
#                 args.min_face_person_ratio,
#                 args.min_face_conf,
#                 args.min_person_conf,
#             )
#             valid = True

#             if valid:
#                 save_path = base_stem_path.with_name(f"{base_stem_path.stem}_{saved:02d}.png")
#                 cv2.imwrite(str(save_path), img)
#                 print(f"[OK] {job_name}: saved {save_path.name}")
#                 saved += 1
#             else:
#                 fail_log[reason] += 1

#     print(f"[DONE] {job_name}: quota reached")


# # ============================================================
# # MAIN
# # ============================================================
# def main(args):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"[INFO] Using device: {device}")

#     with open(args.json_path, "r", encoding="utf-8") as f:
#         prompt_data = json.load(f)

#     print("[INFO] Loading SDXL Base...")
#     pipe = DiffusionPipeline.from_pretrained(
#         "stabilityai/stable-diffusion-xl-base-1.0",
#         torch_dtype=torch.float16,
#         use_safetensors=True,
#         variant="fp16",
#     ).to(device)

#     # pipe.set_progress_bar_config(disable=True)
#     pipe.enable_xformers_memory_efficient_attention()
#     pipe.enable_attention_slicing("max")
#     pipe.enable_vae_slicing()

#     pipe.safety_checker = None

#     from diffusers import DPMSolverMultistepScheduler

#     pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

#     if args.use_refiner:
#         print("[INFO] Loading SDXL Refiner...")
#         refiner = DiffusionPipeline.from_pretrained(
#             "stabilityai/stable-diffusion-xl-refiner-1.0",
#             text_encoder_2=pipe.text_encoder_2,
#             vae=pipe.vae,
#             torch_dtype=torch.float16,
#             use_safetensors=True,
#             variant="fp16",
#         ).to(device)

#     print("[INFO] Loading YOLO models...")
#     yolo_person = YOLO(args.yolo_person_model).to("cpu")
#     yolo_face = YOLO(args.yolo_face_model).to("cpu")

#     fail_log = Counter()
#     output_root = Path(args.output_root)
#     output_root.mkdir(exist_ok=True)

#     for job_title, job_block in prompt_data.items():
#         job_dir = output_root / job_title.replace(" ", "_")
#         job_dir.mkdir(exist_ok=True)

#         for idx, key in enumerate(job_block):
#             base_stem_path = job_dir / f"{job_title}_{idx:04d}"
        
#             generate_valid_set(
#                 pipe,
#                 yolo_person,
#                 yolo_face,
#                 job_block[key],
#                 base_stem_path,
#                 args.images_per_prompt,
#                 args.sd_batch_size,
#                 args.img_size,
#                 fail_log,
#                 args,
#             )

#     print("\n[FAILURE STATS]")
#     for reason, count in fail_log.items():
#         print(f"  {reason}: {count}")


# # ============================================================
# # ARGUMENT PARSER
# # ============================================================
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--json-path", type=str, required=True)
#     parser.add_argument("--output-root", type=str, default="generated_dataset")

#     parser.add_argument("--images-per-prompt", type=int, default=6)
#     parser.add_argument("--sd-batch-size", type=int, default=2)
#     parser.add_argument("--img-size", type=int, default=512)

#     parser.add_argument("--yolo-person-model", type=str, default="yolov12s.pt")
#     parser.add_argument("--yolo-face-model", type=str, default="yolov12l-face.pt")

#     parser.add_argument("--iou-thresh", type=float, default=0.08)
#     parser.add_argument("--min-face-person-ratio", type=float, default=0.025)
#     parser.add_argument("--min-face-conf", type=float, default=0.45)
#     parser.add_argument("--min-person-conf", type=float, default=0.40)

#     parser.add_argument("--use-refiner", action="store_true")

#     args = parser.parse_args()
#     main(args)


import json
import torch
import cv2
import numpy as np
from pathlib import Path
from collections import Counter
from diffusers import DiffusionPipeline
from ultralytics import YOLO
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ============================================================
# GEOMETRIC UTILITIES
# ============================================================
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


# ============================================================
# ASYNC VALIDATION WORKER
# ============================================================
class ValidationWorker:
    def __init__(self, yolo_person_path, yolo_face_path, num_workers=2):
        self.num_workers = num_workers
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.workers = []
        self.stop_event = threading.Event()
        
        # Start worker threads
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(yolo_person_path, yolo_face_path, i),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
    
    def _worker_loop(self, yolo_person_path, yolo_face_path, worker_id):
        # Each worker gets its own YOLO models
        yolo_person = YOLO(yolo_person_path)
        yolo_face = YOLO(yolo_face_path)
        
        while not self.stop_event.is_set():
            try:
                task = self.task_queue.get(timeout=0.1)
                if task is None:
                    break
                
                idx, img_bgr, args = task
                valid, reason = self._validate(
                    img_bgr, yolo_person, yolo_face, args
                )
                self.result_queue.put((idx, valid, reason))
                self.task_queue.task_done()
            except queue.Empty:
                continue
    
    def _validate(self, img_bgr, yolo_person, yolo_face, args):
        rp = yolo_person(img_bgr, verbose=False, device="cpu")[0]
        if rp.boxes is None or len(rp.boxes) == 0:
            return False, "no_person"

        person_boxes = [
            b for b in rp.boxes.data.cpu().numpy()
            if int(b[5]) == 0 and float(b[4]) >= args.min_person_conf
        ]

        if len(person_boxes) == 0:
            return False, "no_person_conf"

        rf = yolo_face(img_bgr, verbose=False, device="cpu")[0]
        if rf.boxes is None or len(rf.boxes) == 0:
            return False, "no_face"

        face_boxes = [
            b for b in rf.boxes.data.cpu().numpy()
            if float(b[4]) >= args.min_face_conf
        ]

        if len(face_boxes) != 1:
            return False, f"{len(face_boxes)}_faces"

        fx1, fy1, fx2, fy2, _, _ = face_boxes[0]
        face_area = (fx2 - fx1) * (fy2 - fy1)

        fcx = (fx1 + fx2) / 2
        fcy = (fy1 + fy2) / 2

        for pb in person_boxes:
            px1, py1, px2, py2, _, _ = pb
            person_area = (px2 - px1) * (py2 - py1)

            centroid_inside = (px1 <= fcx <= px2 and py1 <= fcy <= py2)
            iou = box_iou([fx1, fy1, fx2, fy2], [px1, py1, px2, py2])

            if centroid_inside or iou >= args.iou_thresh:
                if (face_area / float(person_area)) >= args.min_face_person_ratio:
                    return True, "ok"
                else:
                    return False, "face_not_primary"

        return False, "face_not_on_person"
    
    def submit(self, idx, img_bgr, args):
        self.task_queue.put((idx, img_bgr, args))
    
    def get_result(self, timeout=None):
        return self.result_queue.get(timeout=timeout)
    
    def shutdown(self):
        self.stop_event.set()
        for _ in self.workers:
            self.task_queue.put(None)
        for worker in self.workers:
            worker.join()


# ============================================================
# OPTIMIZED BATCH GENERATION
# ============================================================
def generate_batch(pipe, prompt, batch_size, img_size):
    prompts = [prompt] * batch_size
    
    # Use faster inference settings
    with torch.inference_mode():
        out = pipe(
            prompts,
            height=img_size,
            width=img_size,
            num_inference_steps=25,  # Reduced from 30
            guidance_scale=7.0,
        ).images

    imgs_bgr = []
    for img in out:
        arr = np.array(img)
        imgs_bgr.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

    return imgs_bgr


# ============================================================
# GENERATE WITH ASYNC VALIDATION
# ============================================================
def generate_valid_set(
    pipe,
    validation_worker,
    prompt,
    base_stem_path,
    images_per_prompt,
    sd_batch_size,
    img_size,
    fail_log,
    args,
):
    saved = 0
    job_name = base_stem_path.name
    
    # Pre-generate larger batches and validate asynchronously
    validation_buffer = {}  # idx -> (img, submitted)
    next_idx = 0

    print(prompt)

    while saved < images_per_prompt:
        # Generate images
        current_bs = min(sd_batch_size, images_per_prompt - saved + 5)  # Overshoot slightly
        
        print(f"[GEN] {job_name}: batch_size={current_bs}, saved={saved}/{images_per_prompt}")
        batch_imgs = generate_batch(pipe, prompt, current_bs, img_size)

        # Submit all images for validation
        for img in batch_imgs:
            validation_worker.submit(next_idx, img, args)
            validation_buffer[next_idx] = (img, True)
            next_idx += 1

        # Collect validation results
        while saved < images_per_prompt and validation_buffer:
            try:
                idx, valid, reason = validation_worker.get_result(timeout=1.0)
                
                if idx in validation_buffer:
                    img, _ = validation_buffer.pop(idx)

                    if valid:
                        save_path = base_stem_path.with_name(
                            f"{base_stem_path.stem}_{saved:02d}.png"
                        )
                        cv2.imwrite(str(save_path), img)
                        print(f"[OK] {job_name}: saved {save_path.name}")
                        saved += 1
                    else:
                        print(f"[FAILED]: {reason}")
                        fail_log[reason] += 1
            except queue.Empty:
                break

    # Clear remaining buffer
    while validation_buffer:
        try:
            idx, valid, reason = validation_worker.get_result(timeout=1.0)
            if idx in validation_buffer:
                validation_buffer.pop(idx)
                if not valid:
                    fail_log[reason] += 1
        except queue.Empty:
            break

    print(f"[DONE] {job_name}: quota reached")


# ============================================================
# MAIN
# ============================================================
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    with open(args.json_path, "r", encoding="utf-8") as f:
        prompt_data = json.load(f)

    print("[INFO] Loading SDXL Base...")
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to(device)

    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_attention_slicing("max")
    pipe.vae.enable_slicing()
    pipe.safety_checker = None

    # Use faster scheduler
    from diffusers import EulerAncestralDiscreteScheduler
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config
    )

    # Compile model for faster inference (PyTorch 2.0+)

    # print("[INFO] Compiling model with torch.compile this may take 15-20+ mins")
    # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

    print(f"[INFO] Starting {args.validation_workers} validation workers...")
    validation_worker = ValidationWorker(
        args.yolo_person_model,
        args.yolo_face_model,
        num_workers=args.validation_workers
    )

    fail_log = Counter()
    output_root = Path(args.output_root)
    output_root.mkdir(exist_ok=True)

    try:
        for job_title, job_block in prompt_data.items():
            job_dir = output_root / job_title.replace(" ", "_")
            job_dir.mkdir(exist_ok=True)

            for idx, key in enumerate(job_block):
                base_stem_path = job_dir / f"{job_title}_{idx:04d}"
            
                generate_valid_set(
                    pipe,
                    validation_worker,
                    job_block[key],
                    base_stem_path,
                    args.images_per_prompt,
                    args.sd_batch_size,
                    args.img_size,
                    fail_log,
                    args,
                )
    finally:
        validation_worker.shutdown()

    print("\n[FAILURE STATS]")
    for reason, count in fail_log.items():
        print(f"  {reason}: {count}")


# ============================================================
# ARGUMENT PARSER
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--json-path", type=str, required=True)
    parser.add_argument("--output-root", type=str, default="generated_dataset")

    parser.add_argument("--images-per-prompt", type=int, default=6)
    parser.add_argument("--sd-batch-size", type=int, default=4)  # Increased default
    parser.add_argument("--img-size", type=int, default=512)

    parser.add_argument("--yolo-person-model", type=str, default="yolov12s.pt")
    parser.add_argument("--yolo-face-model", type=str, default="yolov12l-face.pt")

    parser.add_argument("--iou-thresh", type=float, default=0.08)
    parser.add_argument("--min-face-person-ratio", type=float, default=0.025)
    parser.add_argument("--min-face-conf", type=float, default=0.45)
    parser.add_argument("--min-person-conf", type=float, default=0.40)

    parser.add_argument("--validation-workers", type=int, default=2)
    parser.add_argument("--compile-model", action="store_true")

    args = parser.parse_args()
    main(args)