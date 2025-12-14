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
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        out = pipe(
            prompts,
            height=img_size,
            width=img_size,
            num_inference_steps=20,  # Reduced from 25
            guidance_scale=6.5,  # Slightly lower for speed
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
        # Generate images with better overshoot calculation
        remaining = images_per_prompt - saved
        # Estimate we'll keep ~60-70% of images, so overshoot accordingly
        estimated_needed = int(remaining * 1.5) if remaining > 2 else remaining + 2
        current_bs = min(sd_batch_size, estimated_needed)
        
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

                    valid = True

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
    
    # Use faster scheduler
    from diffusers import DPMSolverMultistepScheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="sde-dpmsolver++"
    )
    
    # Use channels-last memory format for better performance
    if args.use_channels_last:
        pipe.unet.to(memory_format=torch.channels_last)
        pipe.vae.to(memory_format=torch.channels_last)

    # Memory optimizations
    if args.enable_xformers:
        pipe.enable_xformers_memory_efficient_attention()
    
    if args.enable_slicing:
        pipe.enable_attention_slicing("max")
        pipe.vae.enable_slicing()
        
    if args.enable_tiling:
        pipe.vae.enable_tiling()
    
    pipe.safety_checker = None

    # Compile model for faster inference (PyTorch 2.0+)
    if hasattr(torch, 'compile') and args.compile_model:
        print("[INFO] Compiling model with torch.compile...")
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        if args.compile_vae:
            pipe.vae.decode = torch.compile(pipe.vae.decode, mode="reduce-overhead", fullgraph=True)

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
    parser.add_argument("--sd-batch-size", type=int, default=2)  # Reduced for 8GB VRAM
    parser.add_argument("--img-size", type=int, default=512)

    parser.add_argument("--yolo-person-model", type=str, default="yolov12s.pt")
    parser.add_argument("--yolo-face-model", type=str, default="yolov12l-face.pt")

    parser.add_argument("--iou-thresh", type=float, default=0.08)
    parser.add_argument("--min-face-person-ratio", type=float, default=0.025)
    parser.add_argument("--min-face-conf", type=float, default=0.45)
    parser.add_argument("--min-person-conf", type=float, default=0.40)

    parser.add_argument("--validation-workers", type=int, default=3)
    
    # Speed optimization flags
    parser.add_argument("--compile-model", action="store_true", help="Use torch.compile on UNet (PyTorch 2.0+)")
    parser.add_argument("--compile-vae", action="store_true", help="Use torch.compile on VAE decoder")
    parser.add_argument("--enable-xformers", action="store_true", default=True, help="Enable xformers")
    parser.add_argument("--enable-slicing", action="store_true", default=True, help="Enable attention slicing")
    parser.add_argument("--enable-tiling", action="store_true", help="Enable VAE tiling (saves VRAM)")
    parser.add_argument("--use-channels-last", action="store_true", help="Use channels-last memory format")

    args = parser.parse_args()
    main(args)


# python generateImages2.py --json-path test_prompts.json --output-root generated_dataset --images-per-prompt 6 --sd-batch-size 2 --img-size 512 --yolo-person-model yolo12s.pt --yolo-face-model yolov12l-face.pt --use-channels-last --enable-tiling --validation-workers 3   