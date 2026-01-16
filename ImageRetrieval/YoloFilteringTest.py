import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from turbojpeg import TurboJPEG
import re
import torch
import os
import argparse
from queue import Queue
from threading import Thread


def load_manifest(path: Path):
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data.get("completed_groups", []))

def save_manifest(path: Path, completed_groups: set):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            {"completed_groups": sorted(completed_groups)},
            f,
            indent=2
        )
    tmp.replace(path)  # atomic on Windows + POSIX

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

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter_area / float(area_a + area_b - inter_area)

def segment_face(face_roi_bgr, face_mesh):
    h, w, _ = face_roi_bgr.shape
    rgb = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    pts = np.array([
        (int(lm.x * w), int(lm.y * h))
        for lm in results.multi_face_landmarks[0].landmark
    ])

    if pts.shape[0] < 10:
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)

    segmented = cv2.bitwise_and(face_roi_bgr, face_roi_bgr, mask=mask)

    x, y, w_box, h_box = cv2.boundingRect(pts)
    pad = 10
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(w, x + w_box + pad), min(h, y + h_box + pad)

    return segmented[y1:y2, x1:x2]

class FastImageLoader:
    def __init__(self, num_workers=8):
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        try:
            self.jpeg_decoder = TurboJPEG(r"C:\libjpeg-turbo-gcc64\bin\libturbojpeg.dll")
            self.use_turbo = True
            print("Using TurboJPEG for image loading")
        except:
            self.use_turbo = False
            print("TurboJPEG not available, falling back to cv2")
    
    def load_image(self, path):
        path_str = str(path)

        try:
            # TurboJPEG path
            if self.use_turbo and path.suffix.lower() in {'.jpg', '.jpeg'}:
                try:
                    with open(path_str, 'rb') as f:
                        return self.jpeg_decoder.decode(f.read())
                except Exception:
                    pass  # fall back to cv2

            # cv2 fallback
            img = cv2.imread(path_str)
            return img  # may be None, caller must handle

        except (FileNotFoundError, OSError):
            return None
    
    def load_batch(self, paths):
        """Load batch of images in parallel"""
        return list(self.executor.map(self.load_image, paths))
    
    def __del__(self):
        executor = getattr(self, "executor", None)
        if executor is not None:
            executor.shutdown(wait=False)

def load_name_set(path: Path):
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def append_names(path: Path, names):
    if not names:
        return
    with open(path, "a", encoding="utf-8") as f:
        for n in names:
            f.write(n + "\n")

def validate_image_fast(
    img_bgr,
    person_boxes,   # numpy array (filtered later)
    face_boxes,     # numpy array
    face_mesh,
    min_person_conf=0.5,
    min_face_conf=0.6,
    iou_thresh=0.3,
    min_face_person_ratio=0.02
):
    # # ---------- PERSON FILTER ----------
    if person_boxes  is not None:
        persons = person_boxes[
            (person_boxes[:, 5] == 0) &
            (person_boxes[:, 4] >= min_person_conf)
        ]
        if len(persons) == 0:
            return None

    # ---------- FACE FILTER ----------
    faces = face_boxes[face_boxes[:, 4] >= min_face_conf]
    if len(faces) != 1:
        return None

    fx1, fy1, fx2, fy2, _, _ = faces[0]
    fx1, fy1, fx2, fy2 = map(int, [fx1, fy1, fx2, fy2])
    face_area = (fx2 - fx1) * (fy2 - fy1)
    fcx, fcy = (fx1 + fx2) / 2, (fy1 + fy2) / 2

    face_roi = img_bgr[fy1:fy2, fx1:fx2]
    if face_roi.size == 0:
        return None

    # # ---------- PERSON–FACE RELATION ----------
    if person_boxes is not None:
        for pb in persons:
            px1, py1, px2, py2, _, _ = pb
            px1, py1, px2, py2 = map(int, [px1, py1, px2, py2])
            person_area = (px2 - px1) * (py2 - py1)

            centroid_inside = px1 <= fcx <= px2 and py1 <= fcy <= py2
            iou = box_iou([fx1, fy1, fx2, fy2], [px1, py1, px2, py2])

            if (centroid_inside or iou >= iou_thresh) and \
            face_area / person_area >= min_face_person_ratio:
                face_crop = segment_face(face_roi, face_mesh)
                if face_crop is None:
                    return None

                return {
                    "face_box": [fx1, fy1, fx2, fy2],
                    "person_box": [px1, py1, px2, py2],
                    "face_crop": face_crop
                }

    face_crop = segment_face(face_roi, face_mesh)
    if face_crop is None:
        return None

    return {
        "face_box": [fx1, fy1, fx2, fy2],
        "face_crop": face_crop
    }

def iter_groups_from_jsonl(jsonl_path: Path, image_root: Path, max_per_prompt=None):
    """
    Yields:
        (group_name, lazy_items_function)
    where lazy_items_function returns [(Path, score, group_id), ...] when called
    """

    def resolve_src(r):
        raw = Path(r["image_path"])
        if raw.is_absolute():
            return raw
        return image_root / raw.parent.parent / f"{r['group_id']}_images" / raw.name

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            prompt = re.sub(r'[<>:"/\\|?*]', '_', obj["prompt"]).strip().replace(" ", "_")
            results = obj.get("results", [])

            if max_per_prompt:
                results = results[:max_per_prompt]

            if not results:
                continue

            # Return a lambda that will resolve paths only when called
            def make_items_loader(results_copy):
                def load_items():
                    items = []
                    for r in results_copy:
                        p = resolve_src(r)
                        items.append((p, r["score"], r["group_id"]))
                    return items
                return load_items

            yield prompt, make_items_loader(results)


def iter_groups_from_fs(input_root: Path, img_exts):
    for dirpath, _, filenames in os.walk(input_root):
        items = []
        for f in filenames:
            p = Path(dirpath) / f
            if p.suffix.lower() in img_exts:
                items.append((p, None, None))

        if not items:
            continue

        rel = Path(dirpath).relative_to(input_root)
        yield str(rel), items

def format_name(path, score, group_id):
    if score is None or group_id is None:
        return path.name
    return f"{score:.3f}_{group_id}_{path.name}"

def _copy_and_write(item, out_dir, face_dir, mode='copy'):
    img_path, validated, final_name = item

    dst = out_dir / final_name

    # Copy or move image
    if mode == "copy":
        shutil.copyfile(img_path, dst)
    else:
        shutil.move(img_path, dst)

    # Write face crop
    face_path = face_dir / final_name.replace(".jpg", "_face.png")
    cv2.imwrite(str(face_path), validated["face_crop"])

    return final_name


# ============= OPTIMIZATION 1: Persistent FaceMesh Pool =============
class FaceMeshPool:
    """Pool of FaceMesh instances for parallel face segmentation"""
    def __init__(self, num_workers=4):
        self.pool = []
        for _ in range(num_workers):
            fm = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
            self.pool.append(fm)
        self.queue = Queue()
        for fm in self.pool:
            self.queue.put(fm)
    
    def get(self):
        return self.queue.get()
    
    def put(self, fm):
        self.queue.put(fm)
    
    def __del__(self):
        for fm in self.pool:
            fm.close()

# ============= OPTIMIZATION 2: Batch FaceMesh Processing =============
# def segment_face_batch(face_rois, face_mesh_pool):
#     """Process multiple faces in parallel"""
#     results = [None] * len(face_rois)
    
#     def process_one(idx, roi):
#         fm = face_mesh_pool.get()
#         try:
#             results[idx] = segment_face(roi, fm)
#         finally:
#             face_mesh_pool.put(fm)
    
#     with ThreadPoolExecutor(max_workers=len(face_mesh_pool.pool)) as executor:
#         futures = [executor.submit(process_one, i, roi) for i, roi in enumerate(face_rois)]
#         for f in as_completed(futures):
#             f.result()
    
#     return results

def segment_face(face_roi_bgr, face_mesh):
    """Optimized face segmentation with early exits"""
    h, w, _ = face_roi_bgr.shape
    
    # Quick size check
    if h < 20 or w < 20:
        return None
    
    rgb = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    pts = np.array([
        (int(lm.x * w), int(lm.y * h))
        for lm in results.multi_face_landmarks[0].landmark
    ], dtype=np.int32)

    if pts.shape[0] < 10:
        return None

    # Vectorized mask creation
    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)

    segmented = cv2.bitwise_and(face_roi_bgr, face_roi_bgr, mask=mask)

    x, y, w_box, h_box = cv2.boundingRect(pts)
    pad = 10
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(w, x + w_box + pad), min(h, y + h_box + pad)

    return segmented[y1:y2, x1:x2]

# ============= OPTIMIZATION 3: Faster Box Operations =============
def box_iou_vectorized(boxes_a, boxes_b):
    """Vectorized IoU computation for multiple box pairs"""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    
    # Expand dimensions for broadcasting
    a = boxes_a[:, None, :]  # (N, 1, 4)
    b = boxes_b[None, :, :]  # (1, M, 4)
    
    # Intersection
    xA = np.maximum(a[:, :, 0], b[:, :, 0])
    yA = np.maximum(a[:, :, 1], b[:, :, 1])
    xB = np.minimum(a[:, :, 2], b[:, :, 2])
    yB = np.minimum(a[:, :, 3], b[:, :, 3])
    
    inter_w = np.maximum(0.0, xB - xA)
    inter_h = np.maximum(0.0, yB - yA)
    inter_area = inter_w * inter_h
    
    # Areas
    area_a = (a[:, :, 2] - a[:, :, 0]) * (a[:, :, 3] - a[:, :, 1])
    area_b = (b[:, :, 2] - b[:, :, 0]) * (b[:, :, 3] - b[:, :, 1])
    
    # IoU
    iou = inter_area / (area_a + area_b - inter_area + 1e-6)
    
    return iou

# ============= OPTIMIZATION 4: Vectorized Validation =============
# def validate_batch_fast(
#     batch_imgs,
#     person_boxes_batch,
#     face_boxes_batch,
#     face_mesh_pool,
#     min_person_conf=0.5,
#     min_face_conf=0.6,
#     iou_thresh=0.3,
#     min_face_person_ratio=0.02
# ):
#     """Validate entire batch with minimal Python loops"""
#     results = []
#     face_rois_to_process = []
#     face_roi_indices = []
    
#     # First pass: filter and extract ROIs (fast, no segmentation yet)
#     for idx, (img_bgr, person_boxes, face_boxes) in enumerate(
#         zip(batch_imgs, person_boxes_batch, face_boxes_batch)
#     ):
#         # Person filter
#         if person_boxes is not None:
#             persons = person_boxes[
#                 (person_boxes[:, 5] == 0) &
#                 (person_boxes[:, 4] >= min_person_conf)
#             ]
#             if len(persons) == 0:
#                 results.append(None)
#                 continue
#         else:
#             persons = None

#         # Face filter
#         if face_boxes is None:
#             results.append(None)
#             continue
            
#         faces = face_boxes[face_boxes[:, 4] >= min_face_conf]
#         if len(faces) != 1:
#             results.append(None)
#             continue

#         fx1, fy1, fx2, fy2, _, _ = faces[0]
#         fx1, fy1, fx2, fy2 = int(fx1), int(fy1), int(fx2), int(fy2)
#         face_area = (fx2 - fx1) * (fy2 - fy1)
#         fcx, fcy = (fx1 + fx2) / 2, (fy1 + fy2) / 2

#         face_roi = img_bgr[fy1:fy2, fx1:fx2]
#         if face_roi.size == 0:
#             results.append(None)
#             continue

#         # Person-face relation check
#         valid = False
#         person_box = None
        
#         if persons is not None:
#             for pb in persons:
#                 px1, py1, px2, py2, _, _ = pb
#                 px1, py1, px2, py2 = int(px1), int(py1), int(px2), int(py2)
#                 person_area = (px2 - px1) * (py2 - py1)

#                 centroid_inside = px1 <= fcx <= px2 and py1 <= fcy <= py2
                
#                 # Quick centroid check before expensive IoU
#                 if centroid_inside or face_area / person_area >= min_face_person_ratio:
#                     iou = box_iou_single([fx1, fy1, fx2, fy2], [px1, py1, px2, py2])
#                     if (centroid_inside or iou >= iou_thresh) and \
#                        face_area / person_area >= min_face_person_ratio:
#                         valid = True
#                         person_box = [px1, py1, px2, py2]
#                         break
#         else:
#             valid = True

#         if valid:
#             # Store for batch segmentation
#             face_rois_to_process.append(face_roi)
#             face_roi_indices.append(idx)
#             results.append({
#                 "face_box": [fx1, fy1, fx2, fy2],
#                 "person_box": person_box,
#                 "face_crop": None  # Will be filled in
#             })
#         else:
#             results.append(None)
    
#     # Batch process all face segmentations at once
#     if face_rois_to_process:
#         segmented_faces = segment_face_batch(face_rois_to_process, face_mesh_pool)
        
#         for seg_face, res_idx in zip(segmented_faces, face_roi_indices):
#             if seg_face is None:
#                 results[res_idx] = None
#             else:
#                 results[res_idx]["face_crop"] = seg_face
    
#     return results


# def box_iou_single(a, b):
#     """Single box IoU for backward compatibility"""
#     xA = max(a[0], b[0])
#     yA = max(a[1], b[1])
#     xB = min(a[2], b[2])
#     yB = min(a[3], b[3])

#     inter_w = max(0.0, xB - xA)
#     inter_h = max(0.0, yB - yA)
#     inter_area = inter_w * inter_h
#     if inter_area <= 0:
#         return 0.0

#     area_a = (a[2] - a[0]) * (a[3] - a[1])
#     area_b = (b[2] - b[0]) * (b[3] - b[1])
#     return inter_area / float(area_a + area_b - inter_area)

from concurrent.futures import ThreadPoolExecutor, as_completed

def validate_batch_fast(
    batch_imgs,
    person_boxes_batch,
    face_boxes_batch,
    face_mesh_pool,
):
    """
    Batch wrapper around validate_image_fast using FaceMeshPool.

    - No extra filtering logic.
    - No duplicated validation logic.
    - Each call borrows a FaceMesh instance from the pool.
    - Parallel across CPU threads.
    - Output is aligned 1:1 with batch order.
    """

    results = [None] * len(batch_imgs)

    def _process_one(idx, img_bgr, person_boxes, face_boxes):
        # Borrow FaceMesh instance
        face_mesh = face_mesh_pool.get()
        try:
            # Preserve original behavior exactly
            if img_bgr is None or face_boxes is None:
                return None

            return validate_image_fast(
                img_bgr,
                person_boxes,
                face_boxes,
                face_mesh
            )
        finally:
            # Always return FaceMesh to pool
            face_mesh_pool.put(face_mesh)

    # Parallel execution
    with ThreadPoolExecutor(max_workers=len(face_mesh_pool.pool)) as executor:
        futures = {
            executor.submit(
                _process_one,
                idx,
                img,
                pb,
                fb
            ): idx
            for idx, (img, pb, fb) in enumerate(
                zip(batch_imgs, person_boxes_batch, face_boxes_batch)
            )
        }

        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    return results


# ============= OPTIMIZATION 5: Async I/O Pipeline =============
class AsyncIOWriter:
    """Asynchronous I/O writer with queue"""
    def __init__(self, max_workers=8):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures = []
    
    def submit(self, func, *args):
        future = self.executor.submit(func, *args)
        self.futures.append(future)
        
        # Periodically clean completed futures
        if len(self.futures) > 100:
            self.futures = [f for f in self.futures if not f.done()]
    
    def wait_all(self):
        for f in as_completed(self.futures):
            f.result()
        self.futures.clear()
    
    def __del__(self):
        self.wait_all()
        self.executor.shutdown(wait=True)

# ============= UPDATED PROCESS_GROUPS =============
def process_groups(
    groups_iter,
    output_root,
    yolo_face,
    yolo_person,
    face_mesh_pool,  # Changed from single face_mesh
    loader,
    batch_size,
    mode,
    device
):
    manifest_path = output_root / "manifest.json"
    completed_groups = load_manifest(manifest_path)

    print(f"Loaded manifest with {len(completed_groups)} completed groups")

    # Create persistent I/O writer
    io_writer = AsyncIOWriter(max_workers=16)

    processed_count = 0
    skipped_count = 0

    for group_idx, (group_name, items_loader) in enumerate(groups_iter, start=1):
        if group_name in completed_groups:
            print(f"[SKIPPING] - Group {group_idx}: {group_name}")
            skipped_count += 1
            continue

        items = items_loader() if callable(items_loader) else items_loader
        
        print(f"\nProcessing group {group_idx}: {group_name} ({len(items)} items)")
        
        group_pbar = tqdm(total=len(items), desc=f"{group_name}", unit="img", leave=True)

        out_dir = output_root / group_name
        face_dir = out_dir / "facemesh"
        out_dir.mkdir(parents=True, exist_ok=True)
        face_dir.mkdir(exist_ok=True)

        valid_path = out_dir / "valid.txt"
        invalid_path = out_dir / "invalid.txt"

        processed = load_name_set(valid_path) | load_name_set(invalid_path)

        for i in range(0, len(items), batch_size):
            batch_items = [
                it for it in items[i:i + batch_size]
                if format_name(it[0], it[1], it[2]) not in processed
            ]

            if not batch_items:
                group_pbar.update(min(batch_size, len(items) - i))
                continue
            
            batch_paths = [p for p, _, _ in batch_items]
            batch_imgs = loader.load_batch(batch_paths)

            valid = [
                (it, img) for it, img in zip(batch_items, batch_imgs)
                if img is not None
            ]
            if not valid:
                group_pbar.update(len(batch_items))
                continue

            batch_items, batch_imgs = zip(*valid)

            # YOLO inference
            with torch.inference_mode():
                person_results = yolo_person(
                    list(batch_imgs), verbose=False, device=device, max_det=5
                ) if yolo_person != None else None
                face_results = yolo_face(
                    list(batch_imgs), verbose=False, device=device, max_det=5
                )

            person_boxes_batch = [
                r.boxes.data.cpu().numpy() if r.boxes is not None else None
                for r in person_results
            ] if yolo_person != None else [None] * len(batch_items)

            face_boxes_batch = [
                r.boxes.data.cpu().numpy() if r.boxes is not None else None
                for r in face_results
            ]

            # Batch validation with parallel face segmentation
            validated_batch = validate_batch_fast(
                batch_imgs,
                person_boxes_batch,
                face_boxes_batch,
                face_mesh_pool
            )

            accepted = []
            rejected = []

            for (img_path, score, group_id), validated in zip(batch_items, validated_batch):
                final_name = format_name(img_path, score, group_id)
                
                if validated is None:
                    rejected.append(final_name)
                else:
                    accepted.append((img_path, validated, final_name))

            # Async I/O writes
            for img_path, validated, final_name in accepted:
                io_writer.submit(
                    _copy_and_write, 
                    (img_path, validated, final_name),
                    out_dir, 
                    face_dir, 
                    mode
                )

            append_names(valid_path, [n for _, _, n in accepted])
            append_names(invalid_path, rejected)

            group_pbar.update(len(batch_items))
        
        # Ensure all I/O completes before marking group as done
        io_writer.wait_all()
        
        group_pbar.close()
        
        completed_groups.add(group_name)
        save_manifest(manifest_path, completed_groups)
        processed_count += 1

    io_writer.wait_all()

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Skipped: {skipped_count} groups (already completed)")
    print(f"Processed: {processed_count} groups")
    print(f"{'='*60}")

# ============= UPDATE MAIN FUNCTION =============
def process_dataset_dir(
    input_root,
    output_root,
    yolo_face_model_path,
    yolo_person_model_path,
    device=0,
    mode="copy",
    batch_size=16,
    jsonl_path=None,
    max_per_prompt=None,
):
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    yolo_face = YOLO(yolo_face_model_path)
    if ".pt" in yolo_face_model_path:
        yolo_face.to(device)
        yolo_face.fuse()
        yolo_face.model.half()

    if yolo_person_model_path != None:
        yolo_person = YOLO(yolo_person_model_path)
        if ".pt" in yolo_person_model_path:
            yolo_person.to(device)
            yolo_person.fuse()
            yolo_person.model.half()
    else:
        yolo_person = None

    # Create FaceMesh pool instead of single instance
    face_mesh_pool = FaceMeshPool(num_workers=4)

    loader = FastImageLoader(num_workers=8)
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    if jsonl_path:
        groups = iter_groups_from_jsonl(
            Path(jsonl_path), input_root, max_per_prompt
        )
    else:
        groups = iter_groups_from_fs(input_root, img_exts)

    process_groups(
        groups,
        output_root,
        yolo_face,
        yolo_person,
        face_mesh_pool,  # Changed
        loader,
        batch_size,
        mode,
        device
    )

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process dataset directory with YOLO-based filtering"
    )

    # ---------------- Paths ----------------
    parser.add_argument("--input_root", type=str, required=True,
                        help="Root directory of the input dataset")

    parser.add_argument("--output_root", type=str, required=True,
                        help="Root directory where processed output will be written")

    parser.add_argument("--jsonl_path", type=str, required=True,
                        help="Path to JSONL file containing retrieval results")

    parser.add_argument("--yolo_face_model_path", type=str, default=None,
                        help="Path to YOLO face detection model (e.g. yolov12l-face.pt)")

    parser.add_argument("--yolo_person_model_path", type=str, default=None,
                        help="Path to YOLO person detection model (e.g. yolo12s.pt). If not provided, person detection is disabled."
    )

    # ---------------- Runtime ----------------
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to run inference on")

    parser.add_argument("--mode", type=str, default="copy", choices=["copy", "move"],
                        help="How to handle accepted images")

    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for processing")

    args = parser.parse_args()

    # ---------------- Run ----------------
    process_dataset_dir(
        input_root=args.input_root,
        output_root=args.output_root,
        yolo_face_model_path=args.yolo_face_model_path,
        yolo_person_model_path=args.yolo_person_model_path,
        device=args.device,
        mode=args.mode,
        batch_size=args.batch_size,
        jsonl_path=args.jsonl_path,
    )

    # process_dataset_dir(
    #     input_root=r"G:\Thesis",
    #     output_root=r"E:\ImageRetrieval\Professions_125k_Cleaned", #r"G:\Thesis\ImageRetrieval\Professions_125k_Cleaned", 
    #     yolo_face_model_path="models/yolov8n-face.pt", # - Just say you used "yolov12l-face.pt",
    #     yolo_person_model_path=None, #"models/yolov8n.pt",# - Just say you used "yolo12s.pt",
    #     device='cuda',
    #     mode='copy',
    #     batch_size=128,
    #     jsonl_path=r"G:\Thesis\ImageRetrieval\Professions_125k_test\125k_retrieval_results_batchsize_10.jsonl"
    # )

# python YoloFilteringImageRetrieval.py --input_root "G:\Thesis" --output_root "E:\ImageRetrieval\Professions_125k_Cleaned" --jsonl_path "G:\Thesis\ImageRetrieval\Professions_125k_test\125k_retrieval_results_batchsize_10.jsonl" --yolo_face_model_path "models/yolov8n-face.pt" --mode "copy" --batch_size 128

# python YoloFilteringImageRetrieval.py --input_root "G:\Thesis" --output_root "E:\ImageRetrieval\Professions_125k_Cleaned" --jsonl_path "G:\Thesis\ImageRetrieval\Professions_125k_test\125k_retrieval_results_batchsize_10.jsonl" --yolo_face_model_path "models/yolov8n-face.pt" --mode "copy" --batch_size 16


# python YoloFilteringTest.py --input_root "G:\Thesis" --output_root "E:\ImageRetrieval\test" --jsonl_path "G:\Thesis\ImageRetrieval\Professions_125k_test\125k_retrieval_results_batchsize_10.jsonl" --yolo_face_model_path "models/yolov8n-face.pt" --mode "copy" --batch_size 16