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
import time

def load_manifest(path: Path):
    if not path.exists():
        print("Doesnt Exist")
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
    # if person_boxes != None:
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
    # if person_boxes != None:
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


def process_groups(
    groups_iter,
    output_root,
    yolo_face,
    yolo_person,
    face_mesh,
    loader,
    batch_size,
    mode,
    device
):
    manifest_path = output_root / "manifest.json"
    completed_groups = load_manifest(manifest_path)
    print(manifest_path)
    print(completed_groups)

    print(f"Loaded manifest with {len(completed_groups)} completed groups")

    processed_count = 0
    skipped_count = 0

    for group_idx, (group_name, items_loader) in enumerate(groups_iter, start=1):
        
        # Fast skip check - no expensive operations
        if group_name in completed_groups:
            print(f"[SKIPPING] - Group {group_idx}: {group_name}")
            skipped_count += 1
            continue

        # Only now do we actually load the items (resolve paths, etc.)
        items = items_loader() if callable(items_loader) else items_loader
        
        print(f"\nProcessing group {group_idx}: {group_name} ({len(items)} items)")
        
        group_pbar = tqdm(total=len(items), desc=f"{group_name}", unit="img", leave=True, mininterval=0.0)

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
            # print(f"Batch Length: {len(batch_items)}")

            if not batch_items:
                # print("Batch Skipped as its already fully processed")
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

            with torch.inference_mode():
                person_results = yolo_person(list(batch_imgs), verbose=False, device=device, max_det=5) if yolo_person != None else None
                face_results = yolo_face(list(batch_imgs), verbose=False, device=device, max_det=5)

            person_boxes_batch = [
                r.boxes.data.cpu().numpy() if r.boxes is not None else None
                for r in person_results
            ] if yolo_person != None else [None] * len(batch_items)

            face_boxes_batch = [
                r.boxes.data.cpu().numpy() if r.boxes is not None else None
                for r in face_results
            ]

            accepted = []
            rejected = []

            for (img_path, score, group_id), img_bgr, pb, fb in zip(
                batch_items, batch_imgs, person_boxes_batch, face_boxes_batch
            ):
                final_name = format_name(img_path, score, group_id)

                if img_bgr is None or fb is None:
                    rejected.append(final_name)
                    continue
                
                validated = validate_image_fast(img_bgr, pb, fb, face_mesh)
                if validated is None:
                    rejected.append(final_name)
                    continue

                accepted.append((img_path, validated, final_name))

            max_workers = min(32, (os.cpu_count() or 8) * 2)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_copy_and_write, item, out_dir, face_dir, mode) for item in accepted]

                for f in as_completed(futures):
                    f.result()

            append_names(valid_path, [n for _, _, n in accepted])
            append_names(invalid_path, rejected)

            group_pbar.update(len(batch_items))
        
        group_pbar.close()
        
        completed_groups.add(group_name)
        save_manifest(manifest_path, completed_groups)
        processed_count += 1

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Skipped: {skipped_count} groups (already completed)")
    print(f"Processed: {processed_count} groups")
    print(f"{'='*60}")

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

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

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
        face_mesh,
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

# python.exe YoloFilteringImageRetrieval.py --input_root "" --output_root "" --jsonl_path "l" --yolo_face_model_path "" --yolo_person_model_path "" --mode "copy" --batch_size 128