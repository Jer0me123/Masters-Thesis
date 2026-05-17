import os
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# ============================================================
# CONFIG
# ============================================================

NUM_JOINTS = 17
TARGET_SHORT_SIDE = 384   # canonical processing size
DEPTH_NORM_EPS = 1e-6

COCO_SKELETON = [
    (15,13),(13,11),(16,14),(14,12),
    (11,12),(5,11),(6,12),
    (5,6),(5,7),(6,8),
    (7,9),(8,10),
    (1,2),(0,1),(0,2),(1,3),(2,4)
]

# ============================================================
# IMAGE UTILS
# ============================================================

def resize_keep_aspect(img, target_short):
    h, w = img.shape[:2]
    scale = target_short / min(h, w)
    nh, nw = int(h * scale), int(w * scale)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

# ============================================================
# DEPTH NORMALIZATION (per-person)
# ============================================================

def normalize_depths(depths, vis):
    mask = vis.astype(bool)
    if not mask.any():
        return np.zeros_like(depths)

    d = depths[mask]
    d = d - d.mean()
    std = d.std()
    if std < DEPTH_NORM_EPS:
        std = 1.0
    depths[mask] = depths[mask] / std
    return depths

# ============================================================
# VISUALIZATION
# ============================================================

def draw_pose_on_depth(depth_img, keypoints):
    vis = cv2.cvtColor(depth_img, cv2.COLOR_GRAY2BGR)
    kps = np.array(keypoints).reshape(NUM_JOINTS, 3)

    for x, y, v in kps:
        if v > 0:
            cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)

    for i, j in COCO_SKELETON:
        if kps[i,2] > 0 and kps[j,2] > 0:
            cv2.line(
                vis,
                (int(kps[i,0]), int(kps[i,1])),
                (int(kps[j,0]), int(kps[j,1])),
                (0, 255, 255),
                2
            )
    return vis

def load_and_resize(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    return resize_keep_aspect(img, TARGET_SHORT_SIDE)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Models ----
    pose_model = YOLO(f"models/yolov8{args.pose_model}-pose.pt")
    pose_model.to(device)
    pose_model.fuse()

    depth_name = f"depth-anything/Depth-Anything-V2-{args.depth_model}-hf"
    depth_processor = AutoImageProcessor.from_pretrained(depth_name, use_fast=True)
    depth_model = AutoModelForDepthEstimation.from_pretrained(depth_name).to(device).eval()

    out_jsonl = Path(args.output_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if args.viz_dir:
        viz_dir = Path(args.viz_dir)
        viz_dir.mkdir(parents=True, exist_ok=True)

    processed = set()
    if out_jsonl.exists():
        with open(out_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    processed.add(rec["image"])
                except Exception:
                    pass

    exclude = {d.lower() for d in args.exclude_dirs}

    images = []
    for p in Path(args.image_dir).rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".jpg",".png",".jpeg"}:
            continue
        if any(part.lower() in exclude for part in p.parts):
            continue
        if p.name in processed:
            continue
        images.append(p)

    print(f"[Resume] Found {len(processed)} already processed images")
    print(f"[Resume] Remaining images: {len(images)}")

    with open(out_jsonl, "a", encoding="utf-8") as out:
        for i in tqdm(range(0, len(images), args.batch_size), desc="Pose+Depth extraction"):
            batch_paths = images[i:i + args.batch_size]

            # -------------------------
            # Parallel image loading
            # -------------------------
            with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
                imgs = list(pool.map(load_and_resize, batch_paths))

            valid = [(p, img) for p, img in zip(batch_paths, imgs) if img is not None]
            if not valid:
                continue

            paths, imgs = zip(*valid)

            # -------------------------
            # Pose (batched)
            # -------------------------
            pose_results = pose_model(list(imgs), conf=args.conf, verbose=False)

            # -------------------------
            # Depth (batched)
            # -------------------------
            rgbs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
            # inputs = depth_processor(images=rgbs, return_tensors="pt").to(device)
            inputs = depth_processor(
                images=rgbs,
                return_tensors="pt",
                do_resize=False,
                do_pad=False,
            ).to(device)


            with torch.no_grad():
                depths_pred = depth_model(**inputs).predicted_depth

            # -------------------------
            # Per-image processing
            # -------------------------
            for img_path, img, pose_res, depth_tensor in zip(
                paths, imgs, pose_results, depths_pred
            ):

                if pose_res.keypoints is None or len(pose_res.boxes) == 0:
                    continue

                best = np.argmax(pose_res.boxes.conf.cpu().numpy())
                kps = pose_res.keypoints.data[best].cpu().numpy()  # (17,3)

                keypoints_xyv = []
                for x, y, c in kps:
                    if c >= args.kp_conf:
                        keypoints_xyv.extend([float(x), float(y), 2])
                    else:
                        keypoints_xyv.extend([0.0, 0.0, 0])

                # ---- depth map ----
                # depth = depth_tensor[0].cpu().numpy()
                depth = depth_tensor.cpu().numpy()
                depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
                depth = (depth * 255).astype(np.uint8)

                # ---- sample depth at pose ----
                depths = np.zeros(NUM_JOINTS, dtype=np.float32)
                vis = np.zeros(NUM_JOINTS, dtype=np.int32)

                H, W = depth.shape
                for j in range(NUM_JOINTS):
                    x, y, v = keypoints_xyv[3*j:3*j+3]
                    if v > 0:
                        xi = int(np.clip(round(x), 0, W - 1))
                        yi = int(np.clip(round(y), 0, H - 1))
                        depths[j] = depth[yi, xi]
                        vis[j] = 1

                depths_norm = normalize_depths(depths.copy(), vis)

                rec = {
                    "image": img_path.name,
                    "joint_depths_raw": depths.tolist(),
                    "joint_depths_normalized": depths_norm.tolist(),
                    "joint_visibility": vis.tolist(),
                    "keypoints_with_visibility": keypoints_xyv
                }

                out.write(json.dumps(rec) + "\n")

                # ---- Visualization ----
                if args.draw:
                    vis_img = draw_pose_on_depth(depth, keypoints_xyv)

                    if args.viz_resize:
                        vis_img = cv2.resize(
                            vis_img,
                            (args.viz_resize, args.viz_resize),
                            interpolation=cv2.INTER_AREA
                        )

                    rel = img_path.relative_to(args.image_dir)
                    out_path = viz_dir / rel.parent / f"{img_path.stem}_pose_on_depth.png"
                    out_path.parent.mkdir(parents=True, exist_ok=True)

                    cv2.imwrite(str(out_path), vis_img)


    print(f"\nSaved pose+depth vectors to: {out_jsonl}")
    if args.draw:
        print(f"Saved visualizations to: {viz_dir}")

# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Combined Pose + Depth Extraction")
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--exclude_dirs", nargs="*", default=["facemesh", "depth", "pose"], help="Directory names to exclude during recursive scan")
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--pose_model", default="l", choices=["n","s","m","l","x"])
    ap.add_argument("--depth_model", default="Small", choices=["Small","Base","Large","Giant"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--kp_conf", type=float, default=0.5)
    ap.add_argument("--draw", action="store_true")
    ap.add_argument("--viz_dir")
    ap.add_argument("--viz_resize", type=int, help="Resize visualization (e.g. 224)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()
    main(args)


# ===========================================================
# EXAMPLE USAGE
# python DepthAtPosePoints.py --image_dir "path/to/input" --output_jsonl "path/to/jsonloutput" --pose_model l --depth_model Small --conf 0.25 --kp_conf 0.5 --draw --viz_resize 224 --viz_dir "path/to/output" --batch_size 1 --num_workers 8 --exclude_dirs facemesh