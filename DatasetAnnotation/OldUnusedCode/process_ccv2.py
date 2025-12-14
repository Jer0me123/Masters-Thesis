# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
# os.environ["GLOG_minloglevel"] = "3"
# os.environ["absl.logging"] = "ERROR"

# import json
# import cv2
# import gc
# import numpy as np
# import pandas as pd
# from multiprocessing import Pool, cpu_count
# import multiprocessing
# import mediapipe as mp
# from tqdm import tqdm

# # ============================================================
# # WINDOWS MULTIPROCESSING FIX
# # ============================================================
# multiprocessing.set_start_method("spawn", force=True)

# # ============================================================
# # CONFIG
# # ============================================================

# CCV2_ROOT = r"G:\Thesis\CasualConversationv2_Dataset\Images"
# CCV2_JSON = r"G:\Thesis\CasualConversationv2_Dataset\Annotations\CasualConversationsV2.json"

# confidence_filter = None  # OR ["low"], ["medium"], ["high"]

# if confidence_filter:
#     CCV2_OUTPUT_DIR = rf"G:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2_{confidence_filter}"
# else:
#     CCV2_OUTPUT_DIR = r"G:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2_2"

# os.makedirs(CCV2_OUTPUT_DIR, exist_ok=True)


# # ============================================================
# # SKIN TONE PARSER
# # ============================================================

# def parse_mst_label(mst_dict):
#     scale_str = mst_dict.get("scale", "")
#     digits = ''.join(c for c in scale_str if c.isdigit())
#     return int(digits) if digits.isdigit() else None


# # ============================================================
# # FACE SEGMENTATION
# # ============================================================

# def segment_face(image, face_mesh):
#     h_img, w_img, _ = image.shape
#     rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb)

#     if not results.multi_face_landmarks:
#         return None

#     landmarks = results.multi_face_landmarks[0]
#     points = np.array([(int(lm.x * w_img), int(lm.y * h_img))
#                        for lm in landmarks.landmark])

#     x, y, w_box, h_box = cv2.boundingRect(points)
#     if w_box < 50 or h_box < 50:
#         return None

#     aspect = h_box / w_box
#     if aspect < 0.6 or aspect > 2.5:
#         return None

#     # Mask
#     mask = np.zeros((h_img, w_img), dtype=np.uint8)
#     hull = cv2.convexHull(points)
#     cv2.fillConvexPoly(mask, hull, 255)

#     segmented = cv2.bitwise_and(image, image, mask=mask)

#     # Padded crop
#     pad = 10
#     x = max(0, x - pad)
#     y = max(0, y - pad)
#     w_box = min(w_img, x + w_box + 2*pad) - x
#     h_box = min(h_img, y + h_box + 2*pad) - y

#     return segmented[y:y + h_box, x:x + w_box]


# # ============================================================
# # WORKER FUNCTION (FIXED)
# # ============================================================

# def process_subject(row):
#     """Process a single subject with proper resource cleanup"""
    
#     subject_id = row["subject_id"]
#     mst_label = row["mst_label"]
#     confidence = row["confidence"]

#     input_dir = os.path.join(CCV2_ROOT, subject_id)
#     if not os.path.isdir(input_dir):
#         return []

#     output_dir = os.path.join(CCV2_OUTPUT_DIR, subject_id)
#     os.makedirs(output_dir, exist_ok=True)

#     # Create face mesh LOCALLY (not global)
#     try:
#         face_mesh = mp.solutions.face_mesh.FaceMesh(
#             static_image_mode=True,
#             max_num_faces=1,
#             refine_landmarks=True,
#             min_detection_confidence=0.5
#         )
#     except Exception as e:
#         return []

#     results = []
#     image_files = [
#         f for f in os.listdir(input_dir)
#         if f.lower().endswith((".jpg", ".jpeg", ".png"))
#     ]

#     try:
#         for fname in image_files:
#             in_path = os.path.join(input_dir, fname)
#             out_name = f"{subject_id}_{fname}"
#             out_path = os.path.join(output_dir, out_name)

#             img = cv2.imread(in_path)
#             if img is None:
#                 continue

#             crop = segment_face(img, face_mesh)
            
#             # Clean up image memory immediately
#             del img
            
#             if crop is None:
#                 continue

#             cv2.imwrite(out_path, crop)
#             del crop

#             results.append({
#                 "subject_id": subject_id,
#                 "mst_label": mst_label,
#                 "confidence": confidence,
#                 "original_image": fname,
#                 "cropped_image": out_name
#             })

#     except Exception as e:
#         pass
#     finally:
#         # CRITICAL: Close face mesh to free resources
#         face_mesh.close()
#         gc.collect()

#     return results


# # ============================================================
# # MAIN PIPELINE
# # ============================================================

# def process_casual_conversations_v2(confidence_filter=None):
#     print("\n=== Processing CCV2 ===")

#     # Load JSON (limit to 10 for testing)
#     with open(CCV2_JSON, "r", encoding="utf-8") as f:
#         data = json.load(f)
    
#     # Group by subject_id and take first entry for each subject
#     subjects_dict = {}
    
#     for entry in data:
#         subject_id = entry["subject_id"]
        
#         # Skip if we've already seen this subject
#         if subject_id in subjects_dict:
#             continue
            
#         mst = entry.get("monk_skin_tone", {})
#         mst_label = parse_mst_label(mst)
#         confidence = mst.get("confidence", "").lower()

#         if mst_label is None:
#             continue
#         if confidence_filter and confidence not in confidence_filter:
#             continue
        
#         subjects_dict[subject_id] = {
#             "subject_id": subject_id,
#             "mst_label": mst_label,
#             "confidence": confidence
#         }
    
#     subjects = list(subjects_dict.values())
#     print(f"Loaded {len(subjects)} unique subjects from {len(data)} video entries.")

#     # Skip subjects already processed
#     already_done = {
#         d for d in os.listdir(CCV2_OUTPUT_DIR)
#         if os.path.isdir(os.path.join(CCV2_OUTPUT_DIR, d))
#         and len(os.listdir(os.path.join(CCV2_OUTPUT_DIR, d))) > 0
#     }

#     print(f"Found {len(already_done)} previously processed subjects.")

#     subjects = [s for s in subjects if s["subject_id"] not in already_done]
#     print(f"{len(subjects)} subjects remaining to process.\n")

#     if len(subjects) == 0:
#         print("Nothing to do.")
#         return

#     # REDUCED worker count to prevent resource exhaustion
#     num_workers = min(2, cpu_count())  # Changed from 4 to 2
#     print(f"Starting multiprocessing with {num_workers} workers...")

#     all_records = []
#     total = len(subjects)

#     try:
#         with Pool(processes=num_workers) as pool:
#             for res in tqdm(
#                 pool.imap_unordered(process_subject, subjects, chunksize=1),  # Changed from 2 to 1
#                 total=total,
#                 desc="Processing subjects",
#                 mininterval=0.5
#             ):
#                 if res:
#                     all_records.extend(res)
#                 # Force garbage collection after each subject
#                 gc.collect()

#     except KeyboardInterrupt:
#         print("\nInterrupted! Terminating workers...")
#         pool.terminate()
#         pool.join()
#         raise SystemExit
#     except Exception as e:
#         print(f"\nError occurred: {e}")
#         pool.terminate()
#         pool.join()
#         raise

#     # Save outputs
#     df = pd.DataFrame(all_records)
#     out_csv = os.path.join(CCV2_OUTPUT_DIR, "ccv2_per_image.csv")
#     df.to_csv(out_csv, index=False)

#     print(f"\nDone. Saved {len(df)} rows.")
#     print(f"CSV output: {out_csv}")


# # ============================================================
# # RUN
# # ============================================================

# if __name__ == "__main__":
#     process_casual_conversations_v2(confidence_filter=confidence_filter)