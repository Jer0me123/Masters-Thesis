# Image Retrieval & Filtering Pipeline

## Overview

This directory contains the retrieval and post-processing pipeline used to construct a profession-conditioned subset from precomputed RE-LAION CLIP embeddings.

The pipeline consists of:

1. CLIP embedding similarity search (FAISS / Non-FAISS)
2. Retrieval result generation (.jsonl)
3. YOLO-based face/person filtering
4. MediaPipe FaceMesh segmentation
5. Resume-safe structured dataset materialisation

The scripts and notebooks in this directory have been re-tested and verified to be functioning correctly.

---

## Files

### `ImageRetrieval.ipynb`

Performs CLIP embedding search over precomputed RE-LAION embeddings.

Supports:
- FAISS-based similarity search (production method)
- Non-FAISS search (testing/debugging only)

Functionality:
- Encodes text prompts using CLIP
- Searches embedding index for similar images
- Retrieves top-k matches per prompt
- Outputs retrieval results to `.jsonl`

Output:
E:\ImageRetrieval\Professions_125k_ISCO_Aligned\ISCO_aligned_125k_retrieval_results_batchsize_10.jsonl

This file contains:
- Prompt
- Retrieved image identifiers
- Similarity scores
- Shard/group information

---

### `YoloFilteringImageRetrieval.py`

Performs post-retrieval filtering and cleaning of CLIP-retrieved profession images.

Implements:
1. YOLO face detection  
2. Optional YOLO person detection  
3. MediaPipe FaceMesh segmentation  
4. Batched inference processing  
5. Resume-safe group tracking  

Functionality:
- Loads images referenced in retrieval `.jsonl`
- Filters images to ensure valid face detection
- Optionally ensures face is associated with a detected person
- Generates segmented face crops via FaceMesh
- Logs valid/invalid images
- Writes cleaned images to structured profession directories

Output:
E:\ImageRetrieval\Professions_125k_ISCO_Aligned\

Contains:
- Profession-grouped directories
- Filtered images
- FaceMesh crops
- Valid / invalid tracking files

---

### `Setup.ipynb`

Documents and performs environment setup required for:
- FAISS
- PyTorch
- Ultralytics YOLO
- MediaPipe
- CLIP (Transformers)
- Supporting dependencies

Ensures reproducibility of the retrieval and filtering pipeline.

---

## Pipeline Outputs

Running the retrieval and filtering pipeline produces:

### 1. Retrieval Results

Generates `.jsonl` file containing:
- Retrieved image list per prompt
- Similarity scores
- FAISS-based similarity search outputs

Location:
E:\ImageRetrieval\Professions_125k_ISCO_Aligned\ISCO_aligned_125k_retrieval_results_batchsize_10.jsonl

---

### 2. Filtered & Structured Profession Dataset

Using the generated `.jsonl` file:
- Images are retrieved from disk
- Filtered via face detection & optional person detection
- FaceMesh segmentation is derived
- Clean images are organised into profession-specific folders

Location:
E:\ImageRetrieval\Professions_125k_ISCO_Aligned\

---

## Notes

- The FAISS-based retrieval approach is the primary production method.
- The non-FAISS approach in `ImageRetrieval.ipynb` is retained for testing and validation.
- YOLO + FaceMesh filtering ensures structural consistency of human subjects prior to downstream demographic and fairness analysis.
- Resume-safe processing enables large-scale profession subsets (e.g., 125k per group) to be processed reliably.


<!-- This text file outlines the purpose of each .ipynb & .py file in this directory. 

ImageRetrieval.ipynb - This notebook performs Non-Faiss / Faiss CLIP embedding search over the precomputed RE-LAION embeddings and retrieves similar images for text prompts (Non-Faiss approach only used for testing)
YoloFilteringImageRetrieval.py - This python script performs post-retrieval filtering and cleaning of the CLIP-retrieved profession images using:
    1. YOLO face detection
    2. Optional YOLO person detection
    3. MediaPipe FaceMesh segmentation
    4. Batch processing
    5. Resume-safe group tracking
Setup.ipynb - This file outlines the Environment setup

The code in the above.ipynb & .py has been re-teseted and confirmed to be working. 

Furthermore running the relevant notebook produces the following: 

1. Generates a .jsonl files with the list of images retrieved according to the specified prompts along with their similarity score using FAISS image similarity search -> E:\ImageRetrieval\Professions_125k_ISCO_Aligned\ISCO_aligned_125k_retrieval_results_batchsize_10.jsonl
2. Retrieves images using the generated.jsonl file filtered out via facedetection & posedetection, alongside deriving facemeshs of retrieved images -> E:\ImageRetrieval\Professions_125k_ISCO_Aligned -->