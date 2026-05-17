# Spurious Feature Image Transformation Pipeline

## Overview

This directory contains the pipeline for generating spurious feature image variants used to probe dataset classifiers and demographic attribute models.

Each subdirectory corresponds to a distinct image transformation type. The scripts in this directory have been re-tested and verified to be functioning correctly.

---

## Directory Structure
```
5_Spurious_Feature/
├── Depth/
├── EdgeDetection/
├── HighLowPassFiltering/
├── ImageCaptioning/
├── ObjectDetection/
├── Occlusion/
├── PoseEstimation/
├── SemanticSegmentation/
├── ShufflingAndColour/
└── VAE/
```

---

## `Depth/`

### `Depth.py`

Generates depth map images using Depth-Anything-V2. Depth is normalised to 0–255 and saved as a 3-channel PNG to match classifier input expectations. Supports resumable processing and optional output resizing.

```

### `DepthAtPosePoints.py`

Combines YOLO pose estimation with Depth-Anything-V2 to extract depth values at each of the 17 COCO skeleton keypoints per person. Outputs a JSONL file with raw and normalised joint depth vectors alongside keypoint coordinates and visibility flags.

```

---

## `EdgeDetection/`

### `EdgeDetection.py`

Generates edge detection images using Canny edge detection and optionally SAM (Segment Anything Model). SAM was evaluated but not used in production due to runtime cost; Canny edges were used as the final transformation.

```

---

## `HighLowPassFiltering/`

### `HighLowPassFiltering.py`

Applies frequency-domain high-pass and low-pass filtering via 2D FFT following the protocol from the reference paper. Used to measure classifier reliance on low- vs high-frequency visual cues.

```

---

## `ImageCaptioning/`

### `ImageCaptioningTesting.ipynb`

Evaluates 10 image captioning models across speed and caption quality to select the best model for large-scale annotation. The reference paper uses LLaVA-1.5; BLIP-large was selected as the production model based on speed-quality trade-off.

### `ImageCaptioning.py`

Generates image captions using BLIP-image-captioning-large. Applies word remappings from `mappings.txt` to produce gender-neutral captions.

### `mappings.txt`

Word-level remapping rules applied during caption generation to neutralise gender-explicit language (e.g. `woman → person`, `she → they`).

---

## `ObjectDetection/`

### `object_detection_testing.ipynb`

Evaluates object detection models across COCO, OpenImages v7, and LVIS label sets to select the best model for large-scale annotation. YOLOv8x-OIV7 was selected as the production model.

### `ObjectDetection.py`

Detects objects in images using YOLOv8x-OIV7 and produces annotated images (bounding boxes on original or white background). Supports label remapping and class exclusion to remove gender-explicit or human body-part classes.

### `run.bat`

Batch runner executing `ObjectDetection.py` across the SD and Re-LAION-5B datasets with the restricted label configuration.

---

## `Occlusion/`

### `Occlusion.py`

Generates occlusion image variants using instance segmentation. Produces five occlusion types: `Full_NoBg`, `MaskSegm`, `MaskSegm_NoBg`, `MaskRect`, `MaskRect_NoBg`. Requires images containing people; COCO is excluded.

### `occlusionModelsHelper.py`

Helper module containing occlusion model classes (Mask2Former, YOLACT) and shared configuration used by `Occlusion.py`.

### `evaluation.py`

Evaluation script used to assess occlusion model quality and inference speed to select the most appropriate model.

---

## `PoseEstimation/`

### `PoseModelEvaluation.py`

Evaluates multiple pose estimation models (YOLOv8, MoveNet, MediaPipe, HRNet) on the COCO 2017 validation set using AP, AR, FPS, and image-level detection rate. Used to select the production pose model.

### `YoloHyperparamTuning.py`

Grid search hyperparameter tuning suite for YOLOv8 pose estimation, optimising confidence threshold, IOU threshold, and keypoint confidence threshold against COCO AP/AR metrics.

### `PoseDetection.py`

Detects poses using YOLOv8 within the specified images, producing COCO-format keypoints and visualised pose overlays. Requires images containing people; COCO is excluded.

---

## `SemanticSegmentation/`

### `SemanticSegmentation.py`

Generates semantic segmentation images using Mask2Former (ADE20K, 150 classes). Segments are coloured using the ADE20K palette and saved as RGB PNGs. Supports resumable processing and optional output resizing.

---

## `ShufflingAndColour/`

### `PixelPatchShufflingMeanRGB.py`

Generates pixel-shuffled images, patch-shuffled images, and mean RGB images. Also extracts per-image mean RGB values. Used to measure classifier reliance on colour and spatial arrangement cues.

---

## `VAE/`

### `VAE.py`

Reconstructs images through the KL-regularised VAE from the latent-diffusion repository, isolating low-level texture and colour information while discarding high-level semantic content. Uses the same VAE and config as the reference paper.

> **Note:** Requires the `latent-diffusion` and `taming-transformers` repositories. `VAE.py` clones them automatically on first run. The `kl-f4` model checkpoint must be downloaded separately via `Setup.ipynb`.

---

## `Setup.ipynb`

Each subdirectory contains its own `Setup.ipynb` documenting the environment dependencies required for that transformation pipeline.