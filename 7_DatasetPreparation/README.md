# Dataset Preparation Pipeline

## Overview

This directory contains the pipeline for constructing train/val/test splits used to train spurious feature identification models across gender, skin tone, and dataset classification tasks.

The pipeline consists of:

1. Base split creation from annotated datasets
2. Auxiliary feature enrichment (object detections, pose keypoints, mean RGB, captions)
3. Dataset classification split construction (Re-LAION-5B vs SD)
4. Pseudo-label split construction (negative control)
5. COCO extension for multi-class dataset classification

The scripts and notebooks in this directory have been re-tested and verified to be functioning correctly.

---

## Files

### `1_CreateSplits.py`

Creates the base stratified train/val/test splits from annotated JSONL files.

Supports label selection (`gender`, `mst_label`), confidence filtering, face/non-face annotation source selection, and optional class balancing.

---

### `2_CreateObject_OHE_Splits.py`

Enriches existing splits with object detection one-hot encoded feature vectors. Uses a fixed class list (`openimagesv7_classes_raw.json`) to produce a consistent 601-dimensional binary presence vector per image.

---

### `3_CreatePoses_MeanRGB_OHE_Splits.py`

Adds auxiliary features to existing splits including pose keypoints, mean RGB statistics, depth features, and object encodings.

---

### `4_CreateCaption_Splits.py`

Enriches existing splits with image captions from a single JSONL caption file. Supports `raw` and `remapped` caption types.

---

### `5_ExtendImagePathsWithSuffix.py`

Updates image paths within split JSON files to reference transformed image variants by inserting a suffix before the file extension (e.g. `_depth`, `_patch_shuffle`).

---

### `6_MergeSplitsForDatasetClassification.py`

Merges two dataset splits into a binary dataset classification split (Re-LAION-5B vs SD). Supports optional class balancing, downsampling, and split membership enforcement via a reference split.

---

### `7_PseudoDatasetSplitsForDatasetClassification.py`

Assigns arbitrary random dataset labels to existing image splits to serve as a negative control experiment for dataset classification.

Output: UniversalSplits\PseudoDatasetClassification\

---

### `8_ExtendWithCoco.py`

Extends binary dataset classification splits with COCO images as a third class, converting the task from binary to multi-class.

---

### `DebiasedImagesCreateSplits.py`

Generates an `annotations.jsonl` file for the debiased image dataset by parsing gender and MST skin tone labels from the ITI-GEN output folder naming convention.

---

### `captions_map.json`

Maps dataset label indices to their corresponding caption JSONL file paths. Used by `New_4_CreateCaption_Splits.py` in multi-dataset mode.

---

### `openimagesv7_classes_raw.json`

Fixed class list of 601 Open Images v7 object categories used by `2_CreateObject_OHE_Splits.py` to produce consistent one-hot feature vectors.