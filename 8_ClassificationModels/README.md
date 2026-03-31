# Spurious Feature Analysis & Demographic Analysis Pipeline

## Overview

This directory contains the pipeline for training spurious feature probes and performing demographic distribution analysis across Re-LAION-5B and SDv1.5 datasets.

The pipeline consists of:

1. Demographic distribution analysis and figure generation
2. Dataset classification using ConvNeXt-Tiny
3. Demographic attribute probing using ResNet-50, Logistic Regression, MLP, and Sentence-T5

The scripts in this directory have been re-tested and verified to be functioning correctly.

---

## Files

### `DemographicAnalysis.py`

Extracts demographic statistics from annotation JSONL files and produces all figures and CSV summaries for the evaluation section.

Outputs figures:
- `aggregate_gender.png`
- `aggregate_skintone_bins.png`
- `aggregate_mst_distribution.png`
- `isco_gender.png`
- `isco_skintone_bins.png`
- `isco_mst_heatmap.png`

Outputs CSVs:
- Aggregate and per-profession gender and skin tone distributions
- ISCO-08 grouped gender and skin tone breakdowns
- Top gender/skin tone skew and amplification analysis

---

### `DatasetClassification_gradient_accum.py`

Replicates the ConvNeXt-Tiny dataset classification experiment from the reference paper (Section 3.1). Trains a binary or multi-class classifier to distinguish between Re-LAION-5B, SDv1.5, and COCO images.

Enhancements over the paper: early stopping, AUC metrics, bootstrapping, and gradient accumulation for large effective batch sizes.

Reference accuracy: 82.0%

---

### `ResNet50_Classification.py`

Replicates the ResNet-50 demographic attribute probe from the paper. Trains image-based classifiers for gender and skin tone prediction using fixed train/val/test splits and validation-based hyperparameter selection.

Also supports ConvNeXt-Tiny as an alternative backbone. Includes checkpoint/resume support, per-seed output files, and a `summary.json` aggregating results across seeds.

---

### `LogisticRegression.py`

Replicates the logistic regression probe from the paper. Operates on pre-extracted numeric feature vectors (object OHE, mean RGB, pose keypoints) to test whether low-level spurious features are sufficient to predict demographic attributes.

---

### `MLP.py`

Replicates the MLP probe from the paper. Operates on pre-extracted numeric features (e.g. body keypoints) to isolate and measure the contribution of pose-related artifacts to demographic predictability, removing all pixel-level appearance cues.

---

### `SentenceT5_Probe.py`

Replicates the Sentence-T5 caption probe from the paper. Encodes image captions using a SentenceTransformer and trains a linear head to predict demographic attributes from caption embeddings alone.

Implements the cosine LR scheduler with linear warmup matching the paper's training recipe exactly.

---

### `Setup.ipynb`

Documents and performs environment setup required for:
- PyTorch / torchvision
- scikit-learn
- Supporting dependencies