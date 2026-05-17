# Debiasing Pipeline

## Overview

This directory contains the pipeline for generating demographically debiased images using ITI-GEN, ControlNet, and colour alignment.

The pipeline consists of:

1. CCv2 dataset preparation for ITI-GEN training
2. ITI-GEN model training (via `ITI-GEN/train_iti_gen.py`)
3. Debiased image generation using two alternative sampling strategies

The scripts and notebooks in this directory have been re-tested and verified to be functioning correctly.

> **Note:** The ITI-GEN scripts require the ITI-GEN repository to be present. Clone it from [https://github.com/humansensinglab/ITI-GEN](https://github.com/humansensinglab/ITI-GEN) into the `ITI-GEN/` subdirectory before running.

---

## Files

### `prepare_ccv2_skintone_gender.py`

Prepares the CCv2 dataset for ITI-GEN training by creating per-label image directories from the CCv2 annotations.

Reads gender and MST skin tone labels from the CCv2 JSON and creates symlinked image views organised by label, producing the folder structure expected by `train_iti_gen.py`.

---

### `ITI-GEN/train_iti_gen.py`

Prerequisite training script (from the ITI-GEN repository). Trains ITI-GEN token embeddings for gender and skin tone debiasing using the prepared CCv2 data.

---

### `ITIGen-ControlNet-ChamferDebiasing_AlignedLikeZeroShot.py`

Debiased image generation pipeline using ITI-GEN + ControlNet + Chamfer colour alignment.

The diffusion sampling applies colour projection inside each DDPM backward step using iterative Chamfer distance matching, aligned with the `zero-shot.py` sampling logic from the ITI-GEN repository.

Supports ControlNet conditioning types: `pose`, `canny`, `depth`, `seg`.

---

### `ITIGen-ControlNet-SW-Guidance.py`

Debiased image generation pipeline using ITI-GEN + ControlNet + Sliced Wasserstein colour guidance.

Uses SW-guidance applied at each denoising step with configurable loss types (`mean_cov`, `SWD`, `DSWD`, `GSWD`, `ISEBSW`). Supports time travel blocks and optional SARAH gradient optimisation.

Supports ControlNet conditioning types: `pose`, `canny`, `depth`, `seg`.

---

Both generation scripts share the same ITI-GEN and ControlNet conditioning components and differ only in the diffusion sampling strategy and colour alignment mechanism.

---

### `Setup.ipynb`

Documents and performs environment setup required for:
- Stable Diffusion / Diffusers
- ControlNet
- ITI-GEN dependencies
- PyTorch3D (optional, for Chamfer distance)
- Supporting packages