This text file outlines the purpose of each .ipynb & .py file in this directory. 

ITI-GEN/ - This directory contains the main debiasing code
    1. ITIGen-ControlNet-ChamferDebiasing_AlignedLikeZeroShot.py & ITIGen-ControlNet-SW-Guidance.py -> These pipelines share identical ITI-GEN and ControlNet conditioning components, differing only in the diffusion sampling strategy and colour alignment mechanism.
    2. train_iti_gen.py -> This script serves as a prerequiste to the above .py scripts as it trains the ITI-GEN model to facilitate debiasing, these trained models are stored in ckpts/
    3. prepend.py -> This script is not longer used as its logic was integrated into the scripts in 1. however it was used to prepend the prompts with the necessary debiased embeddings.
    4. data/ -> This stores the training data used to train the ITI-GEN debiased models
    5. generation.py / evaluation.py -> These are scripts that came with the ITI-GEN repo and are not used.
prepare_ccv2_skintone_gender.py -> This scripts was used to prepare the ccv2 dataset for ITI-GEN training using the gender / skintone annotations that were derived prior
Testing\ (ColourDebiasing / FairGen / PoseDebiasing / unified-concept-editing / ImageOutputs) -> These are all directories relating to testing / repo used during initial debiasing method exploration.
Setup.ipynb - This file outlines the Environment setup & Model Downloads

The code in the above.ipynb & .py has been re-teseted and confirmed to be working. 

Furthermore running the relevant notebooks produces the following: 

1. Generates a training dataset using the CCv2 annotated images -> ITI-GEN\data\CCv2_Gender_benchmark
2. Trains ITI-GEN debiased models for gender / skintone -> ITI-GEN\ckpts\an_image_of_a_person_CCv2_Gender_CCv2_MSTE_SkinTone
3. Generated debiased images in terms of gender / skintone / pose / depth / segmentation / colour / objects -> ITI-GEN\outputs\


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

```

---

### `ITI-GEN/train_iti_gen.py`

Prerequisite training script (from the ITI-GEN repository). Trains ITI-GEN token embeddings for gender and skin tone debiasing using the prepared CCv2 data.

```

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

```

---

### `Setup.ipynb`

Documents and performs environment setup required for:
- Stable Diffusion / Diffusers
- ControlNet
- ITI-GEN dependencies
- PyTorch3D (optional, for Chamfer distance)
- Supporting packages