# Image Generation Pipeline

## Overview

This directory contains the pipeline for generating synthetic profession-conditioned images using Stable Diffusion v1.5.

The pipeline consists of:

1. Model comparison and selection across SD variants
2. Profession-conditioned image generation with face/person validation
3. Unconditioned image generation for dataset classification testing

The scripts and notebooks in this directory have been re-tested and verified to be functioning correctly.

---

## Files

### `ImageGenerationTesting.ipynb`

Compares multiple text-to-image models and configurations to identify the best balance of speed and quality.

Models evaluated:
- Stable Diffusion v1.5 (baseline)
- Stable Diffusion v1.5 + LCM LoRA
- Stable Diffusion v1.2
- SDXL / SDXL + LoRA / SDXL-Turbo
- SSD-1B / SSD-1B + LoRA
- PixArt-Alpha / PixArt-Alpha LCM
- Segmind Vega / Vega RT

Conclusion: SD v1.5 + LCM LoRA selected as the production model.

---

### `ImageGeneration.py`

Generates profession-conditioned images using SD v1.5 + LCM LoRA with post-generation validation.

Validation pipeline:
- YOLO person detection
- YOLO face detection (exactly one face required)
- MediaPipe FaceMesh segmentation
- Spatial consistency between face and person bounding boxes
- Aesthetic quality scoring via CLIP embeddings

Output:
E:\ImageRetrieval\StableDiffusionGeneratedImages\valid\{profession}\
E:\ImageRetrieval\StableDiffusionGeneratedImages\invalid\{profession}\
E:\ImageRetrieval\StableDiffusionGeneratedImages\ImageGenMetadata.csv

---

### `prompts.json`

Defines the profession list, prompt template, and negative prompt used by `ImageGeneration.py`.

Prompt template:
`full-body realistic photo of a {profession}, standing, in a professional setting appropriate for their occupation, sharp focus, 35mm lens, natural professional lighting`

---

### `Setup.ipynb`

Documents and performs environment setup required for:
- Stable Diffusion / Diffusers
- LCM LoRA weights
- Ultralytics YOLO
- MediaPipe
- OpenCLIP (aesthetic scoring)
- Supporting dependencies