# CLIP Embedding Pipeline

## Overview

This directory contains the pipeline for:
1. Downloading Re-LAION-5B parquet shards
2. Downloading corresponding images
3. Deriving CLIP embeddings

## Files

### `hf_token.txt`
Contains:
- HuggingFace token used in Setup.ipynb.

### `Setup.ipynb`
Performs:
- Environment setup
- Downloads Re-LAION5B .parquet files

Output: 
- {OUTPUT_DIR}\RE-LAION-5B_Dataset\relaion2B-en-research-safe

### `Utilities.ipynb`
Contains:
- create_symlink() utility function

### `ImageDownloadAndEmbeddings.ipynb`
Performs:
- Image download from parquet
- CLIP embedding extraction

Outputs:
- Images → {OUTPUT_DIR}\{0000-0003}_images
- Embeddings → {OUTPUT_DIR}\all_images_openai_clip_vit_large_patch14
