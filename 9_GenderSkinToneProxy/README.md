# Gender & Skin Tone Reference Demographics Pipeline

## Overview

This directory contains the pipeline for deriving real-world gender demographic baselines used as reference markers in the demographic analysis.

The pipeline consists of:

1. Profession-to-ISCO-08 mapping
2. ILOSTAT global gender demographic download and processing
3. US BLS gender demographic matching
4. FairFace-to-MST skin tone mapping via face segmentation and annotation

The scripts in this directory have been re-tested and verified to be functioning correctly.

---

## Directory Structure
```
GenderSkinToneProxy/
├── Global/               — ILOSTAT global gender demographics pipeline
├── MappingRaceToSkinTone/ — FairFace race-to-MST skin tone mapping pipeline
└── US/                   — US BLS gender demographics pipeline
```

---

## `Global/`

### `1_ProfessionToISCOMapping.py`

Maps the thesis profession list to ISCO-08 codes using the WageIndicator WISCO database. Downloads the WISCO Excel file automatically if not present. Applies hard-coded remappings for ambiguous titles (e.g. DJ → Disc Jockey).

---

### `2_Download_ILOSTAT_Data.py`

Downloads occupation-level employment by sex data from the ILOSTAT SDMX API at ISCO-08 two-digit granularity.

---

### `3_DeriveGlobalGenderDemographics.py`

Joins the ISCO-mapped profession list with ILOSTAT employment data to derive per-profession and per-ISCO-group gender splits. Supports `all`, `latest-per-country`, and `latest-global` year selection modes.

---

## `US/`

### `PracticalBLSDemographics.py`

Matches the thesis profession list against US Bureau of Labor Statistics CPS Table 11 occupation data using fuzzy matching to derive US-specific gender proportions per profession.

---

## `MappingRaceToSkinTone/`

This subdirectory derives a race-to-MST skin tone mapping using FairFace as a bridge dataset. FairFace provides race labels; annotating it with the VGG16 MST model produces an empirical distribution of MST scores per race group.

### `1_DownloadingFairFaceDataset.py`

Downloads the FairFace dataset from HuggingFace and saves images organised by race and gender.

---

### `2_SegmentFaces.py`

Applies MediaPipe FaceMesh segmentation to FairFace images, producing face crops organised by split, race, and gender.

---

### `3_AnnotateDatasetsUsingSkinToneModel.py`

Annotates segmented FairFace images using the trained VGG16 MST skin tone model to produce MST label predictions per image. CLI invocations are retained as reference.

---

### `4_DeriveDistribution.py`

Computes the distribution of MST labels per race group from the annotation JSONL, producing the race-to-MST mapping used in the thesis methodology.