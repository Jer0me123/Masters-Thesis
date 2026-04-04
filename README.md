# Evaluating the Quality of the LAION-2B Dataset: Insights into Large-Scale Image Dataset Assessment

### Author:
Jerome Agius
### Supervisor:
Dr Dylan Seychell

## Abstract
<div align="justify">
Generative Artificial Intelligence (AI) systems have enabled the large-scale creation of images, with these technologies becoming increasingly accessible to the public. However, the development of these systems is largely dependent on the existence of massive web-scraped datasets serving as training data, such as Re-LAION-5B, which forms the focus of this research alongside the Stable Diffusion v1.5 model trained on it. Due to the nature of web-scraped training data, there has been increased concern regarding the ethical implications imposed, particularly related to bias and unintended correlations embedded within the training corpus. Specifically, non-semantic or spurious features present within the training dataset may serve as shortcuts for various downstream tasks, negatively impacting model behaviour and fairness outcomes. While these shortcuts can emerge for any task, demographic attributes represent a particularly concerning application, alongside other attributes unrelated to the inherent semantic content of the dataset.

This thesis investigates the demographic bias within the Re-LAION-5B dataset and resultant generative images produced by the Stable Diffusion v1.5 model, with a direct focus on identifying spurious features within these two components and determining whether the generative model amplifies, alters, or mitigates those spurious features identified within the respective training corpus. The study adopts a classifier-based approach derived from prior research, in which discriminative models are trained to distinguish between Re-LAION-5B and Stable Diffusion v1.5 generated images. These transformations systematically reduce the semantic information available to the model and force reliance on specific low-level image features, thereby exposing the presence of dataset specific spurious patterns that can be exploited by the model for image source identification.

Building upon this classifier based analysis, the relevant features used in the discrimination task are further annotated in terms of gender and skin-tone attributes. This secondary analysis investigates whether the previously identified dataset specific spurious features demonstrate correlations with these sensitive demographic attributes, allowing for the detection of harmful features that may serve as model shortcuts for deriving gender or skin-tone classifications. To further contextualise these findings and assess the real world implications of observed biases, the demographic distributions of retrieved and generated images across 200 occupation categories are compared against real world demographic statistics, revealing patterns of representation bias and potential amplification effects in the generative model.

Finally, based on the spurious features identified in the prior sections, this thesis implements and evaluates a combined set of inference-time interventions for debiasing the Stable Diffusion v1.5 model without requiring full model retraining. These interventions demonstrably reduce spurious demographic correlations across low-level, geometric, and semantic representations, confirming that inference-time debiasing can produce measurable and substantive departures from the demographic and spurious feature profile of standard generation. By explicitly linking dataset characteristics, spurious feature learning mechanisms, and generative bias outcomes, this work contributes both methodological insights for evaluating large-scale multimodal datasets and practical interventions for the development of fairer text-to-image generation systems.
</div>

## Repository Content

* [Thesis](../main/Masters_Thesis_Write_Up/) - This directory contains the LaTeX source files for the thesis.
* [1 — CLIP Embedding Pipeline](../main/1_CLIPEmbeddingPipeline/) - Downloads Re-LAION-5B parquet shards, retrieves images, and extracts CLIP (ViT-L/14) embeddings across four shards (~43M images total).
* [2 — Image Retrieval](../main/2_ImageRetrieval/) - FAISS-based CLIP similarity search over precomputed embeddings using 200 occupation prompts, with YOLO face/person filtering and MediaPipe FaceMesh segmentation.
* [3 — Image Generation](../main/3_ImageGeneration/) - Stable Diffusion v1.5 + LCM LoRA profession-conditioned image generation with face/person validation.
* [4 — Dataset Annotation](../main/4_DatasetAnnotation/) - Gender and skin tone annotation of Re-LAION-5B and SD-generated images using the Realistic Gender Classifier and a trained VGG16 MST regression model.
* [5 — Spurious Feature Pipeline](../main/5_SpuriousFeature/) - Generates 15 image transformation variants per dataset (depth, edge detection, high/low pass filtering, captioning, object detection, occlusion, pose estimation, semantic segmentation, pixel/patch shuffling, mean RGB, VAE reconstruction).
* [6 — Debiasing](../main/6_Debiasing/) - ITI-GEN + ControlNet inference-time debiasing using Chamfer colour alignment and Sliced Wasserstein guidance.
* [7 — Dataset Preparation](../main/7_DatasetPreparation/) - Constructs train/val/test splits for spurious feature probe experiments across gender, skin tone, and dataset classification tasks.
* [8 — Spurious Feature Analysis](../main/8_SpuriousFeatureAnalysis/) - Trains and evaluates spurious feature probes (ConvNeXt-Tiny, ResNet-50, Logistic Regression, MLP, Sentence-T5) and produces all demographic distribution figures and results.
* [Gender & Skin Tone Proxy](../main/GenderSkinToneProxy/) - Derives real-world gender demographic baselines from US BLS and ILOSTAT data, and establishes a race-to-MST skin tone mapping via FairFace annotation.

---

## Environment

Each subdirectory contains a `Setup.ipynb` documenting the specific dependencies required for that stage of the pipeline. Python 3.10.8 was used throughout. GPU training and inference was performed using CUDA 12.4.

<hr>
<p align="center">
  <img src="../main/University-of-Malta.png" alt="University of Malta" width="200"/>
</p>
