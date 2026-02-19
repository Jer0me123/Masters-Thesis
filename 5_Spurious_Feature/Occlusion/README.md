This text file outlines the purpose of each .ipynb & .py file in this directory. 

Setup.ipynb - This file outlines the Environment setup
evaluation.py - This python script containes a variety of occlusion Models tested on inference speed & output quality to determine which Model is most applicable.
occlusionModelsHelper.py - This file contains various occlusion model classes used in the Occlusion.py file.
Occlusion.py - This python script creates the Full_NoBg, MaskSegm, MaskSegm_NoBg, MaskRect, MaskRect_NoBg occlusion images.

The code in the above.ipynb & .py has been re-teseted and confirmed to be working. 

Furthermore running the relevant notebooks produces the following: 

1. Full_NoBg MaskSegm MaskSegm_NoBg MaskRect MaskRect_NoBg images of the specified directories -> F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\Occlusion |F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\Occlusion (Coco is not used here since this requires images to 100% contain people)