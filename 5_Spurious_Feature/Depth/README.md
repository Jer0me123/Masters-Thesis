This text file outlines the purpose of each .ipynb & .py file in this directory. 

Setup.ipynb - This file outlines the Environment setup
Depth.py - This python script creates depth images of the specified directory
DepthAtPosePoints.py - This python script derives the depth values according to detected poses for depth image of the specifieed directory

The code in the above.ipynb & .py has been re-teseted and confirmed to be working. 

Furthermore running the relevant notebooks produces the following: 

1. Depth images of the specified directories -> F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\Depth | F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\Depth | F:\ImageRetrieval\SpuriousFeatureImages\Coco\Depth
2. Depth values at the 17 COCO Skeleton points -> F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\DepthAtPose | F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\DepthAtPose