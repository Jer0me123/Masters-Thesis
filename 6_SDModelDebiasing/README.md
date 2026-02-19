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