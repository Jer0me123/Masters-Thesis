This text file outlines the purpose of each .ipynb & .py file in this directory. 

ImageGenerationTesting.ipynb - This notebook performs comparison across several models / models + speed LoRa trained on the Re-Laion5B datasets to identify the best overall model
ImageGeneration.py - This python script performs image generation using the best performing model derived in ImageGenerationTesting.ipynb and the prompts defined in prompts.json. Generated images are split into valid / invalid being filtered in accordance with fac & pose detection. 
ImageGeneration_NoValidation.py - This python script performs image generation using the best performing model derived in ImageGenerationTesting.ipynb and the prompts defined in prompts.json. This does not have the validation logic of the other script and is aimed at generating arbitrary images to use a test set for ensuring the dataset classifiaction models work on generic non-person depicting images. 
Setup.ipynb - This file outlines the Environment setup & Model Downloads

The code in the above.ipynb & .py has been re-teseted and confirmed to be working. 

Furthermore running the relevant notebooks produces the following: 

1. Generates synthetic images aligned with the provided prompts -> E:\ImageRetrieval\StableDiffusionGeneratedImages