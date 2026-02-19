This text file outlines the purpose of each .ipynb & .py file in this directory. 

Setup.ipynb - This file outlines the Environment setup & Dataset download
models/ - This directory includes all the .py files implementing the captioning models used:
    1. Blip Image Captioning
    2. Florence 2 Image Captioning
    3. Kosmos 2 Image Captioning
    4. Tiny Image Captioning
    5. Vit GPT 2 Image Captioning
ImageCaptioningTesting.ipynb - This notebook test 10 differing captioning models, to derive the best performing one in terms of speed and output quality
ImageCaptioning.py - This python script annotates the specified images using the blip-image-captioning-large model
mappings.txt - This file outlines word mappings used in caption generation to remap gender explicit wording 

The code in the above.ipynb & .py has been re-teseted and confirmed to be working. 

Furthermore running the relevant notebooks produces the following: 

1. Image captions of the specified directories -> F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\ImageCaptioning\captions.jsonl | F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\ImageCaptioning\captions.jsonl | F:\ImageRetrieval\SpuriousFeatureImages\Coco\ImageCaptioning\captions.jsonl