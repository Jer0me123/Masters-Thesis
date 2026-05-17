## Overview

- Setup.ipynb - This file outlines the Environment setup
- exclude_classes / _restricted.txt - This file denotes a list of classes that should be ignored if detected in an image (ex: human arm)
- label_remap / _restricted.json - This file denotes a list of classes that if detected will be mapped to other class names, mainly used to avoid gender explicit classes (ex: "Man": "Person")
- object_detection_testing.ipynb - This notebook containes a variety of Object Detection Models tested on inference speed & accuracy to determine which Model is appropriate for captioning Million scale Images.
- ObjectDetection.py - This python script detects objects in images and annotates the image using bounding boxes, can output bounding boxes on the base image or entirely white image.