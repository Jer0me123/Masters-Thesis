@echo off
setlocal

@REM DEPTH

"C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\5_Spurious_Feature\Depth\.venv\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\5_Spurious_Feature\Depth\Depth.py" --image_dir "F:\ImageRetrieval\TestingArbitrary10k\CocoUnlabelled_Random10k" --resize 224 224 --batch_size 16 --num_workers 8 --output_dir "F:\ImageRetrieval\TestingArbitrary10k\SpuriousFeatureImages\CocoUnlabelled_Random10k\Depth" --device cuda --model_size Small
if errorlevel 1 exit /b %errorlevel%

@REM EDGE

"C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\5_Spurious_Feature\EdgeDetection\.venv\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\5_Spurious_Feature\EdgeDetection\EdgeDetection.py" --image_dir "F:\ImageRetrieval\TestingArbitrary10k\CocoUnlabelled_Random10k" --edge_method canny --resize 224 224 --batch_size 16 --num_workers 8 --output_dir "F:\ImageRetrieval\TestingArbitrary10k\SpuriousFeatureImages\CocoUnlabelled_Random10k\EdgeDetection" --device cuda
if errorlevel 1 exit /b %errorlevel%

@REM HighPassFilter

"C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\5_Spurious_Feature\HighLowPassFiltering\.venv\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\5_Spurious_Feature\HighLowPassFiltering\HighLowPassFiltering.py" --image_dir "F:\ImageRetrieval\TestingArbitrary10k\CocoUnlabelled_Random10k" --resize 224 224 --batch_size 16 --num_workers 8 --output_dir "F:\ImageRetrieval\TestingArbitrary10k\SpuriousFeatureImages\CocoUnlabelled_Random10k\High&LowPassFilter" --radius 40 --do_low --do_high --filter_type ideal
if errorlevel 1 exit /b %errorlevel%

@REM Image Captioning

"C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\5_Spurious_Feature\ImageCaptioning\.venv\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\5_Spurious_Feature\ImageCaptioning\ImageCaptioning.py" --image_dir "F:\ImageRetrieval\TestingArbitrary10k\CocoUnlabelled_Random10k" --batch_size 8 --num_workers 8 --output_file "F:\ImageRetrieval\TestingArbitrary10k\SpuriousFeatureImages\CocoUnlabelled_Random10k\ImageCaptioning\captions.jsonl" --prompt "An image of" --max_length 30 --word_mapping_file "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\5_Spurious_Feature\ImageCaptioning\mappings.txt"
if errorlevel 1 exit /b %errorlevel%

@REM Object Detection

"C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\5_Spurious_Feature\ObjectDetection\.venv\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\5_Spurious_Feature\ObjectDetection\ObjectDetection.py" --image_dir "F:\ImageRetrieval\TestingArbitrary10k\CocoUnlabelled_Random10k" --output_dir "F:\ImageRetrieval\TestingArbitrary10k\SpuriousFeatureImages\CocoUnlabelled_Random10k\ObjectDetection" --model_path "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\5_Spurious_Feature\ObjectDetection\yolov8x-oiv7.pt" --color_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\5_Spurious_Feature\ObjectDetection\openImagesv7_color_map.json" --batch_size 16 --chunk_size 32 --imgsz 640 --conf_thresh 0.1 --iou_thresh 0.5 --resize 224 224 --operation normal white_background --label_remap "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\5_Spurious_Feature\ObjectDetection\label_remap.json" --exclude_classes_file "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\5_Spurious_Feature\ObjectDetection\exclude_classes.txt"
if errorlevel 1 exit /b %errorlevel%

"C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\5_Spurious_Feature\ObjectDetection\.venv\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\5_Spurious_Feature\ObjectDetection\ObjectDetection.py" --image_dir "F:\ImageRetrieval\TestingArbitrary10k\CocoUnlabelled_Random10k" --output_dir "F:\ImageRetrieval\TestingArbitrary10k\SpuriousFeatureImages\CocoUnlabelled_Random10k\ObjectDetection_LabelRestricted" --model_path "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\5_Spurious_Feature\ObjectDetection\yolov8x-oiv7.pt" --color_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\5_Spurious_Feature\ObjectDetection\openImagesv7_color_map.json" --batch_size 16 --chunk_size 32 --imgsz 640 --conf_thresh 0.1 --iou_thresh 0.5 --resize 224 224 --operation normal white_background --label_remap "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\5_Spurious_Feature\ObjectDetection\label_remap_restricted.json" --exclude_classes_file "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\5_Spurious_Feature\ObjectDetection\exclude_classes.txt"
if errorlevel 1 exit /b %errorlevel%

@REM Occlusion (For the sake of completion)

"C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\5_Spurious_Feature\Occlusion\.venv\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\5_Spurious_Feature\Occlusion\Occlusion.py" --model mask2former_coco --image_dir "F:\ImageRetrieval\TestingArbitrary10k\CocoUnlabelled_Random10k" --output_dir "F:\ImageRetrieval\TestingArbitrary10k\SpuriousFeatureImages\CocoUnlabelled_Random10k\Occlusion" --batch_size 4 --fixed_size 224 224 --resize 224 224 --operations Full_NoBg MaskSegm MaskSegm_NoBg MaskRect MaskRect_NoBg
if errorlevel 1 exit /b %errorlevel%

@REM Pose Estimation (Skipped These are mostly images of non-people)

@REM Semantic Segmentation

"C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\5_Spurious_Feature\SemanticSegmentation\.venv\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\5_Spurious_Feature\SemanticSegmentation\SemanticSegmentation.py" --image_dir "F:\ImageRetrieval\TestingArbitrary10k\CocoUnlabelled_Random10k" --output_dir "F:\ImageRetrieval\TestingArbitrary10k\SpuriousFeatureImages\CocoUnlabelled_Random10k\SemanticSegmentation" --resize 224 224 --batch_size 16 --num_workers 8 --fixed_size 512 512
if errorlevel 1 exit /b %errorlevel%

@REM Shuffling&Colour

"C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\5_Spurious_Feature\ShufflingAndColour\.venv\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\5_Spurious_Feature\ShufflingAndColour\PixelPatchShufflingMeanRGB.py" --image_dir "F:\ImageRetrieval\TestingArbitrary10k\CocoUnlabelled_Random10k" --output_dir "F:\ImageRetrieval\TestingArbitrary10k\SpuriousFeatureImages\CocoUnlabelled_Random10k\Shuffling&Colour" --resize 224 224 --batch_size 16 --num_workers 8 --patch_sizes 2 4 8 16 --do_pixel_shuffle --do_patch_shuffle --do_mean_rgb
if errorlevel 1 exit /b %errorlevel%

@REM VAE

cd "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\5_Spurious_Feature\VAE"

"C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\5_Spurious_Feature\VAE\.venv\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\5_Spurious_Feature\VAE\VAE.py" --image_dir "F:\ImageRetrieval\TestingArbitrary10k\CocoUnlabelled_Random10k" --output_dir "F:\ImageRetrieval\TestingArbitrary10k\SpuriousFeatureImages\CocoUnlabelled_Random10k\VAE" --resize 224 224 --batch_size 16
if errorlevel 1 exit /b %errorlevel%