@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo Running Object Detection - Stable Diffusion
echo ==========================================

python ObjectDetection.py --image_dir "E:\ImageRetrieval\StableDiffusionGeneratedImages\valid" --output_dir "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\ObjectDetection_LabelRestricted" --model_path "yolov8x-oiv7.pt" --color_json "openImagesv7_color_map.json" --batch_size 16 --chunk_size 32 --imgsz 640 --conf_thresh 0.1 --iou_thresh 0.5 --exclude_dirs face_crops --resize 224 224 --operation normal white_background --label_remap "label_remap_restricted.json" --exclude_classes_file "exclude_classes.txt"

IF ERRORLEVEL 1 (
    echo ❌ First job failed. Stopping.
    exit /b 1
)

echo.
echo ==========================================
echo Running Object Detection - Professions
echo ==========================================

python ObjectDetection.py --image_dir "F:\ImageRetrieval\Professions_125k_ISCO_Aligned_1k_Subset" --output_dir "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\ObjectDetection_LabelRestricted" --model_path "yolov8x-oiv7.pt" --color_json "openImagesv7_color_map.json" --batch_size 16 --chunk_size 32 --imgsz 640 --conf_thresh 0.1 --iou_thresh 0.5 --exclude_dirs facemesh --resize 224 224 --operation normal white_background --label_remap "label_remap_restricted.json" --exclude_classes_file "exclude_classes.txt"

echo.
echo ✅ All jobs completed successfully.
pause