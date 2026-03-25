@echo off
setlocal enabledelayedexpansion

REM Enable Ctrl+C handling
if not defined _BATCH_SIGNAL_HANDLER (
  set _BATCH_SIGNAL_HANDLER=1
  %ComSpec% /C "%~f0" %*
  exit /b
)

echo ============================================
echo VGG16 FULL Experiment Suite
echo ============================================

call ..\..\.venv\Scripts\activate
chcp 65001
set PYTHONIOENCODING=utf-8
set TQDM_MINITERS=50

REM ============================================================
REM ================== LAB CLASSIFICATION ======================
REM ============================================================

@REM REM ------------------------------------------------------------
@REM REM 1️⃣ LAB 4-Class (Original Background)
@REM REM ------------------------------------------------------------

@REM set SAVE_DIR=F:\VGG_MST_Testing\Models\VGG16_4Classification_LAB
@REM if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"

@REM echo Running LAB 4-Class (Original BG)...

@REM python -u vgg16_mst_classification_regression_rgb_lab.py ^
@REM   --mode classification ^
@REM   --arch vgg16 ^
@REM   --input_space lab ^
@REM   --csv_path "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\annotations_filtered_MH_with_dark.csv" ^
@REM   --image_dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2" ^
@REM   --label_mapping "label_mapping_4class.json" ^
@REM   --use_ordinal_loss ^
@REM   --distance_weight 0.2 ^
@REM   --focal_gamma 0 ^
@REM   --label_smoothing 0 ^
@REM   --no_contrastive ^
@REM   --dropout 0.5 ^
@REM   --compute_lab_stats ^
@REM   --balance_strategy sampling ^
@REM   --max_samples_per_class 1500 ^
@REM   --batch_size 32 ^
@REM   --epochs 50 ^
@REM   --lr 1e-4 ^
@REM   --weight_decay 2e-4 ^
@REM   --save_dir "%SAVE_DIR%" ^
@REM   --save_split_path "%SAVE_DIR%\train_val_split.json" ^
@REM   --show_metrics_every 1 ^
@REM   --resume ^
@REM   --gpu 0 > "%SAVE_DIR%\train_log.txt" 2>&1

@REM if %errorlevel% neq 0 exit /b %errorlevel%

@REM REM ------------------------------------------------------------
@REM REM 2️⃣ LAB 4-Class (Fixed Background)
@REM REM ------------------------------------------------------------

@REM set SAVE_DIR=F:\VGG_MST_Testing\Models\VGG16_4Classification_LAB_FixedBG
@REM if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"

@REM echo Running LAB 4-Class (Fixed BG)...

@REM python -u vgg16_mst_classification_regression_rgb_lab.py ^
@REM   --mode classification ^
@REM   --arch vgg16 ^
@REM   --input_space lab ^
@REM   --csv_path "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\annotations_filtered_MH_with_dark.csv" ^
@REM   --image_dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2_BGFixed" ^
@REM   --label_mapping "label_mapping_4class.json" ^
@REM   --use_ordinal_loss ^
@REM   --distance_weight 0.2 ^
@REM   --focal_gamma 0 ^
@REM   --label_smoothing 0 ^
@REM   --no_contrastive ^
@REM   --dropout 0.5 ^
@REM   --compute_lab_stats ^
@REM   --balance_strategy sampling ^
@REM   --max_samples_per_class 1500 ^
@REM   --batch_size 32 ^
@REM   --epochs 50 ^
@REM   --lr 1e-4 ^
@REM   --weight_decay 2e-4 ^
@REM   --save_dir "%SAVE_DIR%" ^
@REM   --save_split_path "%SAVE_DIR%\train_val_split.json" ^
@REM   --show_metrics_every 1 ^
@REM   --resume ^
@REM   --gpu 0 > "%SAVE_DIR%\train_log.txt" 2>&1

@REM if %errorlevel% neq 0 exit /b %errorlevel%

@REM REM ------------------------------------------------------------
@REM REM 3️⃣ LAB 3-Class (Fixed Background)
@REM REM ------------------------------------------------------------

@REM set SAVE_DIR=F:\VGG_MST_Testing\Models\VGG16_3Classification_LAB_FixedBG
@REM if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"

@REM echo Running LAB 3-Class (Fixed BG)...

@REM python -u vgg16_mst_classification_regression_rgb_lab.py ^
@REM   --mode classification ^
@REM   --arch vgg16 ^
@REM   --input_space lab ^
@REM   --csv_path "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\annotations_filtered_MH_with_dark.csv" ^
@REM   --image_dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2_BGFixed" ^
@REM   --label_mapping "label_mapping_3class.json" ^
@REM   --use_ordinal_loss ^
@REM   --distance_weight 0.2 ^
@REM   --focal_gamma 0 ^
@REM   --label_smoothing 0 ^
@REM   --no_contrastive ^
@REM   --dropout 0.5 ^
@REM   --compute_lab_stats ^
@REM   --balance_strategy sampling ^
@REM   --max_samples_per_class 1500 ^
@REM   --batch_size 32 ^
@REM   --epochs 50 ^
@REM   --lr 1e-4 ^
@REM   --weight_decay 2e-4 ^
@REM   --save_dir "%SAVE_DIR%" ^
@REM   --save_split_path "%SAVE_DIR%\train_val_split.json" ^
@REM   --show_metrics_every 1 ^
@REM   --resume ^
@REM   --gpu 0 > "%SAVE_DIR%\train_log.txt" 2>&1

@REM if %errorlevel% neq 0 exit /b %errorlevel%

@REM REM ------------------------------------------------------------
@REM REM 3️⃣ LAB 3-Class (Fixed Background)
@REM REM ------------------------------------------------------------

@REM set SAVE_DIR=F:\VGG_MST_Testing\Models\VGG16_3Regression_LAB_FixedBG
@REM if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"

@REM echo Running Regression LAB 3-Class (Fixed BG)...

@REM "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\4_DatasetAnnotation\.venv\Scripts\python.exe" -u vgg16_mst_classification_regression_rgb_lab.py ^
@REM   --mode regression ^
@REM   --arch vgg16 ^
@REM   --input_space lab ^
@REM   --csv_path "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\annotations_filtered_MH_with_dark.csv" ^
@REM   --image_dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2_BGFixed" ^
@REM   --label_mapping "label_mapping_3class.json" ^
@REM   --use_ordinal_loss ^
@REM   --distance_weight 0.2 ^
@REM   --focal_gamma 0 ^
@REM   --label_smoothing 0 ^
@REM   --no_contrastive ^
@REM   --dropout 0.5 ^
@REM   --compute_lab_stats ^
@REM   --balance_strategy sampling ^
@REM   --max_samples_per_class 1500 ^
@REM   --batch_size 32 ^
@REM   --epochs 50 ^
@REM   --lr 1e-4 ^
@REM   --weight_decay 2e-4 ^
@REM   --save_dir "%SAVE_DIR%" ^
@REM   --save_split_path "%SAVE_DIR%\train_val_split.json" ^
@REM   --show_metrics_every 1 ^
@REM   --resume ^
@REM   --gpu 0 > "%SAVE_DIR%\train_log.txt" 2>&1

@REM if %errorlevel% neq 0 exit /b %errorlevel%

REM ------------------------------------------------------------
REM 3️⃣ LAB 3-Class (Normal Background)
REM ------------------------------------------------------------

set SAVE_DIR=F:\VGG_MST_Testing\Models\VGG16_3Classification_LAB
if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"

echo Running LAB 3-Class (Fixed BG)...

"C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\4_DatasetAnnotation\.venv\Scripts\python.exe" -u vgg16_mst_classification_regression_rgb_lab.py ^
  --mode classification ^
  --arch vgg16 ^
  --input_space lab ^
  --csv_path "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\annotations_filtered_MH_with_dark.csv" ^
  --image_dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2" ^
  --label_mapping "label_mapping_3class.json" ^
  --use_ordinal_loss ^
  --distance_weight 0.2 ^
  --focal_gamma 0 ^
  --label_smoothing 0 ^
  --no_contrastive ^
  --dropout 0.5 ^
  --compute_lab_stats ^
  --balance_strategy sampling ^
  --max_samples_per_class 1500 ^
  --batch_size 32 ^
  --epochs 50 ^
  --lr 1e-4 ^
  --weight_decay 2e-4 ^
  --save_dir "%SAVE_DIR%" ^
  --save_split_path "%SAVE_DIR%\train_val_split.json" ^
  --show_metrics_every 1 ^
  --resume ^
  --gpu 0 > "%SAVE_DIR%\train_log.txt" 2>&1

if %errorlevel% neq 0 exit /b %errorlevel%

REM ------------------------------------------------------------
REM 3️⃣ LAB 3-Class (Normal Background)
REM ------------------------------------------------------------

set SAVE_DIR=F:\VGG_MST_Testing\Models\VGG16_3Regression_LAB
if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"

echo Running Regression LAB 3-Class (Fixed BG)...

"C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\4_DatasetAnnotation\.venv\Scripts\python.exe" -u vgg16_mst_classification_regression_rgb_lab.py ^
  --mode regression ^
  --arch vgg16 ^
  --input_space lab ^
  --csv_path "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\annotations_filtered_MH_with_dark.csv" ^
  --image_dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2" ^
  --label_mapping "label_mapping_3class.json" ^
  --use_ordinal_loss ^
  --distance_weight 0.2 ^
  --focal_gamma 0 ^
  --label_smoothing 0 ^
  --no_contrastive ^
  --dropout 0.5 ^
  --compute_lab_stats ^
  --balance_strategy sampling ^
  --max_samples_per_class 1500 ^
  --batch_size 32 ^
  --epochs 50 ^
  --lr 1e-4 ^
  --weight_decay 2e-4 ^
  --save_dir "%SAVE_DIR%" ^
  --save_split_path "%SAVE_DIR%\train_val_split.json" ^
  --show_metrics_every 1 ^
  --resume ^
  --gpu 0 > "%SAVE_DIR%\train_log.txt" 2>&1

if %errorlevel% neq 0 exit /b %errorlevel%


@REM REM ------------------------------------------------------------
@REM REM 4️⃣ LAB 10-Class (Fixed Background)
@REM REM ------------------------------------------------------------

@REM set SAVE_DIR=F:\VGG_MST_Testing\Models\VGG16_10Classification_LAB_FixedBG
@REM if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"

@REM echo Running LAB 10-Class (Fixed BG)...

@REM python -u vgg16_mst_classification_regression_rgb_lab.py ^
@REM   --mode classification ^
@REM   --arch vgg16 ^
@REM   --input_space lab ^
@REM   --csv_path "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\annotations_filtered_MH_with_dark.csv" ^
@REM   --image_dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2_BGFixed" ^
@REM   --label_mapping "label_mapping_10class.json" ^
@REM   --use_ordinal_loss ^
@REM   --distance_weight 0.2 ^
@REM   --focal_gamma 0 ^
@REM   --label_smoothing 0 ^
@REM   --no_contrastive ^
@REM   --dropout 0.5 ^
@REM   --compute_lab_stats ^
@REM   --balance_strategy sampling ^
@REM   --max_samples_per_class 1500 ^
@REM   --batch_size 32 ^
@REM   --epochs 50 ^
@REM   --lr 1e-4 ^
@REM   --weight_decay 2e-4 ^
@REM   --save_dir "%SAVE_DIR%" ^
@REM   --save_split_path "%SAVE_DIR%\train_val_split.json" ^
@REM   --show_metrics_every 1 ^
@REM   --resume ^
@REM   --gpu 0 > "%SAVE_DIR%\train_log.txt" 2>&1

@REM if %errorlevel% neq 0 exit /b %errorlevel%

@REM REM ------------------------------------------------------------
@REM REM 4️⃣ LAB 10-Class (Original Background)
@REM REM ------------------------------------------------------------

@REM set SAVE_DIR=F:\VGG_MST_Testing\Models\VGG16_10Classification_LAB
@REM if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"

@REM echo Running LAB 10-Class (Original BG)...

@REM "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\4_DatasetAnnotation\.venv\Scripts\python.exe" -u vgg16_mst_classification_regression_rgb_lab.py ^
@REM   --mode classification ^
@REM   --arch vgg16 ^
@REM   --input_space lab ^
@REM   --csv_path "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\annotations_filtered_MH_with_dark.csv" ^
@REM   --image_dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2" ^
@REM   --label_mapping "label_mapping_10class.json" ^
@REM   --use_ordinal_loss ^
@REM   --distance_weight 0.2 ^
@REM   --focal_gamma 0 ^
@REM   --label_smoothing 0 ^
@REM   --no_contrastive ^
@REM   --dropout 0.5 ^
@REM   --compute_lab_stats ^
@REM   --balance_strategy sampling ^
@REM   --max_samples_per_class 1500 ^
@REM   --batch_size 32 ^
@REM   --epochs 50 ^
@REM   --lr 1e-4 ^
@REM   --weight_decay 2e-4 ^
@REM   --save_dir "%SAVE_DIR%" ^
@REM   --save_split_path "%SAVE_DIR%\train_val_split.json" ^
@REM   --show_metrics_every 1 ^
@REM   --resume ^
@REM   --gpu 0 > "%SAVE_DIR%\train_log.txt" 2>&1

@REM if %errorlevel% neq 0 exit /b %errorlevel%

@REM REM ============================================================
@REM REM ================== RGB CLASSIFICATION ======================
@REM REM ============================================================

@REM REM ------------------------------------------------------------
@REM REM 5️⃣ RGB 4-Class (Original Background)
@REM REM ------------------------------------------------------------

@REM set SAVE_DIR=F:\VGG_MST_Testing\Models\VGG16_4Classification_RGB
@REM if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"

@REM echo Running RGB 4-Class (Original BG)...

@REM python -u vgg16_mst_classification_regression_rgb_lab.py ^
@REM   --mode classification ^
@REM   --arch vgg16 ^
@REM   --input_space rgb ^
@REM   --csv_path "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\annotations_filtered_MH_with_dark.csv" ^
@REM   --image_dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2" ^
@REM   --label_mapping "label_mapping_4class.json" ^
@REM   --use_ordinal_loss ^
@REM   --distance_weight 0.2 ^
@REM   --focal_gamma 0 ^
@REM   --label_smoothing 0 ^
@REM   --no_contrastive ^
@REM   --dropout 0.5 ^
@REM   --compute_lab_stats ^
@REM   --balance_strategy sampling ^
@REM   --max_samples_per_class 1500 ^
@REM   --batch_size 32 ^
@REM   --epochs 50 ^
@REM   --lr 1e-4 ^
@REM   --weight_decay 2e-4 ^
@REM   --save_dir "%SAVE_DIR%" ^
@REM   --save_split_path "%SAVE_DIR%\train_val_split.json" ^
@REM   --show_metrics_every 1 ^
@REM   --resume ^
@REM   --gpu 0 > "%SAVE_DIR%\train_log.txt" 2>&1

@REM if %errorlevel% neq 0 exit /b %errorlevel%

@REM REM ------------------------------------------------------------
@REM REM 6️⃣ RGB 4-Class (Fixed Background)
@REM REM ------------------------------------------------------------

@REM set SAVE_DIR=F:\VGG_MST_Testing\Models\VGG16_4Classification_RGB_FixedBG
@REM if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"

@REM echo Running RGB 4-Class (Fixed BG)...

@REM python -u vgg16_mst_classification_regression_rgb_lab.py ^
@REM   --mode classification ^
@REM   --arch vgg16 ^
@REM   --input_space rgb ^
@REM   --csv_path "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\annotations_filtered_MH_with_dark.csv" ^
@REM   --image_dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2_BGFixed" ^
@REM   --label_mapping "label_mapping_4class.json" ^
@REM   --use_ordinal_loss ^
@REM   --distance_weight 0.2 ^
@REM   --focal_gamma 0 ^
@REM   --label_smoothing 0 ^
@REM   --no_contrastive ^
@REM   --dropout 0.5 ^
@REM   --compute_lab_stats ^
@REM   --balance_strategy sampling ^
@REM   --max_samples_per_class 1500 ^
@REM   --batch_size 32 ^
@REM   --epochs 50 ^
@REM   --lr 1e-4 ^
@REM   --weight_decay 2e-4 ^
@REM   --save_dir "%SAVE_DIR%" ^
@REM   --save_split_path "%SAVE_DIR%\train_val_split.json" ^
@REM   --show_metrics_every 1 ^
@REM   --resume ^
@REM   --gpu 0 > "%SAVE_DIR%\train_log.txt" 2>&1

@REM if %errorlevel% neq 0 exit /b %errorlevel%

@REM REM ------------------------------------------------------------
@REM REM 7️⃣ RGB 3-Class (Fixed Background)
@REM REM ------------------------------------------------------------

@REM set SAVE_DIR=F:\VGG_MST_Testing\Models\VGG16_3Classification_RGB_FixedBG
@REM if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"

@REM echo Running RGB 3-Class (Fixed BG)...

@REM python -u vgg16_mst_classification_regression_rgb_lab.py ^
@REM   --mode classification ^
@REM   --arch vgg16 ^
@REM   --input_space rgb ^
@REM   --csv_path "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\annotations_filtered_MH_with_dark.csv" ^
@REM   --image_dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2_BGFixed" ^
@REM   --label_mapping "label_mapping_3class.json" ^
@REM   --use_ordinal_loss ^
@REM   --distance_weight 0.2 ^
@REM   --focal_gamma 0 ^
@REM   --label_smoothing 0 ^
@REM   --no_contrastive ^
@REM   --dropout 0.5 ^
@REM   --compute_lab_stats ^
@REM   --balance_strategy sampling ^
@REM   --max_samples_per_class 1500 ^
@REM   --batch_size 32 ^
@REM   --epochs 50 ^
@REM   --lr 1e-4 ^
@REM   --weight_decay 2e-4 ^
@REM   --save_dir "%SAVE_DIR%" ^
@REM   --save_split_path "%SAVE_DIR%\train_val_split.json" ^
@REM   --show_metrics_every 1 ^
@REM   --resume ^
@REM   --gpu 0 > "%SAVE_DIR%\train_log.txt" 2>&1

@REM if %errorlevel% neq 0 exit /b %errorlevel%


@REM REM ------------------------------------------------------------
@REM REM 3️⃣ LAB 3-Class (Fixed Background)
@REM REM ------------------------------------------------------------

@REM set SAVE_DIR=F:\VGG_MST_Testing\Models\VGG16_3Regression_RGB_FixedBG
@REM if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"

@REM echo Running Regression RGB 3-Class (Fixed BG)...

@REM "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\4_DatasetAnnotation\.venv\Scripts\python.exe" -u vgg16_mst_classification_regression_rgb_lab.py ^
@REM   --mode regression ^
@REM   --arch vgg16 ^
@REM   --input_space rgb ^
@REM   --csv_path "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\annotations_filtered_MH_with_dark.csv" ^
@REM   --image_dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2_BGFixed" ^
@REM   --label_mapping "label_mapping_3class.json" ^
@REM   --use_ordinal_loss ^
@REM   --distance_weight 0.2 ^
@REM   --focal_gamma 0 ^
@REM   --label_smoothing 0 ^
@REM   --no_contrastive ^
@REM   --dropout 0.5 ^
@REM   --compute_lab_stats ^
@REM   --balance_strategy sampling ^
@REM   --max_samples_per_class 1500 ^
@REM   --batch_size 32 ^
@REM   --epochs 50 ^
@REM   --lr 1e-4 ^
@REM   --weight_decay 2e-4 ^
@REM   --save_dir "%SAVE_DIR%" ^
@REM   --save_split_path "%SAVE_DIR%\train_val_split.json" ^
@REM   --show_metrics_every 1 ^
@REM   --resume ^
@REM   --gpu 0 > "%SAVE_DIR%\train_log.txt" 2>&1

@REM if %errorlevel% neq 0 exit /b %errorlevel%

REM ------------------------------------------------------------
REM 7️⃣ RGB 3-Class (Fixed Background)
REM ------------------------------------------------------------

set SAVE_DIR=F:\VGG_MST_Testing\Models\VGG16_3Classification_RGB
if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"

echo Running RGB 3-Class (Fixed BG)...

"C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\4_DatasetAnnotation\.venv\Scripts\python.exe" -u vgg16_mst_classification_regression_rgb_lab.py ^
  --mode classification ^
  --arch vgg16 ^
  --input_space rgb ^
  --csv_path "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\annotations_filtered_MH_with_dark.csv" ^
  --image_dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2" ^
  --label_mapping "label_mapping_3class.json" ^
  --use_ordinal_loss ^
  --distance_weight 0.2 ^
  --focal_gamma 0 ^
  --label_smoothing 0 ^
  --no_contrastive ^
  --dropout 0.5 ^
  --compute_lab_stats ^
  --balance_strategy sampling ^
  --max_samples_per_class 1500 ^
  --batch_size 32 ^
  --epochs 50 ^
  --lr 1e-4 ^
  --weight_decay 2e-4 ^
  --save_dir "%SAVE_DIR%" ^
  --save_split_path "%SAVE_DIR%\train_val_split.json" ^
  --show_metrics_every 1 ^
  --resume ^
  --gpu 0 > "%SAVE_DIR%\train_log.txt" 2>&1

if %errorlevel% neq 0 exit /b %errorlevel%


REM ------------------------------------------------------------
REM 3️⃣ RGB 3-Class (Fixed Background)
REM ------------------------------------------------------------

set SAVE_DIR=F:\VGG_MST_Testing\Models\VGG16_3Regression_RGB
if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"

echo Running Regression RGB 3-Class (Fixed BG)...

"C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\4_DatasetAnnotation\.venv\Scripts\python.exe" -u vgg16_mst_classification_regression_rgb_lab.py ^
  --mode regression ^
  --arch vgg16 ^
  --input_space rgb ^
  --csv_path "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\annotations_filtered_MH_with_dark.csv" ^
  --image_dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2" ^
  --label_mapping "label_mapping_3class.json" ^
  --use_ordinal_loss ^
  --distance_weight 0.2 ^
  --focal_gamma 0 ^
  --label_smoothing 0 ^
  --no_contrastive ^
  --dropout 0.5 ^
  --compute_lab_stats ^
  --balance_strategy sampling ^
  --max_samples_per_class 1500 ^
  --batch_size 32 ^
  --epochs 50 ^
  --lr 1e-4 ^
  --weight_decay 2e-4 ^
  --save_dir "%SAVE_DIR%" ^
  --save_split_path "%SAVE_DIR%\train_val_split.json" ^
  --show_metrics_every 1 ^
  --resume ^
  --gpu 0 > "%SAVE_DIR%\train_log.txt" 2>&1

if %errorlevel% neq 0 exit /b %errorlevel%

@REM REM ------------------------------------------------------------
@REM REM 8️⃣ RGB 10-Class (Fixed Background)
@REM REM ------------------------------------------------------------

@REM set SAVE_DIR=F:\VGG_MST_Testing\Models\VGG16_10Classification_RGB_FixedBG
@REM if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"

@REM echo Running RGB 10-Class (Fixed BG)...

@REM python -u vgg16_mst_classification_regression_rgb_lab.py ^
@REM   --mode classification ^
@REM   --arch vgg16 ^
@REM   --input_space rgb ^
@REM   --csv_path "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\annotations_filtered_MH_with_dark.csv" ^
@REM   --image_dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2_BGFixed" ^
@REM   --label_mapping "label_mapping_10class.json" ^
@REM   --use_ordinal_loss ^
@REM   --distance_weight 0.2 ^
@REM   --focal_gamma 0 ^
@REM   --label_smoothing 0 ^
@REM   --no_contrastive ^
@REM   --dropout 0.5 ^
@REM   --compute_lab_stats ^
@REM   --balance_strategy sampling ^
@REM   --max_samples_per_class 1500 ^
@REM   --batch_size 32 ^
@REM   --epochs 50 ^
@REM   --lr 1e-4 ^
@REM   --weight_decay 2e-4 ^
@REM   --save_dir "%SAVE_DIR%" ^
@REM   --save_split_path "%SAVE_DIR%\train_val_split.json" ^
@REM   --show_metrics_every 1 ^
@REM   --resume ^
@REM   --gpu 0 > "%SAVE_DIR%\train_log.txt" 2>&1

@REM if %errorlevel% neq 0 exit /b %errorlevel%

@REM REM ------------------------------------------------------------
@REM REM 8️⃣ RGB 10-Class (Original Background)
@REM REM ------------------------------------------------------------

@REM set SAVE_DIR=F:\VGG_MST_Testing\Models\VGG16_10Classification_RGB
@REM if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"

@REM echo Running RGB 10-Class (Original BG)...

@REM "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\4_DatasetAnnotation\.venv\Scripts\python.exe" -u vgg16_mst_classification_regression_rgb_lab.py ^
@REM   --mode classification ^
@REM   --arch vgg16 ^
@REM   --input_space rgb ^
@REM   --csv_path "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\annotations_filtered_MH_with_dark.csv" ^
@REM   --image_dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2" ^
@REM   --label_mapping "label_mapping_10class.json" ^
@REM   --use_ordinal_loss ^
@REM   --distance_weight 0.2 ^
@REM   --focal_gamma 0 ^
@REM   --label_smoothing 0 ^
@REM   --no_contrastive ^
@REM   --dropout 0.5 ^
@REM   --compute_lab_stats ^
@REM   --balance_strategy sampling ^
@REM   --max_samples_per_class 1500 ^
@REM   --batch_size 32 ^
@REM   --epochs 50 ^
@REM   --lr 1e-4 ^
@REM   --weight_decay 2e-4 ^
@REM   --save_dir "%SAVE_DIR%" ^
@REM   --save_split_path "%SAVE_DIR%\train_val_split.json" ^
@REM   --show_metrics_every 1 ^
@REM   --resume ^
@REM   --gpu 0 > "%SAVE_DIR%\train_log.txt" 2>&1

@REM if %errorlevel% neq 0 exit /b %errorlevel%

REM ============================================================
REM ================== LAB REGRESSION ==========================
REM ============================================================

@REM REM ------------------------------------------------------------
@REM REM 9️⃣ LAB 10-Class Regression (Fixed Background)
@REM REM ------------------------------------------------------------

@REM set SAVE_DIR=F:\VGG_MST_Testing\Models\VGG16_10Regression_LAB_FixedBG
@REM if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"

@REM echo Running LAB 10-Class Regression (Fixed BG)...

@REM python -u vgg16_mst_classification_regression_rgb_lab.py ^
@REM   --mode regression ^
@REM   --arch vgg16 ^
@REM   --input_space lab ^
@REM   --csv_path "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\annotations_filtered_MH_with_dark.csv" ^
@REM   --image_dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2_BGFixed" ^
@REM   --label_mapping "label_mapping_10class.json" ^
@REM   --use_ordinal_loss ^
@REM   --distance_weight 0.2 ^
@REM   --focal_gamma 0 ^
@REM   --label_smoothing 0 ^
@REM   --no_contrastive ^
@REM   --dropout 0.5 ^
@REM   --compute_lab_stats ^
@REM   --balance_strategy sampling ^
@REM   --max_samples_per_class 1500 ^
@REM   --batch_size 32 ^
@REM   --epochs 50 ^
@REM   --lr 1e-4 ^
@REM   --weight_decay 2e-4 ^
@REM   --save_dir "%SAVE_DIR%" ^
@REM   --save_split_path "%SAVE_DIR%\train_val_split.json" ^
@REM   --show_metrics_every 1 ^
@REM   --resume ^
@REM   --gpu 0 > "%SAVE_DIR%\train_log.txt" 2>&1

@REM if %errorlevel% neq 0 exit /b %errorlevel%

@REM REM ------------------------------------------------------------
@REM REM 9️⃣ LAB 10-Class Regression (Fixed Background)
@REM REM ------------------------------------------------------------

@REM set SAVE_DIR=F:\VGG_MST_Testing\Models\VGG16_10Regression_LAB2
@REM if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"

@REM echo Running LAB 10-Class Regression (Fixed BG)...

@REM "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\4_DatasetAnnotation\.venv\Scripts\python.exe" -u vgg16_mst_classification_regression_rgb_lab.py ^
@REM   --mode regression ^
@REM   --arch vgg16 ^
@REM   --input_space lab ^
@REM   --csv_path "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\annotations_filtered_MH_with_dark.csv" ^
@REM   --image_dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2" ^
@REM   --label_mapping "label_mapping_10class.json" ^
@REM   --use_ordinal_loss ^
@REM   --distance_weight 0.2 ^
@REM   --focal_gamma 0 ^
@REM   --label_smoothing 0 ^
@REM   --no_contrastive ^
@REM   --dropout 0.5 ^
@REM   --compute_lab_stats ^
@REM   --balance_strategy sampling ^
@REM   --max_samples_per_class 1500 ^
@REM   --batch_size 32 ^
@REM   --epochs 50 ^
@REM   --lr 1e-4 ^
@REM   --weight_decay 2e-4 ^
@REM   --save_dir "%SAVE_DIR%" ^
@REM   --save_split_path "%SAVE_DIR%\train_val_split.json" ^
@REM   --show_metrics_every 1 ^
@REM   --resume ^
@REM   --gpu 0 > "%SAVE_DIR%\train_log.txt" 2>&1

@REM if %errorlevel% neq 0 exit /b %errorlevel%

REM ============================================================
REM ================== RGB REGRESSION ==========================
REM ============================================================

@REM REM ------------------------------------------------------------
@REM REM 🔟 RGB 10-Class Regression (Fixed Background)
@REM REM ------------------------------------------------------------

@REM set SAVE_DIR=F:\VGG_MST_Testing\Models\VGG16_10Regression_RGB_FixedBG
@REM if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"

@REM echo Running RGB 10-Class Regression (Fixed BG)...

@REM python -u vgg16_mst_classification_regression_rgb_lab.py ^
@REM   --mode regression ^
@REM   --arch vgg16 ^
@REM   --input_space rgb ^
@REM   --csv_path "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\annotations_filtered_MH_with_dark.csv" ^
@REM   --image_dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2_BGFixed" ^
@REM   --label_mapping "label_mapping_10class.json" ^
@REM   --use_ordinal_loss ^
@REM   --distance_weight 0.2 ^
@REM   --focal_gamma 0 ^
@REM   --label_smoothing 0 ^
@REM   --no_contrastive ^
@REM   --dropout 0.5 ^
@REM   --compute_lab_stats ^
@REM   --balance_strategy sampling ^
@REM   --max_samples_per_class 1500 ^
@REM   --batch_size 32 ^
@REM   --epochs 50 ^
@REM   --lr 1e-4 ^
@REM   --weight_decay 2e-4 ^
@REM   --save_dir "%SAVE_DIR%" ^
@REM   --save_split_path "%SAVE_DIR%\train_val_split.json" ^
@REM   --show_metrics_every 1 ^
@REM   --resume ^
@REM   --gpu 0 > "%SAVE_DIR%\train_log.txt" 2>&1

@REM if %errorlevel% neq 0 exit /b %errorlevel%


@REM REM ------------------------------------------------------------
@REM REM 🔟 RGB 10-Class Regression (Fixed Background)
@REM REM ------------------------------------------------------------

@REM set SAVE_DIR=F:\VGG_MST_Testing\Models\VGG16_10Regression_RGB
@REM if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"

@REM echo Running RGB 10-Class Regression (Fixed BG)...

@REM "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\4_DatasetAnnotation\.venv\Scripts\python.exe" -u vgg16_mst_classification_regression_rgb_lab.py ^
@REM   --mode regression ^
@REM   --arch vgg16 ^
@REM   --input_space rgb ^
@REM   --csv_path "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2\annotations_filtered_MH_with_dark.csv" ^
@REM   --image_dir "F:\Thesis\CasualConversationv2_Dataset\Segmented_CCV2" ^
@REM   --label_mapping "label_mapping_10class.json" ^
@REM   --use_ordinal_loss ^
@REM   --distance_weight 0.2 ^
@REM   --focal_gamma 0 ^
@REM   --label_smoothing 0 ^
@REM   --no_contrastive ^
@REM   --dropout 0.5 ^
@REM   --compute_lab_stats ^
@REM   --balance_strategy sampling ^
@REM   --max_samples_per_class 1500 ^
@REM   --batch_size 32 ^
@REM   --epochs 50 ^
@REM   --lr 1e-4 ^
@REM   --weight_decay 2e-4 ^
@REM   --save_dir "%SAVE_DIR%" ^
@REM   --save_split_path "%SAVE_DIR%\train_val_split.json" ^
@REM   --show_metrics_every 1 ^
@REM   --resume ^
@REM   --gpu 0 > "%SAVE_DIR%\train_log.txt" 2>&1

@REM if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ============================================
echo ALL RUNS COMPLETE
echo ============================================

pause