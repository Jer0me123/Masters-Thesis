@echo off
setlocal EnableDelayedExpansion

REM ============================================================
REM Environment
REM ============================================================
set PYTHONUTF8=1
set "PYTHON=python"
set "SCRIPT=DatasetClassification_gradient_accum.py"

REM ============================================================
REM Define seed sets here
REM - Works for single seed: call :run_seed_set 0
REM - Works for multiple seeds: call :run_seed_set 0 1 2
REM ============================================================

@REM call :run_seed_set 0
call :run_seed_set 0 1 2
REM call :run_seed_set 0 1 2

echo.
echo ============================================================
echo ALL SEEDS COMPLETED
echo ============================================================
pause
exit /b


REM ============================================================
REM Subroutine: run one seed set
REM ============================================================
:run_seed_set
set "SEEDS=%*"

echo.
echo ============================================================
echo STARTING SEEDS: %SEEDS%
echo ============================================================

REM ============================================================
REM Helper macro: run experiment
REM ============================================================

@REM REM --- Base ---
@REM echo === Running Base | seeds %SEEDS% ===
@REM %PYTHON% %SCRIPT% ^
@REM   --splits_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\7_DatasetPreparation\UniversalSplits\DatasetClassification\splits_face_combined_stratified.json" ^
@REM   --batch_size 16 --early_stopping --patience 2 --bootstrap_ci ^
@REM   --output_dir "outputs_dataset_classification\Base\seeds_%SEEDS%" ^
@REM   --seeds %SEEDS%

@REM REM --- Pseudo1 (ISCO) ---
@REM echo === Running Pseudo1_ISCO | seeds %SEEDS% ===
@REM %PYTHON% %SCRIPT% ^
@REM   --splits_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\7_DatasetPreparation\UniversalSplits\PseudoDatasetClassification\ISCO_splits_face_combined_stratified.json" ^
@REM   --batch_size 16 --early_stopping --patience 2 --bootstrap_ci ^
@REM   --output_dir "outputs_dataset_classification\Pseudo1\seeds_%SEEDS%" ^
@REM   --seeds %SEEDS%

@REM REM --- Pseudo2 (SD) ---
@REM echo === Running Pseudo2_SD | seeds %SEEDS% ===
@REM %PYTHON% %SCRIPT% ^
@REM   --splits_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\7_DatasetPreparation\UniversalSplits\PseudoDatasetClassification\SD_splits_face_combined_stratified.json" ^
@REM   --batch_size 16 --early_stopping --patience 2 --bootstrap_ci ^
@REM   --output_dir "outputs_dataset_classification\Pseudo2\seeds_%SEEDS%" ^
@REM   --seeds %SEEDS%

@REM REM --- Depth ---
@REM %PYTHON% %SCRIPT% ^
@REM   --splits_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\7_DatasetPreparation\UniversalSplits\DatasetClassification\splits_face_combined_stratified_depth.json" ^
@REM   --batch_size 16 --early_stopping --patience 2 --bootstrap_ci ^
@REM   --output_dir "outputs_dataset_classification\Depth\seeds_%SEEDS%" ^
@REM   --seeds %SEEDS%

REM --- Edge ---
%PYTHON% %SCRIPT% ^
  --splits_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\7_DatasetPreparation\UniversalSplits\DatasetClassification\splits_face_combined_stratified_edge.json" ^
  --batch_size 16 --early_stopping --patience 2 --bootstrap_ci ^
  --output_dir "outputs_dataset_classification\Edge\seeds_%SEEDS%" ^
  --seeds %SEEDS%

REM --- HighFilter ---
%PYTHON% %SCRIPT% ^
  --splits_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\7_DatasetPreparation\UniversalSplits\DatasetClassification\splits_face_combined_stratified_highFilter.json" ^
  --batch_size 16 --early_stopping --patience 2 --bootstrap_ci ^
  --output_dir "outputs_dataset_classification\HighFilter\seeds_%SEEDS%" ^
  --seeds %SEEDS%

@REM REM --- LowFilter ---
@REM %PYTHON% %SCRIPT% ^
@REM   --splits_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\7_DatasetPreparation\UniversalSplits\DatasetClassification\splits_face_combined_stratified_lowFilter.json" ^
@REM   --batch_size 16 --early_stopping --patience 2 --bootstrap_ci ^
@REM   --output_dir "outputs_dataset_classification\LowFilter\seeds_%SEEDS%" ^
@REM   --seeds %SEEDS%

@REM REM --- MeanRGB ---
@REM %PYTHON% %SCRIPT% ^
@REM   --splits_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\7_DatasetPreparation\UniversalSplits\DatasetClassification\splits_face_combined_stratified_meanRGB.json" ^
@REM   --batch_size 16 --early_stopping --patience 2 --bootstrap_ci ^
@REM   --output_dir "outputs_dataset_classification\MeanRGB\seeds_%SEEDS%" ^
@REM   --seeds %SEEDS%

@REM REM --- ObjectDetection ---
@REM %PYTHON% %SCRIPT% ^
@REM   --splits_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\7_DatasetPreparation\UniversalSplits\DatasetClassification\splits_face_combined_stratified_objectDetection.json" ^
@REM   --batch_size 16 --early_stopping --patience 2 --bootstrap_ci ^
@REM   --output_dir "outputs_dataset_classification\ObjectDetection\seeds_%SEEDS%" ^
@REM   --seeds %SEEDS%

@REM REM --- PatchShufflePS2 ---
@REM %PYTHON% %SCRIPT% ^
@REM   --splits_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\7_DatasetPreparation\UniversalSplits\DatasetClassification\splits_face_combined_stratified_patchShufflePS2.json" ^
@REM   --batch_size 16 --early_stopping --patience 2 --bootstrap_ci ^
@REM   --output_dir "outputs_dataset_classification\PatchShufflePS2\seeds_%SEEDS%" ^
@REM   --seeds %SEEDS%

@REM REM --- PatchShufflePS4 ---
@REM %PYTHON% %SCRIPT% ^
@REM   --splits_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\7_DatasetPreparation\UniversalSplits\DatasetClassification\splits_face_combined_stratified_patchShufflePS4.json" ^
@REM   --batch_size 16 --early_stopping --patience 2 --bootstrap_ci ^
@REM   --output_dir "outputs_dataset_classification\PatchShufflePS4\seeds_%SEEDS%" ^
@REM   --seeds %SEEDS%

@REM REM --- PatchShufflePS8 ---
@REM %PYTHON% %SCRIPT% ^
@REM   --splits_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\7_DatasetPreparation\UniversalSplits\DatasetClassification\splits_face_combined_stratified_patchShufflePS8.json" ^
@REM   --batch_size 16 --early_stopping --patience 2 --bootstrap_ci ^
@REM   --output_dir "outputs_dataset_classification\PatchShufflePS8\seeds_%SEEDS%" ^
@REM   --seeds %SEEDS%

@REM REM --- PatchShufflePS16 ---
@REM %PYTHON% %SCRIPT% ^
@REM   --splits_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\7_DatasetPreparation\UniversalSplits\DatasetClassification\splits_face_combined_stratified_patchShufflePS16.json" ^
@REM   --batch_size 16 --early_stopping --patience 2 --bootstrap_ci ^
@REM   --output_dir "outputs_dataset_classification\PatchShufflePS16\seeds_%SEEDS%" ^
@REM   --seeds %SEEDS%

@REM REM --- PixelShuffle ---
@REM %PYTHON% %SCRIPT% ^
@REM   --splits_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\7_DatasetPreparation\UniversalSplits\DatasetClassification\splits_face_combined_stratified_pixelShuffle.json" ^
@REM   --batch_size 16 --early_stopping --patience 2 --bootstrap_ci ^
@REM   --output_dir "outputs_dataset_classification\PixelShuffle\seeds_%SEEDS%" ^
@REM   --seeds %SEEDS%

@REM REM --- SemanticSegmentation ---
@REM %PYTHON% %SCRIPT% ^
@REM   --splits_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\7_DatasetPreparation\UniversalSplits\DatasetClassification\splits_face_combined_stratified_semanticSegmentation.json" ^
@REM   --batch_size 16 --early_stopping --patience 2 --bootstrap_ci ^
@REM   --output_dir "outputs_dataset_classification\SemanticSegmentation\seeds_%SEEDS%" ^
@REM   --seeds %SEEDS%

@REM REM --- VAE ---
@REM %PYTHON% %SCRIPT% ^
@REM   --splits_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\7_DatasetPreparation\UniversalSplits\DatasetClassification\splits_face_combined_stratified_vae.json" ^
@REM   --batch_size 16 --early_stopping --patience 2 --bootstrap_ci ^
@REM   --output_dir "outputs_dataset_classification\VAE\seeds_%SEEDS%" ^
@REM   --seeds %SEEDS%

exit /b
