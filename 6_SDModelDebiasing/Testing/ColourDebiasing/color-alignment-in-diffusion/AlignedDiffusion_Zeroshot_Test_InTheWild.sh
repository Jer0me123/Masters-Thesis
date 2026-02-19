huggingface-cli login --token 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
wandb online

CUDA_VISIBLE_DEVICES="0" python test.py \
  --dataset_name="jackyhate/text-to-image-2M" --train_split_only \
  --cache_dir="./cache" \
  --resolution=512 \
  --output_dir="WithProjection-Zeroshot-t2i_2m-512" \
  --eval_batch_size=4 \
  --logger="wandb" \
  --resume_from_checkpoint latest \
  --prediction_type="adapted_noise" \
  --ddpm_num_steps=1000 \
  --ddpm_num_inference_steps=50 \
  --bdpm_beta_schedule="scaled_linear" \
  --beta_start=0.00085 \
  --beta_end=0.0120 \
  --model_type="UNet2DConditionModel" \
  --proportion_empty_prompts=0 \
  --projection_threshold=200 \
  --with_pred_sample_projection


set CUDA_VISIBLE_DEVICES=0

python test.py --dataset_name "jackyhate/text-to-image-2M" --train_split_only --cache_dir "D:\HuggingFace_Models" --resolution 512 --output_dir "WithProjection-Zeroshot-t2i_2m-512" --eval_batch_size 1 --prediction_type "adapted_noise" --ddpm_num_steps 1000 --ddpm_num_inference_steps 50 --bdpm_beta_schedule "scaled_linear" --beta_start 0.00085 --beta_end 0.0120 --model_type "UNet2DConditionModel" --proportion_empty_prompts 0 --projection_threshold 200 --with_pred_sample_projection

python test.py --dataset_name imagefolder --dataset_config_name local_color_dataset --train_split_only --resolution 512 --eval_batch_size 1 --model_type UNet2DConditionModel --with_pred_sample_projection --projection_threshold 200 --output_dir "WithProjection-Zeroshot-t2i_2m-512" --prediction_type "adapted_noise" --ddpm_num_steps 1000 --ddpm_num_inference_steps 50 --bdpm_beta_schedule "scaled_linear" --beta_start 0.00085 --beta_end 0.0120 --model_type "UNet2DConditionModel" --proportion_empty_prompts 0 --projection_threshold 200 --with_pred_sample_projection



python test.py --dataset_name imagefolder --train_split_only --dataset_config_name local_color_dataset --cache_dir "D:\HuggingFace_Models" --resolution 512 --eval_batch_size 4 --resume_from_checkpoint latest --prediction_type adapted_noise --ddpm_num_steps=1000 --ddpm_num_inference_steps=50 --bdpm_beta_schedule="scaled_linear" --beta_start=0.00085 --beta_end=0.0120 --model_type UNet2DConditionModel --proportion_empty_prompts 0 --projection_threshold=200 --with_pred_sample_projection --output_dir "WithProjection-Zeroshot-t2i_2m-512"   

# D:\conda_envs\ColourDebiasingEnv

# conda activate ColourDebiasingEnv
# cd color-alignment-in-diffusion