huggingface-cli login --token {your_token}
wandb online

CUDA_VISIBLE_DEVICES="0" python test.py \
  --dataset_name="jackyhate/text-to-image-2M" --train_split_only \
  --cache_dir="./cache" \
  --resolution=512 \
  --output_dir="WithProjection-WithShuffling-Finetuning-t2i_2m-512" \
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
  --with_noisy_sample_projection \
  --with_shuffling \
  --test_on_manually_drawn_samples