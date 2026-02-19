huggingface-cli login --token {your_token}
wandb online

accelerate launch --multi_gpu --main_process_port=29505 train.py \
  --dataset_name="jackyhate/text-to-image-2M" --train_split_only \
  --cache_dir="./cache" \
  --resolution=512 \
  --output_dir="WithProjection-WithShuffling-Finetuning-t2i_2m-512" \
  --train_batch_size=4 \
  --eval_batch_size=4 \
  --num_epochs=100 \
  --learning_rate=1e-5 \
  --lr_warmup_steps=200 \
  --use_8bit_adam \
  --use_ema \
  --logger="wandb" \
  --save_images_steps=1600 \
  --save_fully_diffused_images_steps=1600 \
  --checkpointing_steps=8001 \
  --resume_from_checkpoint latest \
  --prediction_type="adapted_noise" \
  --ddpm_beta_schedule="scaled_linear" \
  --ddpm_num_steps=1000 \
  --ddpm_num_inference_steps=50 \
  --beta_start=0.00085 \
  --beta_end=0.0120 \
  --model_type="UNet2DConditionModel" \
  --proportion_empty_prompts=0.1 \
  --with_shuffling \
  --with_projection \
  --projection_threshold=200