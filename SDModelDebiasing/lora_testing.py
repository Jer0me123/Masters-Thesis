# # """
# # LoRA Fine-tuning Script for Debiasing Stable Diffusion 1.5
# # This script fine-tunes SD 1.5 with a balanced dataset to debias professions
# # across gender and race attributes.
# # """

# # import torch
# # from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
# # from diffusers.loaders import AttnProcsLayers
# # from diffusers.models.attention_processor import LoRAAttnProcessor
# # from torch.utils.data import Dataset, DataLoader
# # from PIL import Image
# # import json
# # from pathlib import Path
# # from tqdm import tqdm

# # # =============================================================================
# # # STEP 1: Create Balanced Dataset
# # # =============================================================================

# # class BalancedProfessionDataset(Dataset):
# #     """
# #     Dataset structure:
# #     data/
# #       ├── train.json
# #       └── images/
# #             ├── image_0001.png
# #             ├── image_0002.png
# #             └── ...
    
# #     train.json format:
# #     [
# #       {"image": "images/image_0001.png", "caption": "a photo of a white male doctor"},
# #       {"image": "images/image_0002.png", "caption": "a photo of a white female doctor"},
# #       ...
# #     ]
# #     """
    
# #     def __init__(self, json_path, image_size=512):
# #         with open(json_path, 'r') as f:
# #             self.data = json.load(f)
# #         self.image_size = image_size
        
# #     def __len__(self):
# #         return len(self.data)
    
# #     def __getitem__(self, idx):
# #         item = self.data[idx]
# #         image = Image.open(item['image']).convert('RGB')
# #         image = image.resize((self.image_size, self.image_size))
# #         caption = item['caption']
        
# #         # Convert to tensor
# #         image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
# #         image = (image - 0.5) * 2  # Normalize to [-1, 1]
        
# #         return {"pixel_values": image, "caption": caption}


# # # =============================================================================
# # # STEP 2: Generate Balanced Training Data (Optional - Use SD itself)
# # # =============================================================================

# # def generate_balanced_dataset(
# #     professions=["doctor", "nurse", "teacher", "engineer", "lawyer"],
# #     demographics=["white male", "white female", "black male", "black female", 
# #                   "asian male", "asian female"],
# #     images_per_combination=20,  # 20 images per demo x profession
# #     output_dir="data/balanced_professions"
# # ):
# #     """
# #     Generate a balanced dataset using SD 1.5 itself.
# #     Total images: len(professions) * len(demographics) * images_per_combination
# #     Example: 5 professions * 6 demographics * 20 = 600 images
# #     """
    
# #     output_dir = Path(output_dir)
# #     output_dir.mkdir(parents=True, exist_ok=True)
# #     (output_dir / "images").mkdir(exist_ok=True)
    
# #     # Load base SD 1.5
# #     pipe = StableDiffusionPipeline.from_pretrained(
# #         "runwayml/stable-diffusion-v1-5",
# #         torch_dtype=torch.float16
# #     ).to("cuda")
    
# #     metadata = []
# #     img_idx = 0
    
# #     for profession in professions:
# #         for demographic in demographics:
# #             prompt = f"a professional photo of a {demographic} {profession}, high quality portrait"
            
# #             for i in range(images_per_combination):
# #                 # Generate image
# #                 image = pipe(
# #                     prompt,
# #                     num_inference_steps=50,
# #                     guidance_scale=7.5,
# #                     generator=torch.Generator("cuda").manual_seed(img_idx)
# #                 ).images[0]
                
# #                 # Save image
# #                 img_path = f"images/img_{img_idx:05d}.png"
# #                 image.save(output_dir / img_path)
                
# #                 # Add to metadata
# #                 metadata.append({
# #                     "image": img_path,
# #                     "caption": f"a photo of a {demographic} {profession}",
# #                     "profession": profession,
# #                     "demographic": demographic
# #                 })
                
# #                 img_idx += 1
                
# #             print(f"Generated {images_per_combination} images for {demographic} {profession}")
    
# #     # Save metadata
# #     with open(output_dir / "train.json", 'w') as f:
# #         json.dump(metadata, f, indent=2)
    
# #     print(f"\nDataset created: {len(metadata)} total images")
# #     print(f"Saved to: {output_dir}")
# #     return output_dir


# # # =============================================================================
# # # STEP 3: Set up LoRA Training
# # # =============================================================================

# # def setup_lora_layers(unet, rank=4):
# #     """
# #     Add LoRA layers to the UNet cross-attention layers.
# #     rank: LoRA rank (4-8 typical, lower = faster/less capacity)
# #     """
# #     lora_attn_procs = {}
    
# #     for name in unet.attn_processors.keys():
# #         cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
# #         if name.startswith("mid_block"):
# #             hidden_size = unet.config.block_out_channels[-1]
# #         elif name.startswith("up_blocks"):
# #             block_id = int(name[len("up_blocks.")])
# #             hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
# #         elif name.startswith("down_blocks"):
# #             block_id = int(name[len("down_blocks.")])
# #             hidden_size = unet.config.block_out_channels[block_id]
        
# #         lora_attn_procs[name] = LoRAAttnProcessor(
# #             hidden_size=hidden_size,
# #             cross_attention_dim=cross_attention_dim,
# #             rank=rank
# #         )
    
# #     unet.set_attn_processor(lora_attn_procs)
# #     return lora_attn_procs


# # def train_lora(
# #     model_id="runwayml/stable-diffusion-v1-5",
# #     dataset_path="data/balanced_professions/train.json",
# #     output_dir="outputs/lora_debiased",
# #     lora_rank=4,
# #     learning_rate=1e-4,
# #     num_epochs=10,
# #     batch_size=4,
# #     gradient_accumulation_steps=4,
# #     mixed_precision="fp16"
# # ):
# #     """
# #     Train LoRA adapter for debiasing.
    
# #     Parameters:
# #     -----------
# #     lora_rank: 4-8 typical (higher = more capacity but slower)
# #     learning_rate: 1e-4 to 5e-4 typical
# #     num_epochs: 5-20 depending on dataset size
# #     batch_size: Adjust based on GPU memory
# #     gradient_accumulation_steps: Effective batch_size = batch_size * this
# #     """
    
# #     from accelerate import Accelerator
# #     from transformers import CLIPTextModel, CLIPTokenizer
# #     from diffusers import AutoencoderKL, UNet2DConditionModel
# #     from diffusers.optimization import get_scheduler
# #     import numpy as np
    
# #     # Initialize accelerator
# #     accelerator = Accelerator(
# #         gradient_accumulation_steps=gradient_accumulation_steps,
# #         mixed_precision=mixed_precision
# #     )
    
# #     # Load models
# #     print("Loading models...")
# #     tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
# #     text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
# #     vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
# #     unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    
# #     # Freeze base models
# #     vae.requires_grad_(False)
# #     text_encoder.requires_grad_(False)
# #     unet.requires_grad_(False)
    
# #     # Setup LoRA
# #     print(f"Setting up LoRA with rank={lora_rank}...")
# #     lora_attn_procs = setup_lora_layers(unet, rank=lora_rank)
    
# #     # Get trainable parameters (only LoRA)
# #     lora_layers = AttnProcsLayers(unet.attn_processors)
# #     trainable_params = lora_layers.parameters()
    
# #     print(f"Trainable LoRA parameters: {sum(p.numel() for p in trainable_params):,}")
    
# #     # Setup optimizer
# #     optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    
# #     # Load dataset
# #     print("Loading dataset...")
# #     dataset = BalancedProfessionDataset(dataset_path)
# #     train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
# #     # Learning rate scheduler
# #     lr_scheduler = get_scheduler(
# #         "cosine",
# #         optimizer=optimizer,
# #         num_warmup_steps=100,
# #         num_training_steps=len(train_dataloader) * num_epochs
# #     )
    
# #     # Prepare for training
# #     unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
# #         unet, optimizer, train_dataloader, lr_scheduler
# #     )
    
# #     # Training loop
# #     print(f"\nStarting training for {num_epochs} epochs...")
# #     global_step = 0
    
# #     for epoch in range(num_epochs):
# #         unet.train()
# #         progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
# #         for step, batch in enumerate(progress_bar):
# #             with accelerator.accumulate(unet):
# #                 # Encode images to latent space
# #                 latents = vae.encode(batch["pixel_values"].to(dtype=vae.dtype)).latent_dist.sample()
# #                 latents = latents * vae.config.scaling_factor
                
# #                 # Sample noise
# #                 noise = torch.randn_like(latents)
# #                 timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device)
                
# #                 # Add noise to latents
# #                 noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
# #                 # Get text embeddings
# #                 input_ids = tokenizer(
# #                     batch["caption"],
# #                     padding="max_length",
# #                     truncation=True,
# #                     max_length=tokenizer.model_max_length,
# #                     return_tensors="pt"
# #                 ).input_ids.to(latents.device)
                
# #                 encoder_hidden_states = text_encoder(input_ids)[0]
                
# #                 # Predict noise
# #                 noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
# #                 # Calculate loss
# #                 loss = torch.nn.functional.mse_loss(noise_pred, noise)
                
# #                 # Backprop
# #                 accelerator.backward(loss)
# #                 optimizer.step()
# #                 lr_scheduler.step()
# #                 optimizer.zero_grad()
                
# #                 global_step += 1
# #                 progress_bar.set_postfix({"loss": loss.item()})
        
# #         # Save checkpoint
# #         if accelerator.is_main_process:
# #             save_path = Path(output_dir) / f"checkpoint-{epoch+1}"
# #             save_path.mkdir(parents=True, exist_ok=True)
            
# #             # Save LoRA weights
# #             unet.save_attn_procs(save_path)
# #             print(f"Saved checkpoint to {save_path}")
    
# #     print("\nTraining complete!")
# #     return output_dir


# # # =============================================================================
# # # STEP 4: Use the Fine-tuned Model
# # # =============================================================================

# # def load_debiased_model(base_model_id, lora_weights_path):
# #     """Load SD 1.5 with debiasing LoRA applied."""
    
# #     pipe = StableDiffusionPipeline.from_pretrained(
# #         base_model_id,
# #         torch_dtype=torch.float16
# #     ).to("cuda")
    
# #     # Load LoRA weights
# #     pipe.unet.load_attn_procs(lora_weights_path)
    
# #     return pipe


# # def generate_debiased_images(prompt, num_images=6):
# #     """Generate images with debiased model."""
    
# #     pipe = load_debiased_model(
# #         "runwayml/stable-diffusion-v1-5",
# #         "outputs/lora_debiased/checkpoint-10"
# #     )
    
# #     images = []
# #     for i in range(num_images):
# #         image = pipe(
# #             prompt,
# #             num_inference_steps=50,
# #             guidance_scale=7.5,
# #             generator=torch.Generator("cuda").manual_seed(i)
# #         ).images[0]
# #         images.append(image)
    
# #     return images


# # # =============================================================================
# # # USAGE EXAMPLES
# # # =============================================================================

# # if __name__ == "__main__":
    
# #     # Example 1: Generate balanced training dataset
# #     print("Step 1: Generating balanced dataset...")
# #     dataset_dir = generate_balanced_dataset(
# #         professions=["doctor", "nurse", "teacher", "engineer", "lawyer"],
# #         demographics=["white male", "white female", "black male", "black female", "asian male", "asian female"],
# #         images_per_combination=1,#20,  # 600 total images
# #         output_dir="data/balanced_professions"
# #     )
    
# #     # Example 2: Train LoRA
# #     print("\nStep 2: Training LoRA...")
# #     train_lora(
# #         dataset_path="data/balanced_professions/train.json",
# #         output_dir="outputs/lora_debiased",
# #         lora_rank=4,  # Small rank = faster training
# #         learning_rate=1e-4,
# #         num_epochs=10,
# #         batch_size=4
# #     )
    
# #     # Example 3: Generate images with debiased model
# #     print("\nStep 3: Testing debiased model...")
# #     images = generate_debiased_images("a photo of a doctor", num_images=12)
    
# #     # Save results
# #     for i, img in enumerate(images):
# #         img.save(f"test_doctor_{i}.png")
    
# #     print("Done! Check test_doctor_*.png for results")

# """
# LoRA Fine-tuning Script for Debiasing Stable Diffusion 1.5
# This script fine-tunes SD 1.5 with a balanced dataset to debias professions
# across gender and race attributes using the modern PEFT integration.
# """

# import torch
# import numpy as np
# import json
# from pathlib import Path
# from tqdm import tqdm
# from PIL import Image

# from torch.utils.data import Dataset, DataLoader
# from diffusers import (
#     StableDiffusionPipeline, 
#     DDPMScheduler,
#     UNet2DConditionModel,
#     AutoencoderKL
# )
# from diffusers.optimization import get_scheduler
# from transformers import CLIPTextModel, CLIPTokenizer
# from accelerate import Accelerator
# from peft import LoraConfig

# # =============================================================================
# # STEP 1: Create Balanced Dataset Class
# # =============================================================================

# class BalancedProfessionDataset(Dataset):
#     def __init__(self, json_path, image_size=512):
#         with open(json_path, 'r') as f:
#             self.data = json.load(f)
#         self.image_size = image_size
        
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         item = self.data[idx]
#         image = Image.open(item['image']).convert('RGB')
#         image = image.resize((self.image_size, self.image_size))
#         caption = item['caption']
        
#         # Convert to tensor
#         image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
#         image = (image - 0.5) * 2  # Normalize to [-1, 1]
        
#         return {"pixel_values": image, "caption": caption}


# # =============================================================================
# # STEP 2: Generate Balanced Training Data
# # =============================================================================

# def generate_balanced_dataset(
#     professions=["doctor", "nurse", "teacher", "engineer", "lawyer"],
#     demographics=["white male", "white female", "black male", "black female", 
#                   "asian male", "asian female"],
#     images_per_combination=20,
#     output_dir="data/balanced_professions"
# ):
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#     (output_dir / "images").mkdir(exist_ok=True)
    
#     print(f"Generating dataset in {output_dir}...")
    
#     # Load base SD 1.5 for data generation
#     pipe = StableDiffusionPipeline.from_pretrained(
#         "runwayml/stable-diffusion-v1-5",
#         torch_dtype=torch.float16
#     ).to("cuda")
#     pipe.set_progress_bar_config(disable=True)
    
#     metadata = []
#     img_idx = 0
    
#     for profession in professions:
#         for demographic in demographics:
#             prompt = f"a professional photo of a {demographic} {profession}, high quality portrait"
            
#             # Batch generation for speed
#             for i in range(images_per_combination):
#                 image = pipe(
#                     prompt,
#                     num_inference_steps=30, # Reduced for speed
#                     guidance_scale=7.5,
#                 ).images[0]
                
#                 img_path = f"images/img_{img_idx:05d}.png"
#                 image.save(output_dir / img_path)
                
#                 metadata.append({
#                     "image": str(output_dir / img_path),
#                     "caption": f"a photo of a {demographic} {profession}",
#                     "profession": profession,
#                     "demographic": demographic
#                 })
#                 img_idx += 1
            
#             print(f"Generated {images_per_combination} images for {demographic} {profession}")
    
#     # Save metadata
#     with open(output_dir / "train.json", 'w') as f:
#         json.dump(metadata, f, indent=2)
    
#     # Clear memory
#     del pipe
#     torch.cuda.empty_cache()
    
#     return output_dir


# # =============================================================================
# # STEP 3: Train LoRA (Fixed for PEFT)
# # =============================================================================

# # def train_lora(
# #     model_id="runwayml/stable-diffusion-v1-5",
# #     dataset_path="data/balanced_professions/train.json",
# #     output_dir="outputs/lora_debiased",
# #     lora_rank=4,
# #     learning_rate=1e-4,
# #     num_epochs=5,
# #     batch_size=1,
# #     gradient_accumulation_steps=4,
# #     mixed_precision="fp16"
# # ):
# #     # Initialize accelerator
# #     accelerator = Accelerator(
# #         gradient_accumulation_steps=gradient_accumulation_steps,
# #         mixed_precision=mixed_precision
# #     )
    
# #     # Load models
# #     print("Loading base models...")
# #     tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
# #     text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
# #     vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
# #     unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
# #     noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
# #     # Freeze base models
# #     vae.requires_grad_(False)
# #     text_encoder.requires_grad_(False)
# #     unet.requires_grad_(False)
    
# #     # --- FIX: Set up LoRA using PEFT ---
# #     print(f"Setting up LoRA with rank={lora_rank}...")
# #     lora_config = LoraConfig(
# #         r=lora_rank,
# #         lora_alpha=lora_rank,
# #         init_lora_weights="gaussian",
# #         target_modules=["to_k", "to_q", "to_v", "to_out.0"],
# #     )
# #     unet.add_adapter(lora_config)
    
# #     # Filter for trainable parameters
# #     trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
# #     print(f"Trainable LoRA parameters: {sum(p.numel() for p in trainable_params):,}")
    
# #     # Optimizer
# #     optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    
# #     # Dataset
# #     print("Loading dataset...")
# #     dataset = BalancedProfessionDataset(dataset_path)
# #     train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
# #     # Scheduler
# #     lr_scheduler = get_scheduler(
# #         "cosine",
# #         optimizer=optimizer,
# #         num_warmup_steps=50,
# #         num_training_steps=len(train_dataloader) * num_epochs
# #     )
    
# #     # Prepare with Accelerator
# #     unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
# #         unet, optimizer, train_dataloader, lr_scheduler
# #     )
    
# #     # Move static models to device
# #     weight_dtype = torch.float32
# #     if accelerator.mixed_precision == "fp16":
# #         weight_dtype = torch.float16
        
# #     text_encoder.to(accelerator.device, dtype=weight_dtype)
# #     vae.to(accelerator.device, dtype=weight_dtype)
    
# #     # Training Loop
# #     print(f"\nStarting training for {num_epochs} epochs...")
# #     global_step = 0
    
# #     for epoch in range(num_epochs):
# #         unet.train()
# #         progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
# #         for step, batch in enumerate(progress_bar):
# #             with accelerator.accumulate(unet):
# #                 # Convert images to latents
# #                 latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
# #                 latents = latents * vae.config.scaling_factor
                
# #                 # Sample noise
# #                 noise = torch.randn_like(latents)
# #                 timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
                
# #                 # Add noise
# #                 noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
# #                 # Get text embeddings
# #                 input_ids = tokenizer(
# #                     batch["caption"],
# #                     padding="max_length",
# #                     truncation=True,
# #                     max_length=tokenizer.model_max_length,
# #                     return_tensors="pt"
# #                 ).input_ids.to(latents.device)
                
# #                 encoder_hidden_states = text_encoder(input_ids)[0]
                
# #                 # Predict noise
# #                 noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
# #                 # Loss
# #                 loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="mean")
                
# #                 # Backprop
# #                 accelerator.backward(loss)
# #                 optimizer.step()
# #                 lr_scheduler.step()
# #                 optimizer.zero_grad()
                
# #                 global_step += 1
# #                 progress_bar.set_postfix({"loss": loss.item()})
        
# #         # Save checkpoint per epoch
# #         if accelerator.is_main_process:
# #             save_path = Path(output_dir) / f"checkpoint-{epoch+1}"
# #             save_path.mkdir(parents=True, exist_ok=True)
            
# #             # Unwrap and save compatible LoRA weights
# #             unwrapped_unet = accelerator.unwrap_model(unet)
# #             unwrapped_unet.save_attn_procs(save_path)
# #             print(f"Saved checkpoint to {save_path}")
            
# #     print("\nTraining complete!")
# #     return output_dir

# def train_lora(
#     model_id="runwayml/stable-diffusion-v1-5",
#     dataset_path="data/balanced_professions/train.json",
#     output_dir="outputs/lora_debiased",
#     lora_rank=4,
#     learning_rate=1e-4,
#     num_epochs=10,
#     batch_size=1,
#     gradient_accumulation_steps=4,
#     mixed_precision="fp16"
# ):
#     # Initialize accelerator
#     accelerator = Accelerator(
#         gradient_accumulation_steps=gradient_accumulation_steps,
#         mixed_precision=mixed_precision
#     )
    
#     # Load models
#     print("Loading base models...")
#     tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
#     text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
#     vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
#     unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
#     noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
#     # Freeze base models
#     vae.requires_grad_(False)
#     text_encoder.requires_grad_(False)
#     unet.requires_grad_(False)

#     # --- OPTIMIZATION 1: Enable Gradient Checkpointing ---
#     unet.enable_gradient_checkpointing()
    
#     # Set up LoRA
#     print(f"Setting up LoRA with rank={lora_rank}...")
#     lora_config = LoraConfig(
#         r=lora_rank,
#         lora_alpha=lora_rank,
#         init_lora_weights="gaussian",
#         target_modules=["to_k", "to_q", "to_v", "to_out.0"],
#     )
#     unet.add_adapter(lora_config)
    
#     # Filter for trainable parameters
#     trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
#     print(f"Trainable LoRA parameters: {sum(p.numel() for p in trainable_params):,}")
    
#     # Optimizer
#     optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    
#     # Dataset
#     print("Loading dataset...")
#     dataset = BalancedProfessionDataset(dataset_path)
#     train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
#     # Scheduler
#     lr_scheduler = get_scheduler(
#         "cosine",
#         optimizer=optimizer,
#         num_warmup_steps=50,
#         num_training_steps=len(train_dataloader) * num_epochs
#     )
    
#     # Prepare with Accelerator
#     # Note: We do NOT prepare VAE/Text Encoder as they are used for pre-processing only
#     unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
#         unet, optimizer, train_dataloader, lr_scheduler
#     )
    
#     weight_dtype = torch.float32
#     if accelerator.mixed_precision == "fp16":
#         weight_dtype = torch.float16

#     # Move VAE and Text Encoder to GPU temporarily
#     text_encoder.to("cuda", dtype=weight_dtype)
#     vae.to("cuda", dtype=weight_dtype)
    
#     # Training Loop
#     print(f"\nStarting training for {num_epochs} epochs...")
#     global_step = 0
    
#     for epoch in range(num_epochs):
#         unet.train()
#         progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
#         for step, batch in enumerate(progress_bar):
#             with accelerator.accumulate(unet):
#                 # 1. Get Latents (VAE)
#                 with torch.no_grad():
#                     # Move batch to GPU
#                     pixel_values = batch["pixel_values"].to("cuda", dtype=weight_dtype)
#                     latents = vae.encode(pixel_values).latent_dist.sample()
#                     latents = latents * vae.config.scaling_factor
#                     del pixel_values
                
#                 # 2. Get Text Embeddings
#                 with torch.no_grad():
#                     input_ids = tokenizer(
#                         batch["caption"],
#                         padding="max_length",
#                         truncation=True,
#                         max_length=tokenizer.model_max_length,
#                         return_tensors="pt"
#                     ).input_ids.to("cuda")
                    
#                     encoder_hidden_states = text_encoder(input_ids)[0]
#                     del input_ids

#                 # 3. Sample Noise
#                 noise = torch.randn_like(latents)
#                 timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
#                 noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
#                 # 4. UNet Forward Pass
#                 # Ensure latents requires_grad is enabled for Gradient Checkpointing compatibility
#                 noisy_latents.requires_grad_(True)
                
#                 noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
#                 # --- FIX: Cast to float32 for Loss Calculation ---
#                 loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                
#                 accelerator.backward(loss)
#                 optimizer.step()
#                 lr_scheduler.step()
#                 optimizer.zero_grad()
                
#                 global_step += 1
#                 progress_bar.set_postfix({"loss": loss.item()})
                
#                 # Cleanup
#                 del latents, noisy_latents, noise, encoder_hidden_states, noise_pred, loss
        
#         if accelerator.is_main_process:
#             save_path = Path(output_dir) / f"checkpoint-{epoch+1}"
#             save_path.mkdir(parents=True, exist_ok=True)
#             unwrapped_unet = accelerator.unwrap_model(unet)
#             unwrapped_unet.save_attn_procs(save_path)
#             print(f"Saved checkpoint to {save_path}")
            
#     print("\nTraining complete!")
    
#     # Cleanup for inference
#     del unet, vae, text_encoder, optimizer
#     torch.cuda.empty_cache()
    
#     return output_dir

# # =============================================================================
# # STEP 4: Inference with Debiased Model
# # =============================================================================

# def generate_debiased_images(prompt, lora_path, num_images=1):
#     print(f"Loading model with LoRA from {lora_path}...")
    
#     pipe = StableDiffusionPipeline.from_pretrained(
#         "runwayml/stable-diffusion-v1-5",
#         torch_dtype=torch.float16
#     ).to("cuda")
    
#     # --- FIX: Modern loading method ---
#     pipe.load_lora_weights(lora_path)
    
#     images = []
#     for i in range(num_images):
#         image = pipe(
#             prompt,
#             num_inference_steps=30,
#             guidance_scale=7.5
#         ).images[0]
#         images.append(image)
        
#     return images


# # =============================================================================
# # MAIN EXECUTION
# # =============================================================================

# if __name__ == "__main__":
    
#     # 1. Generate Data (Set images_per_combination=20 for real training)
#     print("--- Step 1: Data Generation ---")
#     # dataset_dir = generate_balanced_dataset(
#     #     images_per_combination=20, # Kept low for testing purposes
#     #     output_dir="data/balanced_professions"
#     # )
    
#     # 2. Train LoRA
#     print("\n--- Step 2: Training ---")
#     final_output = train_lora(
#         dataset_path="data/balanced_professions/train.json",
#         output_dir="outputs/lora_debiased",
#         lora_rank=4,
#         num_epochs=10,      # Kept low for testing
#         batch_size=4
#     )

#     # final_output = r"outputs/lora_debiased"

#     # 3. Test Inference
#     print("\n--- Step 3: Inference ---")
#     # Pointing to the last checkpoint saved
#     checkpoint_path = Path(final_output) / "checkpoint-10" 
    
#     test_images = generate_debiased_images(
#         prompt="a photo of a doctor",
#         lora_path=checkpoint_path,
#         num_images=2
#     )
    
#     for i, img in enumerate(test_images):
#         save_name = f"test_doctor_debiased_{i}.png"
#         img.save(save_name)
#         print(f"Saved {save_name}")



"""
Optimized LoRA Fine-tuning Script for 8GB VRAM
Combines modern PEFT integration with low-memory optimizations.
"""

import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Diffusers & Transformers
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model

# =============================================================================
# STEP 1: Dataset Class
# =============================================================================

class BalancedProfessionDataset(Dataset):
    def __init__(self, json_path, image_size=512):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.image_size = image_size
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image']).convert('RGB')
        image = image.resize((self.image_size, self.image_size))
        caption = item['caption']
        
        # Normalize to [-1, 1]
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image = (image - 0.5) * 2
        
        return {"pixel_values": image, "caption": caption}

# =============================================================================
# STEP 2: Data Generation
# =============================================================================

def generate_balanced_dataset(
    professions=["doctor"],
    demographics=["white male", "white female", "black male", "black female", "asian male", "asian female"],
    images_per_combination=20, # Reduced default for speed
    output_dir="data/balanced_doctors"
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    
    print("Loading SD 1.5 for data generation...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")
    
    metadata = []
    img_idx = 0
    
    for profession in professions:
        for demographic in demographics:
            print(f"Generating: {demographic} {profession}...")
            prompt = f"a professional photo of a {demographic} {profession}, high quality portrait"
            
            for _ in range(images_per_combination):
                image = pipe(prompt, num_inference_steps=30).images[0]
                
                img_name = f"img_{img_idx:05d}.png"
                img_path = output_dir / "images" / img_name
                image.save(img_path)
                
                metadata.append({
                    "image": str(img_path),
                    "caption": f"a photo of a {demographic} {profession}",
                    "profession": profession,
                    "demographic": demographic
                })
                img_idx += 1
                
    with open(output_dir / "train.json", 'w') as f:
        json.dump(metadata, f, indent=2)
        
    # Free memory
    del pipe
    torch.cuda.empty_cache()
    return output_dir

# =============================================================================
# STEP 3: Optimized Low-VRAM Training
# =============================================================================

def train_lora_optimized(
    model_id="runwayml/stable-diffusion-v1-5",
    dataset_path="data/balanced_doctors/train.json",
    output_dir="outputs/lora_debiased_optimized",
    lora_rank=4,
    num_epochs=10,
    batch_size=1
):
    print("\n" + "="*50)
    print("Starting Optimized LoRA Training (8GB VRAM Safe)")
    print("="*50)

    # 1. Load Components (CPU first to save memory)
    print("Loading models...")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    # We load these on CPU initially. We will move them to GPU only when needed.
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")

    # Freeze frozen models
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # 2. VRAM OPTIMIZATION: Gradient Checkpointing
    # This is the magic switch that makes training possible on 8GB
    print("Enabling gradient checkpointing...")
    unet.enable_gradient_checkpointing()

    # 3. Setup LoRA
    print(f"Injecting LoRA adapters (Rank {lora_rank})...")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Move UNet to GPU (It fits now because we haven't moved VAE/Text Encoder yet)
    unet.to("cuda")

    # 4. Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)
    
    # 5. Dataset
    dataset = BalancedProfessionDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 6. Training Loop
    unet.train()
    
    # We use a simplified scalar for mixed precision
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        progress_bar = tqdm(dataloader)
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # --- MEMORY HACK: Encode on GPU, then immediately offload ---
            
            # A. Encode Images (VAE)
            # Move VAE to GPU momentarily
            vae.to("cuda") 
            with torch.no_grad():
                pixel_values = batch["pixel_values"].to("cuda")
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
            # Move VAE back to CPU (or simply delete the reference from VRAM)
            vae.to("cpu") 
            del pixel_values
            torch.cuda.empty_cache()

            # B. Encode Text (CLIP)
            # Move Text Encoder to GPU momentarily
            text_encoder.to("cuda")
            with torch.no_grad():
                tokens = tokenizer(
                    batch["caption"], 
                    padding="max_length", 
                    max_length=tokenizer.model_max_length, 
                    truncation=True, 
                    return_tensors="pt"
                ).input_ids.to("cuda")
                encoder_hidden_states = text_encoder(tokens)[0]
            # Move Text Encoder back to CPU
            text_encoder.to("cpu")
            del tokens
            torch.cuda.empty_cache()

            # C. Train UNet (The only thing remaining on GPU)
            latents = latents.to("cuda")
            encoder_hidden_states = encoder_hidden_states.to("cuda")
            
            # Sample noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device="cuda").long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Ensure gradients can flow (required for checkpointing)
            noisy_latents.requires_grad_(True)

            # Mixed Precision Context
            with torch.cuda.amp.autocast():
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                # Ensure float32 for loss to avoid "Half" errors
                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            progress_bar.set_postfix({"loss": loss.item()})
            
            # Cleanup step variables
            del latents, noisy_latents, noise, encoder_hidden_states, noise_pred, loss
        
        # Save checkpoint
        save_path = Path(output_dir) / f"checkpoint-epoch-{epoch+1}"
        unet.save_pretrained(save_path)
        print(f"Saved checkpoint to {save_path}")

    print("Training Complete!")
    return output_dir

# =============================================================================
# STEP 4: Inference
# =============================================================================

def generate_test_images(lora_path, prompt="a photo of a doctor", num_images=4):
    print(f"Loading LoRA from {lora_path} for testing...")
    
    # Load Base
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")
    
    # Load Adapter
    pipe.load_lora_weights(lora_path)
    
    print("Generating images...")
    images = []
    for i in range(num_images):
        img = pipe(prompt, num_inference_steps=30).images[0]
        img.save(f"test_result_{i}.png")
        images.append(img)
    print("Test images saved.")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # 1. Generate Data (Enable this once, then you can comment it out)
    # dataset_dir = generate_balanced_dataset(images_per_combination=20)
    
    dataset_dir = r"data/balanced_professions"
    # 2. Train
    final_model = train_lora_optimized(
        dataset_path=f"{dataset_dir}/train.json",
        output_dir="outputs/lora_debiased",
        lora_rank=4,
        num_epochs=5,
        batch_size=2,
    )
    
    # 3. Test
    generate_test_images(
        lora_path=Path(final_model) / "checkpoint-epoch-5",
        prompt="a photo of a doctor"
    )