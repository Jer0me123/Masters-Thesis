"""
Validation script to compare baseline SD 1.5 vs your LoRA model
and measure demographic balance quantitatively.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline

# =============================================================================
# STEP 1: Generate Comparison Images
# =============================================================================

def generate_comparison(
    baseline_model="runwayml/stable-diffusion-v1-5",
    lora_path="outputs/lora_debiased/checkpoint-epoch-5",
    prompt="a photo of a doctor",
    num_images=100,
    output_dir="validation_comparison"
):
    """
    Generate images from both baseline and LoRA models for comparison.
    """
    from peft import PeftModel
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    (output_dir / "baseline").mkdir(exist_ok=True)
    (output_dir / "lora").mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("COMPARISON: Baseline SD 1.5 vs LoRA-Debiased Model")
    print("="*60)
    
    # Load baseline model
    print("\n1. Loading baseline SD 1.5...")
    baseline_pipe = StableDiffusionPipeline.from_pretrained(
        baseline_model,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")
    
    # Generate baseline images
    print(f"\n2. Generating {num_images} baseline images...")
    for i in tqdm(range(num_images)):
        image = baseline_pipe(
            prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            generator=torch.Generator("cuda").manual_seed(i)
        ).images[0]
        image.save(output_dir / "baseline" / f"baseline_{i:03d}.png")
    
    del baseline_pipe
    torch.cuda.empty_cache()
    
    # Load LoRA model with PEFT
    print("\n3. Loading LoRA-debiased model...")
    lora_pipe = StableDiffusionPipeline.from_pretrained(
        baseline_model,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    
    # Load PEFT LoRA correctly
    print(f"Loading PEFT LoRA from {lora_path}...")
    lora_pipe.unet = PeftModel.from_pretrained(lora_pipe.unet, lora_path)
    lora_pipe = lora_pipe.to("cuda")
    
    # Generate LoRA images
    print(f"\n4. Generating {num_images} LoRA images...")
    for i in tqdm(range(num_images)):
        image = lora_pipe(
            prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            generator=torch.Generator("cuda").manual_seed(i)
        ).images[0]
        image.save(output_dir / "lora" / f"lora_{i:03d}.png")
    
    print(f"\n✓ Images saved to {output_dir}/")
    print(f"  - Baseline: {output_dir}/baseline/")
    print(f"  - LoRA: {output_dir}/lora/")
    
    return output_dir


# =============================================================================
# STEP 2: Visual Grid Comparison
# =============================================================================

def create_visual_grid(
    output_dir="validation_comparison",
    num_samples=12,
    save_path="comparison_grid.png"
):
    """
    Create a side-by-side grid comparing baseline vs LoRA.
    """
    output_dir = Path(output_dir)
    
    print("\nCreating visual comparison grid...")
    
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples*2, 4))
    
    for i in range(num_samples):
        # Baseline
        baseline_img = Image.open(output_dir / "baseline" / f"baseline_{i:03d}.png")
        axes[0, i].imshow(baseline_img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel("Baseline SD 1.5", fontsize=12, rotation=0, ha='right')
        
        # LoRA
        lora_img = Image.open(output_dir / "lora" / f"lora_{i:03d}.png")
        axes[1, i].imshow(lora_img)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel("LoRA Debiased", fontsize=12, rotation=0, ha='right')
    
    plt.suptitle("Baseline vs LoRA-Debiased: 'a photo of a doctor'", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison grid saved to {save_path}")
    
    plt.close()


# =============================================================================
# STEP 3: Manual Demographic Counting Template
# =============================================================================

def create_counting_template(output_dir="validation_comparison", num_images=100):
    """
    Create a CSV template for manual demographic counting.
    """
    import csv
    
    output_dir = Path(output_dir)
    
    # Create templates for both baseline and LoRA
    for model_type in ["baseline", "lora"]:
        csv_path = output_dir / f"{model_type}_demographics.csv"
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_id', 'gender', 'race', 'notes'])
            
            for i in range(num_images):
                writer.writerow([f"{model_type}_{i:03d}.png", '', '', ''])
        
        print(f"✓ Created counting template: {csv_path}")
    
    print("\nInstructions:")
    print("1. Open the CSV files in Excel/Google Sheets")
    print("2. For each image, fill in:")
    print("   - gender: male, female, unclear")
    print("   - race: white, black, asian, unclear")
    print("3. Save the CSV when done")
    print("4. Run analyze_manual_counts() to see results")


def analyze_manual_counts(output_dir="validation_comparison"):
    """
    Analyze the manually filled demographic counts.
    """
    import csv
    import pandas as pd
    
    output_dir = Path(output_dir)
    
    results = {}
    
    for model_type in ["baseline", "lora"]:
        csv_path = output_dir / f"{model_type}_demographics.csv"
        
        if not csv_path.exists():
            print(f"⚠ {csv_path} not found. Please fill it out first.")
            continue
        
        df = pd.read_csv(csv_path)
        
        # Count demographics
        gender_counts = df['gender'].value_counts()
        race_counts = df['race'].value_counts()
        
        # Combined counts
        df['demographic'] = df['race'] + ' ' + df['gender']
        combined_counts = df['demographic'].value_counts()
        
        results[model_type] = {
            'gender': gender_counts,
            'race': race_counts,
            'combined': combined_counts
        }
    
    # Print results
    print("\n" + "="*60)
    print("DEMOGRAPHIC DISTRIBUTION ANALYSIS")
    print("="*60)
    
    for model_type, counts in results.items():
        print(f"\n{model_type.upper()} MODEL:")
        print("-" * 40)
        
        print("\nGender Distribution:")
        for gender, count in counts['gender'].items():
            if gender and gender != 'unclear':
                percentage = (count / counts['gender'].sum()) * 100
                print(f"  {gender}: {count} ({percentage:.1f}%)")
        
        print("\nRace Distribution:")
        for race, count in counts['race'].items():
            if race and race != 'unclear':
                percentage = (count / counts['race'].sum()) * 100
                print(f"  {race}: {count} ({percentage:.1f}%)")
        
        print("\nCombined Demographics:")
        for demo, count in counts['combined'].items():
            if 'unclear' not in demo.lower():
                percentage = (count / counts['combined'].sum()) * 100
                print(f"  {demo}: {count} ({percentage:.1f}%)")
    
    # Calculate balance metrics
    print("\n" + "="*60)
    print("BALANCE METRICS")
    print("="*60)
    
    for model_type, counts in results.items():
        print(f"\n{model_type.upper()}:")
        
        # Gender balance (ideal: 50/50)
        if 'male' in counts['gender'] and 'female' in counts['gender']:
            male_pct = counts['gender']['male'] / counts['gender'].sum() * 100
            female_pct = counts['gender']['female'] / counts['gender'].sum() * 100
            gender_imbalance = abs(male_pct - 50)
            print(f"  Gender deviation from 50/50: {gender_imbalance:.1f}%")
        
        # Race balance (ideal: 33/33/33)
        race_target = 100 / 3  # 33.33%
        race_deviations = []
        for race in ['white', 'black', 'asian']:
            if race in counts['race']:
                pct = counts['race'][race] / counts['race'].sum() * 100
                deviation = abs(pct - race_target)
                race_deviations.append(deviation)
        if race_deviations:
            avg_race_deviation = np.mean(race_deviations)
            print(f"  Average race deviation from 33.33%: {avg_race_deviation:.1f}%")
    
    return results


# =============================================================================
# STEP 4: Quick Visual Check
# =============================================================================

def quick_visual_check(lora_path="outputs/lora_debiased/checkpoint-epoch-5"):
    """
    Generate a quick 4x4 grid to visually spot-check diversity.
    """
    from peft import PeftModel
    from diffusers import UNet2DConditionModel
    
    print("\nGenerating quick visual check (16 images)...")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None
    )
    
    # Load PEFT LoRA weights correctly
    print(f"Loading PEFT LoRA from {lora_path}...")
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    pipe = pipe.to("cuda")
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    
    for i in range(16):
        row = i // 4
        col = i % 4
        
        image = pipe(
            "a photo of a doctor",
            num_inference_steps=50,
            guidance_scale=7.5,
            generator=torch.Generator("cuda").manual_seed(i)
        ).images[0]
        
        axes[row, col].imshow(image)
        axes[row, col].axis('off')
        axes[row, col].set_title(f"Seed {i}", fontsize=10)
    
    plt.suptitle("LoRA Model: Doctor Diversity Check", fontsize=14)
    plt.tight_layout()
    plt.savefig("quick_diversity_check.png", dpi=150, bbox_inches='tight')
    print("✓ Saved quick_diversity_check.png")
    
    plt.show()


# =============================================================================
# STEP 5: Compare Multiple Checkpoints
# =============================================================================

def compare_epochs(
    base_path="outputs/lora_debiased",
    epochs=[1, 3, 5],
    prompt="a photo of a white male doctor",
    num_images=3
):
    """
    Compare different epoch checkpoints to see training progression.
    """
    from peft import PeftModel
    
    print("\n" + "="*60)
    print("COMPARING TRAINING EPOCHS")
    print("="*60)
    
    fig, axes = plt.subplots(len(epochs), num_images, figsize=(num_images*3, len(epochs)*3))
    
    for epoch_idx, epoch in enumerate(epochs):
        checkpoint = Path(base_path) / f"checkpoint-epoch-{epoch}"
        
        if not checkpoint.exists():
            print(f"⚠ Checkpoint {checkpoint} not found, skipping...")
            continue
        
        print(f"\nLoading epoch {epoch} checkpoint...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None
        )
        
        # Load PEFT LoRA correctly
        pipe.unet = PeftModel.from_pretrained(pipe.unet, checkpoint)
        pipe = pipe.to("cuda")
        
        for img_idx in range(num_images):
            image = pipe(
                prompt,
                num_inference_steps=50,
                guidance_scale=7.5,
                generator=torch.Generator("cuda").manual_seed(img_idx)
            ).images[0]
            
            axes[epoch_idx, img_idx].imshow(image)
            axes[epoch_idx, img_idx].axis('off')
            
            if img_idx == 0:
                axes[epoch_idx, img_idx].set_ylabel(f"Epoch {epoch}", fontsize=12)
        
        del pipe
        torch.cuda.empty_cache()
    
    plt.suptitle("Training Progress: Different Epochs", fontsize=14)
    plt.tight_layout()
    plt.savefig("epoch_comparison.png", dpi=150, bbox_inches='tight')
    print("\n✓ Saved epoch_comparison.png")
    plt.close()


# =============================================================================
# MAIN VALIDATION WORKFLOW
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("LoRA DEBIASING VALIDATION SUITE")
    print("="*60)
    
    # Step 1: Quick visual check (fastest)
    print("\n[1/5] Quick Visual Check (16 images)")
    quick_visual_check(lora_path="outputs/lora_debiased/checkpoint-epoch-5")
    
    # Step 2: Compare epochs (if you have multiple)
    print("\n[2/5] Epoch Comparison")
    compare_epochs(epochs=[1, 3, 5])
    
    # Step 3: Generate comparison dataset
    print("\n[3/5] Generating Baseline vs LoRA Comparison")
    response = input("Generate 100 images for comparison? (y/n): ")
    if response.lower() == 'y':
        output_dir = generate_comparison(
            lora_path="outputs/lora_debiased/checkpoint-epoch-5",
            num_images=100
        )
        create_visual_grid(output_dir=output_dir)
    
    # Step 4: Create counting template
    print("\n[4/5] Manual Counting Template")
    response = input("Create CSV template for manual demographic counting? (y/n): ")
    if response.lower() == 'y':
        create_counting_template()
    
    # Step 5: Analyze results (after manual counting)
    print("\n[5/5] Analyze Manual Counts")
    response = input("Analyze filled CSV files? (y/n): ")
    if response.lower() == 'y':
        analyze_manual_counts()
    
    print("\n" + "="*60)
    print("✓ VALIDATION COMPLETE")
    print("="*60)

# python train_iti_gen.py --prompt='a headshot of a person' --attr-list='Male,Skin_tone,Age' --epochs=30 --save-ckpt-per-epochs=10