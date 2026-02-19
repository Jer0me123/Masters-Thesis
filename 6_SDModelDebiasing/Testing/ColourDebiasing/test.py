import torch
from PIL import Image
from diffusers import StableDiffusionAdapterPipeline, T2IAdapter
import numpy as np

# --- 1. Color Transfer Helper Function ---
def apply_color_profile(target_image, reference_image, blend=1.0):
    """
    Forces the target_image to match the Mean and Standard Deviation 
    (Average Color and Contrast) of the reference_image in RGB space.
    """
    target = np.array(target_image).astype("float32")
    ref = np.array(reference_image.resize(target_image.size)).astype("float32")

    mu_t = target.mean(axis=(0, 1)) 
    sig_t = target.std(axis=(0, 1)) 
    mu_r = ref.mean(axis=(0, 1))    
    sig_r = ref.std(axis=(0, 1))    

    sig_t = np.maximum(sig_t, 1e-5)

    corrected = (target - mu_t) * (sig_r / sig_t) + mu_r
    corrected = np.clip(corrected, 0, 255).astype("uint8")
    
    if blend < 1.0:
        corrected = (corrected * blend) + (target * (1 - blend))
        corrected = corrected.astype("uint8")

    return Image.fromarray(corrected)

# --- 2. Main Generation Function ---
def generate_with_color_guidance(
    prompt: str,
    reference_image_path: str,
    output_path: str = "output.png",
    seed: int = 45  # <--- NEW PARAMETER
):
    # Load Adapter
    adapter_id = "TencentARC/t2iadapter_color_sd14v1"
    adapter = T2IAdapter.from_pretrained(adapter_id, torch_dtype=torch.float16)

    # Load Pipeline
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionAdapterPipeline.from_pretrained(
        model_id,
        adapter=adapter,
        torch_dtype=torch.float16
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)

    # Process Reference
    ref_image = Image.open(reference_image_path).convert("RGB")
    ref_image = ref_image.resize((512, 512))
    
    # Resize to 64x64 for better structure retention (as discussed previously)
    # or keep 16x16 if you prefer looser guidance.
    color_map = ref_image.resize((64, 64), resample=Image.Resampling.BILINEAR)
    color_map = color_map.resize((512, 512), resample=Image.Resampling.NEAREST)

    # --- NEW: SET THE SEED ---
    print(f"Generating with seed: {seed}")
    generator = torch.Generator(device=device).manual_seed(seed)
    # -------------------------

    print(f"Generating: '{prompt}'...")
    result = pipe(
        prompt=prompt,
        image=color_map, 
        adapter_conditioning_scale=1.0, 
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=generator  # <--- PASS THE GENERATOR HERE
    ).images[0]

    # Force Average Color Alignment
    print("Applying strict color alignment...")
    original_ref = Image.open(reference_image_path).convert("RGB")
    final_image = apply_color_profile(result, original_ref, blend=1.0)

    result.save(output_path)
    print(f"Saved to {output_path}")

# --- Example Usage ---
if __name__ == "__main__":
    try:
        generate_with_color_guidance(
            prompt="Image of a doctor",
            reference_image_path=r"C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\SDModelDebiasing\ColourDebiasing\color-alignment-in-diffusion\local_color_dataset\train\pastel.jpeg",
            seed=45 # You can change this number to get different (but reproducible) results
        )
    except FileNotFoundError:
        print("Error: Please provide a valid 'reference.jpg' or update the path.")