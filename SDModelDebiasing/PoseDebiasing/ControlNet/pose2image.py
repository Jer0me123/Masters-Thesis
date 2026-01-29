from share import *
import config

import os
import cv2
import einops
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.openpose import OpenposeDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


# -----------------------------
# Model initialization (unchanged)
# -----------------------------

apply_openpose = OpenposeDetector()

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_openpose.pth', location='cpu'))
# model = model.cuda()
model.eval()

ddim_sampler = DDIMSampler(model)
ddim_sampler.model = ddim_sampler.model.cuda()

# Disable aggressive VRAM swapping for speed
config.save_memory = True #False

# -----------------------------
# Core processing function
# -----------------------------

@torch.no_grad()
def process(
    input_image: np.ndarray,
    prompt: str,
    a_prompt: str = "best quality, extremely detailed",
    n_prompt: str = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
    num_samples: int = 1,
    image_resolution: int = 512,
    detect_resolution: int = 512,
    ddim_steps: int = 50,
    guess_mode: bool = False,
    strength: float = 1.0,
    scale: float = 9.0,
    seed: int = 45, #-1,
    eta: float = 0.0,
):
    input_image = HWC3(input_image)

    detected_map, _ = apply_openpose(
        resize_image(input_image, detect_resolution)
    )
    detected_map = HWC3(detected_map)

    img = resize_image(input_image, image_resolution)
    H, W, _ = img.shape

    detected_map = cv2.resize(
        detected_map, (W, H), interpolation=cv2.INTER_NEAREST
    )

    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, "b h w c -> b c h w").clone()

    if seed == -1:
        seed = random.randint(0, 65535)
    seed_everything(seed)

    c_cross = model.get_learned_conditioning(
        [prompt + ", " + a_prompt] * num_samples
    ).cuda()

    uc_cross = model.get_learned_conditioning(
        [n_prompt] * num_samples
    ).cuda()

    cond = {
        "c_concat": [control],
        "c_crossattn": [c_cross],
    }

    un_cond = {
        "c_concat": None if guess_mode else [control],
        "c_crossattn": [uc_cross],
    }

    shape = (4, H // 8, W // 8)

    model.control_scales = (
        [strength * (0.825 ** float(12 - i)) for i in range(13)]
        if guess_mode
        else [strength] * 13
    )

    print(
        "model:", next(model.parameters()).device,
        "control:", control.device,
        "cond:", cond["c_crossattn"][0].device,
    )

    samples, _ = ddim_sampler.sample(
        ddim_steps,
        num_samples,
        shape,
        cond,
        verbose=False,
        eta=eta,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=un_cond,
    )

    x_samples = model.decode_first_stage(samples)
    x_samples = (
        einops.rearrange(x_samples, "b c h w -> b h w c")
        * 127.5
        + 127.5
    ).cpu().numpy().clip(0, 255).astype(np.uint8)

    return detected_map, x_samples


# -----------------------------
# CLI / entry point
# -----------------------------

def main():
    input_path = r"C:\Users\User\OneDrive\Desktop\Austria\IMG_20220904_163400.jpg"          # <-- change
    output_dir = "outputs"
    prompt = "a coloured photo of a doctor"

    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(input_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pose_map, images = process(
        input_image=img,
        prompt=prompt,
    )

    cv2.imwrite(
        os.path.join(output_dir, "pose.png"),
        cv2.cvtColor(pose_map, cv2.COLOR_RGB2BGR),
    )

    for i, im in enumerate(images):
        cv2.imwrite(
            os.path.join(output_dir, f"sample_{i:02d}.png"),
            cv2.cvtColor(im, cv2.COLOR_RGB2BGR),
        )

    print(f"Saved {len(images)} images to {output_dir}")


if __name__ == "__main__":
    main()
