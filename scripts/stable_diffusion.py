import torch
import numpy as np
from PIL import Image
from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline


class StableDiffusionModel():
    def __init__(self) -> None:
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            requires_safety_checker=False,
            safety_checker=None,
            variant='fp16',
            torch_dtype=torch.float32,
        )
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

    def diffusion_inpaint(self, image, mask, positive_prompt, negative_prompt, w_orig, h_orig):
        inpaint_image = self.pipe(
            num_inference_steps=50,
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            guidance_scale=12.5,
            strength=1.0
        ).images[0]

        # inpaint_image = inpaint_image
        inpaint_image = inpaint_image.resize((w_orig, h_orig))
        return np.array(inpaint_image)
    

stable_diffusion = StableDiffusionModel()