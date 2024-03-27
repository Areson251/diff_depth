import torch
import numpy as np
from PIL import Image
from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline


class StableDiffusionModel():
    def __init__(self) -> None:
        self.device = torch.device("mps")
        print("DEVICE FOR SD: ", self.device)

        # self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
        #     "runwayml/stable-diffusion-v1-5",
        #     # "stabilityai/stable-diffusion-2-inpainting",
        #     requires_safety_checker=False,
        #     safety_checker=None,
        #     variant='fp16',
        #     torch_dtype=torch.float32,
        # # )
        # ).to(self.device)

        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        # )
        ).to(self.device)

        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

    def diffusion_inpaint(self, image, mask, 
                          positive_prompt, negative_prompt, 
                          w_orig, h_orig, 
                          iter_number, guidance_scale):
        
        inpaint_images = self.pipe(
            num_inference_steps=iter_number,
            prompt=positive_prompt,
            # negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            guidance_scale=guidance_scale,
            strength=1.0
        ).images

        print("GENERATED IMAGE COUNT: ", len(inpaint_images))
        inpaint_image = inpaint_images[0]

        inpaint_image = inpaint_image.resize((w_orig, h_orig))
        return np.array(inpaint_image)
    

# stable_diffusion = StableDiffusionModel()