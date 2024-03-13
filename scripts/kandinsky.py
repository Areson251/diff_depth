import torch
import numpy as np
from diffusers import AutoPipelineForInpainting


class KandinskyModel():
    def __init__(self) -> None:
        self.pipe = AutoPipelineForInpainting.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder-inpaint", 
        # torch_dtype=torch.float16
        )
        # self.pipe.enable_model_cpu_offload()
        
        # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
        # self.pipe.enable_xformers_memory_efficient_attention()

    def diffusion_inpaint(self, image, mask, positive_prompt, negative_prompt, w_orig, h_orig):
        inpaint_image = self.pipe(
            num_inference_steps=30,
            prompt=positive_prompt, 
            negative_prompt=negative_prompt,
            image=image, 
            mask_image=mask, 
            guidance_scale=9.0,
            # strength=1.0,
            ).images[0]
        
        inpaint_image = inpaint_image.resize((w_orig, h_orig))
        return np.array(inpaint_image)



kandinsky = KandinskyModel()