import os
import numpy as np
import torch
from PIL import Image
import gradio as gr
from .stable_diffusion import StableDiffusionModel
from .kandinsky import KandinskyModel

class GradioWindow():
    def __init__(self) -> None:
        self.path_to_orig_imgs = "images/orig_imgs"
        self.path_to_prompts = "images/prompts.txt"
        self.path_to_negative_prompts = "images/negative_prompts.txt"

        self.stable_diffusion = StableDiffusionModel()
        self.kandinsky = KandinskyModel()

        self.original_img = None
        self.masks = None

        self.main()

    def main(self):
        with gr.Blocks() as self.demo:
            self.input_img = gr.ImageEditor(
                type="pil",
                label="Input",
            )

            with gr.Row():
                self.im_out_1 = gr.Image(type="pil", label="original")
                self.im_out_2 = gr.Image(type="pil", label="mask")
                self.im_out_3 = gr.Image(type="pil", label="composite")

            with gr.Row():
                self.positive_prompt = gr.Textbox(label="Positive prompt")
                self.negative_prompt = gr.Textbox(label="Negative prompt")
                # self.iter_number = gr.Number(label="Steps")
                # self.guidance_scale = gr.Number(label="Guidance Scale")
                self.enter_prompt = gr.Button("Enter prompt")

            with gr.Row():
                self.kandinsky_image = gr.Image(label="Kandinsky")
                self.stable_diffusion_image = gr.Image(label="Stable Diffusion")

            # Connect the UI and logic
            self.input_img.change(
                self.get_mask, 
                outputs=[self.im_out_1, self.im_out_2, self.im_out_3], 
                inputs=self.input_img
            )

            # TODO: rewrite to cycle for each model
            self.enter_prompt.click(
                self.inpaint_image,
                inputs=[self.positive_prompt, self.negative_prompt],
                outputs=[self.kandinsky_image, self.stable_diffusion_image],
            )

     # Define the logic
    def get_mask(self, input_img):
        self.original_img = input_img["background"]
        mask = input_img["layers"][0]
        mask = np.array(Image.fromarray(np.uint8(mask)).convert("L"))
        self.masks = np.where(mask != 0, 255, 0)
        return [self.original_img, self.masks, input_img["composite"]]
    
    def prepare_input(self, image, mask):
        print(np.array(image).shape, np.array(mask).shape)
        image = Image.fromarray(np.uint8(image)).convert("RGB")
        w_orig, h_orig = image.size
        image = image.resize((512, 512))
        mask = Image.fromarray(np.uint8(mask)).resize((512, 512), Image.NEAREST)
        print(np.unique(np.array(mask)))
        print(np.array(image).shape, np.array(mask).shape)
        return image, mask, w_orig, h_orig

    def inpaint_image(self, positive_prompt, negative_prompt):
        image, mask, w_orig, h_orig = self.prepare_input(self.original_img, self.masks)

        # TODO: write common AutoPipelineForInpainting for all models
        self.kandinsky_image = self.kandinsky.diffusion_inpaint(
            image, mask, positive_prompt, negative_prompt, w_orig, h_orig
        )

        self.stable_diffusion_image = self.stable_diffusion.diffusion_inpaint(
            image, mask, positive_prompt, negative_prompt, w_orig, h_orig
        )
        return self.kandinsky_image, self.stable_diffusion_image
    

window = GradioWindow()
