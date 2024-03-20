import os
import numpy as np
import torch
from PIL import Image
import gradio as gr
from .stable_diffusion import StableDiffusionModel
from .kandinsky import KandinskyModel
import logging

class GradioWindow():
    def __init__(self) -> None:
        self.path_to_orig_imgs = "images/orig_imgs"
        self.path_to_output_imgs = "images/output_imgs"
        self.path_to_prompts = "images/prompts.txt"
        self.path_to_negative_prompts = "images/negative_prompts.txt"

        self.original_img = None
        self.masks = None
        self.prompts = None

        self.folders = [self.path_to_orig_imgs, self.path_to_output_imgs]

        # self.stable_diffusion = StableDiffusionModel()
        self.kandinsky = KandinskyModel()

        self.check_folders()
        self.read_prompts()
        self.main()

    def check_folders(self):
        for path in self.folders:
            if not os.path.exists(path):
                os.makedirs(path)

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
                # self.positive_prompt = gr.Textbox(label="Positive prompt")
                # self.negative_prompt = gr.Textbox(label="Negative prompt")
                self.iter_number = gr.Number(value=20, label="Steps")
                self.guidance_scale = gr.Number(value=0.7, label="Guidance Scale")
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
                # inputs=[self.positive_prompt, self.negative_prompt],
                inputs=[self.iter_number, self.guidance_scale],
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

    def inpaint_image(self, iter_number, guidance_scale):
        image, mask, w_orig, h_orig = self.prepare_input(self.original_img, self.masks)

        for prompt in self.prompts:
            # try:
            #     # TODO: write common AutoPipelineForInpainting for all models
            #     # TODO: use some bib for logs
            #     self.stable_diffusion_image = self.stable_diffusion.diffusion_inpaint(
            #         image, mask, prompt, None, w_orig, h_orig, 
            #         iter_number, guidance_scale,
            #     )
            #     self.save_img(self.stable_diffusion_image, "SDXL_"+prompt)
            # except Exception as error:
            #     print("ERROR WITH GENERATING IMAGE VIA SDXL: ", error)

            try:
                self.kandinsky_image = self.kandinsky.diffusion_inpaint(
                    image, mask, prompt, None, w_orig, h_orig, 
                    iter_number, guidance_scale,
                )
                self.save_img(self.kandinsky_image, "KAND_"+prompt)
            except Exception as error:
                print("ERROR WITH GENERATING IMAGE VIA KANDINSKY: ", error)

        return self.kandinsky_image, self.stable_diffusion_image
    
    def read_prompts(self):
        with open(self.path_to_prompts, 'r') as file:
            self.prompts = [line.rstrip() for line in file]

    def save_img(self, img, prompt):
        im = Image.fromarray(img)
        im.save(os.path.join(self.path_to_output_imgs, prompt) + ".png")
        print("SAVED: ", prompt)
    

window = GradioWindow()
