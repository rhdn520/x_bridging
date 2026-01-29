from dotenv import load_dotenv
load_dotenv()
from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
import torch

# load model
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)
pipeline = pipeline.to("cuda")

# load local low-resolution image
# 🔽 change this path to your local file
image_path = "test.jpeg"
low_res_img = Image.open(image_path).convert("RGB")
low_res_img = low_res_img.resize((1280, 669))

prompt = "a website with a table data in it."

# upscale
upscaled_image = pipeline(
    prompt=prompt,
    image=low_res_img
).images[0]

# save result
upscaled_image.save("upsampled_web.png")