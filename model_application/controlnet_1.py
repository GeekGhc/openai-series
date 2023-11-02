import cv2
import numpy as np
import matplotlib.pyplot as plt
from diffusers.utils import load_image
from PIL import Image

image_file = "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
original_image = load_image(image_file)


# 根据设置的 low_threshold 和 high_threshold 对图片进行边缘检测
def get_canny_image(original_image, low_threshold=100, high_threshold=200):
    image = np.array(original_image)

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image


canny_image = get_canny_image(original_image)


def display_images(image1, image2):
    # Combine the images horizontally
    combined_image = Image.new('RGB', (image1.width + image2.width, max(image1.height, image2.height)))
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (image1.width, 0))
    # Display the combined image
    plt.imshow(combined_image)
    plt.axis('off')
    plt.show()


display_images(original_image, canny_image)

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch

# 加载sd-controlnet-canny：基于一系列的边缘检测图片和原始的 Stable Diffusion 训练出来的额外的模型
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
# 基础的 Stable Diffusion 1.5 的模型
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)

# 允许GPU显存不够时，把不用的模型从显存移到内存
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()  # 通过xformers加速模型推理

# 加载完pipline后开始绘图
prompt = ", best quality, extremely detailed"
prompt = [t + prompt for t in ["Audrey Hepburn", "Elizabeth Taylor", "Scarlett Johansson", "Taylor Swift"]]
generator = [torch.Generator(device="cpu").manual_seed(42) for i in range(len(prompt))]

output = pipe(
    prompt,
    canny_image,
    negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * 4,
    num_inference_steps=20,
    generator=generator,
)


# 呈现4张图片

def draw_image_grids(images, rows, cols):
    # Create a rows x cols grid for displaying the images
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for row in range(rows):
        for col in range(cols):
            axes[row, col].imshow(images[col + row * cols])
    for ax in axes.flatten():
        ax.axis('off')
    # Display the grid
    plt.show()


draw_image_grids(output.images, 2, 2)
