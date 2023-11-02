from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image
import matplotlib.pyplot as plt

openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

image_file1 = "./data/rodin.jpg"
original_image1 = load_image(image_file1)
openpose_image1 = openpose(original_image1)

image_file2 = "./data/discobolos.jpg"
original_image2 = load_image(image_file2)
openpose_image2 = openpose(original_image2)


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


images = [original_image1, openpose_image1, original_image2, openpose_image2]
draw_image_grids(images, 2, 2)
