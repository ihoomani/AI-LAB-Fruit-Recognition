from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def read_fruit_image(file_name):
    """Read an image and convert it to RGB."""
    with Image.open(file_name) as img:
        return np.asarray(img.convert("RGB"))


def display_fruit_image(pixels):
    """Display the image."""
    plt.figure(figsize=(5, 4))
    plt.imshow(pixels)
    plt.title("Input Fruit Image")
    plt.axis("off")
    plt.show()


def get_color_summary(pixels):
    """Calculate average RGB values."""
    normalized = pixels / 255.0
    averages = np.mean(normalized, axis=(0, 1))
    return np.round(averages, 3)


def analyze_image(file_name):
    path = Path(file_name)

    if not path.exists() or not path.is_file():
        print("Error: Image file not found.")
        return

    try:
        pixels = read_fruit_image(path)
    except (OSError, ValueError) as error:
        print("Could not open the image:", error)
        return

    height, width, _ = pixels.shape
    colors = get_color_summary(pixels)

    display_fruit_image(pixels)

    print("Image Analysis Report")
    print("---------------------")
    print(f"Image dimensions: {width} x {height}")
    print(f"Average red value:   {colors[0]}")
    print(f"Average green value: {colors[1]}")
    print(f"Average blue value:  {colors[2]}")

    print("\nThese values describe the image colors.")
    print("They do not identify the fruit automatically.")


if __name__ == "__main__":
    file_name = input("Enter the path of a fruit image: ").strip()
    analyze_image(file_name)
