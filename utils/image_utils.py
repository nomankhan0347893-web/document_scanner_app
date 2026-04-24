import cv2
import numpy as np
import os

# Utility function to load an image from a specified path
def load_image(image_path):

    # Loading the image using OpenCV

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    else:
        return image

# Utility function to convert an image to grayscale
def convert_to_grayscale(image):
    
    #color to grayscale reduce complexity and improve performance
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

# function to resize all images for faster processing while kepping aspect ratio
def resize_image(image, max_dim=800):
    height, width = image.shape[:2]
    if (height, width) > (max_dim, max_dim):
        scale = max_dim / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized=cv2.resize(image, (new_width, new_height))
        return resized,scale
    return image,1.0

if __name__ == "__main__":

    image_path = r"E:\desktop data\mentorship program devsil\document_scanner\dataset\input_images"
    output_path = r"E:\desktop data\mentorship program devsil\document_scanner\dataset\output_images"

    # 🔥 IMPORTANT FIX
    os.makedirs(output_path, exist_ok=True)

    for image_file in os.listdir(image_path):

        image = load_image(os.path.join(image_path, image_file))
        gray_image = convert_to_grayscale(image)
        scaled_image,scale_factor= resize_image(gray_image)

        gray_image = np.clip(scaled_image, 0, 255).astype("uint8")

        output_file = os.path.join(output_path, "gray_scale_" + image_file)

        success = cv2.imwrite(output_file, gray_image)
        print("Saved:", success, output_file)

        cv2.imshow("Original Image", image)
        cv2.imshow("Grayscale Image", gray_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()