import cv2
import numpy as np
import argparse

def quantize_image(imagew_path, k=5):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Reshape the image to a 2D array of pixels
    pixels = image.reshape((-1, 3))
    
    # Convert to float32, required by cv2.kmeans
    pixels = np.float32(pixels)
    
    # Define criteria and apply kmeans
    # Criteria: Stop after 100 iterations or if the epsilon (0.2) is reached
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert centers back to uint8
    centers = np.uint8(centers)
    
    # Map the labels to the corresponding cluster centers
    quantized_pixels = centers[labels.flatten()]
    
    # Reshape back to the original image shape
    quantized_image = quantized_pixels.reshape(image.shape)
    
    # Display the original and quantized images
    cv2.imshow('Original Image', image)
    cv2.imshow(f'Quantized Image (K={k})', quantized_image)
    
    print("Press any key in the image window to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    image_path = r"C:\Users\fra-fisso\Downloads\eldenring.jpg"
    k = 10
    
    quantize_image(image_path, k)
