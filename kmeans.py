import cv2
import numpy as np
import argparse

def quantize_colors_kmeans(colors, k=10, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs K-Means clustering on an array of colors (e.g. unique colors).
    Returns (labels, centers) where centers is uint8 shape (k, 3).
    """
    colors_f = np.asarray(colors, dtype=np.float32)
    # Using KMEANS_PP_CENTERS instead of random centers gives significantly better clusters
    _, labels, centers = cv2.kmeans(colors_f, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)  # type: ignore[arg-type]
    centers_arr = np.asarray(centers, dtype=np.uint8)
    labels_arr = np.asarray(labels, dtype=np.int32)
    return labels_arr, centers_arr

def quantize_colors_kmeans_lab(colors_bgr, k=10, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs K-Means++ clustering on colors in the perceptually uniform CIELAB color space.
    This guarantees that the clusters map to what human eyes actually perceive as distinct colors.
    Returns (labels, centers_bgr).
    """
    # Convert BGR flat pixels to LAB image array
    colors_bgr_img = np.asarray(colors_bgr, dtype=np.uint8).reshape(-1, 1, 3)
    colors_lab_img = cv2.cvtColor(colors_bgr_img, cv2.COLOR_BGR2LAB)
    
    colors_lab_f = np.asarray(colors_lab_img, dtype=np.float32).reshape(-1, 3)
    
    # Run K-Means++ in LAB space (PP_CENTERS is much better than RANDOM_CENTERS)
    _, labels, centers_lab = cv2.kmeans(colors_lab_f, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)  # type: ignore
    
    # Convert LAB centers back to BGR
    centers_lab_img = np.asarray(centers_lab, dtype=np.uint8).reshape(1, -1, 3)
    centers_bgr_img = cv2.cvtColor(centers_lab_img, cv2.COLOR_LAB2BGR)
    
    centers_bgr = np.asarray(centers_bgr_img.reshape(-1, 3), dtype=np.uint8)
    labels_arr = np.asarray(labels, dtype=np.int32).flatten()
    return labels_arr, centers_bgr


def quantize_image(image_path, k=5, use_color_set=False):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Reshape the image to a 2D array of pixels
    pixels = np.asarray(image.reshape((-1, 3)), dtype=np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    if use_color_set:
        unique_colors = np.unique(pixels, axis=0)
        _, centers = quantize_colors_kmeans(unique_colors, k=k, criteria=criteria)
        # Assign all pixels to their nearest cluster center in Euclidean color space
        dists = np.sum((pixels[:, np.newaxis, :] - centers[np.newaxis, :, :].astype(np.float32)) ** 2, axis=2)
        closest_indices = np.argmin(dists, axis=1)
        quantized_pixels = centers[closest_indices]
    else:
        labels, centers = quantize_colors_kmeans(pixels, k=k, criteria=criteria)
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
    from pathlib import Path
    image_path = str(Path(__file__).resolve().parent / "eldenring.jpg")
    k = 10
    
    quantize_image(image_path, k)
