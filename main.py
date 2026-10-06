import sys
from pathlib import Path

# Ensure local workspace directory is in import search path
_WORKSPACE_ROOT = str(Path(__file__).resolve().parent)
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)

import cv2
import numpy as np
from estimate_pixels import scan_and_explore
from kmeans import quantize_colors_kmeans_lab


def create_palette_swatch(centers, swatch_size=48, padding=6, cols=8):
    """
    Creates a visual swatch image displaying the K palette colors.
    """
    k = len(centers)
    rows = int(np.ceil(k / float(cols)))
    
    total_w = cols * (swatch_size + padding) + padding
    total_h = rows * (swatch_size + padding) + padding
    
    swatch_img = np.full((total_h, total_w, 3), 35, dtype=np.uint8)
    
    for i, color in enumerate(centers):
        r = i // cols
        c = i % cols
        x1 = padding + c * (swatch_size + padding)
        y1 = padding + r * (swatch_size + padding)
        x2 = x1 + swatch_size
        y2 = y1 + swatch_size
        
        cv2.rectangle(swatch_img, (x1, y1), (x2, y2), [int(v) for v in color], -1)
        cv2.rectangle(swatch_img, (x1, y1), (x2, y2), (200, 200, 200), 1)
        
        # Color index text
        text = str(i + 1)
        b_val, g_val, r_val = float(color[0]), float(color[1]), float(color[2])
        luminance = 0.299 * r_val + 0.587 * g_val + 0.114 * b_val
        text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)
        cv2.putText(swatch_img, text, (x1 + 4, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1, cv2.LINE_AA)
        
    return swatch_img


def process_image(image_path, k=16, save_output=True, show_images=True):
    # 1. Read original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from '{image_path}'")
        return None

    h, w = image.shape[:2]
    print("=" * 80)
    print(f"Step 1: Detecting true pixel grid using estimate_pixels...")
    print(f"Original Resolution : {w}x{h}")
    print("=" * 80)

    # 2. Automatically detect true pixel grid dimensions using estimate_pixels
    best = scan_and_explore(image_path, show_images=False)
    if best is None:
        print("Failed to detect pixel grid.")
        return None

    grid_w, grid_h = best["dim"]
    scale = best["scale"]
    print(f"Detected True Grid  : {grid_w}x{grid_h} (Scale: {scale:.3f}, {1.0/scale:.2f}x)")

    # 3. Downscale original image to the true pixel grid
    # We MUST use INTER_AREA for downscaling. Nearest Neighbor throws away pixels and creates
    # horrible aliasing/noise when converting a high-res image into a pixel art grid!
    downscaled = cv2.resize(image, (grid_w, grid_h), interpolation=cv2.INTER_AREA)

    # 4. Prepare pixels for K-Means
    flat_pixels = downscaled.reshape((-1, 3))
    total_pixel_count = len(flat_pixels)
    
    print(f"\nStep 2: Preparing pixels for Perceptual K-Means++...")
    print(f"Total grid cells    : {total_pixel_count:,} (Clustering on the full pixel distribution)")

    # 5. Run Perceptual K-Means++ on the full pixel distribution
    print(f"\nStep 3: Running Perceptual CIELAB K-Means++ (K={k}) on the image pixels...")
    _, centers = quantize_colors_kmeans_lab(flat_pixels, k=k)
    centers = np.asarray(centers, dtype=np.uint8)

    # Sort palette colors by luminance for organized presentation
    r_chan = centers[:, 2].astype(np.float32)
    g_chan = centers[:, 1].astype(np.float32)
    b_chan = centers[:, 0].astype(np.float32)
    luminances = 0.299 * r_chan + 0.587 * g_chan + 0.114 * b_chan
    sorted_order = np.argsort(luminances)
    centers = centers[sorted_order]

    print(f"Discovered {k} Palette Colors (BGR):")
    for i, c in enumerate(centers):
        b, g, r = [int(v) for v in c]
        hex_code = f"#{r:02X}{g:02X}{b:02X}"
        print(f"  Color #{i+1:02d}: BGR=({b:3d}, {g:3d}, {r:3d}) | HEX={hex_code}")

    # 6. Map all pixels in the downscaled grid to their nearest discovered palette color
    print(f"\nStep 4: Mapping grid cells to discovered palette colors...")
    diffs = flat_pixels[:, np.newaxis, :].astype(np.float32) - centers[np.newaxis, :, :].astype(np.float32)
    squared_distances = np.sum(diffs ** 2, axis=2)
    nearest_palette_indices = np.argmin(squared_distances, axis=1)

    quantized_grid = centers[nearest_palette_indices].reshape((grid_h, grid_w, 3))

    # 8. Create palette swatch
    palette_swatch = create_palette_swatch(centers, swatch_size=42, padding=6, cols=min(k, 8))

    # Optional: Save outputs
    if save_output:
        cv2.imwrite("quantized_grid.png", quantized_grid)
        cv2.imwrite("palette_swatch.png", palette_swatch)
        print("\nSaved output files:")
        print(f"  - 'quantized_grid.png' ({grid_w}x{grid_h} native pixel art)")
        print(f"  - 'palette_swatch.png' (Color palette preview)")

    # 9. Display results
    if show_images:
        print("\nDisplaying windows (Press any key to close):")
        win_orig = "Original Image"
        win_quant = f"Quantized Pixel Art ({grid_w}x{grid_h}, K={k})"
        win_palette = f"Discovered Palette (K={k})"

        cv2.namedWindow(win_orig, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(win_quant, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(win_palette, cv2.WINDOW_AUTOSIZE)

        import ctypes
        user32 = ctypes.windll.user32
        screen_h = user32.GetSystemMetrics(1)
        
        # Calculate the nearest integer scale multiplier to 90% of screen height
        scale_factor = max(1, round((screen_h * 0.9) / grid_h))
        
        target_w = grid_w * scale_factor
        target_h = grid_h * scale_factor

        # Scale the HD original image to match the display window size using Nearest Neighbor
        disp_orig = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        
        # Explicitly upscale the pixel art using Nearest Neighbor. Since scale_factor is an integer,
        # every pixel scales up perfectly into a crisp NxN block with no distortion.
        disp_quant = cv2.resize(quantized_grid, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        cv2.imshow(win_orig, disp_orig)
        cv2.imshow(win_quant, disp_quant)
        cv2.imshow(win_palette, palette_swatch)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        "grid": quantized_grid,
        "palette": centers,
        "dim": (grid_w, grid_h)
    }


if __name__ == '__main__':
    # Hardcoded configuration
    image_path = str(Path(__file__).resolve().parent / "eldenring.jpg")
    k = 10  # Number of palette colors to extract

    process_image(image_path=image_path, k=k, save_output=True)
