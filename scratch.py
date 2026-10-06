import cv2
import numpy as np

def test(image_path):
    image = cv2.imread(image_path)
    if image is None: 
        print("Image not found")
        return
    
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    
    col_grad = np.sum(grad_mag, axis=0)
    row_grad = np.sum(grad_mag, axis=1)
    mean_grad = float(np.mean(grad_mag))
    penalty_per_pixel = mean_grad * 1.5  # Adjust penalty as needed
    
    smaller_dim = min(w, h)
    other_dim = max(w, h)
    smaller_is_w = (w < h)
    
    results = []
    
    for d in range(smaller_dim, 0, -1):
        if (d * other_dim) % smaller_dim != 0: continue
        other_d = (d * other_dim) // smaller_dim
        gw, gh = (d, other_d) if smaller_is_w else (other_d, d)
        
        bx = np.clip(np.round(np.arange(1, gw) * (w / float(gw))).astype(int), 0, w - 1)
        by = np.clip(np.round(np.arange(1, gh) * (h / float(gh))).astype(int), 0, h - 1)
        
        vert_sum = float(np.sum(col_grad[bx]))
        horiz_sum = float(np.sum(row_grad[by]))
        overlap_sum = float(np.sum(grad_mag[np.ix_(by, bx)])) if len(by) > 0 and len(bx) > 0 else 0.0
        boundary_sum = vert_sum + horiz_sum - overlap_sum
        boundary_pixels = float(len(bx) * h + len(by) * w - len(bx) * len(by))
        
        score = boundary_sum - penalty_per_pixel * boundary_pixels
        
        if boundary_pixels > 0:
            results.append({
                "dim": (gw, gh),
                "score": score,
                "b_mean": boundary_sum / boundary_pixels
            })
        
    results.sort(key=lambda x: x["score"], reverse=True)
    print("Top 10 results:")
    for r in results[:10]:
        print(f"{r['dim'][0]:>4}x{r['dim'][1]:<4} : score = {r['score']:>10.1f} | mean = {r['b_mean']:.2f}")

test('eldenring.jpg')
