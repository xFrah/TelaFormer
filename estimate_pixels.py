from pathlib import Path
import cv2
import numpy as np


class InteractiveViewer:
    """
    Interactive image viewer with discrete aspect-ratio-preserving grid steps,
    stepping the smaller dimension down to 1 px,
    mouse-wheel zoom (centered at cursor), drag-to-pan, view toggling,
    and razor-thin 1-monitor-pixel grid lines rendered directly in screen space.
    """
    def __init__(
        self,
        image,
        initial_scale=0.50,
        step=1,
        down_interp=cv2.INTER_NEAREST,
        up_interp=cv2.INTER_NEAREST,
        grid_color=(0, 255, 0),
        grid_alpha=0.5,
        window_name="TelaFormer - Interactive Pixel Grid Explorer"
    ):
        self.image = image
        self.h, self.w = image.shape[:2]
        self.down_interp = down_interp
        self.up_interp = up_interp
        self.grid_color = grid_color
        self.grid_alpha = grid_alpha
        self.window_name = window_name

        # Precompute all integer grid dimensions (gw, gh) that maintain aspect ratio perfectly
        if self.w <= self.h:
            s_dim, o_dim, s_is_w = self.w, self.h, True
        else:
            s_dim, o_dim, s_is_w = self.h, self.w, False

        self.valid_dims = []
        for d in range(1, s_dim + 1):
            if (d * o_dim) % s_dim == 0:
                other_d = (d * o_dim) // s_dim
                gw, gh = (d, other_d) if s_is_w else (other_d, d)
                sc = d / float(s_dim)
                self.valid_dims.append((gw, gh, sc))

        if not self.valid_dims:
            self.valid_dims = [(self.w, self.h, 1.0)]

        # Find initial index closest to initial_scale
        scales = [v[2] for v in self.valid_dims]
        self.dim_idx = int(np.argmin([abs(s - initial_scale) for s in scales]))
        self.grid_w, self.grid_h, self.scale = self.valid_dims[self.dim_idx]
        self.min_scale = self.valid_dims[0][2]

        # 0: Original, 1: Reconstructed, 2: Recon + Grid, 3: Orig + Grid
        self.view_mode = 2
        self.view_names = [
            "Original (Clean)",
            "Reconstructed (Clean)",
            "Reconstructed with Grid",
            "Original with Grid"
        ]

        self.zoom = 1.0
        self.center_x = self.w / 2.0
        self.center_y = self.h / 2.0

        self.is_dragging = False
        self.drag_start_mouse = (0, 0)
        self.drag_start_center = (self.center_x, self.center_y)
        self.show_hud = True

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        disp_w = min(1280, self.w)
        disp_h = int(disp_w * (self.h / self.w))
        cv2.resizeWindow(self.window_name, disp_w, disp_h)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        self._recompute_scale()

    def _recompute_scale(self):
        if self.scale >= 0.9999 or (self.grid_w == self.w and self.grid_h == self.h):
            self.reconstructed = self.image.copy()
            self.downscaled = self.image.copy()
        else:
            self.downscaled = cv2.resize(self.image, (self.grid_w, self.grid_h), interpolation=self.down_interp)
            self.reconstructed = cv2.resize(self.downscaled, (self.w, self.h), interpolation=self.up_interp)

        self.base_images = [
            self.image,
            self.reconstructed,
            self.reconstructed,
            self.image
        ]

    def step_dim(self, delta_steps):
        new_idx = int(np.clip(self.dim_idx + delta_steps, 0, len(self.valid_dims) - 1))
        if new_idx != self.dim_idx:
            self.dim_idx = new_idx
            self.grid_w, self.grid_h, self.scale = self.valid_dims[self.dim_idx]
            self._recompute_scale()
            self.update_display()
            factor_str = f"{1.0 / self.scale:.2f}x"
            print(f"Scale: {self.scale:.3f} ({factor_str}) | Grid: {self.grid_w}x{self.grid_h}")

    def change_scale(self, delta):
        target_scale = self.scale + delta
        scales = [v[2] for v in self.valid_dims]
        new_idx = int(np.argmin([abs(s - target_scale) for s in scales]))
        if new_idx != self.dim_idx:
            self.dim_idx = new_idx
            self.grid_w, self.grid_h, self.scale = self.valid_dims[self.dim_idx]
            self._recompute_scale()
            self.update_display()
            factor_str = f"{1.0 / self.scale:.2f}x"
            print(f"Scale: {self.scale:.3f} ({factor_str}) | Grid: {self.grid_w}x{self.grid_h}")

    def _mouse_callback(self, event, x, y, flags, param):
        crop_w = self.w / self.zoom
        crop_h = self.h / self.zoom
        x1 = self.center_x - crop_w / 2.0
        y1 = self.center_y - crop_h / 2.0

        orig_x = x1 + (x / max(1, self.w)) * crop_w
        orig_y = y1 + (y / max(1, self.h)) * crop_h

        if event == cv2.EVENT_MOUSEWHEEL:
            factor = 1.25 if flags > 0 else 0.8
            self.zoom_at(factor, orig_x, orig_y, x, y)
            self.update_display()

        elif event == cv2.EVENT_LBUTTONDOWN:
            self.is_dragging = True
            self.drag_start_mouse = (x, y)
            self.drag_start_center = (self.center_x, self.center_y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_dragging:
                dx = x - self.drag_start_mouse[0]
                dy = y - self.drag_start_mouse[1]
                self.center_x = self.drag_start_center[0] - dx * (crop_w / self.w)
                self.center_y = self.drag_start_center[1] - dy * (crop_h / self.h)
                self._clamp_center()
                self.update_display()

        elif event == cv2.EVENT_LBUTTONUP:
            self.is_dragging = False

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.zoom = 1.0
            self.center_x = self.w / 2.0
            self.center_y = self.h / 2.0
            self.update_display()

    def zoom_at(self, factor, orig_x, orig_y, win_x, win_y):
        new_zoom = float(np.clip(self.zoom * factor, 1.0, 100.0))
        if new_zoom == self.zoom:
            return
        new_crop_w = self.w / new_zoom
        new_crop_h = self.h / new_zoom

        new_x1 = orig_x - (win_x / max(1, self.w)) * new_crop_w
        new_y1 = orig_y - (win_y / max(1, self.h)) * new_crop_h

        self.center_x = new_x1 + new_crop_w / 2.0
        self.center_y = new_y1 + new_crop_h / 2.0
        self.zoom = new_zoom
        self._clamp_center()

    def _clamp_center(self):
        crop_w = self.w / self.zoom
        crop_h = self.h / self.zoom
        self.center_x = float(np.clip(self.center_x, crop_w / 2.0, self.w - crop_w / 2.0))
        self.center_y = float(np.clip(self.center_y, crop_h / 2.0, self.h - crop_h / 2.0))

    def update_display(self):
        base_img = self.base_images[self.view_mode]
        current_title = self.view_names[self.view_mode]

        crop_w = self.w / self.zoom
        crop_h = self.h / self.zoom

        x1 = max(0, min(self.w - 1, int(round(self.center_x - crop_w / 2.0))))
        y1 = max(0, min(self.h - 1, int(round(self.center_y - crop_h / 2.0))))
        x2 = max(x1 + 1, min(self.w, int(round(x1 + crop_w))))
        y2 = max(y1 + 1, min(self.h, int(round(y1 + crop_h))))

        cw = float(x2 - x1)
        ch = float(y2 - y1)
        cell_w = self.w / float(self.grid_w)
        cell_h = self.h / float(self.grid_h)

        if self.view_mode in (1, 2) and (self.grid_w < self.w or self.grid_h < self.h):
            # Direct single-pass projection from downscaled pixels to screen display:
            # Eliminates intermediate quantization artifacts, ensuring EVERY grid cell
            # is mathematically 100% ONE uniform solid color with zero color bleeding!
            x_indices = np.clip(np.floor((np.arange(self.w) * (cw / self.w) + x1) / cell_w).astype(int), 0, self.grid_w - 1)
            y_indices = np.clip(np.floor((np.arange(self.h) * (ch / self.h) + y1) / cell_h).astype(int), 0, self.grid_h - 1)
            rendered = self.downscaled[np.ix_(y_indices, x_indices)]
        else:
            cropped = base_img[y1:y2, x1:x2]
            rendered = cv2.resize(cropped, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        # Draw grid lines directly in monitor screen pixels (ALWAYS exactly 1 monitor pixel wide and perfectly equidistant)
        if self.view_mode in (2, 3) and (self.grid_w < self.w or self.grid_h < self.h):
            overlay = rendered.copy()

            # Visible index range in X
            k_min_x = max(0, int(np.floor(x1 / cell_w)))
            k_max_x = min(self.grid_w, int(np.ceil(x2 / cell_w)))
            for k in range(k_min_x, k_max_x + 1):
                gx = k * cell_w
                sx = int(round(((gx - x1) / cw) * self.w))
                if 0 <= sx < self.w:
                    cv2.line(overlay, (sx, 0), (sx, self.h - 1), self.grid_color, 1)

            # Visible index range in Y
            k_min_y = max(0, int(np.floor(y1 / cell_h)))
            k_max_y = min(self.grid_h, int(np.ceil(y2 / cell_h)))
            for k in range(k_min_y, k_max_y + 1):
                gy = k * cell_h
                sy = int(round(((gy - y1) / ch) * self.h))
                if 0 <= sy < self.h:
                    cv2.line(overlay, (0, sy), (self.w - 1, sy), self.grid_color, 1)

            if self.grid_alpha < 1.0:
                cv2.addWeighted(overlay, self.grid_alpha, rendered, 1.0 - self.grid_alpha, 0, dst=rendered)
            else:
                rendered = overlay

        if self.show_hud:
            hud_h = 36
            hud_bg = rendered[self.h - hud_h:self.h, 0:self.w].copy()
            cv2.rectangle(hud_bg, (0, 0), (self.w, hud_h), (20, 20, 20), -1)
            cv2.addWeighted(hud_bg, 0.75, rendered[self.h - hud_h:self.h, 0:self.w], 0.25, 0, rendered[self.h - hud_h:self.h, 0:self.w])

            factor_val = 1.0 / max(1e-5, self.scale)
            info = (
                f"Scale: {self.scale:.3f} ({factor_val:.2f}x) | "
                f"Grid: {self.grid_w}x{self.grid_h} | "
                f"View: [{self.view_mode+1}/4] {current_title} | "
                f"Zoom: {self.zoom:.1f}x | "
                f"[ [ / ] ] Step Dim (Strict Aspect Ratio)"
            )
            cv2.putText(rendered, info, (10, self.h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(self.window_name, rendered)

    def run(self):
        self.update_display()
        print("\n" + "=" * 75)
        print(" Interactive Pixel Grid Explorer Controls (1px Monitor Grid Edition):")
        print("=" * 75)
        print("  - [ / ] or A / D          : Step Grid Dimension (+/- 1 valid step)")
        print("  - { / } (Shift+[ / ])   : Coarse Step Grid Dimension (+/- 5 valid steps)")
        print("  - , / . or < / >          : Step Grid Dimension (+/- 1 valid step)")
        print("  - Mouse Scroll Wheel      : Zoom In / Zoom Out (centered at mouse cursor)")
        print("  - Left Mouse Drag         : Pan around image smoothly")
        print("  - Right Click / 'r'       : Reset zoom & position to 1.0x")
        print("  - '1' / '2' / '3' / '4'   : Switch view mode:")
        print("       1 = Original Image (Clean)")
        print("       2 = Reconstructed Image (Clean)")
        print("       3 = Reconstructed with 1px Monitor Grid")
        print("       4 = Original with 1px Monitor Grid")
        print("  - Space                   : Cycle through views")
        print("  - 'g'                     : Toggle grid overlay on/off")
        print("  - 'h'                     : Toggle bottom status bar")
        print("  - 'q' or ESC              : Close viewer\n" + "=" * 75 + "\n")

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (ord('q'), 27):
                break
            # Step adjustment
            elif key in (ord('['), ord('a')):
                self.step_dim(-1)
            elif key in (ord(']'), ord('d')):
                self.step_dim(+1)
            # Coarse step
            elif key in (ord('{'), ord('A')):
                self.step_dim(-5)
            elif key in (ord('}'), ord('D')):
                self.step_dim(+5)
            # Fine step
            elif key in (ord(','), ord('<')):
                self.step_dim(-1)
            elif key in (ord('.'), ord('>')):
                self.step_dim(+1)
            elif key == ord('r'):
                self.zoom = 1.0
                self.center_x = self.w / 2.0
                self.center_y = self.h / 2.0
                self.update_display()
            elif key in (ord('+'), ord('=')):
                self.zoom_at(1.25, self.center_x, self.center_y, self.w / 2, self.h / 2)
                self.update_display()
            elif key in (ord('-'), ord('_')):
                self.zoom_at(0.8, self.center_x, self.center_y, self.w / 2, self.h / 2)
                self.update_display()
            elif key == ord(' '):
                self.view_mode = (self.view_mode + 1) % len(self.base_images)
                self.update_display()
            elif key == ord('g'):
                if self.view_mode == 1:
                    self.view_mode = 2
                elif self.view_mode == 2:
                    self.view_mode = 1
                elif self.view_mode == 0:
                    self.view_mode = 3
                elif self.view_mode == 3:
                    self.view_mode = 0
                self.update_display()
            elif key == ord('h'):
                self.show_hud = not self.show_hud
                self.update_display()
            elif ord('1') <= key <= ord('4'):
                self.view_mode = key - ord('1')
                self.update_display()

        cv2.destroyAllWindows()


def scan_and_explore(
    image_path,
    step=None,
    up_scale=2,
    down_interp_name="nearest",
    up_interp_name="nearest",
    grid_color=(0, 255, 0),
    grid_alpha=0.5,
    show_images=True
):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from '{image_path}'")
        return None

    h, w = image.shape[:2]
    if w <= h:
        smaller_name = "width"
        smaller_dim = w
        other_dim = h
        smaller_is_w = True
    else:
        smaller_name = "height"
        smaller_dim = h
        other_dim = w
        smaller_is_w = False

    dim_step = step if (isinstance(step, int) and step >= 1) else 1

    print("=" * 85)
    print(f"Loaded image    : '{image_path}'")
    print(f"Resolution      : {w}x{h}")
    print(f"Scanning Method : Option B (Boundary vs. Interior Gradient Ratio with {up_scale}x Subpixel Upscaling)")
    print(f"Step Strategy   : Step smaller dimension ({smaller_name} = {smaller_dim}px down to 1px by {dim_step}px)")
    print(f"Aspect Ratio    : Strictly maintained (skipping combinations that distort aspect ratio)")
    print("=" * 85)

    interp_map = {
        "area": cv2.INTER_AREA,
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
    }
    down_interp = interp_map.get(down_interp_name.lower(), cv2.INTER_NEAREST)
    up_interp = interp_map.get(up_interp_name.lower(), cv2.INTER_NEAREST)

    # 1. Upscale by up_scale using bicubic interpolation for continuous subpixel precision
    w_up = w * up_scale
    h_up = h * up_scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_up = cv2.resize(gray, (w_up, h_up), interpolation=cv2.INTER_CUBIC)

    # 2. Compute subpixel gradient magnitude map
    gx = cv2.Sobel(gray_up, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray_up, cv2.CV_32F, 0, 1, ksize=1)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    # Precompute 1D gradient projections
    col_grad = np.sum(grad_mag, axis=0)  # shape (w_up,)
    row_grad = np.sum(grad_mag, axis=1)  # shape (h_up,)
    total_grad = float(np.sum(grad_mag))
    total_pixels = float(w_up * h_up)

    results = []
    seen_dims = set()

    # 3. Fast scan stepping smaller dimension down to 1px with perfect aspect ratio
    for d in range(smaller_dim, 0, -dim_step):
        # Aspect ratio check: if not maintained perfectly, don't even try it!
        if (d * other_dim) % smaller_dim != 0:
            continue

        other_d = (d * other_dim) // smaller_dim
        gw, gh = (d, other_d) if smaller_is_w else (other_d, d)
        current_scale = d / float(smaller_dim)

        if (gw, gh) not in seen_dims:
            seen_dims.add((gw, gh))

            cell_w_up = w_up / float(gw)
            cell_h_up = h_up / float(gh)

            # Subpixel boundary coordinates
            bx = np.clip(np.round(np.arange(1, gw) * cell_w_up).astype(int), 0, w_up - 1)
            by = np.clip(np.round(np.arange(1, gh) * cell_h_up).astype(int), 0, h_up - 1)

            # Fast 1D summation of boundary gradient
            vert_sum = float(np.sum(col_grad[bx]))
            horiz_sum = float(np.sum(row_grad[by]))

            # Subtract intersection double-count
            overlap_sum = float(np.sum(grad_mag[np.ix_(by, bx)])) if len(by) > 0 and len(bx) > 0 else 0.0
            boundary_sum = vert_sum + horiz_sum - overlap_sum
            boundary_pixels = float(len(bx) * h_up + len(by) * w_up - len(bx) * len(by))

            interior_sum = total_grad - boundary_sum
            interior_pixels = total_pixels - boundary_pixels

            if boundary_pixels > 0 and interior_pixels > 0:
                b_mean = boundary_sum / boundary_pixels
                i_mean = interior_sum / interior_pixels
                ratio_score = b_mean / (i_mean + 1e-6)

                results.append({
                    "scale": current_scale,
                    "dim": (gw, gh),
                    "score": ratio_score,
                    "b_mean": b_mean,
                    "i_mean": i_mean,
                })

    if not results:
        print("No valid scales evaluated.")
        return None

    # Sort results by scale (descending) to find local peaks across the spectrum
    results_by_scale = sorted(results, key=lambda x: x["scale"], reverse=True)
    
    # Identify local peaks in the gradient ratio curve
    peaks = []
    for i in range(len(results_by_scale)):
        curr_score = results_by_scale[i]["score"]
        prev_score = results_by_scale[i - 1]["score"] if i > 0 else 0.0
        next_score = results_by_scale[i + 1]["score"] if i < len(results_by_scale) - 1 else 0.0
        if curr_score >= prev_score and curr_score >= next_score and curr_score > 1.15:
            if results_by_scale[i]["scale"] < 0.9999 or len(results_by_scale) == 1:
                peaks.append(results_by_scale[i])

    # Precompute high-frequency edge energy of original image
    gy_o, gx_o = np.gradient(gray)
    gm_o_sum = float(np.sum(np.sqrt(gx_o ** 2 + gy_o ** 2)))

    # Global maximum ratio
    sorted_by_score = sorted(results, key=lambda x: x["score"], reverse=True)
    global_best = sorted_by_score[0]

    # Fundamental scale is the highest-resolution local peak (highest scale) with strong boundary contrast
    if peaks:
        prominent = [p for p in peaks if p["score"] >= 1.25]
        fundamental_best = prominent[0] if prominent else peaks[0]
    else:
        fundamental_best = global_best

    # Find candidate scales around the fundamental frequency (+/- 0.006)
    neighborhood = [r for r in results if abs(r["scale"] - fundamental_best["scale"]) <= 0.006]

    # Evaluate High-Frequency (HF) detail preservation %
    eval_set = {r["scale"]: r for r in (sorted_by_score[:10] + neighborhood + [fundamental_best, global_best])}
    for r in eval_set.values():
        if "hf_ratio" not in r:
            dw, dh = r["dim"]
            down = cv2.resize(image, (dw, dh), interpolation=down_interp)
            up = cv2.resize(down, (w, h), interpolation=up_interp)
            up_gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gy_u, gx_u = np.gradient(up_gray)
            gm_u_sum = float(np.sum(np.sqrt(gx_u ** 2 + gy_u ** 2)))
            r["hf_ratio"] = float((gm_u_sum / (gm_o_sum + 1e-6)) * 100.0)

    # Among neighborhood candidates, select the one that maximizes high-frequency edge contrast & crispness
    sharp_candidates = [r for r in neighborhood if r.get("hf_ratio", 0.0) >= 88.0 and r.get("score", 0.0) >= 1.15]
    if sharp_candidates:
        # Sort by peak HF detail preservation (highest edge contrast and isotropic square pixels)
        sharp_best = max(sharp_candidates, key=lambda x: x.get("hf_ratio", 0.0))
    else:
        sharp_best = fundamental_best

    print("\n" + "=" * 85)
    print(f"{'Scale':<16} | {'Dimensions':<14} | {'Gradient Ratio':<18} | {'HF Detail %':<14}")
    print("-" * 85)
    for res in sorted_by_score[:10]:
        dim_str = f"{res['dim'][0]}x{res['dim'][1]}"
        factor_str = f"({1.0 / res['scale']:.2f}x)"
        scale_str = f"{res['scale']:.3f} {factor_str}"
        tag = "  <- [Best Match]" if abs(res["scale"] - sharp_best["scale"]) < 1e-4 else ""
        print(f"{scale_str:<16} | {dim_str:<14} | {res['score']:<18.3f} | {res.get('hf_ratio', 0.0):<13.1f}%{tag}")
    
    # Also ensure best match is printed if not in top 10
    if sharp_best["scale"] not in [r["scale"] for r in sorted_by_score[:10]]:
        dim_str = f"{sharp_best['dim'][0]}x{sharp_best['dim'][1]}"
        factor_str = f"({1.0 / sharp_best['scale']:.2f}x)"
        scale_str = f"{sharp_best['scale']:.3f} {factor_str}"
        print(f"{scale_str:<16} | {dim_str:<14} | {sharp_best['score']:<18.3f} | {sharp_best.get('hf_ratio', 0.0):<13.1f}%  <- [Best Match]")
    print("=" * 85)

    best = sharp_best

    print(f"\n[Best Matching Scale]:")
    print(f"  Scale       : {best['scale']:.3f} (Factor: {1.0 / best['scale']:.2f}x)")
    print(f"  Resolution  : {best['dim'][0]}x{best['dim'][1]}")
    print(f"  HF Detail   : {best.get('hf_ratio', 0.0):.1f}% preserved")

    print(f"\nOpening Interactive Viewer at scale {best['scale']:.3f}...")

    if show_images:
        # Display the original reference image in a separate dedicated window
        orig_win_name = f"Original Image ({w}x{h})"
        cv2.namedWindow(orig_win_name, cv2.WINDOW_NORMAL)
        disp_w = min(1280, w)
        disp_h = int(disp_w * (h / w))
        cv2.resizeWindow(orig_win_name, disp_w, disp_h)
        cv2.imshow(orig_win_name, image)

        viewer = InteractiveViewer(
            image=image,
            initial_scale=best["scale"],
            down_interp=down_interp,
            up_interp=up_interp,
            grid_color=grid_color,
            grid_alpha=grid_alpha
        )
        viewer.run()

    return best


if __name__ == '__main__':
    # Hardcoded configuration
    image_path = str(Path(__file__).resolve().parent / "eldenring.jpg")
    up_scale = 3                 # Subpixel upscaling factor (2x for subpixel gradient interpolation)
    down_interp_name = "nearest" # 'nearest' or 'area'
    up_interp_name = "nearest"   # 'nearest'
    grid_color = (0, 255, 0)     # Grid line color (BGR)
    grid_alpha = 0.5             # Grid line transparency (0.0 to 1.0)
    show_images = True

    scan_and_explore(
        image_path=image_path,
        up_scale=up_scale,
        down_interp_name=down_interp_name,
        up_interp_name=up_interp_name,
        grid_color=grid_color,
        grid_alpha=grid_alpha,
        show_images=show_images
    )
