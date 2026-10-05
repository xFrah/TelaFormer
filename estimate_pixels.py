import cv2
import numpy as np


def compute_similarity(original, reconstructed):
    """
    Computes similarity metrics between the original and reconstructed images.
    Returns MSE (Mean Squared Error), MAE (Mean Absolute Error), and PSNR (dB).
    """
    orig_f = original.astype(np.float32)
    recon_f = reconstructed.astype(np.float32)

    mse = float(np.mean((orig_f - recon_f) ** 2))
    mae = float(np.mean(np.abs(orig_f - recon_f)))
    psnr = float(10.0 * np.log10((255.0 ** 2) / (mse + 1e-10)))

    return {
        "mse": mse,
        "mae": mae,
        "psnr": psnr
    }


class InteractiveViewer:
    """
    Interactive image viewer with continuous fractional scale adjustments (step = 0.001),
    mouse-wheel zoom (centered at cursor), drag-to-pan, view toggling,
    and razor-thin 1-monitor-pixel grid lines rendered directly in screen space.
    """
    def __init__(
        self,
        image,
        initial_scale=0.50,
        step=0.001,
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

        self.scale = float(np.clip(initial_scale, 0.001, 1.0))
        self.step = float(step)
        self.min_scale = max(1.0 / self.w, 1.0 / self.h)

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
        self.grid_w = max(1, int(round(self.w * self.scale)))
        self.grid_h = max(1, int(round(self.h * self.scale)))

        if self.scale >= 0.9999 or (self.grid_w == self.w and self.grid_h == self.h):
            self.reconstructed = self.image.copy()
            self.downscaled = self.image.copy()
            self.mse = 0.0
            self.psnr = float('inf')
        else:
            self.downscaled = cv2.resize(self.image, (self.grid_w, self.grid_h), interpolation=self.down_interp)
            self.reconstructed = cv2.resize(self.downscaled, (self.w, self.h), interpolation=self.up_interp)
            diff = self.image.astype(np.float32) - self.reconstructed.astype(np.float32)
            self.mse = float(np.mean(diff ** 2))
            self.psnr = float(10.0 * np.log10((255.0 ** 2) / (self.mse + 1e-10)))

        self.base_images = [
            self.image,
            self.reconstructed,
            self.reconstructed,
            self.image
        ]

    def change_scale(self, delta):
        new_scale = float(np.clip(round(self.scale + delta, 4), self.min_scale, 1.0))
        if new_scale != self.scale:
            self.scale = new_scale
            self._recompute_scale()
            self.update_display()
            factor_str = f"{1.0 / self.scale:.2f}x"
            print(f"Scale: {self.scale:.3f} ({factor_str}) | Grid: {self.grid_w}x{self.grid_h} | MSE: {self.mse:6.1f} | PSNR: {self.psnr:4.1f} dB")

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
                f"MSE: {self.mse:.1f} | "
                f"View: [{self.view_mode+1}/4] {current_title} | "
                f"Zoom: {self.zoom:.1f}x | "
                f"[ [ / ] ] Step: {self.step:+.3f}"
            )
            cv2.putText(rendered, info, (10, self.h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(self.window_name, rendered)

    def run(self):
        self.update_display()
        print("\n" + "=" * 75)
        print(" Interactive Pixel Grid Explorer Controls (1px Monitor Grid Edition):")
        print("=" * 75)
        print(f"  - [ / ] or A / D          : Step Scale by {self.step:+.3f}")
        print(f"  - {{ / }} (Shift+[ / ])   : Coarse Step (+/- 0.05)")
        print(f"  - , / . or < / >          : Fine Step (+/- 0.001)")
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
                self.change_scale(-self.step)
            elif key in (ord(']'), ord('d')):
                self.change_scale(+self.step)
            # Coarse step
            elif key in (ord('{'), ord('A')):
                self.change_scale(-0.05)
            elif key in (ord('}'), ord('D')):
                self.change_scale(+0.05)
            # Fine step
            elif key in (ord(','), ord('<')):
                self.change_scale(-0.001)
            elif key in (ord('.'), ord('>')):
                self.change_scale(+0.001)
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
    step=0.001,
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
    print("=" * 75)
    print(f"Loaded image    : '{image_path}'")
    print(f"Resolution      : {w}x{h}")
    print(f"Scanning scales with step = {step} down to 1px...")
    print("=" * 75)

    interp_map = {
        "area": cv2.INTER_AREA,
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
    }
    down_interp = interp_map.get(down_interp_name.lower(), cv2.INTER_NEAREST)
    up_interp = interp_map.get(up_interp_name.lower(), cv2.INTER_NEAREST)

    results = []
    seen_dims = set()
    current_scale = 1.0 - step

    # Scan down to 1px
    while current_scale > 0:
        dw = max(1, int(round(w * current_scale)))
        dh = max(1, int(round(h * current_scale)))

        if (dw, dh) not in seen_dims:
            seen_dims.add((dw, dh))
            downscaled = cv2.resize(image, (dw, dh), interpolation=down_interp)
            reconstructed = cv2.resize(downscaled, (w, h), interpolation=up_interp)

            metrics = compute_similarity(image, reconstructed)
            results.append({
                "scale": current_scale,
                "dim": (dw, dh),
                "mse": metrics["mse"],
                "mae": metrics["mae"],
                "psnr": metrics["psnr"],
            })

        if dw <= 1 or dh <= 1:
            break
        current_scale -= step

    if not results:
        print("No valid scales evaluated.")
        return None

    sorted_by_mse = sorted(results, key=lambda x: x["mse"])
    best = sorted_by_mse[0]

    print("\n" + "=" * 75)
    print(f"{'Scale':<15} | {'Dimensions':<14} | {'MSE (lower=better)':<20} | {'PSNR (dB)':<10}")
    print("-" * 75)
    for res in sorted_by_mse[:10]:
        dim_str = f"{res['dim'][0]}x{res['dim'][1]}"
        scale_str = f"scale {res['scale']:.3f}"
        print(f"{scale_str:<15} | {dim_str:<14} | {res['mse']:<20.2f} | {res['psnr']:<10.2f}")
    print("=" * 75)

    print(f"\n[Best Matching Scale Selected] (excluding trivial 1:1 original):")
    print(f"  Scale                 : {best['scale']:.3f} (Factor: {1.0 / best['scale']:.2f}x)")
    print(f"  Resolution            : {best['dim'][0]}x{best['dim'][1]}")
    print(f"  Reconstruction MSE    : {best['mse']:.2f}")
    print(f"  Reconstruction PSNR   : {best['psnr']:.2f} dB")
    print("\nOpening Interactive Viewer at best matching scale...")

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
            step=step,
            down_interp=down_interp,
            up_interp=up_interp,
            grid_color=grid_color,
            grid_alpha=grid_alpha
        )
        viewer.run()

    return best


if __name__ == '__main__':
    # Hardcoded configuration
    image_path = r"C:\Users\fra-fisso\Downloads\eldenring.jpg"
    step = 0.001                 # Scale scan step size (0.001)
    down_interp_name = "nearest" # 'nearest' or 'area'
    up_interp_name = "nearest"   # 'nearest'
    grid_color = (0, 255, 0)     # Grid line color (BGR)
    grid_alpha = 0.5             # Grid line transparency (0.0 to 1.0)
    show_images = True

    scan_and_explore(
        image_path=image_path,
        step=step,
        down_interp_name=down_interp_name,
        up_interp_name=up_interp_name,
        grid_color=grid_color,
        grid_alpha=grid_alpha,
        show_images=show_images
    )
