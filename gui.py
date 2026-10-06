import sys
import os
import cv2
import numpy as np
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QSpinBox, QSlider, QFrame,
    QProgressBar, QToolTip, QMenu, QMessageBox, QScrollArea,
    QSizePolicy, QStackedWidget, QComboBox
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QSize, QPoint, QRect, QTimer,
    QPropertyAnimation, QParallelAnimationGroup, QEasingCurve
)
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QColor, QPen, QBrush, QFont,
    QCursor, QAction, QPainterPath
)

from estimate_pixels import scan_and_explore
from kmeans import quantize_colors_kmeans_lab
from main import create_palette_swatch


# ==============================================================================
# Workflow Modes
# ==============================================================================
MODE_DETECT = "detect"      # 1. Detect True Pixel Art (auto-scan native grid)
MODE_PIXELATE = "pixelate"  # 2. Pixelate Photo/Art (downscale normal image to pixel grid)
MODE_EXTRACT = "extract"    # 3. Extract Colors (direct 1:1 image pass-through)


# ==============================================================================
# Asynchronous Background Worker Thread
# ==============================================================================
class ProcessThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, task_type, **kwargs):
        super().__init__()
        self.task_type = task_type
        self.kwargs = kwargs

    def run(self):
        try:
            if self.task_type == "estimate":
                best = scan_and_explore(self.kwargs["image_path"], show_images=False)
                if best is None:
                    self.error.emit("Failed to detect grid.")
                else:
                    self.finished.emit(best)

            elif self.task_type == "quantize":
                image_bgr = self.kwargs["image_bgr"]
                k = self.kwargs["k"]
                grid_w = self.kwargs["grid_w"]
                grid_h = self.kwargs["grid_h"]

                flat_pixels = image_bgr.reshape((-1, 3))
                unique_colors = np.unique(flat_pixels, axis=0)
                num_uniques = len(unique_colors)

                # If the image is already quantized and has <= K unique colors, preserve them exactly!
                if num_uniques <= k:
                    centers = np.asarray(unique_colors, dtype=np.uint8)
                else:
                    _, centers = quantize_colors_kmeans_lab(flat_pixels, k=k)
                    centers = np.asarray(centers, dtype=np.uint8)

                # Sort palette colors by luminance for organized presentation
                r_chan = centers[:, 2].astype(np.float32)
                g_chan = centers[:, 1].astype(np.float32)
                b_chan = centers[:, 0].astype(np.float32)
                luminances = 0.299 * r_chan + 0.587 * g_chan + 0.114 * b_chan
                sorted_order = np.argsort(luminances)
                centers = centers[sorted_order]

                # Map all pixels to their nearest sorted cluster center
                diffs = flat_pixels[:, np.newaxis, :].astype(np.float32) - centers[np.newaxis, :, :].astype(np.float32)
                squared_distances = np.sum(diffs ** 2, axis=2)
                nearest_palette_indices = np.argmin(squared_distances, axis=1)

                quantized_grid = centers[nearest_palette_indices].reshape((grid_h, grid_w, 3))
                self.finished.emit({
                    "quantized_grid": quantized_grid,
                    "centers": centers
                })
        except Exception as e:
            self.error.emit(str(e))


def hex_to_rgba(hex_code, alpha):
    hex_code = hex_code.lstrip('#')
    if len(hex_code) == 6:
        r = int(hex_code[0:2], 16)
        g = int(hex_code[2:4], 16)
        b = int(hex_code[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"
    return hex_code


# ==============================================================================
# Smooth Sliding Stacked Widget for Modal Multi-Step Navigation
# ==============================================================================
class SlidingStackedWidget(QStackedWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.animation_duration = 280  # milliseconds
        self.is_animating = False
        self.anim_group = None
        self.setStyleSheet("QStackedWidget { background-color: #0b0f19; }")

    def slide_to_index(self, new_index):
        if new_index == self.currentIndex() or self.is_animating:
            return

        direction_forward = new_index > self.currentIndex()
        next_widget = self.widget(new_index)
        curr_widget = self.currentWidget()

        if next_widget is None or curr_widget is None:
            self.setCurrentIndex(new_index)
            return

        width = self.width()
        height = self.height()
        offset_x = width if direction_forward else -width

        next_widget.setGeometry(offset_x, 0, width, height)
        next_widget.show()
        next_widget.raise_()

        self.anim_group = QParallelAnimationGroup(self)

        anim_next = QPropertyAnimation(next_widget, b"pos", self)
        anim_next.setDuration(self.animation_duration)
        anim_next.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim_next.setStartValue(QPoint(offset_x, 0))
        anim_next.setEndValue(QPoint(0, 0))
        self.anim_group.addAnimation(anim_next)

        anim_curr = QPropertyAnimation(curr_widget, b"pos", self)
        anim_curr.setDuration(self.animation_duration)
        anim_curr.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim_curr.setStartValue(QPoint(0, 0))
        anim_curr.setEndValue(QPoint(-offset_x, 0))
        self.anim_group.addAnimation(anim_curr)

        self.is_animating = True

        def on_finished():
            self.setCurrentIndex(new_index)
            next_widget.setGeometry(0, 0, self.width(), self.height())
            for i in range(self.count()):
                w = self.widget(i)
                if w is not None:
                    if i != new_index:
                        w.hide()
                        w.move(0, 0)
                    else:
                        w.show()
            self.is_animating = False

        self.anim_group.finished.connect(on_finished)
        self.anim_group.start()


# ==============================================================================
# Interactive Pixel Canvas Widget with Zoom, Pan, Grid & HUD Overlays
# ==============================================================================
class PixelCanvas(QWidget):
    pixelHovered = pyqtSignal(int, int, int, int, int, str)  # x, y, r, g, b, hex
    pixelLeft = pyqtSignal()

    def __init__(self, placeholder_title="No Image Loaded", placeholder_subtitle="Select or drop an image"):
        super().__init__()
        self.setMouseTracking(True)
        self.placeholder_title = placeholder_title
        self.placeholder_subtitle = placeholder_subtitle

        self.cv_img = None
        self.pixmap = None
        self.show_grid = False
        self.grid_dims = None  # (width, height)

        # Zoom & Pan State
        self.zoom = 1.0  # 1.0 = fit to canvas
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.is_dragging = False
        self.last_mouse_pos = QPoint()
        self.hover_info = None  # (x, y, r, g, b, hex_code)

        self.setMinimumSize(320, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background-color: #060912; border-radius: 10px;")

    def set_image(self, cv_img, grid_dims=None):
        self.cv_img = cv_img
        if cv_img is not None:
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.pixmap = QPixmap.fromImage(qt_img)
            self.grid_dims = grid_dims if grid_dims is not None else (w, h)
        else:
            self.pixmap = None
            self.grid_dims = None

        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.hover_info = None
        self.update()

    def set_grid_overlay(self, enabled):
        self.show_grid = enabled
        self.update()

    def reset_view(self):
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()

    def _get_target_rect(self):
        if self.pixmap is None or self.pixmap.isNull():
            return None
        pw = self.pixmap.width()
        ph = self.pixmap.height()
        ww = max(10, self.width() - 16)
        wh = max(10, self.height() - 16)

        base_scale = min(ww / pw, wh / ph)
        effective_scale = base_scale * self.zoom

        target_w = max(1, int(pw * effective_scale))
        target_h = max(1, int(ph * effective_scale))

        center_x = (self.width() / 2.0) + self.pan_x
        center_y = (self.height() / 2.0) + self.pan_y

        target_x = int(center_x - target_w / 2.0)
        target_y = int(center_y - target_h / 2.0)
        return QRect(target_x, target_y, target_w, target_h)

    def wheelEvent(self, a0):
        event = a0
        if self.pixmap is None or self.pixmap.isNull():
            return

        delta = event.angleDelta().y()
        factor = 1.22 if delta > 0 else (1.0 / 1.22)
        new_zoom = float(np.clip(self.zoom * factor, 1.0, 30.0))

        if new_zoom != self.zoom:
            if new_zoom <= 1.03:
                self.zoom = 1.0
                self.pan_x = 0.0
                self.pan_y = 0.0
            else:
                mx = event.position().x() - (self.width() / 2.0)
                my = event.position().y() - (self.height() / 2.0)
                ratio = new_zoom / self.zoom
                self.pan_x = mx - (mx - self.pan_x) * ratio
                self.pan_y = my - (my - self.pan_y) * ratio
                self.zoom = new_zoom
            self.update()

    def mousePressEvent(self, a0):
        event = a0
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            if self.zoom > 1.0:
                self.is_dragging = True
                self.last_mouse_pos = event.pos()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, a0):
        event = a0
        if self.is_dragging:
            delta = event.pos() - self.last_mouse_pos
            self.last_mouse_pos = event.pos()
            self.pan_x += delta.x()
            self.pan_y += delta.y()
            self.update()
            return

        target_rect = self._get_target_rect()
        if self.cv_img is not None and target_rect is not None and target_rect.contains(event.pos()):
            rel_x = event.pos().x() - target_rect.left()
            rel_y = event.pos().y() - target_rect.top()
            h, w = self.cv_img.shape[:2]
            px = int(np.clip((rel_x / target_rect.width()) * w, 0, w - 1))
            py = int(np.clip((rel_y / target_rect.height()) * h, 0, h - 1))
            b, g, r = self.cv_img[py, px]
            hex_code = f"#{int(r):02X}{int(g):02X}{int(b):02X}"
            self.hover_info = (px, py, int(r), int(g), int(b), hex_code)
            self.pixelHovered.emit(px, py, int(r), int(g), int(b), hex_code)
        else:
            self.hover_info = None
            self.pixelLeft.emit()
        self.update()

    def mouseReleaseEvent(self, a0):
        event = a0
        if self.is_dragging:
            self.is_dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseDoubleClickEvent(self, a0):
        self.reset_view()

    def enterEvent(self, event):
        self.update()

    def leaveEvent(self, a0):
        self.hover_info = None
        self.pixelLeft.emit()
        self.update()

    def paintEvent(self, a0):
        event = a0
        painter = QPainter(self)

        clip_path = QPainterPath()
        clip_path.addRoundedRect(1, 1, self.width() - 2, self.height() - 2, 10, 10)
        painter.setClipPath(clip_path)

        painter.fillRect(self.rect(), QColor("#060912"))

        target_rect = self._get_target_rect()

        if self.pixmap is not None and target_rect is not None:
            painter.setPen(QPen(QColor(255, 255, 255, 20), 1))
            painter.drawRect(target_rect.adjusted(-1, -1, 1, 1))

            scaled_pix = self.pixmap.scaled(
                target_rect.size(),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.FastTransformation
            )
            painter.drawPixmap(target_rect.topLeft(), scaled_pix)

            if self.show_grid and self.grid_dims is not None:
                gw, gh = self.grid_dims
                cell_w = target_rect.width() / float(gw)
                cell_h = target_rect.height() / float(gh)

                if cell_w >= 4.0 and cell_h >= 4.0:
                    grid_pen = QPen(QColor(255, 255, 255, 38), 1)
                    painter.setPen(grid_pen)

                    v_left = max(target_rect.left(), 0)
                    v_right = min(target_rect.right(), self.width())
                    v_top = max(target_rect.top(), 0)
                    v_bottom = min(target_rect.bottom(), self.height())

                    start_i = max(1, int((v_left - target_rect.left()) / cell_w))
                    end_i = min(gw, int((v_right - target_rect.left()) / cell_w) + 1)
                    for i in range(start_i, end_i):
                        lx = int(target_rect.left() + i * cell_w)
                        painter.drawLine(lx, v_top, lx, v_bottom)

                    start_j = max(1, int((v_top - target_rect.top()) / cell_h))
                    end_j = min(gh, int((v_bottom - target_rect.top()) / cell_h) + 1)
                    for j in range(start_j, end_j):
                        ly = int(target_rect.top() + j * cell_h)
                        painter.drawLine(v_left, ly, v_right, ly)

            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

            if self.zoom > 1.05 or self.underMouse():
                zoom_text = f"{self.zoom:.1f}×" if self.zoom > 1.05 else "FIT"
                painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                badge_w, badge_h = 44, 20
                bx = self.width() - badge_w - 10
                by = 10
                painter.setPen(QPen(QColor(255, 255, 255, 30), 1))
                painter.setBrush(QColor(15, 23, 42, 210))
                painter.drawRoundedRect(bx, by, badge_w, badge_h, 5, 5)
                painter.setPen(QColor("#38bdf8" if self.zoom > 1.05 else "#94a3b8"))
                painter.drawText(QRect(bx, by, badge_w, badge_h), Qt.AlignmentFlag.AlignCenter, zoom_text)

            if self.underMouse():
                hud_y = self.height() - 28
                if self.hover_info is not None:
                    px, py, r, g, b, hex_c = self.hover_info
                    info_str = f"({px}, {py})  •  {hex_c}  •  RGB({r},{g},{b})"
                    painter.setFont(QFont("Consolas", 9, QFont.Weight.DemiBold))
                    tw = painter.fontMetrics().horizontalAdvance(info_str) + 24
                    painter.setPen(QPen(QColor(56, 189, 248, 90), 1))
                    painter.setBrush(QColor(11, 17, 30, 225))
                    painter.drawRoundedRect(10, hud_y, tw, 20, 5, 5)

                    painter.setBrush(QColor(r, g, b))
                    painter.setPen(QPen(QColor(255, 255, 255, 140), 1))
                    painter.drawRect(15, hud_y + 4, 12, 12)

                    painter.setPen(QColor("#38bdf8"))
                    painter.drawText(QRect(33, hud_y, tw - 33, 20), Qt.AlignmentFlag.AlignVCenter, info_str)
                else:
                    tip_str = "Scroll to Zoom • Drag to Pan • 2× Click to Reset"
                    painter.setFont(QFont("Segoe UI", 8))
                    tw = painter.fontMetrics().horizontalAdvance(tip_str) + 14
                    painter.setPen(QPen(QColor(255, 255, 255, 25), 1))
                    painter.setBrush(QColor(15, 23, 42, 190))
                    painter.drawRoundedRect(10, hud_y, tw, 18, 4, 4)
                    painter.setPen(QColor("#94a3b8"))
                    painter.drawText(QRect(10, hud_y, tw, 18), Qt.AlignmentFlag.AlignCenter, tip_str)

        else:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

            pen_dashed = QPen(QColor(45, 55, 75, 110), 1.5, Qt.PenStyle.DashLine)
            painter.setPen(pen_dashed)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(self.rect().adjusted(14, 14, -14, -14), 10, 10)

            cx = self.width() // 2
            cy = self.height() // 2

            icon_size = 46
            ix = cx - icon_size // 2
            iy = cy - 40
            painter.setPen(QPen(QColor("#334155"), 1.5))
            painter.setBrush(QColor("#0f172a"))
            painter.drawRoundedRect(ix, iy, icon_size, icon_size, 8, 8)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#38bdf8"))
            painter.drawRect(ix + 10, iy + 10, 10, 10)
            painter.setBrush(QColor("#818cf8"))
            painter.drawRect(ix + 24, iy + 10, 12, 10)
            painter.setBrush(QColor("#c084fc"))
            painter.drawRect(ix + 10, iy + 24, 12, 12)
            painter.setBrush(QColor("#34d399"))
            painter.drawRect(ix + 24, iy + 24, 12, 12)

            painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            painter.setPen(QColor("#cbd5e1"))
            painter.drawText(QRect(20, cy + 18, self.width() - 40, 24), Qt.AlignmentFlag.AlignCenter, self.placeholder_title)

            painter.setFont(QFont("Segoe UI", 9))
            painter.setPen(QColor("#64748b"))
            painter.drawText(QRect(20, cy + 44, self.width() - 40, 20), Qt.AlignmentFlag.AlignCenter, self.placeholder_subtitle)


# ==============================================================================
# Glassmorphic Styled Card Container
# ==============================================================================
class StudioCard(QFrame):
    def __init__(self, step_num, title, accent_color="#38bdf8", has_grid_toggle=False):
        super().__init__()
        self.accent_color = accent_color
        self.has_grid_toggle = has_grid_toggle
        self.setObjectName("StudioCard")

        self.card_layout = QVBoxLayout(self)
        self.card_layout.setContentsMargins(14, 14, 14, 14)
        self.card_layout.setSpacing(10)

        # --- Card Header ---
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(2, 0, 2, 2)
        header_layout.setSpacing(8)

        self.pill_step = QLabel(step_num)
        self.pill_step.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bg_rgba = hex_to_rgba(accent_color, 0.15)
        border_rgba = hex_to_rgba(accent_color, 0.35)
        self.pill_step.setStyleSheet(f"""
            QLabel {{
                background-color: {bg_rgba};
                color: {accent_color};
                border: 1px solid {border_rgba};
                border-radius: 11px;
                font-weight: bold;
                font-size: 11px;
                padding: 2px 8px;
                min-width: 22px;
                max-height: 22px;
            }}
        """)
        header_layout.addWidget(self.pill_step)

        self.lbl_title = QLabel(title)
        self.lbl_title.setStyleSheet("""
            QLabel {
                color: #f1f5f9;
                font-size: 14px;
                font-weight: 700;
                letter-spacing: 0.3px;
            }
        """)
        header_layout.addWidget(self.lbl_title)
        header_layout.addStretch()

        self.lbl_badge = QLabel("Idle")
        self.lbl_badge.setStyleSheet("""
            QLabel {
                color: #94a3b8;
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 10px;
                font-size: 11px;
                font-weight: 600;
                padding: 2px 9px;
            }
        """)
        header_layout.addWidget(self.lbl_badge)

        if self.has_grid_toggle:
            self.btn_grid_toggle = QPushButton("# Grid")
            self.btn_grid_toggle.setProperty("class", "card-tool-btn")
            self.btn_grid_toggle.setCheckable(True)
            self.btn_grid_toggle.setToolTip("Toggle Pixel Grid Overlay (Active when zoomed or cell >= 4px)")
            self.btn_grid_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
            header_layout.addWidget(self.btn_grid_toggle)

        self.header_actions = QHBoxLayout()
        self.header_actions.setSpacing(6)
        header_layout.addLayout(self.header_actions)

        self.card_layout.addLayout(header_layout)

        # --- Center Canvas Area ---
        self.canvas = PixelCanvas()
        self.card_layout.addWidget(self.canvas, stretch=1)

        if self.has_grid_toggle:
            self.btn_grid_toggle.toggled.connect(self.canvas.set_grid_overlay)

        # --- Bottom Controls Slot ---
        self.controls_layout = QVBoxLayout()
        self.controls_layout.setContentsMargins(0, 0, 0, 0)
        self.controls_layout.setSpacing(0)
        self.card_layout.addLayout(self.controls_layout)

    def add_header_action(self, widget):
        self.header_actions.addWidget(widget)

    def set_badge(self, text, is_active=False):
        self.lbl_badge.setText(text)
        if is_active:
            bg_rgba = hex_to_rgba(self.accent_color, 0.15)
            border_rgba = hex_to_rgba(self.accent_color, 0.35)
            self.lbl_badge.setStyleSheet(f"""
                QLabel {{
                    color: {self.accent_color};
                    background-color: {bg_rgba};
                    border: 1px solid {border_rgba};
                    border-radius: 10px;
                    font-size: 11px;
                    font-weight: 600;
                    padding: 2px 9px;
                }}
            """)
        else:
            self.lbl_badge.setStyleSheet("""
                QLabel {
                    color: #94a3b8;
                    background-color: #1e293b;
                    border: 1px solid #334155;
                    border-radius: 10px;
                    font-size: 11px;
                    font-weight: 600;
                    padding: 2px 9px;
                }
            """)


# ==============================================================================
# Interactive Candidate Grid Sidebar (Step 2 - Detect Mode)
# ==============================================================================
class GridCandidateItem(QFrame):
    clicked = pyqtSignal(int)

    def __init__(self, index, candidate, is_selected=False):
        super().__init__()
        self.setObjectName("CandidateItem")
        self.index = index
        self.candidate = candidate
        self.is_selected = is_selected
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.init_ui()
        self.update_style()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        # Header: Rank Badge + Gradient Score
        hdr = QHBoxLayout()
        hdr.setSpacing(6)

        if self.index == 0:
            lbl_rank = QLabel("★ Best Match")
            lbl_rank.setStyleSheet("""
                color: #38bdf8;
                background-color: rgba(14, 165, 233, 0.15);
                border: 1px solid rgba(56, 189, 248, 0.35);
                border-radius: 4px;
                padding: 1px 6px;
                font-weight: 800;
                font-size: 10px;
            """)
        else:
            lbl_rank = QLabel(f"#{self.index + 1}")
            lbl_rank.setStyleSheet("""
                color: #94a3b8;
                background-color: #0c121e;
                border: 1px solid #1e293b;
                border-radius: 4px;
                padding: 1px 6px;
                font-weight: 700;
                font-size: 10px;
            """)
        hdr.addWidget(lbl_rank)
        hdr.addStretch()

        score_val = self.candidate.get("score", 0.0)
        lbl_score = QLabel(f"Score {score_val:,.0f}")
        lbl_score.setStyleSheet("color: #64748b; font-size: 10px; font-weight: 600; border: none; background: transparent;")
        hdr.addWidget(lbl_score)
        layout.addLayout(hdr)

        # Main: Dimensions + Downscale Factor
        mid = QHBoxLayout()
        gw, gh = self.candidate["dim"]
        scale = self.candidate["scale"]
        factor = 1.0 / scale if scale > 0 else 1.0

        lbl_dim = QLabel(f"{gw} × {gh} px")
        lbl_dim.setStyleSheet("color: #f1f5f9; font-size: 13px; font-weight: 800; border: none; background: transparent;")
        mid.addWidget(lbl_dim)
        mid.addStretch()

        lbl_factor = QLabel(f"{factor:.2f}×")
        lbl_factor.setStyleSheet("""
            color: #38bdf8;
            background-color: #0c121e;
            border: 1px solid #1e2e4a;
            border-radius: 4px;
            padding: 2px 6px;
            font-weight: 700;
            font-size: 11px;
        """)
        mid.addWidget(lbl_factor)
        layout.addLayout(mid)

        # Footer: Gradient Average per cell edge pixel
        b_mean = self.candidate.get("b_mean", 0.0)
        lbl_sub = QLabel(f"Avg Grad: {b_mean:.1f}/px  •  Scale: {scale:.3f}")
        lbl_sub.setStyleSheet("color: #475569; font-size: 9px; border: none; background: transparent;")
        layout.addWidget(lbl_sub)

    def update_style(self):
        if self.is_selected:
            self.setStyleSheet("""
                QFrame#CandidateItem {
                    background-color: #132238;
                    border: 1px solid #0ea5e9;
                    border-left: 3px solid #38bdf8;
                    border-radius: 8px;
                }
            """)
        else:
            self.setStyleSheet("""
                QFrame#CandidateItem {
                    background-color: #0c121e;
                    border: 1px solid #1e293b;
                    border-left: 1px solid #1e293b;
                    border-radius: 8px;
                }
                QFrame#CandidateItem:hover {
                    background-color: #101a2c;
                    border: 1px solid #334155;
                }
            """)

    def set_selected(self, selected):
        self.is_selected = selected
        self.update_style()

    def mousePressEvent(self, a0):
        event = a0
        if event is not None and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.index)
        super().mousePressEvent(a0)


class GridCandidateSidebar(QFrame):
    candidateSelected = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setObjectName("StudioCard")
        self.setFixedWidth(275)

        self.candidates = []
        self.item_widgets = []
        self.selected_idx = 0

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Header
        hdr = QHBoxLayout()
        hdr.setSpacing(6)

        lbl_icon = QLabel("📐")
        lbl_icon.setStyleSheet("font-size: 13px;")
        hdr.addWidget(lbl_icon)

        lbl_title = QLabel("Candidate Grids")
        lbl_title.setStyleSheet("font-size: 13px; font-weight: 800; color: #f8fafc;")
        hdr.addWidget(lbl_title)

        hdr.addStretch()

        self.btn_rescan = QPushButton("⚡ Rescan")
        self.btn_rescan.setProperty("class", "card-tool-btn")
        self.btn_rescan.setToolTip("Rescan image for discrete pixel grid candidates")
        self.btn_rescan.setCursor(Qt.CursorShape.PointingHandCursor)
        hdr.addWidget(self.btn_rescan)

        self.lbl_count = QLabel("0 matches")
        self.lbl_count.setStyleSheet("""
            QLabel {
                color: #94a3b8;
                background-color: #0c121e;
                border: 1px solid #1e293b;
                border-radius: 8px;
                font-size: 10px;
                font-weight: 600;
                padding: 2px 7px;
            }
        """)
        hdr.addWidget(self.lbl_count)
        layout.addLayout(hdr)

        lbl_hint = QLabel("Select candidate to preview and quantize:")
        lbl_hint.setStyleSheet("color: #64748b; font-size: 10px;")
        layout.addWidget(lbl_hint)

        # Scrollable candidates container
        self.scroll_area = QScrollArea()
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
                background-color: transparent;
            }
            QScrollArea > QWidget {
                border: none;
                background: transparent;
                background-color: transparent;
            }
            QScrollArea > QWidget > QWidget {
                border: none;
                background: transparent;
                background-color: transparent;
            }
            QScrollBar:vertical {
                width: 6px;
                background: #0b0f19;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical {
                background: #1e293b;
                border-radius: 3px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #334155;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        vp = self.scroll_area.viewport()
        if vp is not None:
            vp.setAutoFillBackground(False)
            vp.setStyleSheet("background: transparent; border: none;")

        self.list_container = QWidget()
        self.list_container.setStyleSheet("background: transparent; border: none;")
        self.list_layout = QVBoxLayout(self.list_container)
        self.list_layout.setContentsMargins(0, 0, 4, 0)
        self.list_layout.setSpacing(6)
        self.list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.scroll_area.setWidget(self.list_container)
        layout.addWidget(self.scroll_area, stretch=1)

        self.show_placeholder()

    def show_placeholder(self):
        self.clear()
        lbl_empty = QLabel("⚡ Click 'Scan Discrete Grid' below to detect candidate resolutions.")
        lbl_empty.setWordWrap(True)
        lbl_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_empty.setStyleSheet("color: #64748b; font-size: 11px; padding: 40px 10px;")
        self.list_layout.addWidget(lbl_empty)

    def clear(self):
        while self.list_layout.count():
            item = self.list_layout.takeAt(0)
            if item is not None:
                w = item.widget()
                if w is not None:
                    w.hide()
                    w.setParent(None)
                    w.deleteLater()
        self.item_widgets.clear()
        self.candidates.clear()
        self.lbl_count.setText("0 matches")

    def set_candidates(self, candidates, active_idx=0):
        self.clear()
        self.candidates = candidates
        self.selected_idx = active_idx
        self.lbl_count.setText(f"{len(candidates)} matches")

        for idx, cand in enumerate(candidates):
            item = GridCandidateItem(idx, cand, is_selected=(idx == active_idx))
            item.clicked.connect(self._on_item_clicked)
            self.list_layout.addWidget(item)
            self.item_widgets.append(item)

        self.list_layout.addStretch()

    def _on_item_clicked(self, idx):
        self.set_selected_index(idx)
        self.candidateSelected.emit(idx)

    def set_selected_index(self, idx):
        self.selected_idx = idx
        for i, item in enumerate(self.item_widgets):
            item.set_selected(i == idx)
            if i == idx:
                self.scroll_area.ensureWidgetVisible(item)


# ==============================================================================
# Interactive Color Palette Sidebar & Items
# ==============================================================================
class ColorPaletteItem(QFrame):
    clicked = pyqtSignal(str)

    def __init__(self, idx, bgr_color):
        super().__init__()
        self.idx = idx
        b, g, r = [int(v) for v in bgr_color]
        self.hex_code = f"#{r:02X}{g:02X}{b:02X}"
        self.rgb_text = f"RGB: {r}, {g}, {b}"
        self.setObjectName("PaletteCardItem")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f"Color #{idx + 1}\nHEX: {self.hex_code}\n{self.rgb_text}\nClick to copy HEX")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(10)

        # Color swatch pill / box
        swatch = QFrame()
        swatch.setFixedSize(26, 26)
        swatch.setStyleSheet(f"""
            QFrame {{
                background-color: {self.hex_code};
                border: 1px solid rgba(255, 255, 255, 0.35);
                border-radius: 6px;
            }}
        """)
        layout.addWidget(swatch)

        # Info column (HEX and RGB)
        info_col = QVBoxLayout()
        info_col.setContentsMargins(0, 0, 0, 0)
        info_col.setSpacing(1)

        lbl_hex = QLabel(self.hex_code)
        lbl_hex.setStyleSheet("color: #f8fafc; font-size: 12px; font-weight: 700; font-family: Consolas, 'Segoe UI', monospace; background: transparent; border: none;")
        info_col.addWidget(lbl_hex)

        lbl_rgb = QLabel(self.rgb_text)
        lbl_rgb.setStyleSheet("color: #94a3b8; font-size: 10px; background: transparent; border: none;")
        info_col.addWidget(lbl_rgb)
        layout.addLayout(info_col)

        layout.addStretch()

        # Index badge
        lbl_idx = QLabel(f"#{idx + 1}")
        lbl_idx.setStyleSheet("""
            color: #64748b;
            font-size: 10px;
            font-weight: 600;
            background-color: #060912;
            border: 1px solid #1e293b;
            border-radius: 4px;
            padding: 2px 6px;
        """)
        layout.addWidget(lbl_idx)

        self.setStyleSheet("""
            QFrame#PaletteCardItem {
                background-color: #111726;
                border: 1px solid #1e293b;
                border-radius: 8px;
            }
            QFrame#PaletteCardItem:hover {
                background-color: #182238;
                border: 1px solid #8b5cf6;
            }
        """)

    def mousePressEvent(self, a0):
        if a0 is not None and a0.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.hex_code)
        super().mousePressEvent(a0)


class ColorPaletteSidebar(QFrame):
    colorClicked = pyqtSignal(str)
    exportSwatchRequested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setObjectName("ColorPaletteSidebar")
        self.setFixedWidth(275)
        self.setStyleSheet("""
            QFrame#ColorPaletteSidebar {
                background-color: #0c121e;
                border: 1px solid #1e293b;
                border-radius: 12px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Header: Icon + Title + Count badge
        hdr = QHBoxLayout()
        lbl_icon = QLabel("🎨")
        lbl_icon.setStyleSheet("font-size: 14px; background: transparent; border: none;")
        hdr.addWidget(lbl_icon)

        lbl_title = QLabel("Color Palette")
        lbl_title.setStyleSheet("font-size: 13px; font-weight: 800; color: #f8fafc; background: transparent; border: none;")
        hdr.addWidget(lbl_title)

        hdr.addStretch()

        self.lbl_count = QLabel("0 colors")
        self.lbl_count.setStyleSheet("""
            QLabel {
                color: #a78bfa;
                background-color: #0c121e;
                border: 1px solid #1e293b;
                border-radius: 8px;
                font-size: 10px;
                font-weight: 600;
                padding: 2px 7px;
            }
        """)
        hdr.addWidget(self.lbl_count)
        layout.addLayout(hdr)

        # Toast / Click Hint
        self.lbl_copy_toast = QLabel("Click color card to copy HEX")
        self.lbl_copy_toast.setStyleSheet("color: #64748b; font-size: 10px; font-style: italic; background: transparent; border: none;")
        layout.addWidget(self.lbl_copy_toast)

        # Scrollable color cards list
        self.scroll_area = QScrollArea()
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
                background-color: transparent;
            }
            QScrollArea > QWidget {
                border: none;
                background: transparent;
                background-color: transparent;
            }
            QScrollArea > QWidget > QWidget {
                border: none;
                background: transparent;
                background-color: transparent;
            }
            QScrollBar:vertical {
                width: 6px;
                background: #0b0f19;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical {
                background: #1e293b;
                border-radius: 3px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #334155;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        vp = self.scroll_area.viewport()
        if vp is not None:
            vp.setAutoFillBackground(False)
            vp.setStyleSheet("background: transparent; border: none;")

        self.list_container = QWidget()
        self.list_container.setStyleSheet("background: transparent; border: none;")
        self.list_layout = QVBoxLayout(self.list_container)
        self.list_layout.setContentsMargins(0, 0, 4, 0)
        self.list_layout.setSpacing(6)
        self.list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.scroll_area.setWidget(self.list_container)
        layout.addWidget(self.scroll_area, stretch=1)

        self.show_placeholder()

    def set_palette(self, centers_bgr):
        self.clear()
        if centers_bgr is None or len(centers_bgr) == 0:
            self.show_placeholder()
            return

        k_count = len(centers_bgr)
        self.lbl_count.setText(f"{k_count} colors")
        for idx, col in enumerate(centers_bgr):
            item = ColorPaletteItem(idx, col)
            item.clicked.connect(self._handle_color_click)
            self.list_layout.addWidget(item)

        self.list_layout.addStretch()

    def show_placeholder(self):
        self.clear()
        lbl_empty = QLabel("🎨 Awaiting palette extraction.\nProceed from Step 2 to generate palette.")
        lbl_empty.setWordWrap(True)
        lbl_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_empty.setStyleSheet("color: #64748b; font-size: 11px; padding: 40px 10px; background: transparent; border: none;")
        self.list_layout.addWidget(lbl_empty)

    def clear(self):
        while self.list_layout.count():
            item = self.list_layout.takeAt(0)
            if item is not None:
                w = item.widget()
                if w is not None:
                    w.hide()
                    w.setParent(None)
                    w.deleteLater()
        self.lbl_count.setText("0 colors")

    def _handle_color_click(self, hex_code):
        cb = QApplication.clipboard()
        if cb is not None:
            cb.setText(hex_code)
        self.lbl_copy_toast.setText(f"Copied {hex_code} to clipboard!")
        self.lbl_copy_toast.setStyleSheet("color: #38bdf8; font-size: 10px; font-weight: bold; background: transparent; border: none;")
        QTimer.singleShot(2500, self._reset_toast)
        self.colorClicked.emit(hex_code)

    def _reset_toast(self):
        self.lbl_copy_toast.setText("Click color card to copy HEX")
        self.lbl_copy_toast.setStyleSheet("color: #64748b; font-size: 10px; font-style: italic; background: transparent; border: none;")


PaletteBar = ColorPaletteSidebar


# ==============================================================================
# Step Navigation Stepper Header Bar
# ==============================================================================
class StudioStepper(QFrame):
    stepClicked = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setObjectName("StudioStepper")
        self.setStyleSheet("""
            QFrame#StudioStepper {
                background-color: #111726;
                border: 1px solid #1e293b;
                border-radius: 12px;
                padding: 4px 12px;
            }
            QPushButton.step-btn {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 8px;
                padding: 6px 14px;
                color: #64748b;
                font-weight: 700;
                font-size: 12px;
            }
            QPushButton.step-btn:hover {
                color: #cbd5e1;
                background-color: #1a2336;
            }
            QPushButton.step-btn-active {
                background-color: #1e293b;
                border: 1px solid #38bdf8;
                color: #f8fafc;
                font-weight: 800;
                font-size: 12px;
                border-radius: 8px;
                padding: 6px 14px;
            }
            QPushButton.step-btn-completed {
                background: transparent;
                color: #38bdf8;
                border: 1px solid #1e293b;
                border-radius: 8px;
                padding: 6px 14px;
                font-weight: 700;
                font-size: 12px;
            }
            QPushButton.step-btn-completed:hover {
                background-color: #162235;
                border: 1px solid #38bdf8;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)

        self.btn_step1 = QPushButton("01 · Source & Mode")
        self.btn_step1.setProperty("class", "step-btn-active")
        self.btn_step1.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_step1.clicked.connect(lambda: self.stepClicked.emit(0))
        layout.addWidget(self.btn_step1)

        arrow1 = QLabel("→")
        arrow1.setStyleSheet("color: #334155; font-size: 13px; font-weight: bold;")
        layout.addWidget(arrow1)

        self.btn_step2 = QPushButton("02 · Pixel Grid Processing")
        self.btn_step2.setProperty("class", "step-btn")
        self.btn_step2.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_step2.setEnabled(False)
        self.btn_step2.clicked.connect(lambda: self.stepClicked.emit(1))
        layout.addWidget(self.btn_step2)

        arrow2 = QLabel("→")
        arrow2.setStyleSheet("color: #334155; font-size: 13px; font-weight: bold;")
        layout.addWidget(arrow2)

        self.btn_step3 = QPushButton("03 · Palette & Export Studio")
        self.btn_step3.setProperty("class", "step-btn")
        self.btn_step3.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_step3.setEnabled(False)
        self.btn_step3.clicked.connect(lambda: self.stepClicked.emit(2))
        layout.addWidget(self.btn_step3)

        layout.addStretch()

        self.lbl_mode_chip = QLabel("Mode: Detect Pixel Art")
        self.lbl_mode_chip.setStyleSheet("""
            QLabel {
                background-color: #0c121e;
                color: #0ea5e9;
                border: 1px solid #1e293b;
                border-radius: 8px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 700;
            }
        """)
        layout.addWidget(self.lbl_mode_chip)

    def set_active_step(self, step_idx, step2_unlocked=False, step3_unlocked=False):
        self.btn_step2.setEnabled(step2_unlocked)
        self.btn_step3.setEnabled(step3_unlocked)

        buttons = [self.btn_step1, self.btn_step2, self.btn_step3]
        for idx, btn in enumerate(buttons):
            if idx == step_idx:
                btn.setStyleSheet("""
                    background-color: #1e293b;
                    border: 1px solid #38bdf8;
                    color: #ffffff;
                    font-weight: 800;
                    border-radius: 8px;
                    padding: 6px 14px;
                """)
            elif (idx == 0 and step_idx > 0) or (idx == 1 and step_idx > 1):
                btn.setStyleSheet("""
                    background-color: transparent;
                    color: #38bdf8;
                    border: 1px solid #1e293b;
                    border-radius: 8px;
                    padding: 6px 14px;
                    font-weight: 700;
                """)
            else:
                btn.setStyleSheet("""
                    background-color: transparent;
                    border: 1px solid transparent;
                    color: #64748b;
                    border-radius: 8px;
                    padding: 6px 14px;
                    font-weight: 700;
                """)

    def set_mode_chip(self, mode_name, color="#0ea5e9"):
        self.lbl_mode_chip.setText(f"Mode: {mode_name}")
        self.lbl_mode_chip.setStyleSheet(f"""
            QLabel {{
                background-color: #0c121e;
                color: {color};
                border: 1px solid #1e293b;
                border-radius: 8px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 700;
            }}
        """)


# ==============================================================================
# Main TelaFormer Window
# ==============================================================================
class TelaFormerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TelaFormer Pro — Pixel Art Digitizer Studio")
        self.setMinimumSize(980, 750)
        self.resize(1140, 860)
        self.setAcceptDrops(True)

        self.current_mode = MODE_DETECT

        self.original_path = None
        self.original_image_bgr = None
        self.downscaled_bgr = None
        self.quantized_bgr = None
        self.palette_centers = None
        self.grid_dims = None
        self.scale_factor = None
        self.detected_candidates = []
        self.selected_candidate_idx = 0

        # Pixelate mode aspect ratio tracking
        self.lock_aspect_ratio = True
        self.is_updating_aspect = False

        self.apply_theme()
        self.init_ui()

    def apply_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0b0f19;
                color: #f1f5f9;
                font-family: 'Segoe UI', -apple-system, 'SF Pro Display', Roboto, sans-serif;
            }
            QFrame#StudioCard {
                background-color: #111726;
                border: 1px solid #1e293b;
                border-radius: 12px;
            }
            QFrame#StudioCard:hover {
                border: 1px solid #2d3b55;
            }
            QLabel {
                color: #e2e8f0;
            }

            /* Action Buttons */
            QPushButton.btn-primary {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3b82f6, stop:1 #2563eb);
                color: #ffffff;
                border: 1px solid #60a5fa;
                border-radius: 8px;
                padding: 10px 18px;
                font-weight: 700;
                font-size: 13px;
            }
            QPushButton.btn-primary:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #60a5fa, stop:1 #3b82f6);
            }
            QPushButton.btn-primary:disabled {
                background-color: #1e293b;
                border: 1px solid #334155;
                color: #64748b;
            }

            QPushButton.btn-cyan {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0ea5e9, stop:1 #0284c7);
                color: #ffffff;
                border: 1px solid #38bdf8;
                border-radius: 8px;
                padding: 10px 18px;
                font-weight: 700;
                font-size: 13px;
            }
            QPushButton.btn-cyan:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #38bdf8, stop:1 #0ea5e9);
            }
            QPushButton.btn-cyan:disabled {
                background-color: #1e293b;
                border: 1px solid #334155;
                color: #64748b;
            }

            QPushButton.btn-purple {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #8b5cf6, stop:1 #7c3aed);
                color: #ffffff;
                border: 1px solid #a78bfa;
                border-radius: 8px;
                padding: 10px 18px;
                font-weight: 700;
                font-size: 13px;
            }
            QPushButton.btn-purple:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #a78bfa, stop:1 #8b5cf6);
            }
            QPushButton.btn-purple:disabled {
                background-color: #1e293b;
                border: 1px solid #334155;
                color: #64748b;
            }

            QPushButton.btn-secondary {
                background-color: #1e293b;
                color: #cbd5e1;
                border: 1px solid #334155;
                border-radius: 7px;
                padding: 8px 14px;
                font-weight: 600;
                font-size: 12px;
            }
            QPushButton.btn-secondary:hover {
                background-color: #334155;
                color: #ffffff;
                border: 1px solid #475569;
            }
            QPushButton.btn-secondary:disabled {
                background-color: #131b2a;
                border: 1px solid #202b3c;
                color: #475569;
            }

            /* Canvas Header Tool Buttons */
            QPushButton.card-tool-btn {
                background-color: #1e293b;
                color: #cbd5e1;
                border: 1px solid #334155;
                border-radius: 7px;
                font-size: 11px;
                font-weight: 600;
                padding: 4px 10px;
            }
            QPushButton.card-tool-btn:hover {
                background-color: #2b3952;
                color: #ffffff;
                border: 1px solid #475569;
            }
            QPushButton.card-tool-btn:checked {
                background-color: #0284c7;
                color: #ffffff;
                border: 1px solid #38bdf8;
            }
            QPushButton.card-tool-btn:disabled {
                background-color: #0c121e;
                color: #475569;
                border: 1px solid #1a2333;
            }

            /* Mode Choice Card Buttons */
            QFrame.mode-choice-card {
                background-color: #0c121e;
                border: 1px solid #1e293b;
                border-radius: 10px;
                padding: 10px;
            }
            QFrame.mode-choice-card:hover {
                border: 1px solid #38bdf8;
                background-color: #0e1626;
            }

            /* Preset Chips */
            QPushButton.preset-chip {
                background-color: #162032;
                color: #94a3b8;
                border: 1px solid #26354d;
                border-radius: 5px;
                font-weight: 600;
                font-size: 11px;
                padding: 4px 8px;
                min-width: 24px;
            }
            QPushButton.preset-chip:hover {
                background-color: #24344d;
                color: #f8fafc;
                border: 1px solid #38bdf8;
            }

            /* SpinBoxes & Inputs */
            QSpinBox {
                background-color: #101726;
                color: #f8fafc;
                padding: 4px 22px 4px 8px;
                border: 1px solid #2b3952;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
                min-width: 60px;
            }
            QSpinBox:focus {
                border: 1px solid #38bdf8;
            }
            QSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 18px;
                border-left: 1px solid #1e293b;
                border-top-right-radius: 6px;
            }
            QSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 18px;
                border-left: 1px solid #1e293b;
                border-bottom-right-radius: 6px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #1e293b;
            }

            QComboBox {
                background-color: #101726;
                color: #f8fafc;
                padding: 5px 10px;
                border: 1px solid #2b3952;
                border-radius: 6px;
                font-size: 11px;
                font-weight: 600;
            }
            QComboBox QAbstractItemView {
                background-color: #161f30;
                color: #f1f5f9;
                selection-background-color: #2563eb;
                border: 1px solid #2b3952;
            }

            QSlider::groove:horizontal {
                height: 5px;
                background: #1e293b;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8b5cf6, stop:1 #38bdf8);
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                border: 2px solid #8b5cf6;
                width: 14px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #f8fafc;
                border: 2px solid #a78bfa;
            }

            QProgressBar {
                background-color: #0c121e;
                border: 1px solid #1e293b;
                border-radius: 4px;
                height: 6px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0ea5e9, stop:1 #8b5cf6);
                border-radius: 3px;
            }

            QToolTip {
                background-color: #1e293b;
                color: #f8fafc;
                border: 1px solid #38bdf8;
                padding: 6px 10px;
                border-radius: 6px;
                font-size: 11px;
            }
            QMenu {
                background-color: #161f30;
                color: #f1f5f9;
                border: 1px solid #2b3952;
                border-radius: 8px;
                padding: 6px;
            }
            QMenu::item {
                padding: 6px 20px;
                border-radius: 4px;
                font-size: 12px;
            }
            QMenu::item:selected {
                background-color: #2563eb;
                color: #ffffff;
            }
        """)

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(16, 14, 16, 14)
        main_layout.setSpacing(12)

        # ======================================================================
        # Top Stepper Breadcrumbs Bar
        # ======================================================================
        self.stepper = StudioStepper()
        self.stepper.stepClicked.connect(self.navigate_to_step)
        main_layout.addWidget(self.stepper)

        # ======================================================================
        # Sliding Stacked Workspace Container
        # ======================================================================
        self.stack = SlidingStackedWidget()

        # Build the 3 distinct modal sliding pages
        self.page1 = self._build_page1_source_and_mode()
        self.page2 = self._build_page2_grid_processing()
        self.page3 = self._build_page3_palette_and_export()

        self.stack.addWidget(self.page1)
        self.stack.addWidget(self.page2)
        self.stack.addWidget(self.page3)

        main_layout.addWidget(self.stack, stretch=1)

        # ======================================================================
        # Bottom Status Bar
        # ======================================================================
        status_frame = QFrame()
        status_frame.setStyleSheet("""
            QFrame {
                background-color: #0c121e;
                border: 1px solid #1e293b;
                border-radius: 8px;
                padding: 4px 10px;
            }
        """)
        s_layout = QHBoxLayout(status_frame)
        s_layout.setContentsMargins(4, 2, 4, 2)
        s_layout.setSpacing(10)

        self.lbl_status_dot = QLabel("●")
        self.lbl_status_dot.setStyleSheet("color: #10b981; font-size: 13px;")
        s_layout.addWidget(self.lbl_status_dot)

        self.lbl_status_msg = QLabel("Ready — Drop or select an image to choose workflow mode")
        self.lbl_status_msg.setStyleSheet("color: #94a3b8; font-size: 12px;")
        s_layout.addWidget(self.lbl_status_msg)

        s_layout.addStretch()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFixedWidth(160)
        self.progress_bar.setVisible(False)
        s_layout.addWidget(self.progress_bar)

        self.lbl_nav_hint = QLabel("Step 1 of 3 · Select image and workflow mode")
        self.lbl_nav_hint.setStyleSheet("color: #64748b; font-size: 11px;")
        s_layout.addWidget(self.lbl_nav_hint)

        main_layout.addWidget(status_frame)
        self.setCentralWidget(main_widget)

    # ==========================================================================
    # Step 1: Source & Mode Selection Page
    # ==========================================================================
    def _build_page1_source_and_mode(self):
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(12)

        # Top: Large Canvas Stage
        self.p1_card_canvas = StudioCard("01", "Source Image Canvas", accent_color="#38bdf8", has_grid_toggle=False)
        self.p1_card_canvas.canvas.placeholder_title = "Drag & Drop Image Here"
        self.p1_card_canvas.canvas.placeholder_subtitle = "Supports PNG, JPG, BMP, WEBP"

        # Action toolbar right beneath the canvas inside its card
        p1_actions_row = QHBoxLayout()
        self.btn_load_source = QPushButton("📂 Select Image...")
        self.btn_load_source.setProperty("class", "btn-primary")
        self.btn_load_source.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_load_source.clicked.connect(self.load_image)
        p1_actions_row.addWidget(self.btn_load_source)

        p1_actions_row.addSpacing(10)

        self.lbl_p1_res = QLabel("Resolution: —")
        self.lbl_p1_res.setStyleSheet("""
            color: #cbd5e1;
            font-size: 11px;
            font-weight: bold;
            background-color: #0c121e;
            border: 1px solid #1e293b;
            border-radius: 6px;
            padding: 5px 12px;
        """)
        p1_actions_row.addWidget(self.lbl_p1_res)

        self.lbl_p1_file = QLabel("File: No image selected")
        self.lbl_p1_file.setStyleSheet("""
            color: #64748b;
            font-size: 11px;
            background-color: #0c121e;
            border: 1px solid #1e293b;
            border-radius: 6px;
            padding: 5px 12px;
        """)
        p1_actions_row.addWidget(self.lbl_p1_file)

        p1_actions_row.addStretch()
        self.p1_card_canvas.controls_layout.addLayout(p1_actions_row)

        page_layout.addWidget(self.p1_card_canvas, stretch=1)

        # Bottom: Mode Selection Horizontal Deck
        p1_mode_card = QFrame()
        p1_mode_card.setObjectName("StudioCard")
        mode_layout = QVBoxLayout(p1_mode_card)
        mode_layout.setContentsMargins(14, 12, 14, 12)
        mode_layout.setSpacing(10)

        m_hdr = QHBoxLayout()
        lbl_p1_title = QLabel("Choose Workflow Mode:")
        lbl_p1_title.setStyleSheet("color: #f1f5f9; font-size: 13px; font-weight: 800;")
        m_hdr.addWidget(lbl_p1_title)

        lbl_p1_sub = QLabel("Select an image above, then pick the processing pipeline to enter the studio.")
        lbl_p1_sub.setStyleSheet("color: #94a3b8; font-size: 11px;")
        m_hdr.addWidget(lbl_p1_sub)
        m_hdr.addStretch()
        mode_layout.addLayout(m_hdr)

        # 3 Mode Cards side-by-side in a balanced horizontal deck
        cards_row = QHBoxLayout()
        cards_row.setSpacing(12)

        # Mode 1 Card: Detect Pixel Art
        self.card_mode_detect = self._create_mode_card(
            icon="⚡",
            title="Detect True Pixel Art",
            desc="Scans upscaled screenshot/art to recover hidden native pixel grid and integer downscale ratio automatically.",
            btn_text="Proceed with Grid Detection →",
            callback=lambda: self.select_mode_and_proceed(MODE_DETECT)
        )
        cards_row.addWidget(self.card_mode_detect)

        # Mode 2 Card: Pixelate
        self.card_mode_pixelate = self._create_mode_card(
            icon="🔲",
            title="Pixelate Photo / Art",
            desc="Downscales regular high-res photos or digital paintings into a customizable retro pixel grid.",
            btn_text="Configure Pixelation Grid →",
            callback=lambda: self.select_mode_and_proceed(MODE_PIXELATE)
        )
        cards_row.addWidget(self.card_mode_pixelate)

        # Mode 3 Card: Extract Colors
        self.card_mode_extract = self._create_mode_card(
            icon="🎯",
            title="Extract Colors (1:1 Native)",
            desc="Image is already 1:1 pixel art. Skips downscaling and extracts or manages CIELAB palette directly.",
            btn_text="Extract CIELAB Palette Directly →",
            callback=lambda: self.select_mode_and_proceed(MODE_EXTRACT)
        )
        cards_row.addWidget(self.card_mode_extract)

        mode_layout.addLayout(cards_row)
        page_layout.addWidget(p1_mode_card, stretch=0)
        return page

    def _create_mode_card(self, icon, title, desc, btn_text, callback):
        frame = QFrame()
        frame.setProperty("class", "mode-choice-card")
        f_layout = QVBoxLayout(frame)
        f_layout.setContentsMargins(14, 12, 14, 12)
        f_layout.setSpacing(8)

        hdr = QHBoxLayout()
        lbl_ico = QLabel(icon)
        lbl_ico.setStyleSheet("font-size: 16px;")
        hdr.addWidget(lbl_ico)

        lbl_t = QLabel(title)
        lbl_t.setStyleSheet("color: #f1f5f9; font-size: 13px; font-weight: 700;")
        hdr.addWidget(lbl_t)
        hdr.addStretch()
        f_layout.addLayout(hdr)

        lbl_d = QLabel(desc)
        lbl_d.setStyleSheet("color: #94a3b8; font-size: 11px; line-height: 1.4;")
        lbl_d.setWordWrap(True)
        lbl_d.setMinimumHeight(32)
        f_layout.addWidget(lbl_d)

        btn = QPushButton(btn_text)
        btn.setProperty("class", "btn-secondary")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet("""
            QPushButton {
                background-color: #141c2c;
                color: #cbd5e1;
                border: 1px solid #28354b;
                border-radius: 7px;
                padding: 8px 12px;
                font-weight: 600;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #1e2b42;
                color: #ffffff;
                border: 1px solid #38bdf8;
            }
        """)
        btn.clicked.connect(callback)
        f_layout.addWidget(btn)

        return frame

    # ==========================================================================
    # Step 2: Pixel Grid Processing Page (Vertical Layout)
    # ==========================================================================
    def _build_page2_grid_processing(self):
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(12)

        # Stage Row: Grid Canvas (left, stretch=1) + Candidate Sidebar (right, stretch=0)
        p2_stage_layout = QHBoxLayout()
        p2_stage_layout.setContentsMargins(0, 0, 0, 0)
        p2_stage_layout.setSpacing(12)

        # Left: Grid Canvas Stage
        self.p2_card_canvas = StudioCard("02", "Pixel Grid Canvas", accent_color="#0ea5e9", has_grid_toggle=True)
        self.p2_card_canvas.canvas.placeholder_title = "Awaiting Grid Processing"

        # Canvas actions in header alongside # Grid
        self.btn_p2_copy = QPushButton("📋 Copy")
        self.btn_p2_copy.setProperty("class", "card-tool-btn")
        self.btn_p2_copy.setToolTip("Copy current pixel grid to clipboard")
        self.btn_p2_copy.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_p2_copy.setEnabled(False)
        self.btn_p2_copy.clicked.connect(lambda: self._copy_canvas(self.p2_card_canvas.canvas))
        self.p2_card_canvas.add_header_action(self.btn_p2_copy)

        self.btn_save_grid = QPushButton("💾 Save PNG")
        self.btn_save_grid.setProperty("class", "card-tool-btn")
        self.btn_save_grid.setToolTip("Export raw pixel grid as 1:1 PNG")
        self.btn_save_grid.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save_grid.setEnabled(False)
        self.btn_save_grid.clicked.connect(self.save_grid_image)
        self.p2_card_canvas.add_header_action(self.btn_save_grid)

        p2_stage_layout.addWidget(self.p2_card_canvas, stretch=1)

        # Right: Candidate Grid Dimensions Sidebar (in True Pixel / Detect mode)
        self.p2_sidebar = GridCandidateSidebar()
        self.p2_sidebar.candidateSelected.connect(self.on_candidate_chosen)
        self.btn_estimate = self.p2_sidebar.btn_rescan
        self.btn_estimate.clicked.connect(self.run_estimate)
        p2_stage_layout.addWidget(self.p2_sidebar, stretch=0)

        page_layout.addLayout(p2_stage_layout, stretch=1)

        # Bottom: Mode Controls & Next Action Dock
        p2_ctrl_card = QFrame()
        p2_ctrl_card.setObjectName("StudioCard")
        ctrl_layout = QHBoxLayout(p2_ctrl_card)
        ctrl_layout.setContentsMargins(14, 10, 14, 10)
        ctrl_layout.setSpacing(12)

        # Mode Specific Configuration Stack (Only needed for Pixelate mode)
        self.p2_mode_controls_stack = QStackedWidget()

        # --- Pixelate Photo/Art Controls ---
        p_pixelate = QWidget()
        l_pixelate = QHBoxLayout(p_pixelate)
        l_pixelate.setContentsMargins(0, 0, 0, 0)
        l_pixelate.setSpacing(8)

        lbl_pix_title = QLabel("Grid:")
        lbl_pix_title.setStyleSheet("color: #f1f5f9; font-size: 12px; font-weight: bold;")
        l_pixelate.addWidget(lbl_pix_title)

        lbl_w = QLabel("W:")
        lbl_w.setStyleSheet("color: #94a3b8; font-size: 11px;")
        l_pixelate.addWidget(lbl_w)

        self.spin_pix_w = QSpinBox()
        self.spin_pix_w.setRange(8, 2048)
        self.spin_pix_w.setValue(128)
        self.spin_pix_w.setFixedWidth(58)
        self.spin_pix_w.valueChanged.connect(self._on_pix_w_changed)
        l_pixelate.addWidget(self.spin_pix_w)

        self.btn_aspect_lock = QPushButton("🔗")
        self.btn_aspect_lock.setCheckable(True)
        self.btn_aspect_lock.setChecked(True)
        self.btn_aspect_lock.setFixedSize(24, 24)
        self.btn_aspect_lock.setToolTip("Lock Aspect Ratio")
        self.btn_aspect_lock.setStyleSheet("""
            QPushButton {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 5px;
                font-size: 11px;
            }
            QPushButton:checked {
                background-color: #0284c7;
                border: 1px solid #38bdf8;
            }
        """)
        self.btn_aspect_lock.toggled.connect(self._toggle_aspect_lock)
        l_pixelate.addWidget(self.btn_aspect_lock)

        lbl_h = QLabel("H:")
        lbl_h.setStyleSheet("color: #94a3b8; font-size: 11px;")
        l_pixelate.addWidget(lbl_h)

        self.spin_pix_h = QSpinBox()
        self.spin_pix_h.setRange(8, 2048)
        self.spin_pix_h.setValue(128)
        self.spin_pix_h.setFixedWidth(58)
        self.spin_pix_h.valueChanged.connect(self._on_pix_h_changed)
        l_pixelate.addWidget(self.spin_pix_h)

        l_pixelate.addSpacing(4)
        lbl_pres = QLabel("Presets:")
        lbl_pres.setStyleSheet("color: #64748b; font-size: 11px;")
        l_pixelate.addWidget(lbl_pres)

        for p_res in [32, 64, 128, 160, 256]:
            p_btn = QPushButton(str(p_res))
            p_btn.setProperty("class", "preset-chip")
            p_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            p_btn.clicked.connect(lambda _, r=p_res: self._apply_pixel_preset(r))
            l_pixelate.addWidget(p_btn)

        l_pixelate.addSpacing(4)
        lbl_filt = QLabel("Filter:")
        lbl_filt.setStyleSheet("color: #64748b; font-size: 11px;")
        l_pixelate.addWidget(lbl_filt)

        self.combo_pix_interp = QComboBox()
        self.combo_pix_interp.addItem("Area")
        self.combo_pix_interp.addItem("Nearest")
        self.combo_pix_interp.setToolTip("Interpolation filter: Area for smooth downsampling, Nearest for crisp pixel sampling")
        self.combo_pix_interp.setFixedWidth(85)
        self.combo_pix_interp.currentIndexChanged.connect(lambda: self.apply_pixelate())
        l_pixelate.addWidget(self.combo_pix_interp)

        self.btn_apply_pixelate = QPushButton("🔲 Apply")
        self.btn_apply_pixelate.setProperty("class", "btn-cyan")
        self.btn_apply_pixelate.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_apply_pixelate.clicked.connect(self.apply_pixelate)
        l_pixelate.addWidget(self.btn_apply_pixelate)
        l_pixelate.addStretch()

        self.p2_mode_controls_stack.addWidget(p_pixelate)

        # By default in Detect mode, hide controls stack so Proceed button takes 100% full width
        self.p2_mode_controls_stack.setVisible(False)
        ctrl_layout.addWidget(self.p2_mode_controls_stack, stretch=1)

        self.btn_p2_proceed = QPushButton("Proceed to Palette Extraction →")
        self.btn_p2_proceed.setProperty("class", "btn-purple")
        self.btn_p2_proceed.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_p2_proceed.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn_p2_proceed.setMinimumHeight(44)
        self.btn_p2_proceed.setEnabled(False)
        self.btn_p2_proceed.clicked.connect(lambda: self.navigate_to_step(2))
        ctrl_layout.addWidget(self.btn_p2_proceed, stretch=0)

        page_layout.addWidget(p2_ctrl_card, stretch=0)
        return page

    # ==========================================================================
    # Step 3: CIELAB Palette & Export Studio Page (Vertical Layout)
    # ==========================================================================
    def _build_page3_palette_and_export(self):
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(12)

        # Stage Row: Quantized Canvas (left, stretch=1) + Color Palette Sidebar (right, stretch=0)
        p3_stage_layout = QHBoxLayout()
        p3_stage_layout.setContentsMargins(0, 0, 0, 0)
        p3_stage_layout.setSpacing(12)

        # Left: Quantized Art Canvas Stage
        self.p3_card_canvas = StudioCard("03", "Quantized Pixel Art Canvas", accent_color="#8b5cf6", has_grid_toggle=True)
        self.p3_card_canvas.canvas.placeholder_title = "Awaiting Palette Extraction"
        self.p3_card_canvas.canvas.placeholder_subtitle = "Extract CIELAB palette to preview quantized result"

        self.btn_compare_toggle = QPushButton("👁 Compare")
        self.btn_compare_toggle.setProperty("class", "card-tool-btn")
        self.btn_compare_toggle.setCheckable(True)
        self.btn_compare_toggle.setToolTip("Hold or toggle to compare against raw grid")
        self.btn_compare_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_compare_toggle.toggled.connect(self._toggle_compare_view)
        self.p3_card_canvas.add_header_action(self.btn_compare_toggle)

        self.btn_copy_p3 = QPushButton("📋 Copy")
        self.btn_copy_p3.setProperty("class", "card-tool-btn")
        self.btn_copy_p3.setToolTip("Copy quantized art to clipboard")
        self.btn_copy_p3.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_copy_p3.setEnabled(False)
        self.btn_copy_p3.clicked.connect(lambda: self._copy_canvas(self.p3_card_canvas.canvas))
        self.p3_card_canvas.add_header_action(self.btn_copy_p3)

        p3_stage_layout.addWidget(self.p3_card_canvas, stretch=1)

        # Right: Extracted Color Palette Sidebar (Color list only)
        self.p3_sidebar = ColorPaletteSidebar()
        self.palette_bar = self.p3_sidebar
        self.p3_sidebar.exportSwatchRequested.connect(self.export_palette_swatch)

        p3_stage_layout.addWidget(self.p3_sidebar, stretch=0)
        page_layout.addLayout(p3_stage_layout, stretch=1)

        # Bottom: K Selector & Export Studio Action Dock
        p3_ctrl_card = QFrame()
        p3_ctrl_card.setObjectName("StudioCard")
        ctrl_layout = QHBoxLayout(p3_ctrl_card)
        ctrl_layout.setContentsMargins(14, 10, 14, 10)
        ctrl_layout.setSpacing(12)

        lbl_k = QLabel("Colors (K):")
        lbl_k.setStyleSheet("color: #f1f5f9; font-size: 12px; font-weight: bold;")
        ctrl_layout.addWidget(lbl_k)

        self.spin_k = QSpinBox()
        self.spin_k.setRange(2, 64)
        self.spin_k.setValue(10)
        self.spin_k.setFixedWidth(56)
        ctrl_layout.addWidget(self.spin_k)

        self.slider_k = QSlider(Qt.Orientation.Horizontal)
        self.slider_k.setRange(2, 64)
        self.slider_k.setValue(10)
        self.slider_k.setFixedWidth(130)
        self.slider_k.valueChanged.connect(self._on_k_value_changed)
        self.spin_k.valueChanged.connect(self._on_k_value_changed)
        ctrl_layout.addWidget(self.slider_k)

        self.lbl_p3_k_badge = QLabel("K = 10")
        self.lbl_p3_k_badge.setStyleSheet("""
            background-color: #0c121e;
            color: #8b5cf6;
            border: 1px solid #1e293b;
            border-radius: 6px;
            padding: 4px 8px;
            font-size: 11px;
            font-weight: 700;
        """)
        ctrl_layout.addWidget(self.lbl_p3_k_badge)

        self.btn_quantize = QPushButton("🎨 Re-Extract Palette")
        self.btn_quantize.setProperty("class", "btn-purple")
        self.btn_quantize.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_quantize.setEnabled(False)
        self.btn_quantize.clicked.connect(self.run_quantize)
        ctrl_layout.addWidget(self.btn_quantize)

        ctrl_layout.addStretch()

        self.btn_export_palette = QPushButton("🎨 Save Swatch Sheet")
        self.btn_export_palette.setProperty("class", "btn-secondary")
        self.btn_export_palette.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_export_palette.setEnabled(False)
        self.btn_export_palette.clicked.connect(self.export_palette_swatch)
        ctrl_layout.addWidget(self.btn_export_palette)

        self.btn_save_menu = QPushButton("💾 Save Pixel Art...")
        self.btn_save_menu.setProperty("class", "btn-primary")
        self.btn_save_menu.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save_menu.setEnabled(False)
        self.btn_save_menu.clicked.connect(self.show_export_menu)
        ctrl_layout.addWidget(self.btn_save_menu)

        page_layout.addWidget(p3_ctrl_card, stretch=0)
        return page

    # ==========================================================================
    # Navigation & Step Transition Logic
    # ==========================================================================
    def navigate_to_step(self, step_idx):
        step2_ok = self.original_image_bgr is not None
        step3_ok = self.downscaled_bgr is not None

        if step_idx == 1 and not step2_ok:
            self.set_status("Please select an image before proceeding to Grid Processing.", "#ef4444")
            return
        if step_idx == 2 and not step3_ok:
            self.set_status("Please process the grid before proceeding to Palette Extraction.", "#ef4444")
            return

        self.stepper.set_active_step(step_idx, step2_unlocked=step2_ok, step3_unlocked=step3_ok)
        if step_idx == 1:
            self.p2_sidebar.setVisible(self.current_mode == MODE_DETECT)
            self.p2_mode_controls_stack.setVisible(self.current_mode == MODE_PIXELATE)
        elif step_idx == 2:
            if self.current_mode == MODE_PIXELATE:
                cur_w, cur_h = self.spin_pix_w.value(), self.spin_pix_h.value()
                if self.grid_dims != (cur_w, cur_h) or self.downscaled_bgr is None:
                    self.apply_pixelate()
            if self.downscaled_bgr is not None and self.quantized_bgr is None:
                self.run_quantize()
        self.stack.slide_to_index(step_idx)

        hints = [
            "Step 1 of 3 · Select image and workflow mode",
            "Step 2 of 3 · Tune and confirm pixel grid dimensions",
            "Step 3 of 3 · Extract CIELAB palette and export pixel art"
        ]
        self.lbl_nav_hint.setText(hints[step_idx])

    def select_mode_and_proceed(self, mode):
        if self.original_image_bgr is None:
            self.set_status("Please select an image first (click 'Select Image...' or drop file).", "#ef4444")
            return

        self.current_mode = mode
        self._invalidate_quantization()
        h, w = self.original_image_bgr.shape[:2]

        if mode == MODE_DETECT:
            self.p2_sidebar.setVisible(True)
            self.p2_mode_controls_stack.setVisible(False)
            self.stepper.set_mode_chip("Detect Pixel Art", "#0ea5e9")
            self.btn_estimate.setEnabled(True)
            self.navigate_to_step(1)
            # Auto-run estimation if no candidates detected yet or downscaled_bgr is missing
            if not self.detected_candidates or self.downscaled_bgr is None:
                self.run_estimate()
            elif self.selected_candidate_idx is not None and 0 <= self.selected_candidate_idx < len(self.detected_candidates):
                self.on_candidate_chosen(self.selected_candidate_idx)

        elif mode == MODE_PIXELATE:
            self.detected_candidates = []
            self.p2_sidebar.setVisible(False)
            self.p2_mode_controls_stack.setVisible(True)
            self.p2_mode_controls_stack.setCurrentIndex(0)
            self.stepper.set_mode_chip("Pixelate Photo/Art", "#6366f1")
            self._init_pixelate_dims(w, h)
            self.navigate_to_step(1)
            self.apply_pixelate()

        elif mode == MODE_EXTRACT:
            self.p2_sidebar.setVisible(False)
            self.p2_mode_controls_stack.setVisible(False)
            self.stepper.set_mode_chip("Extract Colors (1:1 Direct)", "#8b5cf6")
            self.navigate_to_step(1)
            self.apply_direct_extract_mode()

    # ==========================================================================
    # Drag & Drop Support
    # ==========================================================================
    def dragEnterEvent(self, a0):
        event = a0
        if event is not None and event.mimeData() is not None:
            if event.mimeData().hasUrls():
                event.acceptProposedAction()

    def dropEvent(self, a0):
        event = a0
        if event is not None and event.mimeData() is not None:
            urls = event.mimeData().urls()
            if urls:
                path = urls[0].toLocalFile()
                if path and Path(path).suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
                    self.load_image_from_path(path)

    # ==========================================================================
    # Image Loading
    # ==========================================================================
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*.*)"
        )
        if path:
            self.load_image_from_path(path)

    def load_sample_image(self):
        sample_path = Path(__file__).resolve().parent / "eldenring.jpg"
        if sample_path.exists():
            self.load_image_from_path(str(sample_path))

    def load_image_from_path(self, path):
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "Load Error", f"Unable to read image from '{path}'")
            return

        self.original_path = path
        self.original_image_bgr = img
        h, w = img.shape[:2]
        file_size_kb = Path(path).stat().st_size / 1024

        self.p1_card_canvas.canvas.set_image(img, grid_dims=(w, h))
        self.p1_card_canvas.set_badge(f"{w} × {h} px", is_active=True)
        self.lbl_p1_res.setText(f"Resolution: {w} × {h} px")
        self.lbl_p1_file.setText(f"{Path(path).name} ({file_size_kb:.0f} KB)")

        # Reset downstream
        self.downscaled_bgr = None
        self.quantized_bgr = None
        self.palette_centers = None
        self.grid_dims = None
        self.detected_candidates = []
        self.selected_candidate_idx = 0
        if hasattr(self, 'p2_sidebar'):
            self.p2_sidebar.show_placeholder()

        self.p2_card_canvas.canvas.set_image(None)
        self.p2_card_canvas.set_badge("Idle", is_active=False)
        self.btn_save_grid.setEnabled(False)
        self.btn_p2_copy.setEnabled(False)
        self.btn_p2_proceed.setEnabled(False)

        self._invalidate_quantization()
        self.btn_quantize.setEnabled(False)

        self.stepper.set_active_step(0, step2_unlocked=True, step3_unlocked=False)
        self.set_status(f"Loaded '{Path(path).name}'. Choose a workflow mode to continue.", "#38bdf8")

    # ==========================================================================
    # Mode 1: Grid Estimation Logic
    # ==========================================================================
    def run_estimate(self):
        if not self.original_path:
            return

        self.btn_estimate.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.set_status("Scanning image with discrete aspect ratio search...", "#0ea5e9", busy=True)
        self.p2_card_canvas.set_badge("Analyzing...", is_active=True)

        self.worker_thread = ProcessThread(task_type="estimate", image_path=self.original_path)
        self.worker_thread.finished.connect(self.on_estimate_done)
        self.worker_thread.error.connect(self.on_estimate_error)
        self.worker_thread.start()

    def on_estimate_done(self, best_result):
        self.progress_bar.setVisible(False)
        self.btn_estimate.setEnabled(True)

        candidates = best_result.get("candidates", [best_result])
        if self.original_image_bgr is not None:
            orig_h, orig_w = self.original_image_bgr.shape[:2]
            # Ensure candidates are strictly downscaled pixel grids (never 1.00x original image)
            valid = [
                c for c in candidates
                if c.get("scale", 1.0) < 0.95 and c.get("dim") != (orig_w, orig_h)
            ]
            if valid:
                candidates = valid
            else:
                candidates = [
                    {"dim": (max(4, int(round(orig_w / f))), max(4, int(round(orig_h / f)))), "scale": 1.0 / f, "score": 0.0}
                    for f in [2, 3, 4]
                ]

        self.detected_candidates = candidates
        self.p2_sidebar.set_candidates(candidates, active_idx=0)

        # Apply top candidate #0
        self.on_candidate_chosen(0)
        self.stepper.set_active_step(1, step2_unlocked=True, step3_unlocked=True)

    def _invalidate_quantization(self):
        self.quantized_bgr = None
        self.palette_centers = None
        if hasattr(self, 'p3_card_canvas'):
            self.p3_card_canvas.canvas.set_image(None)
            self.p3_card_canvas.set_badge("Idle", is_active=False)
        if hasattr(self, 'palette_bar'):
            self.palette_bar.set_palette([])
        if hasattr(self, 'btn_save_menu'):
            self.btn_save_menu.setEnabled(False)
        if hasattr(self, 'btn_export_palette'):
            self.btn_export_palette.setEnabled(False)
        if hasattr(self, 'btn_copy_p3'):
            self.btn_copy_p3.setEnabled(False)
        if hasattr(self, 'btn_quantize'):
            self.btn_quantize.setText("🎨 Re-Extract Palette")

    def on_candidate_chosen(self, idx):
        if not self.detected_candidates or idx < 0 or idx >= len(self.detected_candidates):
            return

        self.selected_candidate_idx = idx
        candidate = self.detected_candidates[idx]

        self.grid_dims = candidate["dim"]
        self.scale_factor = candidate["scale"]
        factor_ratio = 1.0 / self.scale_factor if self.scale_factor > 0 else 1.0

        gw, gh = self.grid_dims
        self.p2_card_canvas.set_badge(f"{gw} × {gh} ({factor_ratio:.2f}×)", is_active=True)

        assert self.original_image_bgr is not None
        self.downscaled_bgr = cv2.resize(self.original_image_bgr, self.grid_dims, interpolation=cv2.INTER_AREA)
        self.p2_card_canvas.canvas.set_image(self.downscaled_bgr, grid_dims=self.grid_dims)

        # Invalidate downstream quantization if grid changes
        self._invalidate_quantization()

        self.btn_save_grid.setEnabled(True)
        self.btn_p2_copy.setEnabled(True)
        self.btn_p2_proceed.setEnabled(True)
        self.btn_quantize.setEnabled(True)

        score_val = candidate.get("score", 0.0)
        self.set_status(f"Selected candidate #{idx + 1}: {gw}×{gh} ({factor_ratio:.2f}×, score: {score_val:,.0f}). Ready for palette extraction.", "#10b981")

    def on_estimate_error(self, err_msg):
        self.progress_bar.setVisible(False)
        self.btn_estimate.setEnabled(True)
        self.p2_card_canvas.set_badge("Failed", is_active=False)
        self.set_status(f"Estimation error: {err_msg}", "#ef4444")
        QMessageBox.warning(self, "Grid Detection Failed", f"Could not detect grid:\n{err_msg}")

    # ==========================================================================
    # Mode 2: Pixelate Logic
    # ==========================================================================
    def _toggle_aspect_lock(self, checked):
        self.lock_aspect_ratio = checked

    def _init_pixelate_dims(self, orig_w, orig_h):
        self.is_updating_aspect = True
        aspect = orig_w / float(orig_h)
        target_w = min(128, orig_w)
        target_h = max(8, int(round(target_w / aspect)))
        self.spin_pix_w.setValue(target_w)
        self.spin_pix_h.setValue(target_h)
        self.is_updating_aspect = False

    def _on_pix_w_changed(self, new_w):
        if self.is_updating_aspect or self.original_image_bgr is None:
            return
        if self.lock_aspect_ratio:
            self.is_updating_aspect = True
            orig_h, orig_w = self.original_image_bgr.shape[:2]
            aspect = orig_w / float(orig_h)
            new_h = max(8, int(round(new_w / aspect)))
            self.spin_pix_h.setValue(new_h)
            self.is_updating_aspect = False
        self.apply_pixelate()

    def _on_pix_h_changed(self, new_h):
        if self.is_updating_aspect or self.original_image_bgr is None:
            return
        if self.lock_aspect_ratio:
            self.is_updating_aspect = True
            orig_h, orig_w = self.original_image_bgr.shape[:2]
            aspect = orig_w / float(orig_h)
            new_w = max(8, int(round(new_h * aspect)))
            self.spin_pix_w.setValue(new_w)
            self.is_updating_aspect = False
        self.apply_pixelate()

    def _apply_pixel_preset(self, target_max_dim):
        if self.original_image_bgr is None:
            return
        orig_h, orig_w = self.original_image_bgr.shape[:2]
        self.is_updating_aspect = True
        if orig_w >= orig_h:
            tw = min(orig_w, target_max_dim)
            th = max(8, int(round(tw * (orig_h / float(orig_w)))))
        else:
            th = min(orig_h, target_max_dim)
            tw = max(8, int(round(th * (orig_w / float(orig_h)))))
        self.spin_pix_w.setValue(tw)
        self.spin_pix_h.setValue(th)
        self.is_updating_aspect = False
        self.apply_pixelate()

    def apply_pixelate(self):
        if self.original_image_bgr is None:
            return

        tw = self.spin_pix_w.value()
        th = self.spin_pix_h.value()
        interp = cv2.INTER_AREA if self.combo_pix_interp.currentIndex() == 0 else cv2.INTER_NEAREST

        self.grid_dims = (tw, th)
        orig_h, orig_w = self.original_image_bgr.shape[:2]
        self.scale_factor = orig_w / float(tw) if tw > 0 else 1.0

        self.downscaled_bgr = cv2.resize(self.original_image_bgr, (tw, th), interpolation=interp)
        self.p2_card_canvas.canvas.set_image(self.downscaled_bgr, grid_dims=self.grid_dims)
        self.p2_card_canvas.set_badge(f"{tw} × {th} ({self.scale_factor:.2f}×)", is_active=True)

        # Invalidate downstream quantization because downscaled grid changed
        self._invalidate_quantization()

        self.btn_save_grid.setEnabled(True)
        self.btn_p2_copy.setEnabled(True)
        self.btn_p2_proceed.setEnabled(True)
        self.btn_quantize.setEnabled(True)

        self.stepper.set_active_step(1, step2_unlocked=True, step3_unlocked=True)
        self.set_status(f"Pixelated to {tw}×{th} ({self.scale_factor:.2f}×). Ready for palette extraction.", "#10b981")

    # ==========================================================================
    # Mode 3: Direct 1:1 Palette Pass-Through Logic
    # ==========================================================================
    def apply_direct_extract_mode(self):
        if self.original_image_bgr is None:
            return

        orig_h, orig_w = self.original_image_bgr.shape[:2]
        self.detected_candidates = []
        self.grid_dims = (orig_w, orig_h)
        self.scale_factor = 1.0
        self.downscaled_bgr = self.original_image_bgr.copy()

        self.p2_card_canvas.canvas.set_image(self.downscaled_bgr, grid_dims=self.grid_dims)
        self.p2_card_canvas.set_badge(f"{orig_w} × {orig_h} (1.00×)", is_active=True)

        # Invalidate downstream quantization because grid changed
        self._invalidate_quantization()

        flat = self.downscaled_bgr.reshape((-1, 3))
        unique_colors = np.unique(flat, axis=0)
        num_uniques = len(unique_colors)

        if num_uniques <= 64:
            self._set_k_preset(num_uniques)

        self.btn_save_grid.setEnabled(True)
        self.btn_p2_copy.setEnabled(True)
        self.btn_p2_proceed.setEnabled(True)
        self.btn_quantize.setEnabled(True)

        self.stepper.set_active_step(1, step2_unlocked=True, step3_unlocked=True)
        self.set_status(f"Direct 1:1 image active ({orig_w}×{orig_h}, {num_uniques:,} unique colors). Ready to extract palette.", "#10b981")

    # ==========================================================================
    # Step 3: CIELAB Palette Quantization Logic
    # ==========================================================================
    def run_quantize(self):
        if self.downscaled_bgr is None or self.grid_dims is None:
            return
        if hasattr(self, 'worker_thread') and self.worker_thread is not None and self.worker_thread.isRunning():
            return

        self.btn_quantize.setEnabled(False)
        self.progress_bar.setVisible(True)

        k_val = self.spin_k.value()
        self.lbl_p3_k_badge.setText(f"K = {k_val}")
        self.set_status(f"Clustering colors with Perceptual CIELAB K-Means++ (K={k_val})...", "#8b5cf6", busy=True)
        self.p3_card_canvas.set_badge(f"K={k_val}...", is_active=True)

        self.worker_thread = ProcessThread(
            task_type="quantize",
            image_bgr=self.downscaled_bgr,
            k=k_val,
            grid_w=self.grid_dims[0],
            grid_h=self.grid_dims[1]
        )
        self.worker_thread.finished.connect(self.on_quantize_done)
        self.worker_thread.error.connect(self.on_quantize_error)
        self.worker_thread.start()

    def on_quantize_done(self, result):
        self.progress_bar.setVisible(False)
        self.btn_quantize.setEnabled(True)
        self.btn_quantize.setText(f"🎨 Re-Extract (K={len(result['centers'])})")

        self.quantized_bgr = result["quantized_grid"]
        self.palette_centers = result["centers"]

        k_val = len(self.palette_centers)
        self.p3_card_canvas.set_badge(f"{k_val} Colors", is_active=True)
        self.p3_card_canvas.canvas.set_image(self.quantized_bgr, grid_dims=self.grid_dims)
        self.palette_bar.set_palette(self.palette_centers)

        self.btn_save_menu.setEnabled(True)
        self.btn_export_palette.setEnabled(True)
        self.btn_copy_p3.setEnabled(True)

        self.set_status(f"Quantization complete! {k_val} CIELAB colors extracted.", "#10b981")

    def on_quantize_error(self, err_msg):
        self.progress_bar.setVisible(False)
        self.btn_quantize.setEnabled(True)
        self.p3_card_canvas.set_badge("Failed", is_active=False)
        self.set_status(f"Quantization error: {err_msg}", "#ef4444")
        QMessageBox.warning(self, "Quantization Failed", f"Color clustering failed:\n{err_msg}")

    def _toggle_compare_view(self, checked):
        if self.quantized_bgr is None or self.downscaled_bgr is None:
            return
        if checked:
            self.p3_card_canvas.canvas.set_image(self.downscaled_bgr, grid_dims=self.grid_dims)
            self.btn_compare_toggle.setText("👁 Showing Raw Grid (Click for Quantized)")
        else:
            self.p3_card_canvas.canvas.set_image(self.quantized_bgr, grid_dims=self.grid_dims)
            self.btn_compare_toggle.setText("👁 Compare with Grid")

    # ==========================================================================
    # Export & Save Routines
    # ==========================================================================
    def save_grid_image(self):
        if self.downscaled_bgr is None or self.grid_dims is None:
            return

        gw, gh = self.grid_dims
        default_name = f"pixel_grid_{gw}x{gh}.png"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Grid Image", default_name, "PNG Image (*.png);;All Files (*.*)"
        )
        if save_path:
            cv2.imwrite(save_path, self.downscaled_bgr)
            self.set_status(f"Saved grid to '{Path(save_path).name}' ({gw}×{gh})", "#10b981")

    def show_export_menu(self):
        if self.quantized_bgr is None or self.grid_dims is None:
            return

        menu = QMenu(self)
        gw, gh = self.grid_dims

        act_1x = QAction(f"💾 1× Native Pixel Art ({gw} × {gh} px)", self)
        act_1x.triggered.connect(lambda: self.save_pixel_art(scale=1))
        menu.addAction(act_1x)

        act_4x = QAction(f"💾 4× Crisp Scaled ({gw * 4} × {gh * 4} px) — Recommended", self)
        act_4x.triggered.connect(lambda: self.save_pixel_art(scale=4))
        menu.addAction(act_4x)

        act_8x = QAction(f"💾 8× Crisp Scaled ({gw * 8} × {gh * 8} px) — Social Media HD", self)
        act_8x.triggered.connect(lambda: self.save_pixel_art(scale=8))
        menu.addAction(act_8x)

        menu.addSeparator()

        act_custom = QAction("💾 Custom Scale...", self)
        act_custom.triggered.connect(lambda: self.save_pixel_art(scale=None))
        menu.addAction(act_custom)

        menu.exec(self.btn_save_menu.mapToGlobal(QPoint(0, self.btn_save_menu.height())))

    def save_pixel_art(self, scale=1):
        if self.quantized_bgr is None or self.grid_dims is None:
            return

        gw, gh = self.grid_dims
        if scale is None:
            scale = 4

        target_w = gw * scale
        target_h = gh * scale

        default_name = f"pixel_art_{target_w}x{target_h}.png"
        save_path, _ = QFileDialog.getSaveFileName(
            self, f"Save {scale}× Pixel Art", default_name, "PNG Image (*.png);;All Files (*.*)"
        )
        if save_path:
            out_img = cv2.resize(self.quantized_bgr, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(save_path, out_img)
            self.set_status(f"Saved pixel art to '{Path(save_path).name}' ({target_w}×{target_h})", "#10b981")

    def export_palette_swatch(self):
        if self.palette_centers is None:
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Palette Swatch", "palette_swatch.png", "PNG Image (*.png);;All Files (*.*)"
        )
        if save_path:
            swatch_img = create_palette_swatch(self.palette_centers, swatch_size=48, padding=6, cols=min(len(self.palette_centers), 8))
            cv2.imwrite(save_path, swatch_img)
            self.set_status(f"Saved palette swatch to '{Path(save_path).name}'", "#10b981")

    # ==========================================================================
    # Helper Utilities
    # ==========================================================================
    def _set_k_preset(self, val):
        self._on_k_value_changed(val)

    def _on_k_value_changed(self, val):
        if self.spin_k.value() != val:
            self.spin_k.blockSignals(True)
            self.spin_k.setValue(val)
            self.spin_k.blockSignals(False)
        if self.slider_k.value() != val:
            self.slider_k.blockSignals(True)
            self.slider_k.setValue(val)
            self.slider_k.blockSignals(False)
        self.lbl_p3_k_badge.setText(f"K = {val}")
        if self.quantized_bgr is not None:
            self.btn_quantize.setText(f"🎨 Re-Extract (K={val})")
        else:
            self.btn_quantize.setText(f"🎨 Extract Palette (K={val})")

    def _copy_canvas(self, canvas_widget):
        cb = QApplication.clipboard()
        if cb is not None and canvas_widget.pixmap and not canvas_widget.pixmap.isNull():
            cb.setPixmap(canvas_widget.pixmap)
            self.set_status("Image copied to clipboard!", "#38bdf8")

    def set_status(self, text, dot_color="#10b981", busy=False):
        self.lbl_status_msg.setText(text)
        self.lbl_status_dot.setStyleSheet(f"color: {dot_color}; font-size: 13px;")


# ==============================================================================
# Application Entry Point
# ==============================================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = TelaFormerApp()
    window.show()
    sys.exit(app.exec())
