import sys
import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QSpinBox, QGroupBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

from estimate_pixels import scan_and_explore
from kmeans import quantize_colors_kmeans_lab


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
                labels, centers = quantize_colors_kmeans_lab(flat_pixels, k=k)
                
                quantized_grid = centers[labels].reshape((grid_h, grid_w, 3))
                self.finished.emit(quantized_grid)
        except Exception as e:
            self.error.emit(str(e))


class ImagePanel(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.vbox = QVBoxLayout()
        
        self.image_label = QLabel("Waiting...")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #1e1e1e; color: #555555; border: 1px solid #333;")
        self.image_label.setMinimumSize(350, 450)
        
        self.vbox.addWidget(self.image_label, stretch=1)
        self.setLayout(self.vbox)

    def set_image(self, cv_img):
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        
        # Scale to fit while keeping aspect ratio. FastTransformation ensures crisp Nearest Neighbor for pixel art!
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.FastTransformation 
        )
        self.image_label.setPixmap(scaled_pixmap)


class TelaFormerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TelaFormer Pro")
        self.setMinimumSize(1200, 600)
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; color: #ffffff; }
            QGroupBox { color: #ffffff; font-weight: bold; border: 1px solid #333; border-radius: 8px; margin-top: 1ex; font-size: 14px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }
            QLabel { color: #dddddd; }
            QPushButton { background-color: #2563eb; color: white; padding: 10px; border-radius: 5px; font-weight: bold; font-size: 13px; }
            QPushButton:disabled { background-color: #1f2937; color: #6b7280; }
            QPushButton:hover:!disabled { background-color: #1d4ed8; }
            QSpinBox { background-color: #1f2937; color: white; padding: 6px; border: 1px solid #374151; border-radius: 4px; font-size: 13px; }
        """)

        self.original_path = None
        self.original_image_bgr = None
        self.downscaled_bgr = None
        self.grid_dims = None

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        panels_layout = QHBoxLayout()
        panels_layout.setSpacing(20)
        
        # --- Panel 1: Original ---
        self.panel1 = ImagePanel("1. Original Image")
        p1_controls = QVBoxLayout()
        self.btn_load = QPushButton("Select HD Image")
        self.btn_load.clicked.connect(self.load_image)
        p1_controls.addWidget(self.btn_load)
        self.panel1.vbox.addLayout(p1_controls)
        
        # --- Panel 2: True Pixel Grid ---
        self.panel2 = ImagePanel("2. Detected Pixel Grid")
        p2_controls = QVBoxLayout()
        self.btn_estimate = QPushButton("Estimate True Pixels")
        self.btn_estimate.clicked.connect(self.run_estimate)
        self.btn_estimate.setEnabled(False)
        self.lbl_estimate_info = QLabel("Ready")
        self.lbl_estimate_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        p2_controls.addWidget(self.btn_estimate)
        p2_controls.addWidget(self.lbl_estimate_info)
        self.panel2.vbox.addLayout(p2_controls)
        
        # --- Panel 3: Quantized Palette ---
        self.panel3 = ImagePanel("3. Quantized Pixel Art")
        p3_controls = QVBoxLayout()
        
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("Colors (K):"))
        self.spin_k = QSpinBox()
        self.spin_k.setRange(2, 256)
        self.spin_k.setValue(10)
        param_layout.addWidget(self.spin_k)
        
        self.btn_quantize = QPushButton("Extract CIELAB Palette")
        self.btn_quantize.clicked.connect(self.run_quantize)
        self.btn_quantize.setEnabled(False)
        
        p3_controls.addLayout(param_layout)
        p3_controls.addWidget(self.btn_quantize)
        self.panel3.vbox.addLayout(p3_controls)
        
        panels_layout.addWidget(self.panel1)
        panels_layout.addWidget(self.panel2)
        panels_layout.addWidget(self.panel3)
        
        main_layout.addLayout(panels_layout)
        self.setCentralWidget(main_widget)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.original_path = path
            self.original_image_bgr = cv2.imread(path)
            self.panel1.set_image(self.original_image_bgr)
            
            self.downscaled_bgr = None
            self.grid_dims = None
            self.panel2.image_label.setText("Waiting...")
            self.panel3.image_label.setText("Waiting...")
            
            self.btn_estimate.setEnabled(True)
            self.btn_quantize.setEnabled(False)
            self.lbl_estimate_info.setText("Image loaded.")

    def run_estimate(self):
        self.btn_estimate.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.lbl_estimate_info.setText("Scanning Image...")
        
        self.worker_thread = ProcessThread(task_type="estimate", image_path=self.original_path)
        self.worker_thread.finished.connect(self.on_estimate_done)
        self.worker_thread.error.connect(self.on_error)
        self.worker_thread.start()

    def on_estimate_done(self, best_result):
        self.btn_estimate.setEnabled(True)
        self.btn_load.setEnabled(True)
        
        self.grid_dims = best_result["dim"]
        scale = best_result["scale"]
        self.lbl_estimate_info.setText(f"Grid: {self.grid_dims[0]}x{self.grid_dims[1]} ({1.0/scale:.2f}x)")
        
        assert self.original_image_bgr is not None
        self.downscaled_bgr = cv2.resize(self.original_image_bgr, self.grid_dims, interpolation=cv2.INTER_AREA)
        self.panel2.set_image(self.downscaled_bgr)
        
        self.btn_quantize.setEnabled(True)

    def run_quantize(self):
        self.btn_quantize.setEnabled(False)
        self.btn_estimate.setEnabled(False)
        self.btn_load.setEnabled(False)
        
        assert self.grid_dims is not None
        assert self.downscaled_bgr is not None
        
        self.worker_thread = ProcessThread(
            task_type="quantize", 
            image_bgr=self.downscaled_bgr,
            k=self.spin_k.value(),
            grid_w=self.grid_dims[0],
            grid_h=self.grid_dims[1]
        )
        self.worker_thread.finished.connect(self.on_quantize_done)
        self.worker_thread.error.connect(self.on_error)
        self.worker_thread.start()

    def on_quantize_done(self, quantized_bgr):
        self.btn_quantize.setEnabled(True)
        self.btn_estimate.setEnabled(True)
        self.btn_load.setEnabled(True)
        
        self.panel3.set_image(quantized_bgr)

    def on_error(self, err_msg):
        self.btn_load.setEnabled(True)
        self.btn_estimate.setEnabled(True)
        self.btn_quantize.setEnabled(True)
        print(f"Error: {err_msg}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TelaFormerApp()
    window.show()
    sys.exit(app.exec())
