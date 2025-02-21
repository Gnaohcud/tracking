import sys
import cv2
import torch
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer, QTime
from ultralytics import YOLO
import time
import math

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("PyTorch CUDA version:", torch.version.cuda)
    print("Device Name:", torch.cuda.get_device_name(0))
else:
    print("No GPU found or not properly configured.")

# Load YOLO model
model = YOLO("G:/pickerball/yolopickleball5.engine")

def merge_nearby_boxes(detections, merge_threshold=50):
    merged_boxes = []
    used = [False] * len(detections)

    for i, (x1, y1, w1, h1) in enumerate(detections):
        if used[i]:
            continue

        cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
        merged_box = [x1, y1, w1, h1]
        used[i] = True

        for j, (x2, y2, w2, h2) in enumerate(detections):
            if i == j or used[j]:
                continue

            cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2
            distance = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

            if distance < merge_threshold:  # Nếu hai box gần nhau, gộp lại
                x_min = min(merged_box[0], x2)
                y_min = min(merged_box[1], y2)
                x_max = max(merged_box[0] + merged_box[2], x2 + w2)
                y_max = max(merged_box[1] + merged_box[3], y2 + h2)

                merged_box = [x_min, y_min, x_max - x_min, y_max - y_min]
                used[j] = True

        merged_boxes.append(merged_box)

    return merged_boxes


# Kalman Tracker Class
class KalmanTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(6, 2)  # [x, y, dx, dy, ddx, ddy] -> [x, y]
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0, 0.1, 0],  
                                               [0, 1, 0, 1, 0, 0.1],  
                                               [0, 0, 1, 0, 0.3, 0],  
                                               [0, 0, 0, 1, 0, 0.3],  
                                               [0, 0, 0, 0, 0.2, 0],  
                                               [0, 0, 0, 0, 0, 0.2]], np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],  
                                              [0, 1, 0, 0, 0, 0]], np.float32)
        
        self.kf.processNoiseCov = np.array([
            [0.1, 0, 0.1, 0, 0.05, 0],  
            [0, 0.1, 0, 0.1, 0, 0.05],  
            [0.1, 0, 0.5, 0, 0.2, 0],  
            [0, 0.1, 0, 0.5, 0, 0.2],  
            [0.05, 0, 0.2, 0, 0.3, 0],  
            [0, 0.05, 0, 0.2, 0, 0.3]
        ], np.float32)

        self.kf.measurementNoiseCov = np.array([[0.01, 0],  
                                                [0, 0.01]], np.float32)

        self.last_prediction = np.array([0, 0], np.float32)

        self.prev_frame = None
        self.prev_pts = None

    def predict(self):
        prediction = self.kf.predict()
        self.last_prediction = prediction[:2]  # [x, y]
        return self.last_prediction

    def update(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measurement)

    def update_with_optical_flow(self, frame):
        if self.prev_frame is None or self.prev_pts is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.prev_pts = np.array([[self.last_prediction]], dtype=np.float32)
            return self.last_prediction

        next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_frame, next_frame, self.prev_pts, None)
        
        if status[0][0] == 1:
            self.update(next_pts[0][0][0], next_pts[0][0][1])
        
        self.prev_frame = next_frame
        self.prev_pts = next_pts
        return self.last_prediction

# Main YOLO Application
class YOLOApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 Detection and Kalman Tracking")
        self.setGeometry(100, 100, 800, 700)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()

        self.mode = 1  # Default: YOLO only

        self.tracker = KalmanTracker()
        self.init_ui()

    def init_ui(self):
        self.video_label = QLabel(self)
        self.video_label.setGeometry(10, 10, 780, 500)
        self.video_label.setStyleSheet("background-color: white;")

        self.fps_label = QLabel("FPS: 0", self)
        self.fps_label.setGeometry(10, 520, 100, 30)

        self.camera_button = QPushButton("Chạy Camera", self)
        self.camera_button.setGeometry(120, 520, 200, 50)
        self.camera_button.clicked.connect(self.start_camera)

        self.video_button = QPushButton("Mở Video", self)
        self.video_button.setGeometry(340, 520, 200, 50)
        self.video_button.clicked.connect(self.open_video)

        self.stop_button = QPushButton("Dừng", self)
        self.stop_button.setGeometry(560, 520, 200, 50)
        self.stop_button.clicked.connect(self.stop_stream)
        self.stop_button.setEnabled(False)

        self.mode1_button = QPushButton("Mode 1: YOLO Only", self)
        self.mode1_button.setGeometry(120, 580, 300, 50)
        self.mode1_button.clicked.connect(self.set_mode1)

        self.mode2_button = QPushButton("Mode 2: YOLO + Kalman", self)
        self.mode2_button.setGeometry(440, 580, 300, 50)
        self.mode2_button.clicked.connect(self.set_mode2)

    def set_mode1(self):
        self.mode = 1
        print("Mode set to YOLO Only")

    def set_mode2(self):
        self.mode = 2
        print("Mode set to YOLO + Kalman")

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.statusBar().showMessage("Không thể mở camera!")
            return

        self.timer.start(30)
        self.camera_button.setEnabled(False)
        self.video_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.frame_count = 0
        self.last_time = time.time()

    def open_video(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "Chọn video", "", "Video Files (*.mp4 *.avi *.mov)")
        if not video_path:
            return

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.statusBar().showMessage("Không thể mở video!")
            return

        self.timer.start(30)
        self.camera_button.setEnabled(False)
        self.video_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.frame_count = 0
        self.last_time = time.time()

    def stop_stream(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.video_label.clear()
        self.camera_button.setEnabled(True)
        self.video_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.tracker = KalmanTracker()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.stop_stream()
            return

        frame = cv2.resize(frame, (780, 500))
        detections = []

        # YOLO Detection
        results = model(frame)
        for r in results[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = r
            class_name = results[0].names[int(cls)]
            if class_name == "ball":
                detections.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))

        detections = merge_nearby_boxes(detections)

        if self.mode == 1:  # YOLO Only
            for x, y, w, h in detections:
                cx, cy = x + w // 2, y + h // 2
                x_pred, y_pred = int(cx - 5), int(cy - 5)
                w_pred, h_pred = 10, 10
                cv2.rectangle(frame, (x_pred, y_pred), (x_pred + w_pred, y_pred + h_pred), (0, 0, 255), 2)

        if self.mode == 2:  # YOLO + Kalman + optical flow
            if detections:
                x, y, w, h = detections[0]
                cx, cy = x + w // 2, y + h // 2
                self.tracker.update(cx, cy)
            else:
                cx, cy = self.tracker.update_with_optical_flow(frame)

            x_pred, y_pred = int(cx - 5), int(cy - 5)
            w_pred, h_pred = 10, 10

            cv2.rectangle(frame, (x_pred, y_pred), (x_pred + w_pred, y_pred + h_pred), (0, 255, 0), 2)

        # Tính toán FPS
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.last_time
        if elapsed_time >= 1.0:
            self.fps = self.frame_count / elapsed_time
            self.fps_label.setText(f"FPS: {self.fps:.2f}")
            self.frame_count = 0
            self.last_time = current_time

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_image = QImage(rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0], rgb_frame.strides[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOApp()
    window.show()
    sys.exit(app.exec())
