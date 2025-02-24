import sys
import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, QTime

# Kiểm tra CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Sử dụng thiết bị:", device)

# Tải model DeepLabV3+ (MobileNetV3)
model = deeplabv3_mobilenet_v3_large(pretrained=True).to(device).eval()

# Thiết lập transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((520, 520)),  # Resize cho phù hợp model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class DeepLabApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepLabV3+ Segmentation")
        self.setGeometry(100, 100, 800, 600)

        # Video setup
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Time tracking for FPS
        self.frame_count = 0
        self.fps = 0
        self.start_time = QTime.currentTime()

        # UI setup
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

    def start_camera(self):
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            self.statusBar().showMessage("Không thể mở camera!")
            return

        self.timer.start(30)
        self.camera_button.setEnabled(False)
        self.video_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        # Reset FPS calculation
        self.frame_count = 0
        self.start_time = QTime.currentTime()

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

        # Reset FPS calculation
        self.frame_count = 0
        self.start_time = QTime.currentTime()

    def stop_stream(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.video_label.clear()
        self.camera_button.setEnabled(True)
        self.video_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.stop_stream()
            return

        # Chạy mô hình DeepLabV3+
        segmented_frame = self.segment_image(frame)

        # Resize frame để hiển thị trong QLabel
        segmented_frame = cv2.resize(segmented_frame, (780, 500))

        # Cập nhật FPS
        self.frame_count += 1
        elapsed_time = self.start_time.elapsed() / 1000  # Thời gian đã trôi qua (giây)
        if elapsed_time > 1.0:
            self.fps = self.frame_count / elapsed_time
            self.fps_label.setText(f"FPS: {self.fps:.2f}")
            self.frame_count = 0
            self.start_time = QTime.currentTime()

        # Chuyển đổi ảnh OpenCV sang QImage
        rgb_frame = cv2.cvtColor(segmented_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # Hiển thị video
        self.video_label.setPixmap(pixmap)

    def segment_image(self, frame):
        """Chạy mô hình DeepLabV3+ trên ảnh đầu vào"""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)['out'][0]  # Lấy output của model
        output_predictions = torch.argmax(output, dim=0).byte().cpu().numpy()

        # Tạo màu cho từng lớp segmentation
        colored_mask = cv2.applyColorMap(output_predictions * 10, cv2.COLORMAP_JET)
        colored_mask = cv2.resize(colored_mask, (frame.shape[1], frame.shape[0]))  # Resize mask về đúng kích thước frame
        blended_frame = cv2.addWeighted(frame, 0.5, colored_mask, 0.5, 0)


        return blended_frame

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()

# Chạy ứng dụng
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DeepLabApp()
    window.show()
    sys.exit(app.exec_())
