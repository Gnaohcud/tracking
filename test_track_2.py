from ultralytics import YOLO 
import cv2
import pickle
import pandas as pd
from tqdm import tqdm
import numpy as np
import math
import time

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
        # Khởi tạo Kalman Filter với 6 trạng thái: [x, y, dx, dy, ddx, ddy] và 2 đo lường: [x, y]
# Ma trận chuyển trạng thái (transition matrix) - Không có gia tốc
        self.kf = cv2.KalmanFilter(4, 2)  

        self.kf.transitionMatrix = np.array([
            [1, 0, 0.5, 0],  # x' = x + dx
            [0, 1, 0, 0.5],  # y' = y + dy
            [0, 0, 0.8, 0],  # dx' = dx
            [0, 0, 0, 0.8]   # dy' = dy
        ], np.float32)

        # Ma trận đo lường (measurement matrix) - Chỉ đo [x, y]
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],  
            [0, 1, 0, 0]
        ], np.float32)

        # Ma trận nhiễu quá trình (process noise) - Giảm độ trễ nhưng vẫn giữ độ mượt
        self.kf.processNoiseCov = np.array([
            [0.1, 0, 0.1, 0],  
            [0, 0.1, 0, 0.1],  
            [0.1, 0, 0.5, 0],  
            [0, 0.1, 0, 0.5]  
        ], np.float32)

        # Ma trận nhiễu đo lường (measurement noise) - Cân bằng giữa độ chính xác và ổn định
        self.kf.measurementNoiseCov = np.array([
            [0.1, 0],
            [0, 0.1]
        ], np.float32)
        
        self.last_prediction = None

        # Các biến dùng cho Optical Flow
        self.prev_frame = None
        self.prev_pts = None
        
        # Biến đếm số frame liên tiếp không phát hiện bóng
        self.lost_frames = 0
        self.max_lost_frames = 10  # Ngưỡng số frame mất tín hiệu

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.3, verbose=False)[0]
        ball_dict = {}
        if len(results.boxes) > 0:
            # Reset counter vì có phát hiện bóng
            self.lost_frames = 0
            
            # Lấy bbox đầu tiên (hoặc có thể xử lý hợp nhất nhiều bbox nếu cần)
            result = results.boxes.xyxy.tolist()[0]
            ball_dict[1] = result
            # Cập nhật Kalman Filter với đo lường từ YOLO
            self.kalman_update(result)
            # Cập nhật Optical Flow với khung hình hiện tại và vị trí trung tâm
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x1, y1, x2, y2 = result[:4]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            self.prev_frame = gray
            self.prev_pts = np.array([[[cx, cy]]], dtype=np.float32)
        else:
            self.lost_frames += 1
            if self.lost_frames >= self.max_lost_frames:
                # Nếu mất bóng quá nhiều frame, không trả về bbox
                ball_dict = {}
            else:
                # Dự đoán vị trí bóng qua Kalman + Optical Flow
                predicted_bbox = self.kalman_predict(frame)
                # (Có thể thêm kiểm tra tọa độ: nếu dự đoán quá sát biên, bỏ qua)
                ball_dict[1] = predicted_bbox
        return ball_dict

    # Các hàm kalman_update, kalman_predict, draw_bboxes không cần thay đổi nhiều
    # (có thể tinh chỉnh thêm trong kalman_predict nếu cần kiểm tra tọa độ dự đoán)
    def kalman_update(self, result):
        x1, y1, x2, y2 = result[:4]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        self.kf.correct(measurement)
        self.last_prediction = self.kf.predict()[:2]
        return self.last_prediction

    def kalman_predict(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_frame is None or self.prev_pts is None:
            if self.last_prediction is None:
                self.last_prediction = self.kf.predict()[:2]
            self.prev_frame = gray
            self.prev_pts = np.array([[[self.last_prediction[0][0], self.last_prediction[1][0]]]], dtype=np.float32)
            cx = self.last_prediction[0][0]
            cy = self.last_prediction[1][0]
        else:
            lk_params = dict(
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_frame, gray, self.prev_pts, None, **lk_params)
            if status is not None and status[0][0] == 1:
                new_point = next_pts[0][0]
                pred_point = np.array([self.last_prediction[0][0], self.last_prediction[1][0]])
                distance = np.linalg.norm(pred_point - new_point)
                if distance < 50:  # Nếu kết quả Optical Flow đáng tin cậy
                    measurement = np.array([[np.float32(new_point[0])], [np.float32(new_point[1])]])
                    self.kf.correct(measurement)
            self.last_prediction = self.kf.predict()[:2]
            self.prev_frame = gray
            self.prev_pts = np.array([[[self.last_prediction[0][0], self.last_prediction[1][0]]]], dtype=np.float32)
            cx = self.last_prediction[0][0]
            cy = self.last_prediction[1][0]
        # Clipping tọa độ dự đoán theo kích thước của frame
        h, w = gray.shape[:2]
        cx = min(max(cx, 0), w)
        cy = min(max(cy, 0), h)
        return [cx - 10, cy - 10, cx + 10, cy + 10]

    def draw_bboxes(self, video_frames, ball_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball ID: {track_id}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)
        return output_video_frames


def main():
    # Thay đổi đường dẫn video tại đây
    video_path = "G:/pickerball/Videos-20250207T032152Z-001/Videos/cutscene.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Không thể mở video.")
        return

    tracker = BallTracker("G:/pickerball/yolopickerball2.pt")
    
    fps = 50
    delay = 1.0 / fps  # thời gian tối thiểu cho mỗi frame
    
    while True:
        start_time = time.time()  # ghi lại thời gian bắt đầu xử lý frame
        
        ret, frame = cap.read()
        if not ret:
            break

        # Nếu cần, resize frame để hiển thị nhỏ lại
        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        
        ball_dict = tracker.detect_frame(frame)
        frame = tracker.draw_bboxes([frame], [ball_dict])[0]

        cv2.imshow("Ball Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Tính toán thời gian xử lý frame và chờ cho đến khi đạt 1/30 giây
        elapsed_time = time.time() - start_time
        sleep_time = delay - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
