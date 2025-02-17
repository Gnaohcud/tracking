from ultralytics import YOLO 
import cv2
import pickle
import pandas as pd
from tqdm import tqdm
import numpy as np

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
        # Khởi tạo Kalman Filter với 6 trạng thái và 2 đo lường
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1, 0, 0.5, 0],
                                             [0, 1, 0, 0.5],
                                             [0, 0, 0.8, 0],
                                             [0, 0, 0, 0.8]
                                            ], np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.array([
            [0.1, 0, 0.1, 0],
            [0, 0.1, 0, 0.1],
            [0.1, 0, 0.5, 0],
            [0, 0.1, 0, 0.5]
        ], np.float32)
        self.kf.measurementNoiseCov = np.array([[0.1, 0],
                                                [0, 0.1]], np.float32)
        self.last_prediction = None

        # Các biến Optical Flow
        self.prev_frame = None
        self.prev_pts = None
                # Biến đếm số frame liên tiếp không phát hiện bóng
        self.lost_frames = 0
        self.max_lost_frames = 10  # Ngưỡng số frame mất tín hiệu

    # def interpolate_ball_positions(self, ball_positions_org):
    #     ball_positions = [x.get(1, []) for x in ball_positions_org]
    #     count = 0
    #     for ball_position in ball_positions:
    #         if ball_position:
    #             count += 1
    #     if count == 0:
    #         return ball_positions_org

    #     # Convert list thành dataframe
    #     df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

    #     # Nội suy các giá trị thiếu
    #     df_ball_positions = df_ball_positions.interpolate()
    #     df_ball_positions = df_ball_positions.bfill()

    #     ball_positions = [{1: x} for x in df_ball_positions.to_numpy().tolist()]

    #     return ball_positions

    # def get_ball_shot_frames(self, ball_positions):
    #     ball_positions = [x.get(1, []) for x in ball_positions]
    #     # Convert list thành dataframe
    #     df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

    #     df_ball_positions['ball_hit'] = 0
    #     df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
    #     df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
    #     df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
    #     minimum_change_frames_for_hit = 25
    #     for i in range(1, len(df_ball_positions) - int(minimum_change_frames_for_hit * 1.2)):
    #         negative_position_change = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[i+1] < 0
    #         positive_position_change = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[i+1] > 0

    #         if negative_position_change or positive_position_change:
    #             change_count = 0
    #             for change_frame in range(i+1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
    #                 negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[change_frame] < 0
    #                 positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[change_frame] > 0

    #                 if negative_position_change and negative_position_change_following_frame:
    #                     change_count += 1
    #                 elif positive_position_change and positive_position_change_following_frame:
    #                     change_count += 1

    #             if change_count > minimum_change_frames_for_hit - 1:
    #                 df_ball_positions['ball_hit'].iloc[i] = 1

    #     frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit'] == 1].index.tolist()

    #     return frame_nums_with_ball_hits

    # def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        
    #     ball_detections = []

    #     # Nếu có stub, có thể load từ file (bỏ comment nếu cần)
    #     # if read_from_stub and stub_path is not None:
    #     #     with open(stub_path, 'rb') as f:
    #     #         ball_detections = pickle.load(f)
    #     #     return ball_detections

    #     for frame in tqdm(frames):
    #         player_dict = self.detect_frame(frame)
    #         ball_detections.append(player_dict)

    #     if stub_path is not None:
    #         with open(stub_path, 'wb') as f:
    #             pickle.dump(ball_detections, f)

    #     return ball_detections

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

    def kalman_update(self, result):
        # result: [x1, y1, x2, y2, ...]
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
