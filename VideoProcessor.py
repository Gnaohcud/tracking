from PickleSwingVision import PickleSwingVision
import TrajectoryPlot
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtGui import QImage
import cv2
import numpy as np
import json
import os
import torch  # để giải phóng bộ nhớ GPU nếu cần
import gc


def cv_to_qimage(cv_img):
    """Convert OpenCV image to QImage."""
    height, width, channel = cv_img.shape
    bytes_per_line = channel * width
    # Convert BGR to RGB
    cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    qimg = QImage(
        cv_img_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
    )
    return qimg

class VideoProcessor(QObject):
    gotImage = Signal(int, QImage)
    xPlotReady = Signal(QImage)
    yPlotReady = Signal(QImage)
    currentFrameChanged = Signal(int)
    ballDetected = Signal(int, float, float)
    ballVisualize = Signal(int, float, float)
    bouncesDetected = Signal(list)
    ballTrajectoryLoaded = Signal(list)
    bouncesLoaded = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Sử dụng current_frame làm số frame tuyệt đối đã xử lý
        self.current_frame = 0
        # frames là buffer chứa các frame mới nhất (sẽ giới hạn kích thước)
        self.frames = []
        self.pickle_vision = PickleSwingVision(
            {
                "path_ball_track_model": "models/model_tracknet.pt",
                "path_bounce_model": "models/ctb_regr_bounce.cbm",
            },
            "cuda",
        )
        self.ball_trajectory = []
        self.keep_playing = False
        self.track_ball = False
        self.thread = QThread()
        self.moveToThread(self.thread)
        self.currentFrameChanged.emit(-1)
        self.thread.start()
        self.video_path = None

    @Slot(str)
    def save(self, path: str):
        if path.startswith("file://"):
            path = path[8:]
        trajectory_path = path + "_ball_trajectory.npy"
        bounces_path = path + "_bounces.npy"
        ball_trajectory_array = np.asarray(self.ball_trajectory)
        np.save(trajectory_path, ball_trajectory_array)
        bounces_array = np.asarray(self.bounces)
        np.save(bounces_path, bounces_array)
        project_object = {
            "video_file": self.video_path,
            "ball_trajectory": trajectory_path, 
            "bounces": bounces_path
        }
        with open(path + ".json", "w") as outfile:
            json.dump(project_object, outfile)

    @Slot(str)
    def load(self, path: str):
        if path.startswith("file://"):
            path = path[8:]
            path = os.path.normpath(path)
        print(os.access(path, os.R_OK))  # True nếu có quyền đọc file
        with open(path + ".json", "r", encoding="cp1252") as f:  # Sử dụng cp1252 khi mở file
             print("File opened successfully!")
             project_object = json.load(f)
        print("JSON loaded successfully!")
        self.read_video(project_object["video_file"])
        self.ball_trajectory = np.load(project_object["ball_trajectory"], allow_pickle=True).tolist()
        self.bounces = np.load(project_object["bounces"], allow_pickle=True).tolist()
        self.ballTrajectoryLoaded.emit(self.ball_trajectory)
        self.bouncesLoaded.emit(self.bounces)

    @Slot()
    def stop(self):
        try:
            self.cap.release()
            self.keep_playing = False
            self.thread.quit()
            self.thread.wait()
        except Exception as e:
            print("Exception when trying to stop thread:", e)

    @Slot(str)
    def read_video(self, path: str):
        if path.startswith("file://"):
            path = path[8:]
        self.cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        self.video_path = path
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frames = []  # reset buffer
        self.current_frame = 0  # reset số frame đã xử lý
        self.processNextFrame()

        
    @Slot(str)
    def trackBallTrajectory(self):
        # Dùng frame mới nhất trong buffer để tracking
        ball_track = []
        ball2 = self.pickle_vision.track_ball2(self.frames[-1])
        if ball2:
            result = ball2[1]
            ball_track.append(((result[0] + result[2]) / 2, (result[1] + result[3]) / 2))
        else:
            ball_track.append((None, None))

        # Đảm bảo ball_trajectory có số phần tử tương ứng với số frame đã xử lý
        while len(self.frames) > len(self.ball_trajectory):
            self.ball_trajectory.append((None, None))

        # Cập nhật ball_trajectory với điểm mới nhất
        self.ball_trajectory.append(ball_track[-1])

        MAX_TRAJECTORY = 30  # Số điểm cần có để chạy bounce detect
        OVERLAP = 15         # Giữ lại 15 điểm cuối sau khi bounce detect

        # Nếu số điểm đã đủ, tiến hành bounce detect
        if len(self.ball_trajectory) >= MAX_TRAJECTORY:
            # Làm mượt quỹ đạo
            x_track, y_track = self.pickle_vision.smooth_ball_track(self.ball_trajectory)
            if self.ball_trajectory[-1] == (None, None) and x_track[-1] is not None:
                self.ball_trajectory[-1] = (x_track[-1], y_track[-1])
            x_track = [x if x is not None else 0 for x in x_track]
            y_track = [y if y is not None else 0 for y in y_track]

            MAX_BOUNCES = 10  # Giới hạn số bounce tối đa
            # Chạy bounce detect trên cửa sổ hiện tại
            self.bounces = self.pickle_vision.bounce_detect(self.ball_trajectory)
            if len(self.bounces) > MAX_BOUNCES:
                self.bounces = self.bounces[-MAX_BOUNCES:]  # Chỉ giữ lại MAX_BOUNCES phần tử cuối

            # Tính chỉ số global của bounce (cửa sổ ball_trajectory tương ứng với các frame từ global_frame_start đến hiện tại)
            global_frame_start = self.current_frame - len(self.ball_trajectory) + 1
            global_bounces = [global_frame_start + idx for idx in self.bounces]
            if len(global_bounces) > MAX_BOUNCES:
                global_bounces = global_bounces[-MAX_BOUNCES:]

            self.bouncesDetected.emit([(gb, x_track[idx], y_track[idx]) for idx, gb in zip(self.bounces, global_bounces)])

            try:
                self.ballDetected.emit(
                    self.current_frame,
                    self.ball_trajectory[-1][0],
                    self.ball_trajectory[-1][1]
                )
            except Exception as e:
                print(f"ball track {self.current_frame} out of range: {len(self.ball_trajectory)} - {e}")

            # Sau khi chạy bounce detect, giữ lại OVERLAP điểm cuối và xóa các điểm cũ
            self.ball_trajectory = self.ball_trajectory[-OVERLAP:]
        else:
            # Nếu chưa đủ 30 điểm, chỉ cập nhật vị trí quả bóng
            try:
                self.ballDetected.emit(
                    self.current_frame,
                    self.ball_trajectory[-1][0],
                    self.ball_trajectory[-1][1]
                )
            except Exception as e:
                print(f"ball track {self.current_frame} out of range: {len(self.ball_trajectory)} - {e}")

        # Giải phóng bộ nhớ GPU nếu sử dụng GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    @Slot(int)
    def replay_frame(self, frame_id: int):
        print("replay at:", frame_id)
        # Tính chỉ số của frame trong buffer hiện tại
        # self.current_frame là số frame tuyệt đối đã xử lý,
        # self.frames chứa MAX_BUFFER frame mới nhất.
        buffer_start = self.current_frame - len(self.frames) + 1
        if frame_id < buffer_start or frame_id > self.current_frame:
            print(f"frame_id {frame_id} ngoài phạm vi buffer ({buffer_start} đến {self.current_frame})")
            return
        buffer_index = frame_id - buffer_start
        self.current_frame = frame_id
        self.track_ball = False
        self.keep_playing = False
        self.currentFrameChanged.emit(self.current_frame)
        image = cv_to_qimage(self.frames[buffer_index])
        self.gotImage.emit(self.current_frame, image)

    @Slot()
    def startPlayLoop(self):
        self.keep_playing = True
        self.track_ball = True
        while self.keep_playing:
            self.processNextFrame()

    def pausePlayLoop(self):
        print("---------------------------- pause play loop")
        self.keep_playing = False
        self.track_ball = False

    @Slot()
    def processNextFrame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frames.append(frame)
            self.current_frame += 1
            # Giới hạn buffer xuống MAX_BUFFER frame để giảm bộ nhớ
            MAX_BUFFER = 3
            if len(self.frames) > MAX_BUFFER:
                self.frames.pop(0)
        else:
            if self.keep_playing:
                self.keep_playing = False
            return
        self.currentFrameChanged.emit(self.current_frame)
        if self.track_ball:
            self.trackBallTrajectory()
        else:
            if self.current_frame <= len(self.ball_trajectory):
                try:
                    self.ballVisualize.emit(
                        self.current_frame,
                        self.ball_trajectory[self.current_frame - 1][0],
                        self.ball_trajectory[self.current_frame - 1][1]
                    )
                except Exception as e:
                    print("Error in ballVisualize emit:", e)
        # Luôn dùng frame mới nhất trong buffer để hiển thị
        image = cv_to_qimage(self.frames[-1])
        self.gotImage.emit(self.current_frame, image)
        if self.current_frame % 100 == 0:
            gc.collect()

    @Slot()
    def get_prev_frame(self):
        if self.current_frame == 0 or len(self.frames) == 0:
            return None
        # Tính chỉ số bắt đầu của buffer (số frame tuyệt đối của frame đầu buffer)
        buffer_start = self.current_frame - len(self.frames) + 1
        if self.current_frame - 1 < buffer_start:
            print("Frame trước không có trong buffer hiện tại.")
            return None
        self.current_frame -= 1
        self.currentFrameChanged.emit(self.current_frame)
        buffer_index = self.current_frame - buffer_start
        image = cv_to_qimage(self.frames[buffer_index])
        self.gotImage.emit(self.current_frame, image)
        if self.current_frame <= len(self.ball_trajectory):
            try:
                self.ballVisualize.emit(
                    self.current_frame,
                    self.ball_trajectory[self.current_frame - 1][0],
                    self.ball_trajectory[self.current_frame - 1][1]
                )
            except Exception as e:
                print("Error in ballVisualize emit in get_prev_frame:", e)

    def get_num_frames(self):
        return len(self.frames)
