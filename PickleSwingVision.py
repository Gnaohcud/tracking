import sys
sys.path.append('./ai')
sys.path.append('./trackers')

from ai.bounce_detector import BounceDetector
from ai.ball_detector import BallDetector
from trackers.ball_tracker import BallTracker
import time

class PickleSwingVision:
    def __init__(self, args, device):
        self.ball_track = None
        self.bounces = None
        self.match_points = None
        print('ball detection')

        # self.ball_detector = BallDetector(args[ "path_ball_track_model" ], device)
        self.ball_detector = None
        self.bounce_detector = BounceDetector(args[ "path_bounce_model" ])
        # self.ball_tracker = BallTracker(model_path='models2/yolo5_last.pt')
        self.ball_tracker = BallTracker(model_path='models/yolopickleball5.pt')
        self._ball_trajectory = {} 
        self._ball_detections = []

    def get_ball_trajectory(self):
        return self._ball_trajectory

    #brief: function to track the ball
    #params: frames: list of 3 consecutive video frames 
    def track_ball(self, frames, frame_index):
        ball_track = self.ball_detector.infer_model(frames)
        self._ball_trajectory[frame_index] = ball_track
        return ball_track

    def track_ball2(self, frame):
        start_time = time.time()  # Bắt đầu đo thời gian
        ball_track = self.ball_tracker.detect_frame(frame)
        elapsed_ms = (time.time() - start_time) * 1000  # chuyển sang milisecond
        print(f"track_ball2 execution time: {elapsed_ms:.1f} ms")
        return ball_track

        # self._ball_detections.append(ball_track)
        #
        # if len(self._ball_detections) > 5:
        #     self._ball_detections = self.ball_tracker.interpolate_ball_positions(self._ball_detections)
        # return self._ball_detections[-1]

    #brief: function to detect bounces
    #params: ball_track: list of (x,y) ball coordinates from 
    # the end of last swing
    #returns: bounces: list of image numbers where ball touches the ground
    # and where the ball touch the racquet
    def bounce_detect(self, ball_track):
        x_ball = [x[0] for x in ball_track]
        y_ball = [x[1] for x in ball_track]
        
        # Bắt đầu đo thời gian
        start_time = time.time()
        bounces = list(self.bounce_detector.predict(x_ball, y_ball))
        elapsed_ms = (time.time() - start_time) * 1000  # chuyển sang ms
        
        print(f"Bounce detection took {elapsed_ms:.1f} ms")
        return bounces


    def smooth_ball_track(self, ball_track):
        x_ball = [x[0] for x in ball_track]
        y_ball = [x[1] for x in ball_track]
        x_ball, y_ball = self.bounce_detector.smooth_predictions(x_ball, y_ball)
        return x_ball, y_ball


