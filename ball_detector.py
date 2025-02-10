from numpy._typing import _16Bit
from numpy.core.fromnumeric import shape
from .tracknet import BallTrackerNet
import torch
import torchvision
import cv2
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm


class BallDetector:
    def __init__(self, path_model=None, device="cuda"):
        self.model = BallTrackerNet(input_channels=9, out_channels=256)
        self.device = device
        if path_model:
            self.model.load_state_dict(
                torch.load(path_model, map_location=device, weights_only=True)
            )
            self.model = self.model.to(device)
            self.model.eval()
        self.width = 640
        self.height = 360

    def convert_to_onnx(self):
        dummy_input = torch.randn(1, 9, self.height, self.width).float().to(self.device)
        torch.onnx.export(
            self.model,
            dummy_input,
            "model.onnx",
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

    def infer_model(self, frames):
        """Run pretrained model on a consecutive list of frames
        :params
            frames: list of consecutive video frames
        :return
            ball_track: list of detected ball points
        """
        ball_track = [(None, None)] * 2
        prev_pred = [None, None]
        print("Ball detector processing")
        scale = []
        for num in range(2, len(frames)):
            img = cv2.resize(frames[num], (self.width, self.height))
            if not scale:
                scale.append(frames[num].shape[1] / self.width)
                scale.append(frames[num].shape[0] / self.height)
            img_prev = cv2.resize(frames[num - 1], (self.width, self.height))
            img_preprev = cv2.resize(frames[num - 2], (self.width, self.height))
            imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
            imgs = imgs.astype(np.float32) / 255.0
            imgs = np.rollaxis(imgs, 2, 0)
            inp = np.expand_dims(imgs, axis=0)
#            print("input shape: ", inp.shape)

            out = self.model(torch.from_numpy(inp).float().to(self.device))
            output = out.argmax(dim=1).detach().cpu().numpy()
            x_pred, y_pred = self.postprocess(output, prev_pred, scale)
            prev_pred = [x_pred, y_pred]
            ball_track.append((x_pred, y_pred))
        return ball_track

    def postprocess(self, feature_map, prev_pred, scale=[2, 2], max_dist=80):
        """
        :params
            feature_map: feature map with shape (1,360,640)
            prev_pred: [x,y] coordinates of ball prediction from previous frame
            scale: scale for conversion to original shape (720,1280)
            max_dist: maximum distance from previous ball detection to remove outliers
        :return
            x,y ball coordinates
        """
        feature_map *= 255
        feature_map = feature_map.reshape((self.height, self.width))
        feature_map = feature_map.astype(np.uint8)
        ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(
            heatmap,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=1,
            param1=50,
            param2=2,
            minRadius=2,
            maxRadius=7,
        )
        x, y = None, None
        if circles is not None:
            if prev_pred[0]:
                for i in range(len(circles[0])):
                    x_temp = circles[0][i][0] * scale[0]
                    y_temp = circles[0][i][1] * scale[1]
                    dist = distance.euclidean((x_temp, y_temp), prev_pred)
                    if dist < max_dist:
                        x, y = x_temp, y_temp
                        break
            else:
                x = circles[0][0][0] * scale[0]
                y = circles[0][0][1] * scale[1]
        return x, y
