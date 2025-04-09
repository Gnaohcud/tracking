import depthai as dai
import cv2
import os
import time  # <== Thêm cái này để giới hạn frame rate

# Tạo thư mục lưu ảnh
os.makedirs("output_frames", exist_ok=True)

# Tạo pipeline
pipeline = dai.Pipeline()

# Camera nodes
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_left = pipeline.create(dai.node.MonoCamera)
cam_right = pipeline.create(dai.node.MonoCamera)

# Cấu hình camera
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)

cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)

cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Output stream
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.video.link(xout_rgb.input)

xout_left = pipeline.create(dai.node.XLinkOut)
xout_left.setStreamName("left")
cam_left.out.link(xout_left.input)

xout_right = pipeline.create(dai.node.XLinkOut)
xout_right.setStreamName("right")
cam_right.out.link(xout_right.input)

# Bắt đầu thiết bị
with dai.Device(pipeline) as device:
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    left_queue = device.getOutputQueue(name="left", maxSize=4, blocking=False)
    right_queue = device.getOutputQueue(name="right", maxSize=4, blocking=False)

    frame_id = 0
    recording = False
    last_save_time = time.time()

    print("Nhấn phím bất kỳ để bắt đầu ghi. Nhấn ESC để dừng và thoát.")

    while True:
        key = cv2.waitKey(1)

        if not recording and key != -1 and key != 27:
            recording = True
            print("⏺️ Bắt đầu ghi frame...")

        if key == 27:
            print("⏹️ Dừng và thoát.")
            break

        rgb_frame = None
        left_frame = None
        right_frame = None

        if rgb_queue.has():
            rgb_frame = rgb_queue.get().getCvFrame()
            cv2.imshow("RGB", rgb_frame)

        if left_queue.has():
            left_frame = left_queue.get().getCvFrame()
            cv2.imshow("Left", left_frame)

        if right_queue.has():
            right_frame = right_queue.get().getCvFrame()
            cv2.imshow("Right", right_frame)

        current_time = time.time()

        # Nếu đang ghi và đã đủ 0.2s kể từ lần lưu cuối
        if recording and current_time - last_save_time >= 0.2:
            if rgb_frame is not None:
                cv2.imwrite(f"output_frames/frame_{frame_id:04d}_rgb.png", rgb_frame)
            if left_frame is not None:
                cv2.imwrite(f"output_frames/frame_{frame_id:04d}_left.png", left_frame)
            if right_frame is not None:
                cv2.imwrite(f"output_frames/frame_{frame_id:04d}_right.png", right_frame)
            frame_id += 1
            last_save_time = current_time

    cv2.destroyAllWindows()

