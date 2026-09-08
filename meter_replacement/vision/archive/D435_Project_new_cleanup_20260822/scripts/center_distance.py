import cv2
import numpy as np
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        depth_width = depth_frame.get_width()
        depth_height = depth_frame.get_height()

        x = depth_width // 2
        y = depth_height // 2

        distance_m = depth_frame.get_distance(x, y)
        distance_cm = distance_m * 100

        cv2.putText(
            color_image,
            f"Center distance: {distance_cm:.1f} cm",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        print("中心点距离:", distance_cm, "厘米")

        cv2.imshow("D435 Center Distance", color_image)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()