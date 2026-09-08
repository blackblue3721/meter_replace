import cv2
import numpy as np
import pyrealsense2 as rs
import time

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

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_show = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        cv2.imshow("Color", color_image)
        cv2.imshow("Depth", depth_show)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        if key == ord("s"):
            ts = int(time.time())
            cv2.imwrite(f"color_{ts}.png", color_image)
            cv2.imwrite(f"depth_view_{ts}.png", depth_show)
            np.save(f"depth_raw_{ts}.npy", depth_image)
            print("已保存图片和深度数据")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()