import cv2
import numpy as np
import pyrealsense2 as rs

clicked_point = None
clicked_distance_cm = None

def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)
align = rs.align(rs.stream.color)

cv2.namedWindow("Click Distance")
cv2.setMouseCallback("Click Distance", mouse_callback)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        if clicked_point is not None:
            x, y = clicked_point
            distance_m = depth_frame.get_distance(x, y)
            clicked_distance_cm = distance_m * 100

            cv2.circle(color_image, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(
                color_image,
                f"{clicked_distance_cm:.1f} cm",
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            print(f"点击点: x={x}, y={y}, 距离={clicked_distance_cm:.1f} cm")

        cv2.imshow("Click Distance", color_image)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()