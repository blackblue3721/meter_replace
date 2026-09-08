import cv2
import numpy as np
import pyrealsense2 as rs

clicked_point = None


def mouse_callback(event, x, y, flags, param):
    global clicked_point

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)


def get_median_distance_cm(depth_frame, x, y, window_size=11):
    half = window_size // 2
    distances_cm = []

    depth_width = depth_frame.get_width()
    depth_height = depth_frame.get_height()

    for yy in range(y - half, y + half + 1):
        for xx in range(x - half, x + half + 1):
            if xx < 0 or yy < 0:
                continue

            if xx >= depth_width or yy >= depth_height:
                continue

            distance_m = depth_frame.get_distance(xx, yy)

            if distance_m > 0:
                distances_cm.append(distance_m * 100)

    if len(distances_cm) == 0:
        return 0

    return float(np.median(distances_cm))


pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

align = rs.align(rs.stream.color)

cv2.namedWindow("Click Median Distance")
cv2.setMouseCallback("Click Median Distance", mouse_callback)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        display_image = color_image.copy()

        if clicked_point is not None:
            x, y = clicked_point

            distance_cm = get_median_distance_cm(depth_frame, x, y, window_size=11)

            cv2.circle(display_image, (x, y), 6, (0, 0, 255), -1)
            cv2.rectangle(
                display_image,
                (x - 5, y - 5),
                (x + 5, y + 5),
                (0, 255, 255),
                1
            )

            if distance_cm > 0:
                text = f"x={x}, y={y}, distance={distance_cm:.1f} cm"
            else:
                text = f"x={x}, y={y}, distance=invalid"

            cv2.putText(
                display_image,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            print(text)

        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        cv2.imshow("Click Median Distance", display_image)
        cv2.imshow("Aligned Depth", depth_colormap)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()