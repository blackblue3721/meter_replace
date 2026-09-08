from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_DIR / "runs/detect/train-4/weights/best.pt"
DEFAULT_LOG_PATH = PROJECT_DIR / "logs/realtime_yolo_d435_log.csv"
DEFAULT_SNAPSHOT_DIR = PROJECT_DIR / "logs/snapshots"


CLASS_NAMES = {
    0: "meter",
    1: "display",
}


DRAW_COLORS = {
    0: (0, 0, 255),
    1: (0, 255, 0),
}


def get_median_depth_m(depth_frame, x: int, y: int, window_size: int = 11) -> float:
    """Return median valid depth in meters around a pixel."""
    half = window_size // 2
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    values = []

    for yy in range(y - half, y + half + 1):
        if yy < 0 or yy >= height:
            continue
        for xx in range(x - half, x + half + 1):
            if xx < 0 or xx >= width:
                continue
            depth_m = depth_frame.get_distance(xx, yy)
            if depth_m > 0:
                values.append(depth_m)

    if not values:
        return 0.0
    return float(np.median(values))


def deproject_pixel_to_camera_xyz(intrinsics, x: int, y: int, depth_m: float) -> tuple[float, float, float] | None:
    """Convert aligned color pixel plus depth to camera coordinates in meters."""
    if depth_m <= 0:
        return None
    point = rs.rs2_deproject_pixel_to_point(intrinsics, [float(x), float(y)], float(depth_m))
    return float(point[0]), float(point[1]), float(point[2])


def collect_detections(result, target_classes: set[int], max_per_class: int) -> list[dict]:
    """Collect top detections for each target class."""
    detections_by_class: dict[int, list[dict]] = {class_id: [] for class_id in target_classes}
    if result.boxes is None:
        return []

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        if class_id not in target_classes:
            continue
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]

        detections_by_class[class_id].append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(class_id, str(class_id)),
                "confidence": confidence,
                "box": (x1, y1, x2, y2),
                "center": (int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))),
            }
        )

    detections: list[dict] = []
    for class_id in sorted(detections_by_class):
        class_detections = sorted(detections_by_class[class_id], key=lambda item: item["confidence"], reverse=True)
        detections.extend(class_detections[:max_per_class])
    return detections


def selected_for_smoothing(detections: list[dict]) -> set[int]:
    """Return object identities of the highest-confidence detection for each class."""
    selected: dict[int, int] = {}
    for index, detection in enumerate(detections):
        class_id = detection["class_id"]
        if class_id not in selected or detection["confidence"] > detections[selected[class_id]]["confidence"]:
            selected[class_id] = index
    return set(selected.values())


def append_log_header_if_needed(log_path: Path) -> None:
    if log_path.exists() and log_path.stat().st_size > 0:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "timestamp",
                "frame_index",
                "class_id",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "center_x",
                "center_y",
                "depth_m",
                "camera_x_m",
                "camera_y_m",
                "camera_z_m",
                "smooth_camera_x_m",
                "smooth_camera_y_m",
                "smooth_camera_z_m",
            ]
        )


def append_log_rows(log_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with log_path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        for row in rows:
            writer.writerow(
                [
                    row["timestamp"],
                    row["frame_index"],
                    row["class_id"],
                    row["class_name"],
                    f"{row['confidence']:.6f}",
                    f"{row['x1']:.2f}",
                    f"{row['y1']:.2f}",
                    f"{row['x2']:.2f}",
                    f"{row['y2']:.2f}",
                    row["center_x"],
                    row["center_y"],
                    f"{row['depth_m']:.6f}",
                    _format_optional_float(row["camera_x_m"]),
                    _format_optional_float(row["camera_y_m"]),
                    _format_optional_float(row["camera_z_m"]),
                    _format_optional_float(row["smooth_camera_x_m"]),
                    _format_optional_float(row["smooth_camera_y_m"]),
                    _format_optional_float(row["smooth_camera_z_m"]),
                ]
            )


def _format_optional_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def smooth_xyz(history: deque[tuple[float, float, float]], xyz: tuple[float, float, float] | None) -> tuple[float, float, float] | None:
    if xyz is None:
        return None
    history.append(xyz)
    values = np.array(history, dtype=np.float32)
    return float(np.median(values[:, 0])), float(np.median(values[:, 1])), float(np.median(values[:, 2]))


def draw_detection(image, detection: dict, depth_m: float, xyz: tuple[float, float, float] | None, smooth: tuple[float, float, float] | None) -> None:
    class_id = detection["class_id"]
    color = DRAW_COLORS.get(class_id, (255, 255, 255))
    x1, y1, x2, y2 = [int(round(value)) for value in detection["box"]]
    cx, cy = detection["center"]

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.circle(image, (cx, cy), 5, color, -1)

    label = f"{detection['class_name']} {detection['confidence']:.2f}"
    cv2.putText(image, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    depth_text = f"pixel=({cx},{cy}) depth={depth_m:.3f}m"
    cv2.putText(image, depth_text, (x1, min(image.shape[0] - 36, y2 + 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    if xyz is not None:
        xyz_text = f"cam=({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f})m"
        cv2.putText(image, xyz_text, (x1, min(image.shape[0] - 12, y2 + 48)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    if smooth is not None:
        smooth_text = f"median=({smooth[0]:.3f},{smooth[1]:.3f},{smooth[2]:.3f})m"
        cv2.putText(image, smooth_text, (x1, min(image.shape[0] - 12, y2 + 72)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def run_realtime(args: argparse.Namespace) -> None:
    model = YOLO(str(args.model))
    append_log_header_if_needed(args.log)
    args.snapshot_dir.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, args.depth_width, args.depth_height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.color_width, args.color_height, rs.format.bgr8, args.fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    meter_history: deque[tuple[float, float, float]] = deque(maxlen=args.smooth_window)
    display_history: deque[tuple[float, float, float]] = deque(maxlen=args.smooth_window)
    histories = {
        0: meter_history,
        1: display_history,
    }

    print("Realtime YOLO + D435 started.")
    print(f"Model: {args.model}")
    print(f"Log: {args.log}")
    print("Keys: q/Esc = quit, s = save snapshot")

    frame_index = 0
    last_print_time = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            display_image = color_image.copy()
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(np.asanyarray(depth_frame.get_data()), alpha=0.03),
                cv2.COLORMAP_JET,
            )

            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

            result = model.predict(
                color_image,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                verbose=False,
                device=args.device if args.device else None,
            )[0]

            detections = collect_detections(result, target_classes={0, 1}, max_per_class=args.max_per_class)
            smoothing_indexes = selected_for_smoothing(detections)
            log_rows = []
            timestamp = time.time()

            for detection_index, detection in enumerate(detections):
                class_id = detection["class_id"]
                cx, cy = detection["center"]
                depth_m = get_median_depth_m(depth_frame, cx, cy, window_size=args.depth_window)
                xyz = deproject_pixel_to_camera_xyz(intrinsics, cx, cy, depth_m)
                smooth = smooth_xyz(histories[class_id], xyz) if detection_index in smoothing_indexes else None

                x1, y1, x2, y2 = detection["box"]
                draw_detection(display_image, detection, depth_m, xyz, smooth)

                log_rows.append(
                    {
                        "timestamp": timestamp,
                        "frame_index": frame_index,
                        "class_id": class_id,
                        "class_name": detection["class_name"],
                        "confidence": detection["confidence"],
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "center_x": cx,
                        "center_y": cy,
                        "depth_m": depth_m,
                        "camera_x_m": None if xyz is None else xyz[0],
                        "camera_y_m": None if xyz is None else xyz[1],
                        "camera_z_m": None if xyz is None else xyz[2],
                        "smooth_camera_x_m": None if smooth is None else smooth[0],
                        "smooth_camera_y_m": None if smooth is None else smooth[1],
                        "smooth_camera_z_m": None if smooth is None else smooth[2],
                    }
                )

            append_log_rows(args.log, log_rows)

            now = time.time()
            if now - last_print_time >= args.print_interval:
                last_print_time = now
                if not log_rows:
                    print(f"frame={frame_index} no detections")
                for row in log_rows:
                    print(
                        f"frame={frame_index} {row['class_name']} conf={row['confidence']:.2f} "
                        f"pixel=({row['center_x']},{row['center_y']}) depth={row['depth_m']:.3f}m "
                        f"cam=({_format_optional_float(row['camera_x_m'])},"
                        f"{_format_optional_float(row['camera_y_m'])},"
                        f"{_format_optional_float(row['camera_z_m'])})"
                    )

            cv2.imshow("YOLO D435 Color", display_image)
            cv2.imshow("Aligned Depth", depth_colormap)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                snapshot_path = args.snapshot_dir / f"snapshot_{frame_index:06d}.jpg"
                cv2.imwrite(str(snapshot_path), display_image)
                print(f"Saved snapshot: {snapshot_path}")

            frame_index += 1

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime YOLO detection with Intel RealSense D435 depth coordinates.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="YOLO model path")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH, help="CSV log path")
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR, help="snapshot output folder")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference image size")
    parser.add_argument("--conf", type=float, default=0.50, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.70, help="YOLO NMS IoU threshold")
    parser.add_argument("--max-per-class", type=int, default=10, help="maximum displayed/logged detections per class")
    parser.add_argument("--device", default="", help="YOLO device, for example 0 or cpu; empty means auto")
    parser.add_argument("--color-width", type=int, default=1280, help="color stream width")
    parser.add_argument("--color-height", type=int, default=720, help="color stream height")
    parser.add_argument("--depth-width", type=int, default=848, help="depth stream width")
    parser.add_argument("--depth-height", type=int, default=480, help="depth stream height")
    parser.add_argument("--fps", type=int, default=30, help="RealSense stream FPS")
    parser.add_argument("--depth-window", type=int, default=11, help="median depth window size in pixels")
    parser.add_argument("--smooth-window", type=int, default=7, help="median smoothing window across frames")
    parser.add_argument("--print-interval", type=float, default=0.5, help="seconds between console prints")
    args = parser.parse_args()

    if args.depth_window <= 0 or args.depth_window % 2 == 0:
        raise ValueError("--depth-window must be a positive odd number")
    if args.smooth_window <= 0:
        raise ValueError("--smooth-window must be greater than 0")
    if args.max_per_class <= 0:
        raise ValueError("--max-per-class must be greater than 0")
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    return args


def main() -> None:
    args = parse_args()
    run_realtime(args)


if __name__ == "__main__":
    main()
