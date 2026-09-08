from __future__ import annotations

import argparse
import csv
import math
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_DIR / "runs/detect/train-4/weights/best.pt"
DEFAULT_LOG_PATH = PROJECT_DIR / "logs/multi_meter_targets.csv"
DEFAULT_ROBOT_READY_PATH = PROJECT_DIR / "logs/robot_ready_targets.csv"
DEFAULT_SNAPSHOT_DIR = PROJECT_DIR / "logs/multi_meter_snapshots"


CLASS_NAMES = {
    0: "meter",
    1: "display",
}


COLORS = {
    0: (0, 0, 255),
    1: (0, 255, 0),
    "target": (255, 0, 255),
    "text": (255, 255, 255),
}


def get_median_depth_m(depth_frame, x: int, y: int, window_size: int) -> float:
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

    return float(np.median(values)) if values else 0.0


def get_box_depth_m(depth_frame, box: tuple[float, float, float, float], window_size: int) -> tuple[float, tuple[int, int] | None]:
    """Try center plus nearby inner points and return a robust median depth."""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    sample_points = [
        (x1 + 0.50 * w, y1 + 0.50 * h),
        (x1 + 0.50 * w, y1 + 0.35 * h),
        (x1 + 0.50 * w, y1 + 0.65 * h),
        (x1 + 0.35 * w, y1 + 0.50 * h),
        (x1 + 0.65 * w, y1 + 0.50 * h),
    ]

    depths = []
    valid_points = []
    for px, py in sample_points:
        x = int(round(px))
        y = int(round(py))
        depth_m = get_median_depth_m(depth_frame, x, y, window_size)
        if depth_m > 0:
            depths.append(depth_m)
            valid_points.append((x, y))

    if not depths:
        return 0.0, None
    return float(np.median(depths)), valid_points[len(valid_points) // 2]


def deproject(intrinsics, pixel: tuple[int, int], depth_m: float) -> tuple[float, float, float] | None:
    if pixel is None or depth_m <= 0:
        return None
    point = rs.rs2_deproject_pixel_to_point(intrinsics, [float(pixel[0]), float(pixel[1])], float(depth_m))
    return float(point[0]), float(point[1]), float(point[2])


def collect_detections(result, conf_floor: float) -> tuple[list[dict], list[dict]]:
    meters = []
    displays = []
    if result.boxes is None:
        return meters, displays

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        if class_id not in (0, 1):
            continue
        confidence = float(box.conf[0].item())
        if confidence < conf_floor:
            continue
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        detection = {
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "confidence": confidence,
            "box": (x1, y1, x2, y2),
            "center": (int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))),
        }
        if class_id == 0:
            meters.append(detection)
        else:
            displays.append(detection)

    meters.sort(key=lambda item: item["confidence"], reverse=True)
    displays.sort(key=lambda item: item["confidence"], reverse=True)
    return meters, displays


def expanded_contains(box: tuple[float, float, float, float], point: tuple[int, int], margin: float) -> bool:
    x1, y1, x2, y2 = box
    x, y = point
    return x1 - margin <= x <= x2 + margin and y1 - margin <= y <= y2 + margin


def distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    return math.hypot(float(p1[0] - p2[0]), float(p1[1] - p2[1]))


def pair_meters_and_displays(meters: list[dict], displays: list[dict], margin: float, max_pair_distance: float) -> list[dict]:
    """Attach each display to the most plausible meter."""
    pairs = []
    used_display_indexes = set()

    for meter in meters:
        meter_center = meter["center"]
        candidates = []
        for index, display in enumerate(displays):
            if index in used_display_indexes:
                continue
            display_center = display["center"]
            inside_score = 0 if expanded_contains(meter["box"], display_center, margin) else 1
            center_distance = distance(meter_center, display_center)
            if inside_score == 1 and center_distance > max_pair_distance:
                continue
            candidates.append((inside_score, center_distance, -display["confidence"], index, display))

        display = None
        if candidates:
            candidates.sort()
            display_index = candidates[0][3]
            display = candidates[0][4]
            used_display_indexes.add(display_index)

        pairs.append(
            {
                "meter": meter,
                "display": display,
            }
        )

    return pairs


def sort_targets_grid(pairs: list[dict], row_tolerance: float) -> list[dict]:
    """Sort detected meters from top to bottom, then left to right within each row."""
    if not pairs:
        return []

    remaining = sorted(pairs, key=lambda pair: pair["meter"]["center"][1])
    rows: list[list[dict]] = []

    for pair in remaining:
        cy = pair["meter"]["center"][1]
        placed = False
        for row in rows:
            row_cy = float(np.mean([item["meter"]["center"][1] for item in row]))
            if abs(cy - row_cy) <= row_tolerance:
                row.append(pair)
                placed = True
                break
        if not placed:
            rows.append([pair])

    ordered = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda pair: pair["meter"]["center"][0]))

    for index, pair in enumerate(ordered, start=1):
        pair["target_id"] = index
    return ordered


def stable_target_key(pair: dict) -> str:
    """Stable-ish key based on grid order."""
    return f"target_{pair['target_id']}"


def median_xyz(history: deque[tuple[float, float, float]], xyz: tuple[float, float, float] | None) -> tuple[float, float, float] | None:
    if xyz is None:
        return None
    history.append(xyz)
    values = np.array(history, dtype=np.float32)
    return float(np.median(values[:, 0])), float(np.median(values[:, 1])), float(np.median(values[:, 2]))


def is_history_stable(history: deque[tuple[float, float, float]], stable_frames: int, threshold_m: float) -> bool:
    if len(history) < stable_frames:
        return False
    recent = np.array(list(history)[-stable_frames:], dtype=np.float32)
    ranges = recent.max(axis=0) - recent.min(axis=0)
    return bool(np.all(ranges <= threshold_m))


def write_log_header(log_path: Path) -> None:
    if log_path.exists() and log_path.stat().st_size > 0:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "timestamp",
                "frame_index",
                "target_id",
                "meter_conf",
                "display_conf",
                "meter_cx",
                "meter_cy",
                "display_cx",
                "display_cy",
                "target_source",
                "depth_m",
                "camera_x_m",
                "camera_y_m",
                "camera_z_m",
                "smooth_x_m",
                "smooth_y_m",
                "smooth_z_m",
                "has_display",
            ]
        )


def append_log(log_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with log_path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        for row in rows:
            writer.writerow(
                [
                    row["timestamp"],
                    row["frame_index"],
                    row["target_id"],
                    f"{row['meter_conf']:.6f}",
                    "" if row["display_conf"] is None else f"{row['display_conf']:.6f}",
                    row["meter_cx"],
                    row["meter_cy"],
                    "" if row["display_cx"] is None else row["display_cx"],
                    "" if row["display_cy"] is None else row["display_cy"],
                    row["target_source"],
                    f"{row['depth_m']:.6f}",
                    _fmt(row["camera_x_m"]),
                    _fmt(row["camera_y_m"]),
                    _fmt(row["camera_z_m"]),
                    _fmt(row["smooth_x_m"]),
                    _fmt(row["smooth_y_m"]),
                    _fmt(row["smooth_z_m"]),
                    int(row["has_display"]),
                ]
            )


def write_robot_ready_targets(ready_path: Path, rows: list[dict]) -> None:
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = ready_path.with_suffix(ready_path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "timestamp",
                "frame_index",
                "target_id",
                "target_source",
                "depth_m",
                "pixel_x",
                "pixel_y",
                "camera_x_m",
                "camera_y_m",
                "camera_z_m",
                "smooth_x_m",
                "smooth_y_m",
                "smooth_z_m",
                "meter_conf",
                "display_conf",
                "stable_frames",
                "status",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["timestamp"],
                    row["frame_index"],
                    row["target_id"],
                    row["target_source"],
                    f"{row['depth_m']:.6f}",
                    row["target_pixel_x"],
                    row["target_pixel_y"],
                    _fmt(row["camera_x_m"]),
                    _fmt(row["camera_y_m"]),
                    _fmt(row["camera_z_m"]),
                    _fmt(row["smooth_x_m"]),
                    _fmt(row["smooth_y_m"]),
                    _fmt(row["smooth_z_m"]),
                    f"{row['meter_conf']:.6f}",
                    "" if row["display_conf"] is None else f"{row['display_conf']:.6f}",
                    row["stable_frames"],
                    row["robot_status"],
                ]
            )
    temporary_path.replace(ready_path)


def send_targets_to_robot_placeholder(ready_rows: list[dict]) -> None:
    """Hook for later robot communication. Keep disabled until calibration is complete."""
    _ = ready_rows


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def draw_box(image, box, label: str, color, thickness: int = 2) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(image, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)


def draw_target(image, pair: dict, row: dict | None) -> None:
    target_id = pair["target_id"]
    meter = pair["meter"]
    display = pair["display"]
    meter_center = meter["center"]

    draw_box(image, meter["box"], f"ID{target_id} meter {meter['confidence']:.2f}", COLORS[0], thickness=3)
    cv2.circle(image, meter_center, 5, COLORS[0], -1)

    if display is not None:
        draw_box(image, display["box"], f"ID{target_id} display {display['confidence']:.2f}", COLORS[1], thickness=2)
        cv2.circle(image, display["center"], 5, COLORS[1], -1)
        cv2.line(image, meter_center, display["center"], (255, 255, 0), 2)

    if row is not None:
        color = (0, 255, 0) if row["robot_ready"] else (0, 255, 255)
        cv2.circle(image, (row["target_pixel_x"], row["target_pixel_y"]), 6, color, -1)


def draw_status_panel(image, rows: list[dict], summary: str) -> None:
    panel_w = 520
    panel_h = 48 + max(1, len(rows)) * 72
    panel_h = min(panel_h, image.shape[0] - 20)
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (panel_w, 10 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.62, image, 0.38, 0, image)

    cv2.putText(image, summary, (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.72, COLORS["text"], 2, cv2.LINE_AA)
    y = 74
    for row in rows:
        status_color = (0, 255, 0) if row["robot_ready"] else (0, 255, 255)
        status = "READY" if row["robot_ready"] else "WAIT"
        depth_text = f"{row['depth_m']:.3f}m" if row["depth_m"] > 0 else "invalid"
        stable_text = (
            f"({row['smooth_x_m']:.3f},{row['smooth_y_m']:.3f},{row['smooth_z_m']:.3f})"
            if row["smooth_x_m"] is not None
            else "(...)"
        )
        line1 = f"ID{row['target_id']} {status} src={row['target_source']} depth={depth_text}"
        line2 = f"stable={stable_text} frames={row['stable_frames']}"
        cv2.putText(image, line1, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2, cv2.LINE_AA)
        cv2.putText(image, line2, (24, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLORS["text"], 1, cv2.LINE_AA)
        y += 72
        if y > 10 + panel_h - 10:
            break


def run(args: argparse.Namespace) -> None:
    model = YOLO(str(args.model))
    write_log_header(args.log)
    write_robot_ready_targets(args.robot_ready_log, [])
    args.snapshot_dir.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, args.depth_width, args.depth_height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.color_width, args.color_height, rs.format.bgr8, args.fps)

    pipeline.start(config)
    align = rs.align(rs.stream.color)

    target_histories: dict[str, deque[tuple[float, float, float]]] = defaultdict(lambda: deque(maxlen=args.smooth_window))
    frame_index = 0
    last_print = 0.0

    print("Realtime multi-meter detection started.")
    print(f"Model: {args.model}")
    print(f"Log: {args.log}")
    print("Keys: q/Esc = quit, s = save snapshot")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            display_image = color_image.copy()
            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

            result = model.predict(
                color_image,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                verbose=False,
                device=args.device if args.device else None,
            )[0]

            meters, displays = collect_detections(result, conf_floor=args.conf)
            meters = meters[: args.max_meters]
            displays = displays[: args.max_displays]
            pairs = pair_meters_and_displays(
                meters,
                displays,
                margin=args.pair_margin,
                max_pair_distance=args.max_pair_distance,
            )
            ordered_targets = sort_targets_grid(pairs, row_tolerance=args.row_tolerance)

            timestamp = time.time()
            rows = []
            row_by_target_id = {}
            robot_ready_rows = []

            for pair in ordered_targets:
                target_id = pair["target_id"]
                meter = pair["meter"]
                display = pair["display"]

                target_detection = display if display is not None and args.coordinate_source == "display" else meter
                target_source = "display" if target_detection is display else "meter"

                depth_m, depth_pixel = get_box_depth_m(depth_frame, target_detection["box"], window_size=args.depth_window)
                if depth_pixel is None:
                    depth_pixel = target_detection["center"]
                xyz = deproject(intrinsics, depth_pixel, depth_m)
                target_key = stable_target_key(pair)
                smooth = median_xyz(target_histories[target_key], xyz)
                stable_frames = min(len(target_histories[target_key]), args.stable_frames)
                robot_ready = (
                    depth_m > 0
                    and xyz is not None
                    and smooth is not None
                    and is_history_stable(target_histories[target_key], args.stable_frames, args.stable_threshold)
                )

                row = {
                    "timestamp": timestamp,
                    "frame_index": frame_index,
                    "target_id": target_id,
                    "meter_conf": meter["confidence"],
                    "display_conf": None if display is None else display["confidence"],
                    "meter_cx": meter["center"][0],
                    "meter_cy": meter["center"][1],
                    "display_cx": None if display is None else display["center"][0],
                    "display_cy": None if display is None else display["center"][1],
                    "target_source": target_source,
                    "target_pixel_x": depth_pixel[0],
                    "target_pixel_y": depth_pixel[1],
                    "depth_m": depth_m,
                    "camera_x_m": None if xyz is None else xyz[0],
                    "camera_y_m": None if xyz is None else xyz[1],
                    "camera_z_m": None if xyz is None else xyz[2],
                    "smooth_x_m": None if smooth is None else smooth[0],
                    "smooth_y_m": None if smooth is None else smooth[1],
                    "smooth_z_m": None if smooth is None else smooth[2],
                    "has_display": display is not None,
                    "stable_frames": stable_frames,
                    "robot_ready": robot_ready,
                    "robot_status": "ready" if robot_ready else "waiting",
                }
                rows.append(row)
                row_by_target_id[target_id] = row
                if robot_ready:
                    robot_ready_rows.append(row)

            append_log(args.log, rows)
            write_robot_ready_targets(args.robot_ready_log, robot_ready_rows)
            send_targets_to_robot_placeholder(robot_ready_rows)

            for pair in ordered_targets:
                draw_target(display_image, pair, row_by_target_id.get(pair["target_id"]))

            summary = f"meters={len(meters)} displays={len(displays)} paired={sum(1 for p in ordered_targets if p['display'] is not None)}"
            draw_status_panel(display_image, rows, summary)

            now = time.time()
            if now - last_print >= args.print_interval:
                last_print = now
                if not rows:
                    print(f"frame={frame_index} no targets")
                else:
                    print(f"frame={frame_index} {summary}")
                    for row in rows:
                        print(
                            f"  ID{row['target_id']} src={row['target_source']} "
                            f"meter=({row['meter_cx']},{row['meter_cy']}) "
                            f"display=({row['display_cx']},{row['display_cy']}) "
                            f"depth={row['depth_m']:.3f}m "
                            f"stable=({_fmt(row['smooth_x_m'])},{_fmt(row['smooth_y_m'])},{_fmt(row['smooth_z_m'])}) "
                            f"robot={row['robot_status']}"
                        )

            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            cv2.imshow("Multi Meter YOLO D435 Color", display_image)
            cv2.imshow("Aligned Depth", depth_colormap)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                snapshot = args.snapshot_dir / f"multi_meter_{frame_index:06d}.jpg"
                cv2.imwrite(str(snapshot), display_image)
                print(f"Saved snapshot: {snapshot}")

            frame_index += 1

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime multi-meter YOLO detection with D435 camera coordinates.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--robot-ready-log", type=Path, default=DEFAULT_ROBOT_READY_PATH)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--device", default="")
    parser.add_argument("--color-width", type=int, default=1280)
    parser.add_argument("--color-height", type=int, default=720)
    parser.add_argument("--depth-width", type=int, default=848)
    parser.add_argument("--depth-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-meters", type=int, default=10)
    parser.add_argument("--max-displays", type=int, default=10)
    parser.add_argument("--pair-margin", type=float, default=40.0)
    parser.add_argument("--max-pair-distance", type=float, default=260.0)
    parser.add_argument("--row-tolerance", type=float, default=140.0)
    parser.add_argument("--coordinate-source", choices=["display", "meter"], default="display")
    parser.add_argument("--depth-window", type=int, default=15)
    parser.add_argument("--smooth-window", type=int, default=15)
    parser.add_argument("--stable-frames", type=int, default=10)
    parser.add_argument("--stable-threshold", type=float, default=0.03)
    parser.add_argument("--print-interval", type=float, default=0.8)
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if args.depth_window <= 0 or args.depth_window % 2 == 0:
        raise ValueError("--depth-window must be a positive odd number")
    if args.smooth_window <= 0:
        raise ValueError("--smooth-window must be greater than 0")
    if args.stable_frames <= 0:
        raise ValueError("--stable-frames must be greater than 0")
    if args.stable_frames > args.smooth_window:
        raise ValueError("--stable-frames must be less than or equal to --smooth-window")
    if args.stable_threshold <= 0:
        raise ValueError("--stable-threshold must be greater than 0")
    if args.max_meters <= 0 or args.max_displays <= 0:
        raise ValueError("--max-meters and --max-displays must be greater than 0")
    return args


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
