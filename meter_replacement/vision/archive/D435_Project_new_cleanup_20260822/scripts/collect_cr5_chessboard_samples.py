from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "calibration"
PATTERN_SIZE = (3, 3)  # Your board is 4x4 squares, so OpenCV sees 3x3 inner corners.
SQUARE_SIZE_MM = 25.0


def ensure_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "images": root / "images",
        "depth": root / "depth",
        "debug": root / "debug",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def samples_csv_path(root: Path) -> Path:
    return root / "calibration_samples.csv"


def ensure_samples_csv(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "sample_id",
                "timestamp",
                "image_path",
                "depth_path",
                "debug_path",
                "robot_x_mm",
                "robot_y_mm",
                "robot_z_mm",
                "robot_rx_deg",
                "robot_ry_deg",
                "robot_rz_deg",
            ]
        )


def next_sample_id(csv_path: Path) -> str:
    if not csv_path.exists():
        return "001"
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8-sig")))
    if not rows:
        return "001"
    max_id = max(int(row["sample_id"]) for row in rows if row.get("sample_id", "").isdigit())
    return f"{max_id + 1:03d}"


def parse_pose(text: str) -> tuple[float, float, float, float, float, float] | None:
    parts = [part.strip() for part in text.replace("，", ",").split(",")]
    if len(parts) != 6:
        return None
    try:
        return tuple(float(part) for part in parts)  # type: ignore[return-value]
    except ValueError:
        return None


def save_intrinsics(root: Path, intrinsics) -> None:
    data = {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "ppx": intrinsics.ppx,
        "ppy": intrinsics.ppy,
        "coeffs": list(intrinsics.coeffs),
        "model": str(intrinsics.model),
    }
    with (root / "camera_intrinsics.json").open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def draw_help(image, found: bool, sample_id: str) -> None:
    status = "FOUND" if found else "NOT FOUND"
    color = (0, 255, 0) if found else (0, 0, 255)
    lines = [
        f"Chessboard {status} | next sample {sample_id}",
        "Keys: c=collect current frame, q/Esc=quit",
        "Board: 4x4 squares -> pattern 3x3 inner corners, square=25mm",
    ]
    y = 30
    for index, line in enumerate(lines):
        line_color = color if index == 0 else (255, 255, 255)
        cv2.putText(image, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, line_color, 2, cv2.LINE_AA)
        y += 28


def append_sample(csv_path: Path, row: list[object]) -> None:
    with csv_path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(row)


def run(args: argparse.Namespace) -> None:
    paths = ensure_dirs(args.output_dir)
    csv_path = samples_csv_path(paths["root"])
    ensure_samples_csv(csv_path)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, args.depth_width, args.depth_height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.color_width, args.color_height, rs.format.bgr8, args.fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    save_intrinsics(paths["root"], color_profile.get_intrinsics())

    print("CR5 chessboard sample collector started.")
    print(f"Output: {paths['root']}")
    print("Move CR5 to a pose, wait until the board is detected, then press c.")
    print("After pressing c, type CR5 current pose as: X,Y,Z,Rx,Ry,Rz")
    print("Example: 350.2,-120.5,420.8,178.2,0.5,91.3")

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            found, corners = cv2.findChessboardCorners(
                gray,
                PATTERN_SIZE,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )

            display = color_image.copy()
            refined = None
            if found:
                refined = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
                cv2.drawChessboardCorners(display, PATTERN_SIZE, refined, found)

            sample_id = next_sample_id(csv_path)
            draw_help(display, found, sample_id)
            cv2.imshow("CR5 Chessboard Collector", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key != ord("c"):
                continue

            if not found:
                print("Current frame has no valid 3x3 chessboard corners. Move/adjust board and try again.")
                continue

            pose_text = input(f"Input CR5 pose for sample {sample_id} as X,Y,Z,Rx,Ry,Rz: ").strip()
            pose = parse_pose(pose_text)
            if pose is None:
                print("Invalid pose. Sample was not saved. Please use six numbers separated by commas.")
                continue

            image_path = paths["images"] / f"sample_{sample_id}.jpg"
            depth_path = paths["depth"] / f"sample_{sample_id}_depth.npy"
            debug_path = paths["debug"] / f"sample_{sample_id}_corners.jpg"

            cv2.imwrite(str(image_path), color_image)
            np.save(str(depth_path), depth_image)
            cv2.imwrite(str(debug_path), display)

            append_sample(
                csv_path,
                [
                    sample_id,
                    f"{time.time():.6f}",
                    str(image_path),
                    str(depth_path),
                    str(debug_path),
                    f"{pose[0]:.6f}",
                    f"{pose[1]:.6f}",
                    f"{pose[2]:.6f}",
                    f"{pose[3]:.6f}",
                    f"{pose[4]:.6f}",
                    f"{pose[5]:.6f}",
                ],
            )
            print(f"Saved sample {sample_id}: {image_path}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect D435 chessboard images and CR5 TCP poses.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--color-width", type=int, default=1280)
    parser.add_argument("--color-height", type=int, default=720)
    parser.add_argument("--depth-width", type=int, default=848)
    parser.add_argument("--depth-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
