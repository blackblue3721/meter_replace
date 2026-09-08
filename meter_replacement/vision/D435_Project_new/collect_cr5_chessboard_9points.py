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
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "calibration_9points_user0"
PATTERN_SIZE = (3, 3)  # 4x4 squares -> 3x3 inner corners.
SQUARE_SIZE_MM = 25.0


def ensure_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "images": root / "images",
        "debug": root / "debug",
        "result": root / "result",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def points_csv_path(root: Path) -> Path:
    return root / "calibration_9points.csv"


def ensure_points_csv(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "pair_id",
                "board_id",
                "point_id",
                "timestamp",
                "image_path",
                "debug_path",
                "camera_x_mm",
                "camera_y_mm",
                "camera_z_mm",
                "robot_x_mm",
                "robot_y_mm",
                "robot_z_mm",
                "robot_rx_deg",
                "robot_ry_deg",
                "robot_rz_deg",
                "reprojection_error_px",
                "user_frame",
                "tool_frame",
                "note",
            ]
        )


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


def next_board_id(csv_path: Path) -> str:
    if not csv_path.exists():
        return "001"
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8-sig")))
    board_ids = [int(row["board_id"]) for row in rows if row.get("board_id", "").isdigit()]
    if not board_ids:
        return "001"
    return f"{max(board_ids) + 1:03d}"


def next_pair_id(csv_path: Path) -> int:
    if not csv_path.exists():
        return 1
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8-sig")))
    pair_ids = [int(row["pair_id"]) for row in rows if row.get("pair_id", "").isdigit()]
    if not pair_ids:
        return 1
    return max(pair_ids) + 1


def parse_pose(text: str) -> tuple[float, float, float, float, float, float] | None:
    parts = [part.strip() for part in text.replace("，", ",").split(",")]
    if len(parts) != 6:
        return None
    try:
        return tuple(float(part) for part in parts)  # type: ignore[return-value]
    except ValueError:
        return None


def make_object_points() -> np.ndarray:
    objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : PATTERN_SIZE[0], 0 : PATTERN_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM
    return objp.astype(np.float64)


def solve_board_pose(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[bool, np.ndarray | None, np.ndarray | None, np.ndarray | None, float | None]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray,
        PATTERN_SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return False, None, None, None, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
    object_points = make_object_points()
    ok, rvec, tvec = cv2.solvePnP(object_points, refined, camera_matrix, dist_coeffs)
    if not ok:
        return False, refined, None, None, None

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    reproj_error = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - refined.reshape(-1, 2), axis=1)))
    return True, refined, rvec, tvec, reproj_error


def camera_points_from_pose(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(rvec)
    object_points = make_object_points()
    return (rotation @ object_points.T).T + tvec.reshape(1, 3)


def draw_help(image: np.ndarray, found: bool, next_board: str) -> None:
    status = "FOUND" if found else "NOT FOUND"
    color = (0, 255, 0) if found else (0, 0, 255)
    lines = [
        f"3x3 inner corners {status} | next board {next_board}",
        "Keys: c=capture this board position, q/Esc=quit",
        "After c: touch P1..P9 shown on image, input User0+Tool2 pose.",
    ]
    y = 30
    for index, line in enumerate(lines):
        line_color = color if index == 0 else (255, 255, 255)
        cv2.putText(image, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, line_color, 2, cv2.LINE_AA)
        y += 28


def draw_labeled_corners(image: np.ndarray, corners: np.ndarray | None) -> np.ndarray:
    display = image.copy()
    if corners is None:
        return display
    cv2.drawChessboardCorners(display, PATTERN_SIZE, corners, True)
    corner_xy = corners.reshape(-1, 2)
    for index, (x, y) in enumerate(corner_xy, start=1):
        label = f"P{index}"
        center = (int(round(x)), int(round(y)))
        cv2.circle(display, center, 7, (0, 0, 255), -1)
        cv2.putText(
            display,
            label,
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return display


def append_point(
    csv_path: Path,
    pair_id: int,
    board_id: str,
    point_id: str,
    image_path: Path,
    debug_path: Path,
    camera_point: np.ndarray,
    pose: tuple[float, float, float, float, float, float],
    reproj_error: float | None,
    user_frame: str,
    tool_frame: str,
    note: str,
) -> None:
    with csv_path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                pair_id,
                board_id,
                point_id,
                f"{time.time():.6f}",
                str(image_path),
                str(debug_path),
                f"{camera_point[0]:.6f}",
                f"{camera_point[1]:.6f}",
                f"{camera_point[2]:.6f}",
                f"{pose[0]:.6f}",
                f"{pose[1]:.6f}",
                f"{pose[2]:.6f}",
                f"{pose[3]:.6f}",
                f"{pose[4]:.6f}",
                f"{pose[5]:.6f}",
                "" if reproj_error is None else f"{reproj_error:.6f}",
                user_frame,
                tool_frame,
                note,
            ]
        )


def collect_board_points(
    args: argparse.Namespace,
    csv_path: Path,
    board_id: str,
    image_path: Path,
    debug_path: Path,
    camera_points: np.ndarray,
    reproj_error: float | None,
) -> None:
    print("")
    print(f"Captured board {board_id}. Now touch the labeled inner corners P1..P9.")
    print("IMPORTANT: DobotStudio must show User 0 + Tool 2, and Tool 2 must be the needle TCP.")
    print("Input format: X,Y,Z,Rx,Ry,Rz")
    print("Type s to skip one point, r to abort this board position, q to finish the program.")
    print("")

    for index in range(9):
        point_id = f"P{index + 1}"
        while True:
            pose_text = input(f"Touch {point_id}, then input CR5 pose: ").strip()
            lowered = pose_text.lower()
            if lowered in {"q", "quit"}:
                raise KeyboardInterrupt
            if lowered in {"r", "redo", "abort"}:
                print(f"Board {board_id} aborted. Already saved points remain in CSV; delete them if needed.")
                return
            if lowered in {"s", "skip"}:
                print(f"Skipped {point_id}.")
                break

            pose = parse_pose(pose_text)
            if pose is None:
                print("Invalid pose. Use six numbers separated by commas, or s/r/q.")
                continue

            pair_id = next_pair_id(csv_path)
            append_point(
                csv_path,
                pair_id,
                board_id,
                point_id,
                image_path,
                debug_path,
                camera_points[index],
                pose,
                reproj_error,
                args.user_frame,
                args.tool_frame,
                args.note,
            )
            print(f"Saved pair {pair_id:04d}: board {board_id} {point_id}")
            break


def run(args: argparse.Namespace) -> None:
    paths = ensure_dirs(args.output_dir)
    csv_path = points_csv_path(paths["root"])
    ensure_points_csv(csv_path)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, args.depth_width, args.depth_height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.color_width, args.color_height, rs.format.bgr8, args.fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_profile.get_intrinsics()
    save_intrinsics(paths["root"], intrinsics)

    camera_matrix = np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float64).reshape(-1, 1)

    print("CR5 9-point chessboard collector started.")
    print(f"Output: {paths['root']}")
    print("One board position gives up to 9 point pairs: P1..P9.")
    print("Recommended: 3-4 board positions, total 27-36 point pairs.")
    print("In DobotStudio use User 0 + Tool 2 before typing robot poses.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            ok, corners, rvec, tvec, reproj_error = solve_board_pose(color_image, camera_matrix, dist_coeffs)

            display = draw_labeled_corners(color_image, corners if ok else None)
            board_id = next_board_id(csv_path)
            draw_help(display, ok, board_id)
            cv2.imshow("CR5 9-Point Chessboard Collector", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key != ord("c"):
                continue

            if not ok or rvec is None or tvec is None:
                print("Current frame has no valid 3x3 chessboard. Adjust the board and try again.")
                continue

            image_path = paths["images"] / f"board_{board_id}.jpg"
            debug_path = paths["debug"] / f"board_{board_id}_points.jpg"
            labeled = draw_labeled_corners(color_image, corners)
            cv2.imwrite(str(image_path), color_image)
            cv2.imwrite(str(debug_path), labeled)

            camera_points = camera_points_from_pose(rvec, tvec)
            collect_board_points(args, csv_path, board_id, image_path, debug_path, camera_points, reproj_error)

    except KeyboardInterrupt:
        print("Collection stopped by user.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect CR5 needle point pairs for all 3x3 inner chessboard corners.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--color-width", type=int, default=1280)
    parser.add_argument("--color-height", type=int, default=720)
    parser.add_argument("--depth-width", type=int, default=848)
    parser.add_argument("--depth-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--user-frame", default="User 0")
    parser.add_argument("--tool-frame", default="Tool 2")
    parser.add_argument("--note", default="manual needle touch, 3x3 inner corners")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
