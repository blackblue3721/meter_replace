from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CALIB_DIR = PROJECT_DIR / "calibration_9points_user0"
PATTERN_SIZE = (3, 3)
SQUARE_SIZE_MM = 25.0


def rigid_transform_3d(source_points: np.ndarray, target_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid
    h = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(h)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = target_centroid - rotation @ source_centroid
    return rotation, translation


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def parse_excluded_points(text: str) -> set[str]:
    excluded = set()
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        excluded.add(item)
    return excluded


def point_key(row: dict[str, str]) -> str:
    return f"{row.get('board_id', '')}:{row.get('point_id', '')}"


def warn_geometry(camera_points: np.ndarray) -> str | None:
    centered = camera_points - np.mean(camera_points, axis=0)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    if len(singular_values) < 3:
        return "Point geometry is degenerate."
    if singular_values[2] < 1e-6:
        return "All camera points are nearly coplanar; Z/depth accuracy may be weak."
    ratio = singular_values[2] / singular_values[0]
    if ratio < 0.03:
        return f"Depth coverage is small (S3/S1={ratio:.4f}); add one board position with 30-60 mm depth change."
    return None


def run(args: argparse.Namespace) -> None:
    csv_path = args.calib_dir / "calibration_9points.csv"
    result_dir = args.calib_dir / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing point-pair CSV: {csv_path}")

    rows = load_rows(csv_path)
    excluded = parse_excluded_points(args.exclude_points)
    used_rows = []
    rejected = []
    camera_points = []
    robot_points = []

    for row in rows:
        key = point_key(row)
        pair_id = row.get("pair_id", "")
        if key in excluded or pair_id in excluded:
            rejected.append({"pair_id": pair_id, "board_id": row.get("board_id", ""), "point_id": row.get("point_id", ""), "reason": "excluded by user"})
            continue
        try:
            reproj_text = row.get("reprojection_error_px", "")
            reproj_error = None if reproj_text == "" else float(reproj_text)
            if reproj_error is not None and reproj_error > args.max_reprojection_error:
                rejected.append({"pair_id": pair_id, "board_id": row.get("board_id", ""), "point_id": row.get("point_id", ""), "reason": f"high reprojection error {reproj_error:.3f}px"})
                continue
            camera_point = np.array(
                [float(row["camera_x_mm"]), float(row["camera_y_mm"]), float(row["camera_z_mm"])],
                dtype=np.float64,
            )
            robot_point = np.array(
                [float(row["robot_x_mm"]), float(row["robot_y_mm"]), float(row["robot_z_mm"])],
                dtype=np.float64,
            )
        except (KeyError, ValueError) as exc:
            rejected.append({"pair_id": pair_id, "board_id": row.get("board_id", ""), "point_id": row.get("point_id", ""), "reason": f"bad row: {exc}"})
            continue

        used_rows.append((row, camera_point, robot_point, reproj_error))
        camera_points.append(camera_point)
        robot_points.append(robot_point)

    if len(camera_points) < 4:
        raise RuntimeError(f"Need at least 4 valid point pairs, got {len(camera_points)}.")

    camera_points_np = np.array(camera_points, dtype=np.float64)
    robot_points_np = np.array(robot_points, dtype=np.float64)
    rotation, translation = rigid_transform_3d(camera_points_np, robot_points_np)

    predicted = (rotation @ camera_points_np.T).T + translation
    errors = np.linalg.norm(predicted - robot_points_np, axis=1)
    geometry_warning = warn_geometry(camera_points_np)

    transform = {
        "description": "CR5 robot base point = rotation_camera_to_robot * D435 camera point + translation_mm",
        "assumption": "Robot XYZ values are recorded in DobotStudio User 0 with Tool 2 needle TCP, and each point_id was touched exactly as labeled in the collector debug image.",
        "unit": "mm",
        "method": "manual needle touch on 3x3 inner chessboard corners over multiple board positions",
        "pattern_size_inner_corners": list(PATTERN_SIZE),
        "square_size_mm": SQUARE_SIZE_MM,
        "rotation_camera_to_robot": rotation.tolist(),
        "translation_mm": translation.tolist(),
        "valid_point_count": len(used_rows),
        "mean_error_mm": float(np.mean(errors)),
        "median_error_mm": float(np.median(errors)),
        "max_error_mm": float(np.max(errors)),
        "geometry_warning": geometry_warning,
        "rejected_points": rejected,
    }

    transform_path = result_dir / "camera_to_cr5_transform.json"
    with transform_path.open("w", encoding="utf-8") as file:
        json.dump(transform, file, indent=2)

    report_path = result_dir / "calibration_9points_report.csv"
    with report_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "pair_id",
                "board_id",
                "point_id",
                "camera_x_mm",
                "camera_y_mm",
                "camera_z_mm",
                "robot_x_mm",
                "robot_y_mm",
                "robot_z_mm",
                "pred_robot_x_mm",
                "pred_robot_y_mm",
                "pred_robot_z_mm",
                "error_mm",
                "reprojection_error_px",
                "image_path",
                "debug_path",
            ]
        )
        for index, (row, camera_point, robot_point, reproj_error) in enumerate(used_rows):
            pred = predicted[index]
            writer.writerow(
                [
                    row.get("pair_id", ""),
                    row.get("board_id", ""),
                    row.get("point_id", ""),
                    f"{camera_point[0]:.6f}",
                    f"{camera_point[1]:.6f}",
                    f"{camera_point[2]:.6f}",
                    f"{robot_point[0]:.6f}",
                    f"{robot_point[1]:.6f}",
                    f"{robot_point[2]:.6f}",
                    f"{pred[0]:.6f}",
                    f"{pred[1]:.6f}",
                    f"{pred[2]:.6f}",
                    f"{errors[index]:.6f}",
                    "" if reproj_error is None else f"{reproj_error:.6f}",
                    row.get("image_path", ""),
                    row.get("debug_path", ""),
                ]
            )

    worst_path = result_dir / "worst_9points.csv"
    order = np.argsort(errors)[::-1]
    with worst_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["rank", "pair_id", "board_id", "point_id", "error_mm", "reprojection_error_px", "debug_path"])
        for rank, index in enumerate(order, start=1):
            row, _, _, reproj_error = used_rows[int(index)]
            writer.writerow(
                [
                    rank,
                    row.get("pair_id", ""),
                    row.get("board_id", ""),
                    row.get("point_id", ""),
                    f"{errors[int(index)]:.6f}",
                    "" if reproj_error is None else f"{reproj_error:.6f}",
                    row.get("debug_path", ""),
                ]
            )

    print(f"Valid point pairs: {len(used_rows)}")
    print(f"Rejected point pairs: {len(rejected)}")
    print(f"Mean error: {np.mean(errors):.3f} mm")
    print(f"Median error: {np.median(errors):.3f} mm")
    print(f"Max error: {np.max(errors):.3f} mm")
    if geometry_warning:
        print(f"Geometry warning: {geometry_warning}")
    print(f"Saved transform: {transform_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved worst-point list: {worst_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate D435 camera to CR5 base transform from 3x3 chessboard point pairs.")
    parser.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB_DIR)
    parser.add_argument("--max-reprojection-error", type=float, default=1.5)
    parser.add_argument(
        "--exclude-points",
        default="",
        help="Comma-separated pair_id or board_id:point_id values to exclude, e.g. 7,002:P5,003:P9",
    )
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
