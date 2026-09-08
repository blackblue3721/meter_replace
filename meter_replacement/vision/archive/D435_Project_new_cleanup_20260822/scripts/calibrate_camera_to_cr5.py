from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CALIB_DIR = PROJECT_DIR / "calibration"
PATTERN_SIZE = (3, 3)
SQUARE_SIZE_MM = 25.0


def load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    camera_matrix = np.array(
        [
            [data["fx"], 0.0, data["ppx"]],
            [0.0, data["fy"], data["ppy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.array(data.get("coeffs", [0, 0, 0, 0, 0]), dtype=np.float64).reshape(-1, 1)
    return camera_matrix, dist_coeffs


def make_object_points() -> np.ndarray:
    objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : PATTERN_SIZE[0], 0 : PATTERN_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM
    return objp.astype(np.float64)


def chessboard_center_camera_mm(image_path: Path, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, debug_dir: Path) -> tuple[np.ndarray | None, float | None]:
    image = cv2.imread(str(image_path))
    if image is None:
        return None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray,
        PATTERN_SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return None, None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    refined = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
    object_points = make_object_points()
    ok, rvec, tvec = cv2.solvePnP(object_points, refined, camera_matrix, dist_coeffs)
    if not ok:
        return None, None

    rotation, _ = cv2.Rodrigues(rvec)
    board_center_obj = np.array([[SQUARE_SIZE_MM], [SQUARE_SIZE_MM], [0.0]], dtype=np.float64)
    center_camera = rotation @ board_center_obj + tvec

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    reproj_error = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - refined.reshape(-1, 2), axis=1)))

    debug = image.copy()
    cv2.drawChessboardCorners(debug, PATTERN_SIZE, refined, True)
    center_px, _ = cv2.projectPoints(board_center_obj.reshape(1, 3), rvec, tvec, camera_matrix, dist_coeffs)
    cx, cy = center_px.reshape(-1, 2)[0]
    cv2.circle(debug, (int(round(cx)), int(round(cy))), 8, (0, 0, 255), -1)
    cv2.putText(debug, "board center", (int(round(cx)) + 10, int(round(cy))), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / f"{image_path.stem}_pnp.jpg"), debug)

    return center_camera.reshape(3), reproj_error


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


def load_samples(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def run(args: argparse.Namespace) -> None:
    csv_path = args.calib_dir / "calibration_samples.csv"
    intrinsics_path = args.calib_dir / "camera_intrinsics.json"
    result_dir = args.calib_dir / "result"
    debug_dir = args.calib_dir / "debug"
    result_dir.mkdir(parents=True, exist_ok=True)

    camera_matrix, dist_coeffs = load_intrinsics(intrinsics_path)
    samples = load_samples(csv_path)
    camera_points = []
    robot_points = []
    used_rows = []
    rejected = []

    for row in samples:
        image_path = Path(row["image_path"])
        center_camera_mm, reproj_error = chessboard_center_camera_mm(image_path, camera_matrix, dist_coeffs, debug_dir)
        if center_camera_mm is None:
            rejected.append((row["sample_id"], "chessboard not found"))
            continue
        if reproj_error is not None and reproj_error > args.max_reprojection_error:
            rejected.append((row["sample_id"], f"high reprojection error {reproj_error:.3f}px"))
            continue

        robot_point = np.array(
            [
                float(row["robot_x_mm"]),
                float(row["robot_y_mm"]),
                float(row["robot_z_mm"]),
            ],
            dtype=np.float64,
        )
        camera_points.append(center_camera_mm)
        robot_points.append(robot_point)
        used_rows.append((row, center_camera_mm, robot_point, reproj_error))

    if len(camera_points) < 4:
        raise RuntimeError(f"Need at least 4 valid samples, got {len(camera_points)}.")

    camera_points_np = np.array(camera_points, dtype=np.float64)
    robot_points_np = np.array(robot_points, dtype=np.float64)
    rotation, translation = rigid_transform_3d(camera_points_np, robot_points_np)

    predicted = (rotation @ camera_points_np.T).T + translation
    errors = np.linalg.norm(predicted - robot_points_np, axis=1)

    transform = {
        "description": "CR5 robot base point = rotation_camera_to_robot * D435 camera point + translation_mm",
        "assumption": "CR5 TCP coordinates recorded in calibration_samples.csv correspond to the chessboard center.",
        "unit": "mm",
        "pattern_size_inner_corners": list(PATTERN_SIZE),
        "square_size_mm": SQUARE_SIZE_MM,
        "rotation_camera_to_robot": rotation.tolist(),
        "translation_mm": translation.tolist(),
        "valid_sample_count": len(used_rows),
        "mean_error_mm": float(np.mean(errors)),
        "max_error_mm": float(np.max(errors)),
        "median_error_mm": float(np.median(errors)),
        "rejected_samples": [{"sample_id": sid, "reason": reason} for sid, reason in rejected],
    }

    transform_path = result_dir / "camera_to_cr5_transform.json"
    with transform_path.open("w", encoding="utf-8") as file:
        json.dump(transform, file, indent=2)

    report_path = result_dir / "calibration_report.csv"
    with report_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "sample_id",
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
            ]
        )
        for index, (row, camera_point, robot_point, reproj_error) in enumerate(used_rows):
            pred = predicted[index]
            writer.writerow(
                [
                    row["sample_id"],
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
                ]
            )

    print(f"Valid samples: {len(used_rows)}")
    print(f"Rejected samples: {len(rejected)}")
    print(f"Mean error: {np.mean(errors):.3f} mm")
    print(f"Median error: {np.median(errors):.3f} mm")
    print(f"Max error: {np.max(errors):.3f} mm")
    print(f"Saved transform: {transform_path}")
    print(f"Saved report: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate D435 camera coordinates to CR5 robot base coordinates.")
    parser.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB_DIR)
    parser.add_argument("--max-reprojection-error", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
