from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_READY_TARGETS = PROJECT_DIR / "logs/robot_ready_targets.csv"
DEFAULT_TRANSFORM = PROJECT_DIR / "calibration/result/camera_to_cr5_transform.json"
DEFAULT_OUTPUT = PROJECT_DIR / "logs/cr5_ready_targets.csv"


def load_transform(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    rotation = np.array(data["rotation_camera_to_robot"], dtype=np.float64)
    translation = np.array(data["translation_mm"], dtype=np.float64).reshape(3)
    return rotation, translation, data


def read_ready_targets(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Ready target file not found: {path}")
    with path.open("r", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def parse_float(row: dict, key: str) -> float | None:
    value = row.get(key, "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def camera_m_to_cr5_mm(camera_xyz_m: np.ndarray, rotation: np.ndarray, translation_mm: np.ndarray) -> np.ndarray:
    camera_xyz_mm = camera_xyz_m * 1000.0
    return rotation @ camera_xyz_mm + translation_mm


def make_approach_point(cr5_xyz_mm: np.ndarray, axis: str, offset_mm: float) -> np.ndarray:
    approach = cr5_xyz_mm.copy()
    axis = axis.lower()
    sign = 1.0
    if axis.startswith("-"):
        sign = -1.0
        axis = axis[1:]

    axis_to_index = {"x": 0, "y": 1, "z": 2}
    if axis not in axis_to_index:
        raise ValueError("--approach-axis must be one of x, y, z, -x, -y, -z")
    approach[axis_to_index[axis]] += sign * offset_mm
    return approach


def convert_targets(args: argparse.Namespace) -> list[dict]:
    rotation, translation_mm, transform_meta = load_transform(args.transform)
    rows = read_ready_targets(args.ready_targets)

    converted = []
    skipped = []

    for row in rows:
        status = row.get("status", "").strip().lower()
        target_id = row.get("target_id", "").strip()
        sx = parse_float(row, "smooth_x_m")
        sy = parse_float(row, "smooth_y_m")
        sz = parse_float(row, "smooth_z_m")
        depth = parse_float(row, "depth_m")

        if status != "ready":
            skipped.append((target_id, "status is not ready"))
            continue
        if sx is None or sy is None or sz is None:
            skipped.append((target_id, "missing smooth camera coordinates"))
            continue
        if depth is None or depth <= 0:
            skipped.append((target_id, "invalid depth"))
            continue

        camera_xyz_m = np.array([sx, sy, sz], dtype=np.float64)
        cr5_xyz_mm = camera_m_to_cr5_mm(camera_xyz_m, rotation, translation_mm)
        approach_xyz_mm = make_approach_point(cr5_xyz_mm, args.approach_axis, args.approach_offset_mm)

        converted.append(
            {
                "target_id": target_id,
                "source_status": status,
                "target_source": row.get("target_source", ""),
                "depth_m": depth,
                "camera_x_m": sx,
                "camera_y_m": sy,
                "camera_z_m": sz,
                "cr5_x_mm": cr5_xyz_mm[0],
                "cr5_y_mm": cr5_xyz_mm[1],
                "cr5_z_mm": cr5_xyz_mm[2],
                "approach_x_mm": approach_xyz_mm[0],
                "approach_y_mm": approach_xyz_mm[1],
                "approach_z_mm": approach_xyz_mm[2],
                "approach_axis": args.approach_axis,
                "approach_offset_mm": args.approach_offset_mm,
                "meter_conf": row.get("meter_conf", ""),
                "display_conf": row.get("display_conf", ""),
                "stable_frames": row.get("stable_frames", ""),
                "transform_mean_error_mm": transform_meta.get("mean_error_mm", ""),
                "transform_max_error_mm": transform_meta.get("max_error_mm", ""),
                "robot_action_status": "verify_manually_first",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = [
            "target_id",
            "source_status",
            "target_source",
            "depth_m",
            "camera_x_m",
            "camera_y_m",
            "camera_z_m",
            "cr5_x_mm",
            "cr5_y_mm",
            "cr5_z_mm",
            "approach_x_mm",
            "approach_y_mm",
            "approach_z_mm",
            "approach_axis",
            "approach_offset_mm",
            "meter_conf",
            "display_conf",
            "stable_frames",
            "transform_mean_error_mm",
            "transform_max_error_mm",
            "robot_action_status",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in converted:
            writer.writerow(
                {
                    key: f"{value:.6f}" if isinstance(value, float) else value
                    for key, value in item.items()
                }
            )

    print(f"Input ready targets: {args.ready_targets}")
    print(f"Input transform: {args.transform}")
    print(f"Output CR5 targets: {args.output}")
    print(f"Converted targets: {len(converted)}")
    if skipped:
        print(f"Skipped targets: {len(skipped)}")
        for target_id, reason in skipped:
            print(f"  ID{target_id}: {reason}")
    for item in converted:
        print(
            f"ID{item['target_id']} "
            f"target=({item['cr5_x_mm']:.2f}, {item['cr5_y_mm']:.2f}, {item['cr5_z_mm']:.2f}) mm "
            f"approach=({item['approach_x_mm']:.2f}, {item['approach_y_mm']:.2f}, {item['approach_z_mm']:.2f}) mm"
        )

    return converted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert D435 ready target coordinates to CR5 robot coordinates.")
    parser.add_argument("--ready-targets", type=Path, default=DEFAULT_READY_TARGETS)
    parser.add_argument("--transform", type=Path, default=DEFAULT_TRANSFORM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--approach-axis",
        default="z",
        help="CR5 axis used for a temporary safe approach point: x, y, z, -x, -y, -z.",
    )
    parser.add_argument("--approach-offset-mm", type=float, default=100.0)
    args = parser.parse_args()

    if args.approach_offset_mm < 0:
        raise ValueError("--approach-offset-mm must be greater than or equal to 0")
    return args


def main() -> None:
    convert_targets(parse_args())


if __name__ == "__main__":
    main()
