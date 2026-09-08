from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "logs/cr5_ready_targets.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "logs/cr5_ready_targets_user2.csv"
DEFAULT_USER_FRAME = PROJECT_DIR / "user2_frame_config.json"


def rot_x(deg: float) -> np.ndarray:
    rad = math.radians(deg)
    c = math.cos(rad)
    s = math.sin(rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def rot_y(deg: float) -> np.ndarray:
    rad = math.radians(deg)
    c = math.cos(rad)
    s = math.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def rot_z(deg: float) -> np.ndarray:
    rad = math.radians(deg)
    c = math.cos(rad)
    s = math.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def euler_to_rotation(rx_deg: float, ry_deg: float, rz_deg: float, convention: str) -> np.ndarray:
    convention = convention.lower().strip()
    rotations = {
        "x": rot_x(rx_deg),
        "y": rot_y(ry_deg),
        "z": rot_z(rz_deg),
    }

    if convention == "rzyx":
        return rotations["z"] @ rotations["y"] @ rotations["x"]
    if convention == "rxyz":
        return rotations["x"] @ rotations["y"] @ rotations["z"]
    if convention == "xyz":
        return rotations["x"] @ rotations["y"] @ rotations["z"]
    if convention == "zyx":
        return rotations["z"] @ rotations["y"] @ rotations["x"]
    raise ValueError("--rotation-convention must be one of: rzyx, rxyz, xyz, zyx")


def load_user_frame(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    required = ["x_mm", "y_mm", "z_mm", "rx_deg", "ry_deg", "rz_deg"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}")
    return data


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input target file not found: {path}")
    with path.open("r", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def parse_point(row: dict[str, str], prefix: str) -> np.ndarray | None:
    keys = [f"{prefix}_x_mm", f"{prefix}_y_mm", f"{prefix}_z_mm"]
    values = []
    for key in keys:
        text = row.get(key, "").strip()
        if not text:
            return None
        try:
            values.append(float(text))
        except ValueError:
            return None
    return np.array(values, dtype=np.float64)


def user0_to_user(user0_point_mm: np.ndarray, user_origin_in_user0: np.ndarray, user_rotation_in_user0: np.ndarray) -> np.ndarray:
    return user_rotation_in_user0.T @ (user0_point_mm - user_origin_in_user0)


def fmt(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def convert(args: argparse.Namespace) -> list[dict[str, object]]:
    frame = load_user_frame(args.user_frame)
    convention = args.rotation_convention or frame.get("rotation_convention", "rzyx")

    user_origin = np.array([frame["x_mm"], frame["y_mm"], frame["z_mm"]], dtype=np.float64)
    user_rotation = euler_to_rotation(frame["rx_deg"], frame["ry_deg"], frame["rz_deg"], convention)
    rows = read_rows(args.input)

    output_rows: list[dict[str, object]] = []
    for row in rows:
        target_user0 = parse_point(row, "cr5")
        approach_user0 = parse_point(row, "approach")
        if target_user0 is None:
            continue

        target_user2 = user0_to_user(target_user0, user_origin, user_rotation)
        approach_user2 = None
        if approach_user0 is not None:
            approach_user2 = user0_to_user(approach_user0, user_origin, user_rotation)

        out: dict[str, object] = dict(row)
        out.update(
            {
                "user_frame_output": frame.get("user_frame_index", 2),
                "user2_x_mm": float(target_user2[0]),
                "user2_y_mm": float(target_user2[1]),
                "user2_z_mm": float(target_user2[2]),
                "user2_approach_x_mm": "" if approach_user2 is None else float(approach_user2[0]),
                "user2_approach_y_mm": "" if approach_user2 is None else float(approach_user2[1]),
                "user2_approach_z_mm": "" if approach_user2 is None else float(approach_user2[2]),
                "user2_frame_x_mm": frame["x_mm"],
                "user2_frame_y_mm": frame["y_mm"],
                "user2_frame_z_mm": frame["z_mm"],
                "user2_frame_rx_deg": frame["rx_deg"],
                "user2_frame_ry_deg": frame["ry_deg"],
                "user2_frame_rz_deg": frame["rz_deg"],
                "rotation_convention": convention,
                "robot_action_status": "verify_user2_manually_first",
            }
        )
        output_rows.append(out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if output_rows:
        base_fields = list(rows[0].keys()) if rows else []
        extra_fields = [
            "user_frame_output",
            "user2_x_mm",
            "user2_y_mm",
            "user2_z_mm",
            "user2_approach_x_mm",
            "user2_approach_y_mm",
            "user2_approach_z_mm",
            "user2_frame_x_mm",
            "user2_frame_y_mm",
            "user2_frame_z_mm",
            "user2_frame_rx_deg",
            "user2_frame_ry_deg",
            "user2_frame_rz_deg",
            "rotation_convention",
        ]
        fieldnames = base_fields + [field for field in extra_fields if field not in base_fields]
    else:
        fieldnames = []

    with args.output.open("w", newline="", encoding="utf-8-sig") as file:
        if fieldnames:
            writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in output_rows:
                writer.writerow({key: fmt(value) for key, value in row.items()})

    print(f"Input User0 targets: {args.input}")
    print(f"User2 frame config: {args.user_frame}")
    print(f"Output User2 targets: {args.output}")
    print(f"Rotation convention: {convention}")
    print(f"Converted targets: {len(output_rows)}")
    for row in output_rows:
        target_id = row.get("target_id", "")
        print(
            f"ID{target_id} "
            f"user2_target=({float(row['user2_x_mm']):.2f}, {float(row['user2_y_mm']):.2f}, {float(row['user2_z_mm']):.2f}) mm "
            f"user2_approach=({float(row['user2_approach_x_mm']):.2f}, {float(row['user2_approach_y_mm']):.2f}, {float(row['user2_approach_z_mm']):.2f}) mm"
        )

    return output_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CR5 target points from User 0/base coordinates to User 2 coordinates.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--user-frame", type=Path, default=DEFAULT_USER_FRAME)
    parser.add_argument(
        "--rotation-convention",
        default=None,
        help="Euler convention for the User2 frame pose. Default reads user2_frame_config.json. Supported: rzyx, rxyz, xyz, zyx.",
    )
    return parser.parse_args()


def main() -> None:
    convert(parse_args())


if __name__ == "__main__":
    main()
