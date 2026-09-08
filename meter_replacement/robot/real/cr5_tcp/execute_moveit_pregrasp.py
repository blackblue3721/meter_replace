#!/usr/bin/env python3
"""Execute the approved straight MoveIt path from commissioned prefix to pregrasp.

Default mode is dry-run.  Execution is limited to one exported trajectory SHA,
requires the prior small segment receipt, verifies that all MoveIt points form
the same monotonic joint-space line used by JointMovJ, and never enables or
recovers the robot automatically.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from commission_j6_step import load_config, read_dashboard_state
from commission_moveit_pregrasp_segment import (
    DEFAULT_TRAJECTORY,
    PROJECT_ROOT,
    execute_joint_target,
    ros_to_vendor_deg,
    verify_digest,
    verify_static_feedback,
)
from preflight_cr5 import read_one_feedback
from read_cr5_state import DEFAULT_CONFIG


PREFIX_RECEIPT = PROJECT_ROOT / "logs/commissioning/cr5_moveit_pregrasp_segment.json"
FULL_RECEIPT = PROJECT_ROOT / "logs/commissioning/cr5_moveit_pregrasp_full.json"
CONFIRMATION = "MOVE_REAL_CR5_TO_PREGRASP"


def save_receipt(payload: dict) -> None:
    FULL_RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    FULL_RECEIPT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--trajectory-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())
    authorization = config.get("commissioning_authorization", {})
    if config.get("access_mode") != "commissioning":
        raise RuntimeError("安全拒绝：当前不处于commissioning模式")
    if authorization.get("scope") != "moveit_pregrasp_full_path_only":
        raise RuntimeError("安全拒绝：当前授权范围不允许完整预抓取路径")
    if not authorization.get("operator_safety_check_confirmed", False):
        raise RuntimeError("安全拒绝：没有现场安全确认记录")
    if authorization.get("full_workflow_authorized", False):
        raise RuntimeError("配置异常：完整工艺仍不应被授权")

    trajectory = json.loads(args.trajectory.resolve().read_text(encoding="utf-8"))
    digest = verify_digest(trajectory)
    if args.trajectory_sha256 != digest:
        raise RuntimeError("安全拒绝：命令行SHA256与轨迹文件不一致")
    if trajectory.get("target_stage") != "pregrasp":
        raise RuntimeError("安全拒绝：只允许pregrasp轨迹")
    if trajectory.get("home_profile") != "real_commissioning":
        raise RuntimeError("安全拒绝：轨迹Home配置不正确")

    if not PREFIX_RECEIPT.exists():
        raise RuntimeError("安全拒绝：缺少已通过的小段执行记录")
    prefix = json.loads(PREFIX_RECEIPT.read_text(encoding="utf-8"))
    if prefix.get("status") != "segment_complete":
        raise RuntimeError("安全拒绝：小段执行尚未成功")
    if prefix.get("trajectory_sha256") != digest:
        raise RuntimeError("安全拒绝：小段记录与待执行轨迹不是同一SHA")

    joint_names = list(trajectory["joint_names"])
    points_rad = np.asarray(
        [point["positions_rad"] for point in trajectory["points"]], dtype=float
    )
    if len(points_rad) < 2:
        raise RuntimeError("安全拒绝：轨迹点不足")
    line = points_rad[-1] - points_rad[0]
    squared_norm = float(line @ line)
    if squared_norm <= 1.0e-12:
        raise RuntimeError("安全拒绝：轨迹起终点相同")
    progress = ((points_rad - points_rad[0]) @ line) / squared_norm
    projected = points_rad[0] + progress[:, None] * line
    maximum_deviation_deg = float(
        np.max(np.abs(np.rad2deg(points_rad - projected)))
    )
    deviation_limit = float(
        config["commissioning_limits"][
            "moveit_joint_line_deviation_tolerance_deg"
        ]
    )
    if np.any(np.diff(progress) < -1.0e-10):
        raise RuntimeError("安全拒绝：MoveIt轨迹进度不是单调的")
    if maximum_deviation_deg > deviation_limit:
        raise RuntimeError(
            f"安全拒绝：MoveIt路径不是JointMovJ直线，偏差={maximum_deviation_deg}°"
        )

    mapping = config["joint_mapping"]
    target_deg = ros_to_vendor_deg(points_rad[-1], joint_names, mapping)
    expected_current_deg = np.asarray(prefix["target_deg"], dtype=float)

    dashboard, dashboard_mode, dashboard_deg = read_dashboard_state(config)
    try:
        actual_deg = verify_static_feedback(config, dashboard_mode, dashboard_deg)
        start_error = float(np.max(np.abs(actual_deg - expected_current_deg)))
        start_tolerance = float(
            config["commissioning_limits"]["moveit_trajectory_start_tolerance_deg"]
        )
        if start_error > start_tolerance:
            raise RuntimeError(
                f"安全拒绝：真机不在已验证小段终点，误差={start_error:.6f}°"
            )

        delta_deg = target_deg - actual_deg
        maximum_delta = float(np.max(np.abs(delta_deg)))
        delta_limit = float(
            config["commissioning_limits"][
                "moveit_pregrasp_full_max_joint_delta_deg"
            ]
        )
        if maximum_delta > delta_limit:
            raise RuntimeError(
                f"安全拒绝：最大关节变化{maximum_delta:.6f}° > {delta_limit:.3f}°"
            )

        speed = int(
            config["commissioning_limits"]["moveit_first_segment_speed_percent"]
        )
        print("CR5 MOVEIT FULL PREGRASP EXECUTION")
        print(f"mode={'EXECUTE' if args.execute else 'DRY-RUN (NO MOTION)'}")
        print(f"trajectory_sha256: {digest}")
        print(f"moveit_points: {len(points_rad)}")
        print(f"joint_line_max_deviation_deg: {maximum_deviation_deg:.12f}")
        print(f"speed_percent: {speed}")
        print("current_deg: " + ", ".join(f"{v:.6f}" for v in actual_deg))
        print("target_deg : " + ", ".join(f"{v:.6f}" for v in target_deg))
        print("delta_deg  : " + ", ".join(f"{v:+.6f}" for v in delta_deg))
        print(f"maximum_delta_deg: {maximum_delta:.6f}")

        if not args.execute:
            print("DRY-RUN PASSED: 未连接30003，未发送运动。")
            return 0
        if args.confirmation != CONFIRMATION:
            raise RuntimeError(f"安全拒绝：必须准确传入{CONFIRMATION}")
        if FULL_RECEIPT.exists():
            prior = json.loads(FULL_RECEIPT.read_text(encoding="utf-8"))
            if prior.get("status") == "pregrasp_complete":
                raise RuntimeError("安全拒绝：预抓取路径已有成功记录，禁止重复执行")
            if prior.get("status") == "command_pending":
                raise RuntimeError("安全拒绝：历史命令状态不明确，禁止重发")

        receipt = {
            "created_at": datetime.now().astimezone().isoformat(),
            "status": "command_pending",
            "trajectory_sha256": digest,
            "origin_deg": actual_deg.tolist(),
            "target_deg": target_deg.tolist(),
        }
        save_receipt(receipt)
        execute_joint_target(config, dashboard, target_deg, speed)

        final_deg = np.asarray(read_one_feedback(config)["q_actual"], dtype=float)
        final_error = float(np.max(np.abs(final_deg - target_deg)))
        tolerance = float(
            config["commissioning_limits"]["moveit_waypoint_target_tolerance_deg"]
        )
        print("actual_deg : " + ", ".join(f"{v:.6f}" for v in final_deg))
        print(f"maximum target error: {final_error:.6f}°")
        if final_error > tolerance:
            raise RuntimeError("预抓取终点反馈误差超限；禁止继续下降")
        receipt["status"] = "pregrasp_complete"
        receipt["completed_at"] = datetime.now().astimezone().isoformat()
        receipt["actual_deg"] = final_deg.tolist()
        save_receipt(receipt)
        print("PREGRASP PASSED: 真机已到电表上方预抓取位；不会自动下降。")
        print(f"receipt={FULL_RECEIPT}")
        return 0
    finally:
        dashboard.close()


if __name__ == "__main__":
    raise SystemExit(main())
