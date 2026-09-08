#!/usr/bin/env python3
"""Guarded JointMovJ executor for reviewed straight MoveIt joint stages."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from commission_j6_step import load_config, read_dashboard_state
from commission_moveit_pregrasp_segment import (
    PROJECT_ROOT,
    execute_joint_target,
    ros_to_vendor_deg,
    verify_digest,
    verify_static_feedback,
)
from preflight_cr5 import read_one_feedback
from read_cr5_state import DEFAULT_CONFIG


STAGE_PROFILES = {
    "semantic_flip": {
        "artifact": PROJECT_ROOT / "logs/real_cr5/semantic_flip_candidate.json",
        "expected_target_stage": "semantic_flip",
        "authorization_scope": "moveit_joint_semantic_flip_only",
        "predecessor_receipt": PROJECT_ROOT / "logs/commissioning/cr5_moveit_lift.json",
        "predecessor_status": "lift_complete",
        "receipt": PROJECT_ROOT / "logs/commissioning/cr5_moveit_semantic_flip.json",
        "completion_status": "semantic_flip_complete",
        "confirmation": "MOVE_REAL_CR5_SEMANTIC_FLIP",
        "maximum_delta_key": "moveit_semantic_flip_max_joint_delta_deg",
        "success_message": (
            "SEMANTIC FLIP PASSED: J5已按审核方向翻转；不会自动转移到电表箱。"
        ),
    },
}


def save_receipt(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=sorted(STAGE_PROFILES), required=True)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--trajectory-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = STAGE_PROFILES[args.stage]
    config = load_config(args.config.resolve())
    authorization = config.get("commissioning_authorization", {})
    if config.get("access_mode") != "commissioning":
        raise RuntimeError("安全拒绝：当前不处于commissioning模式")
    if authorization.get("scope") != profile["authorization_scope"]:
        raise RuntimeError("安全拒绝：当前授权范围不允许此关节阶段")
    if not authorization.get("operator_safety_check_confirmed", False):
        raise RuntimeError("安全拒绝：没有现场安全确认记录")
    if authorization.get("full_workflow_authorized", False):
        raise RuntimeError("配置异常：完整工艺仍不应被授权")

    artifact = (
        args.trajectory.resolve()
        if args.trajectory is not None
        else Path(profile["artifact"]).resolve()
    )
    trajectory = json.loads(artifact.read_text(encoding="utf-8"))
    digest = verify_digest(trajectory)
    if args.trajectory_sha256 != digest:
        raise RuntimeError("安全拒绝：命令行SHA256与轨迹文件不一致")
    if trajectory.get("target_stage") != profile["expected_target_stage"]:
        raise RuntimeError("安全拒绝：轨迹阶段与执行配置不一致")

    predecessor_path = Path(profile["predecessor_receipt"])
    if not predecessor_path.exists():
        raise RuntimeError("安全拒绝：缺少前序实机阶段记录")
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if predecessor.get("status") != profile["predecessor_status"]:
        raise RuntimeError("安全拒绝：前序实机阶段尚未成功")

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
    deviation_deg = float(np.max(np.abs(np.rad2deg(points_rad - projected))))
    limits = config["commissioning_limits"]
    deviation_limit = float(limits["moveit_joint_line_deviation_tolerance_deg"])
    if np.any(np.diff(progress) < -1.0e-10):
        raise RuntimeError("安全拒绝：MoveIt关节轨迹不是单调的")
    if deviation_deg > deviation_limit:
        raise RuntimeError(
            f"安全拒绝：轨迹不是JointMovJ直线，偏差={deviation_deg:.6f}°"
        )

    vendor_points_deg = np.asarray(
        [
            ros_to_vendor_deg(point, joint_names, config["joint_mapping"])
            for point in points_rad
        ]
    )
    dashboard, dashboard_mode, dashboard_deg = read_dashboard_state(config)
    try:
        actual_deg = verify_static_feedback(config, dashboard_mode, dashboard_deg)
        start_error = float(np.max(np.abs(actual_deg - vendor_points_deg[0])))
        start_tolerance = float(limits["moveit_trajectory_start_tolerance_deg"])
        if start_error > start_tolerance:
            raise RuntimeError(
                f"安全拒绝：真机与轨迹起点误差{start_error:.6f}°"
            )

        target_deg = vendor_points_deg[-1]
        if args.stage == "semantic_flip":
            j5_min, j5_max = limits["moveit_semantic_flip_j5_guard_deg"]
            if not float(j5_min) <= float(target_deg[4]) <= float(j5_max):
                raise RuntimeError(
                    "安全拒绝：J5目标超出实机调试保护范围 "
                    f"[{j5_min}, {j5_max}]°，当前={target_deg[4]:.6f}°"
                )
        delta_deg = target_deg - actual_deg
        maximum_delta = float(np.max(np.abs(delta_deg)))
        delta_limit = float(limits[profile["maximum_delta_key"]])
        if maximum_delta > delta_limit:
            raise RuntimeError(
                f"安全拒绝：最大关节变化{maximum_delta:.6f}° > {delta_limit:.3f}°"
            )
        speed = int(limits["moveit_first_segment_speed_percent"])
        print(f"CR5 MOVEIT JOINT STAGE: {args.stage}")
        print(f"mode={'EXECUTE' if args.execute else 'DRY-RUN (NO MOTION)'}")
        print(f"trajectory_sha256: {digest}")
        print(f"moveit_points: {len(points_rad)}")
        print(f"joint_line_max_deviation_deg: {deviation_deg:.12f}")
        print(f"speed_percent: {speed}")
        print("current_deg: " + ", ".join(f"{v:.6f}" for v in actual_deg))
        print("target_deg : " + ", ".join(f"{v:.6f}" for v in target_deg))
        print("delta_deg  : " + ", ".join(f"{v:+.6f}" for v in delta_deg))
        print(f"maximum_delta_deg: {maximum_delta:.6f}")

        if not args.execute:
            print("DRY-RUN PASSED: 未连接30003，未发送运动。")
            return 0
        if args.confirmation != profile["confirmation"]:
            raise RuntimeError(
                f"安全拒绝：必须准确传入{profile['confirmation']}"
            )

        receipt_path = Path(profile["receipt"])
        if receipt_path.exists():
            prior = json.loads(receipt_path.read_text(encoding="utf-8"))
            if prior.get("status") == profile["completion_status"]:
                raise RuntimeError("安全拒绝：此阶段已有成功记录，禁止重复执行")
            if prior.get("status") == "command_pending":
                raise RuntimeError("安全拒绝：历史命令状态不明确，禁止重发")

        receipt = {
            "created_at": datetime.now().astimezone().isoformat(),
            "status": "command_pending",
            "stage": args.stage,
            "trajectory_sha256": digest,
            "origin_deg": actual_deg.tolist(),
            "target_deg": target_deg.tolist(),
        }
        save_receipt(receipt_path, receipt)
        try:
            execute_joint_target(config, dashboard, target_deg, speed)
        except Exception as exc:
            # JointMovJ may be rejected synchronously before anything enters the
            # controller queue (for example -40005: J5 range error).  Do not
            # leave such a command marked as "pending", otherwise a corrected
            # and newly reviewed trajectory cannot be commissioned safely.
            receipt["status"] = "command_rejected_or_failed"
            receipt["failed_at"] = datetime.now().astimezone().isoformat()
            receipt["error"] = str(exc)
            save_receipt(receipt_path, receipt)
            raise

        final_deg = np.asarray(read_one_feedback(config)["q_actual"], dtype=float)
        final_error = float(np.max(np.abs(final_deg - target_deg)))
        tolerance = float(limits["moveit_waypoint_target_tolerance_deg"])
        print("actual_deg : " + ", ".join(f"{v:.6f}" for v in final_deg))
        print(f"maximum target error: {final_error:.6f}°")
        if final_error > tolerance:
            raise RuntimeError("关节阶段终点反馈误差超限；禁止继续")
        receipt["status"] = profile["completion_status"]
        receipt["completed_at"] = datetime.now().astimezone().isoformat()
        receipt["actual_deg"] = final_deg.tolist()
        save_receipt(receipt_path, receipt)
        print(profile["success_message"])
        print(f"receipt={receipt_path}")
        return 0
    finally:
        dashboard.close()


if __name__ == "__main__":
    raise SystemExit(main())
