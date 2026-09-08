#!/usr/bin/env python3
"""Execute only the approved first small segment of an exported MoveIt plan.

The default mode is dry-run.  This tool never enables the robot, never clears
errors, and never continues beyond the single configured waypoint.  It checks
the artifact digest, live start state, controller state, joint mapping, and
per-joint delta before opening the CR5 motion port.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from commission_j1_roundtrip import wait_for_feedback_target
from commission_j6_step import load_config, read_dashboard_state, require_success
from preflight_cr5 import OFFICIAL_SDK_DIR, read_one_feedback, response_code
from read_cr5_state import DEFAULT_CONFIG, scalar


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRAJECTORY = PROJECT_ROOT / "logs/real_cr5/pregrasp_candidate.json"
RECEIPT = PROJECT_ROOT / "logs/commissioning/cr5_moveit_pregrasp_segment.json"
CONFIRMATION = "MOVE_REAL_CR5_PREGRASP_FIRST_SEGMENT"


def verify_digest(payload: dict) -> str:
    expected = str(payload.get("sha256", ""))
    unsigned = dict(payload)
    unsigned.pop("sha256", None)
    canonical = json.dumps(
        unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    actual = hashlib.sha256(canonical).hexdigest()
    if not expected or actual != expected:
        raise RuntimeError(
            f"轨迹摘要校验失败：stored={expected!r}, calculated={actual!r}"
        )
    return actual


def ros_to_vendor_deg(
    ros_positions_rad: np.ndarray,
    joint_names: list[str],
    mapping: dict,
) -> np.ndarray:
    ros_names = list(mapping["ros_joint_names"])
    if joint_names != ros_names:
        raise RuntimeError(
            f"轨迹关节顺序与已验证映射不一致：{joint_names} != {ros_names}"
        )
    sign = np.asarray(mapping["sign"], dtype=float)
    offset = np.asarray(mapping["offset_deg"], dtype=float)
    if np.any(np.abs(sign) != 1.0):
        raise RuntimeError("关节映射 sign 必须全部为 +1 或 -1")
    return (np.rad2deg(ros_positions_rad) - offset) / sign


def verify_static_feedback(
    config: dict,
    dashboard_mode: int,
    dashboard_deg: np.ndarray,
) -> np.ndarray:
    feedback = read_one_feedback(config)
    feedback_deg = np.asarray(feedback["q_actual"], dtype=float)
    mode = int(scalar(feedback, "robot_mode"))
    running = bool(scalar(feedback, "running_status"))
    error = bool(scalar(feedback, "error_status"))
    queue_running = bool(scalar(feedback, "run_queued_cmd"))
    if dashboard_mode != 5 or mode != 5 or running or error or not queue_running:
        raise RuntimeError(
            "安全拒绝：要求机器人已使能、静止、无报警且运动队列已开启；"
            f"mode={dashboard_mode}/{mode}, running={running}, error={error}, "
            f"queue_running={queue_running}"
        )
    if float(np.max(np.abs(dashboard_deg - feedback_deg))) > 0.05:
        raise RuntimeError("安全拒绝：29999与30004关节反馈差异超过0.05°")
    return feedback_deg


def execute_joint_target(
    config: dict,
    dashboard,
    target_deg: np.ndarray,
    speed_percent: int,
) -> None:
    sys.path.insert(0, str(OFFICIAL_SDK_DIR))
    from dobot_api import DobotApiMove  # pylint: disable=import-outside-toplevel

    require_success("SpeedFactor", dashboard.SpeedFactor(speed_percent))
    require_success("SpeedJ", dashboard.SpeedJ(speed_percent))
    require_success("AccJ", dashboard.AccJ(speed_percent))

    move = DobotApiMove(str(config["ip"]), int(config["motion_port"]))
    move.socket_dobot.settimeout(30.0)
    try:
        require_success("JointMovJ", move.JointMovJ(*target_deg.tolist()))
        sync_reply = move.Sync()
        if not sync_reply:
            print("WARNING: Sync应答超时；不重发命令，改用30004反馈核验。")
            wait_for_feedback_target(config, target_deg)
        elif response_code(sync_reply) != 0:
            raise RuntimeError(f"Sync被控制器拒绝：{sync_reply!r}")
    finally:
        move.close()


def save_receipt(
    digest: str,
    waypoint_index: int,
    origin_deg: np.ndarray,
    target_deg: np.ndarray,
    status: str,
) -> None:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": status,
        "trajectory_sha256": digest,
        "waypoint_index": waypoint_index,
        "origin_deg": origin_deg.tolist(),
        "target_deg": target_deg.tolist(),
    }
    RECEIPT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--trajectory-sha256", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())
    authorization = config.get("commissioning_authorization", {})
    if config.get("access_mode") != "commissioning":
        raise RuntimeError("安全拒绝：当前不处于commissioning模式")
    if authorization.get("scope") != "moveit_pregrasp_first_segment_only":
        raise RuntimeError("安全拒绝：当前授权范围不允许MoveIt轨迹小段测试")
    if not authorization.get("operator_safety_check_confirmed", False):
        raise RuntimeError("安全拒绝：没有现场安全确认记录")
    if authorization.get("full_workflow_authorized", False):
        raise RuntimeError("配置异常：本脚本要求full_workflow_authorized=false")

    payload = json.loads(args.trajectory.resolve().read_text(encoding="utf-8"))
    digest = verify_digest(payload)
    if args.trajectory_sha256 and args.trajectory_sha256 != digest:
        raise RuntimeError("命令行轨迹摘要与文件摘要不一致")
    if payload.get("target_stage") != "pregrasp":
        raise RuntimeError("安全拒绝：只允许pregrasp轨迹")
    if payload.get("home_profile") != "real_commissioning":
        raise RuntimeError("安全拒绝：轨迹不是从real_commissioning Home生成")

    limits = config["commissioning_limits"]
    waypoint_index = int(limits["moveit_first_segment_waypoint_index"])
    points = payload["points"]
    if waypoint_index <= 0 or waypoint_index >= len(points):
        raise RuntimeError("配置的首段轨迹点索引无效")
    joint_names = list(payload["joint_names"])
    mapping = config["joint_mapping"]
    if mapping.get("status") != "visual_validation_passed":
        raise RuntimeError("安全拒绝：ROS/厂商关节映射尚未验证")

    artifact_start_rad = np.asarray(
        [payload["live_start_joint_positions_rad"][name] for name in joint_names],
        dtype=float,
    )
    target_rad = np.asarray(points[waypoint_index]["positions_rad"], dtype=float)
    artifact_start_deg = ros_to_vendor_deg(artifact_start_rad, joint_names, mapping)
    target_deg = ros_to_vendor_deg(target_rad, joint_names, mapping)

    dashboard, dashboard_mode, dashboard_deg = read_dashboard_state(config)
    try:
        actual_deg = verify_static_feedback(config, dashboard_mode, dashboard_deg)
        start_error = float(np.max(np.abs(actual_deg - artifact_start_deg)))
        start_tolerance = float(limits["moveit_trajectory_start_tolerance_deg"])
        if start_error > start_tolerance:
            raise RuntimeError(
                f"安全拒绝：真机与轨迹起点误差{start_error:.6f}° "
                f"> {start_tolerance:.3f}°"
            )
        delta_deg = target_deg - actual_deg
        maximum_delta = float(np.max(np.abs(delta_deg)))
        delta_limit = float(limits["moveit_first_segment_max_joint_delta_deg"])
        if maximum_delta > delta_limit:
            raise RuntimeError(
                f"安全拒绝：首段最大关节变化{maximum_delta:.6f}° "
                f"> {delta_limit:.3f}°"
            )

        print("CR5 MOVEIT PREGRASP FIRST-SEGMENT COMMISSIONING")
        print(f"mode={'EXECUTE' if args.execute else 'DRY-RUN (NO MOTION)'}")
        print(f"trajectory_sha256: {digest}")
        print(f"waypoint_index: {waypoint_index}/{len(points) - 1}")
        print("current_deg: " + ", ".join(f"{v:.6f}" for v in actual_deg))
        print("target_deg : " + ", ".join(f"{v:.6f}" for v in target_deg))
        print("delta_deg  : " + ", ".join(f"{v:+.6f}" for v in delta_deg))
        print(f"maximum_delta_deg: {maximum_delta:.6f}")

        if not args.execute:
            print("DRY-RUN PASSED: 未连接30003，未发送运动。")
            return 0
        if RECEIPT.exists():
            prior = json.loads(RECEIPT.read_text(encoding="utf-8"))
            prior_status = str(prior.get("status", "unknown"))
            if prior_status == "segment_complete":
                raise RuntimeError("安全拒绝：首段已有成功记录，禁止重复执行")
            if prior_status == "command_pending":
                raise RuntimeError(
                    "安全拒绝：上次命令状态不明确；必须先运行clear_pending_queue.py"
                )
            if prior_status != "queue_cleared_after_sync_failure":
                raise RuntimeError(f"安全拒绝：未知的历史执行状态{prior_status!r}")
        if args.confirmation != CONFIRMATION:
            raise RuntimeError(f"安全拒绝：必须准确传入{CONFIRMATION}")
        if args.trajectory_sha256 != digest:
            raise RuntimeError("安全拒绝：实机执行必须显式传入准确轨迹SHA256")

        save_receipt(digest, waypoint_index, actual_deg, target_deg, "command_pending")
        execute_joint_target(
            config,
            dashboard,
            target_deg,
            int(limits["moveit_first_segment_speed_percent"]),
        )
        final_deg = np.asarray(read_one_feedback(config)["q_actual"], dtype=float)
        final_error = float(np.max(np.abs(final_deg - target_deg)))
        tolerance = float(limits["moveit_waypoint_target_tolerance_deg"])
        print("actual_deg : " + ", ".join(f"{v:.6f}" for v in final_deg))
        print(f"maximum target error: {final_error:.6f}°")
        if final_error > tolerance:
            raise RuntimeError("首段终点反馈误差超限；禁止继续")
        save_receipt(digest, waypoint_index, actual_deg, target_deg, "segment_complete")
        print("FIRST SEGMENT PASSED: 已到第3号轨迹点；不会自动继续。")
        print(f"receipt={RECEIPT}")
        return 0
    finally:
        dashboard.close()


if __name__ == "__main__":
    raise SystemExit(main())
