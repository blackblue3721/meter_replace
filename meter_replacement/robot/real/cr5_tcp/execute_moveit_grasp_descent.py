#!/usr/bin/env python3
"""Execute one reviewed MoveIt LIN descent through the official CR5 ServoJ API.

Default mode is dry-run.  The tool requires the completed real pregrasp receipt,
an exact trajectory SHA, a live-start match, bounded time intervals and bounded
adjacent joint steps.  It does not enable the robot or command a gripper.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from commission_j6_step import load_config, read_dashboard_state, require_success
from commission_moveit_pregrasp_segment import (
    PROJECT_ROOT,
    ros_to_vendor_deg,
    verify_digest,
    verify_static_feedback,
)
from preflight_cr5 import OFFICIAL_SDK_DIR, read_one_feedback
from read_cr5_state import DEFAULT_CONFIG


DEFAULT_TRAJECTORY = PROJECT_ROOT / "logs/real_cr5/grasp_candidate.json"
PREGRASP_RECEIPT = PROJECT_ROOT / "logs/commissioning/cr5_moveit_pregrasp_full.json"
RECEIPT = PROJECT_ROOT / "logs/commissioning/cr5_moveit_grasp_descent.json"
CONFIRMATION = "MOVE_REAL_CR5_TO_VIRTUAL_GRASP"


def resample_servoj_path(
    vendor_points_deg: np.ndarray,
    times_s: np.ndarray,
    *,
    time_scale: float,
    cycle_s: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """Preserve the reviewed MoveIt waypoints and scale only their timing.

    The previous implementation expanded one reviewed path into hundreds of
    synchronous TCP requests.  That was not the command sequence represented
    by the trajectory SHA and regressed the already commissioned executor.
    """
    if len(vendor_points_deg) < 2 or len(vendor_points_deg) != len(times_s):
        raise RuntimeError("ServoJ重采样失败：轨迹点或时间数据无效")
    if time_scale <= 0.0 or cycle_s <= 0.0:
        raise RuntimeError("ServoJ轨迹准备失败：时间缩放或最小周期无效")

    relative_times = np.asarray(times_s, dtype=float) - float(times_s[0])
    if np.any(np.diff(relative_times) <= 0.0):
        raise RuntimeError("ServoJ轨迹准备失败：轨迹时间必须严格递增")
    scaled_times = relative_times * time_scale
    if np.any(np.diff(scaled_times) < cycle_s):
        raise RuntimeError(
            "ServoJ轨迹准备失败：缩放后的点间隔短于配置的安全最小周期"
        )
    return np.asarray(vendor_points_deg, dtype=float).copy(), scaled_times


def save_receipt(payload: dict) -> None:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--trajectory-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    return parser.parse_args()


def execute_servoj_path(
    config: dict,
    dashboard,
    vendor_points_deg: np.ndarray,
    times_s: np.ndarray,
) -> None:
    sys.path.insert(0, str(OFFICIAL_SDK_DIR))
    from dobot_api import (  # pylint: disable=import-outside-toplevel
        DobotApiMove,
    )

    limits = config["commissioning_limits"]
    speed = int(limits["moveit_servoj_speed_percent"])
    time_scale = float(limits["moveit_servoj_time_scale"])
    servo_cycle_s = float(limits.get("moveit_servoj_cycle_s", 0.03))
    servo_points_deg, servo_times_s = resample_servoj_path(
        vendor_points_deg,
        times_s,
        time_scale=time_scale,
        cycle_s=servo_cycle_s,
    )
    servo_intervals = np.diff(servo_times_s)
    print(
        "ServoJ schedule: "
        f"{len(servo_points_deg)} reviewed MoveIt points, "
        f"interval=[{float(np.min(servo_intervals)):.6f}, "
        f"{float(np.max(servo_intervals)):.6f}]s, "
        f"duration={float(servo_times_s[-1]):.3f}s"
    )
    require_success("SpeedFactor", dashboard.SpeedFactor(speed))

    # Keep the dashboard session alive while streaming, matching Dobot's
    # official three-port example. DobotStudio remote-control commands and an
    # external TCP/IP motion client are not expected to own control together.
    move = DobotApiMove(str(config["ip"]), int(config["motion_port"]))
    move.socket_dobot.settimeout(2.0)
    try:
        # Point zero is the already verified live state.  Send exactly the
        # reviewed MoveIt points represented by the trajectory digest.
        for index in range(1, len(servo_points_deg)):
            interval = float(servo_times_s[index] - servo_times_s[index - 1])
            started = time.monotonic()
            reply = move.ServoJ(
                *servo_points_deg[index].tolist(),
                t=interval,
                lookahead_time=50,
                gain=500,
            )
            require_success(f"ServoJ point {index}", reply)
            remaining = interval - (time.monotonic() - started)
            if remaining > 0.0:
                time.sleep(remaining)
    except Exception:
        # A rejected or interrupted streaming command must not leave a dynamic
        # follow operation active. ResetRobot is the project's established
        # stop-and-clear recovery and sends no new motion target.
        reset_reply = dashboard.ResetRobot()
        print(f"EMERGENCY STREAM RECOVERY: ResetRobot={reset_reply!r}")
        raise
    finally:
        move.close()


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())
    authorization = config.get("commissioning_authorization", {})
    if config.get("access_mode") != "commissioning":
        raise RuntimeError("安全拒绝：当前不处于commissioning模式")
    if authorization.get("scope") != "moveit_grasp_descent_only":
        raise RuntimeError("安全拒绝：当前授权范围不允许抓取下降")
    if not authorization.get("operator_safety_check_confirmed", False):
        raise RuntimeError("安全拒绝：没有现场安全确认记录")
    if authorization.get("full_workflow_authorized", False):
        raise RuntimeError("配置异常：完整工艺仍不应被授权")

    trajectory = json.loads(args.trajectory.resolve().read_text(encoding="utf-8"))
    digest = verify_digest(trajectory)
    if args.trajectory_sha256 != digest:
        raise RuntimeError("安全拒绝：命令行SHA256与轨迹文件不一致")
    if trajectory.get("target_stage") != "grasp_candidate":
        raise RuntimeError("安全拒绝：只允许grasp_candidate轨迹")
    if not PREGRASP_RECEIPT.exists():
        raise RuntimeError("安全拒绝：缺少预抓取成功记录")
    pregrasp = json.loads(PREGRASP_RECEIPT.read_text(encoding="utf-8"))
    if pregrasp.get("status") != "pregrasp_complete":
        raise RuntimeError("安全拒绝：真机尚未完成预抓取")

    joint_names = list(trajectory["joint_names"])
    points_rad = np.asarray(
        [point["positions_rad"] for point in trajectory["points"]], dtype=float
    )
    times_s = np.asarray(
        [point["time_from_start_s"] for point in trajectory["points"]], dtype=float
    )
    if len(points_rad) < 2 or len(points_rad) != len(times_s):
        raise RuntimeError("安全拒绝：轨迹点或时间数据无效")
    intervals = np.diff(times_s)
    if np.any(intervals < 0.02) or np.any(intervals > 0.25):
        raise RuntimeError(f"安全拒绝：ServoJ原始时间间隔异常：{intervals}")

    mapping = config["joint_mapping"]
    vendor_points_deg = np.asarray(
        [ros_to_vendor_deg(point, joint_names, mapping) for point in points_rad]
    )
    maximum_step = float(np.max(np.abs(np.diff(vendor_points_deg, axis=0))))
    step_limit = float(config["commissioning_limits"]["moveit_servoj_max_step_deg"])
    if maximum_step > step_limit:
        raise RuntimeError(
            f"安全拒绝：相邻轨迹点最大变化{maximum_step:.6f}° > {step_limit:.3f}°"
        )

    dashboard, dashboard_mode, dashboard_deg = read_dashboard_state(config)
    try:
        actual_deg = verify_static_feedback(config, dashboard_mode, dashboard_deg)
        start_error = float(np.max(np.abs(actual_deg - vendor_points_deg[0])))
        start_tolerance = float(
            config["commissioning_limits"]["moveit_trajectory_start_tolerance_deg"]
        )
        if start_error > start_tolerance:
            raise RuntimeError(
                f"安全拒绝：真机与下降轨迹起点误差{start_error:.6f}°"
            )

        target_deg = vendor_points_deg[-1]
        total_delta = target_deg - actual_deg
        duration = float(times_s[-1]) * float(
            config["commissioning_limits"]["moveit_servoj_time_scale"]
        )
        print("CR5 MOVEIT VIRTUAL-GRASP DESCENT")
        print(f"mode={'EXECUTE' if args.execute else 'DRY-RUN (NO MOTION)'}")
        print(f"trajectory_sha256: {digest}")
        print(f"points: {len(points_rad)}")
        print(f"scaled_duration_s: {duration:.3f}")
        print(f"maximum_adjacent_step_deg: {maximum_step:.6f}")
        print("current_deg: " + ", ".join(f"{v:.6f}" for v in actual_deg))
        print("target_deg : " + ", ".join(f"{v:.6f}" for v in target_deg))
        print("delta_deg  : " + ", ".join(f"{v:+.6f}" for v in total_delta))

        if not args.execute:
            print("DRY-RUN PASSED: 未连接30003，未发送运动。")
            return 0
        if args.confirmation != CONFIRMATION:
            raise RuntimeError(f"安全拒绝：必须准确传入{CONFIRMATION}")
        if RECEIPT.exists():
            prior = json.loads(RECEIPT.read_text(encoding="utf-8"))
            if prior.get("status") == "grasp_descent_complete":
                raise RuntimeError("安全拒绝：下降已有成功记录，禁止重复执行")
            if prior.get("status") == "command_pending":
                raise RuntimeError("安全拒绝：历史下降命令状态不明确，禁止重发")

        receipt = {
            "created_at": datetime.now().astimezone().isoformat(),
            "status": "command_pending",
            "trajectory_sha256": digest,
            "origin_deg": actual_deg.tolist(),
            "target_deg": target_deg.tolist(),
        }
        save_receipt(receipt)
        execute_servoj_path(config, dashboard, vendor_points_deg, times_s)

        final_deg = np.asarray(read_one_feedback(config)["q_actual"], dtype=float)
        final_error = float(np.max(np.abs(final_deg - target_deg)))
        tolerance = float(
            config["commissioning_limits"]["moveit_servoj_target_tolerance_deg"]
        )
        print("actual_deg : " + ", ".join(f"{v:.6f}" for v in final_deg))
        print(f"maximum target error: {final_error:.6f}°")
        if final_error > tolerance:
            raise RuntimeError("下降终点反馈误差超限；禁止虚拟夹紧和抬升")
        receipt["status"] = "grasp_descent_complete"
        receipt["completed_at"] = datetime.now().astimezone().isoformat()
        receipt["actual_deg"] = final_deg.tolist()
        save_receipt(receipt)
        print("VIRTUAL GRASP POSE PASSED: 已下降到占位夹爪抓取位；未发送夹爪命令。")
        print(f"receipt={RECEIPT}")
        return 0
    finally:
        dashboard.close()


if __name__ == "__main__":
    raise SystemExit(main())
