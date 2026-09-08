#!/usr/bin/env python3
"""真实 CR5 首次 J6 小角度调试程序，默认永远只做 dry-run。

首次动作只旋转法兰轴 J6，其他五轴目标保持为执行前实时角度。程序不负责使能
机器人；只有配置、命令行确认、安全状态和现场人工条件同时满足才连接 30003。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

from preflight_cr5 import (
    OFFICIAL_SDK_DIR,
    parse_first_number_list,
    read_one_feedback,
    response_code,
)
from read_cr5_state import DEFAULT_CONFIG, scalar


EXECUTION_CONFIRMATION = "MOVE_REAL_CR5_J6"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)["real_cr5"]


def require_success(command_name: str, reply: str) -> None:
    code = response_code(reply)
    if code != 0:
        raise RuntimeError(f"{command_name} 被控制器拒绝：{reply}")


def read_dashboard_state(config: dict):
    sys.path.insert(0, str(OFFICIAL_SDK_DIR))
    from dobot_api import DobotApiDashboard  # pylint: disable=import-outside-toplevel

    dashboard = DobotApiDashboard(str(config["ip"]), int(config["dashboard_port"]))
    dashboard.socket_dobot.settimeout(float(config.get("feedback_timeout_s", 2.0)))
    try:
        mode_reply = dashboard.RobotMode()
        angle_reply = dashboard.GetAngle()
        error_reply = dashboard.GetErrorID()
        for name, reply in (
            ("RobotMode", mode_reply),
            ("GetAngle", angle_reply),
            ("GetErrorID", error_reply),
        ):
            require_success(name, reply)
        mode = int(parse_first_number_list(mode_reply)[0])
        angles_deg = parse_first_number_list(angle_reply)
        if angles_deg.size != 6:
            raise RuntimeError("GetAngle 没有返回六个关节角")
    except Exception:
        dashboard.close()
        raise
    return dashboard, mode, angles_deg


def print_plan(current_deg: np.ndarray, target_deg: np.ndarray, config: dict) -> None:
    limits = config["commissioning_limits"]
    print("\n--- 首次实机动作候选 ---")
    print("motion       : JointMovJ (joint-space point-to-point)")
    print(f"speed factor : {limits['speed_factor_percent']}%")
    print("J1..J5       : 保持执行前实时角度")
    print(f"J6           : {current_deg[5]:.6f}° -> {target_deg[5]:.6f}°")
    print("current_deg  : " + ", ".join(f"{v:.6f}" for v in current_deg))
    print("target_deg   : " + ", ".join(f"{v:.6f}" for v in target_deg))


def execute_motion(config: dict, dashboard, target_deg: np.ndarray) -> None:
    """在所有外部安全门通过后发送唯一一条 J6 小角度 JointMovJ。"""
    from dobot_api import DobotApiMove  # pylint: disable=import-outside-toplevel

    limit = int(config["commissioning_limits"]["speed_factor_percent"])
    require_success("SpeedFactor", dashboard.SpeedFactor(limit))
    require_success("SpeedJ", dashboard.SpeedJ(limit))
    require_success("AccJ", dashboard.AccJ(limit))

    move = DobotApiMove(str(config["ip"]), int(config["motion_port"]))
    move.socket_dobot.settimeout(30.0)
    try:
        reply = move.JointMovJ(*target_deg.tolist())
        require_success("JointMovJ", reply)
        sync_reply = move.Sync()
        if response_code(sync_reply) != 0:
            reset_reply = dashboard.ResetRobot()
            raise RuntimeError(
                "Sync failed; ResetRobot recovery was requested to stop and clear "
                f"the queue. Sync={sync_reply!r}, ResetRobot={reset_reply!r}"
            )
    finally:
        move.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--joint6-delta-deg", type=float, default=2.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--confirmation",
        default="",
        help=f"实机执行时必须明确传入 {EXECUTION_CONFIRMATION}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())
    maximum_delta = float(
        config["commissioning_limits"]["maximum_first_test_joint_delta_deg"]
    )
    if abs(args.joint6_delta_deg) < 0.1 or abs(args.joint6_delta_deg) > maximum_delta:
        raise RuntimeError(
            f"J6 调试增量必须在 0.1°..{maximum_delta:.1f}°，当前为 "
            f"{args.joint6_delta_deg:.3f}°"
        )

    print("CR5 FIRST-MOTION COMMISSIONING")
    print(f"mode={'EXECUTE' if args.execute else 'DRY-RUN (NO MOTION)'}")
    dashboard, dashboard_mode, current_deg = read_dashboard_state(config)
    try:
        feedback = read_one_feedback(config)
        feedback_deg = np.asarray(feedback["q_actual"], dtype=float)
        feedback_mode = int(scalar(feedback, "robot_mode"))
        running = bool(scalar(feedback, "running_status"))
        error = bool(scalar(feedback, "error_status"))
        if dashboard_mode != feedback_mode:
            raise RuntimeError("29999/30004 机器人模式不一致")
        if np.max(np.abs(current_deg - feedback_deg)) > 0.05:
            raise RuntimeError("29999/30004 关节反馈差异超过 0.05°")

        target_deg = current_deg.copy()
        target_deg[5] += args.joint6_delta_deg
        print_plan(current_deg, target_deg, config)

        if not args.execute:
            print("\nDRY-RUN PASSED: 未连接 30003，未改变速度，未发送运动。")
            return 0

        if config.get("access_mode") != "commissioning":
            raise RuntimeError(
                "安全拒绝：config/hardware.yaml 的 access_mode 仍是 read_only"
            )
        authorization = config.get("commissioning_authorization", {})
        if authorization.get("scope") != "first_j6_positive_2deg_only":
            raise RuntimeError("安全拒绝：当前调试授权范围不允许本动作")
        if not authorization.get("operator_safety_check_confirmed", False):
            raise RuntimeError("安全拒绝：尚未记录现场操作员安全确认")
        if abs(args.joint6_delta_deg - 2.0) > 1e-9:
            raise RuntimeError("安全拒绝：当前只批准 J6 正向 +2.0°")
        if args.confirmation != EXECUTION_CONFIRMATION:
            raise RuntimeError("安全拒绝：缺少准确的实机运动确认字符串")
        if dashboard_mode != 5:
            raise RuntimeError(
                f"安全拒绝：机器人必须由操作员手动使能并静止（mode=5），当前 mode={dashboard_mode}"
            )
        if running or error:
            raise RuntimeError(
                f"安全拒绝：running={running}, error={error}"
            )

        execute_motion(config, dashboard, target_deg)
        final_feedback = read_one_feedback(config)
        final_deg = np.asarray(final_feedback["q_actual"], dtype=float)
        final_error = np.abs(final_deg - target_deg)
        print("\n--- 执行后反馈 ---")
        print("actual_deg: " + ", ".join(f"{v:.6f}" for v in final_deg))
        print(f"maximum target error: {final_error.max():.6f}°")
        if final_error.max() > 0.10:
            raise RuntimeError("执行终点误差超过 0.10°，禁止继续后续测试")
        print("FIRST MOTION PASSED: J6 小角度动作完成；不会自动执行下一段。")
        return 0
    finally:
        dashboard.close()


if __name__ == "__main__":
    raise SystemExit(main())
