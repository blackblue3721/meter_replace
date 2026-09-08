#!/usr/bin/env python3
"""建立 CR5 TCP 二次开发使能会话，但不发送任何运动命令。

该程序只连接 Dashboard 端口 29999，执行状态查询和 EnableRobot()。
它不会连接运动端口 30003，也不会调用 JointMovJ、MovJ、MovL 或 ServoJ。
"""

from __future__ import annotations

import argparse
import sys
import time

import yaml

from preflight_cr5 import (
    OFFICIAL_SDK_DIR,
    parse_first_number_list,
    read_one_feedback,
    response_code,
)
from read_cr5_state import DEFAULT_CONFIG, scalar


CONFIRMATION = "ENABLE_REAL_CR5_TCP_SESSION"


def require_success(command_name: str, reply: str) -> None:
    if response_code(reply) != 0:
        raise RuntimeError(f"{command_name} 被控制器拒绝：{reply}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", required=True)
    args = parser.parse_args()
    if args.confirmation != CONFIRMATION:
        raise RuntimeError("安全拒绝：TCP 使能确认字符串不正确")

    with DEFAULT_CONFIG.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)["real_cr5"]
    if config.get("access_mode") != "commissioning":
        raise RuntimeError("安全拒绝：当前不处于 commissioning 模式")

    sys.path.insert(0, str(OFFICIAL_SDK_DIR))
    from dobot_api import DobotApiDashboard  # pylint: disable=import-outside-toplevel

    dashboard = DobotApiDashboard(str(config["ip"]), int(config["dashboard_port"]))
    dashboard.socket_dobot.settimeout(float(config.get("feedback_timeout_s", 2.0)))
    try:
        mode_reply = dashboard.RobotMode()
        angle_reply = dashboard.GetAngle()
        error_reply = dashboard.GetErrorID()
        require_success("RobotMode", mode_reply)
        require_success("GetAngle", angle_reply)
        require_success("GetErrorID", error_reply)

        mode_before = int(parse_first_number_list(mode_reply)[0])
        current_deg = parse_first_number_list(angle_reply)
        feedback = read_one_feedback(config)
        feedback_mode = int(scalar(feedback, "robot_mode"))
        running = bool(scalar(feedback, "running_status"))
        error = bool(scalar(feedback, "error_status"))

        print("CR5 TCP SESSION ENABLE / 仅建立二次开发使能会话")
        print(f"mode_before={mode_before}, feedback_mode={feedback_mode}")
        print("current_deg=" + ", ".join(f"{value:.6f}" for value in current_deg))
        print(f"running={running}, error={error}")

        if mode_before != feedback_mode:
            raise RuntimeError("安全拒绝：29999/30004 机器人模式不一致")
        if mode_before != 4:
            raise RuntimeError(
                f"安全拒绝：执行前必须是未使能且静止的 mode=4，当前 mode={mode_before}"
            )
        if running or error:
            raise RuntimeError(
                f"安全拒绝：执行前状态异常，running={running}, error={error}"
            )

        print("Calling EnableRobot() on 29999; no connection to motion port 30003")
        enable_reply = dashboard.EnableRobot()
        if enable_reply:
            require_success("EnableRobot", enable_reply)
        else:
            print(
                "WARNING: EnableRobot acknowledgement timed out; the command "
                "will NOT be resent. Verifying controller feedback instead."
            )

        deadline = time.monotonic() + 5.0
        mode_after = -1
        while time.monotonic() < deadline:
            # 使用独立的 30004 实时反馈判断结果，避免在 EnableRobot 应答
            # 超时后继续复用状态不确定的 29999 套接字，也绝不重发使能。
            after_feedback = read_one_feedback(config)
            mode_after = int(scalar(after_feedback, "robot_mode"))
            if mode_after == 5:
                break
            time.sleep(0.1)

        if mode_after != 5:
            raise RuntimeError(
                f"EnableRobot 已接受，但 5 秒内未进入 ENABLE/idle(mode=5)，当前 mode={mode_after}"
            )

        final_feedback = read_one_feedback(config)
        if bool(scalar(final_feedback, "running_status")):
            raise RuntimeError("异常：使能后检测到机器人正在运动，请立即现场停止")
        if bool(scalar(final_feedback, "error_status")):
            raise RuntimeError("异常：使能后控制器报告错误，请下使能并检查报警")

        print("TCP SESSION ENABLE PASSED: controller is enabled and idle (mode=5)")
        print("No motion port connection was opened and no motion command was sent.")
        return 0
    finally:
        dashboard.close()


if __name__ == "__main__":
    raise SystemExit(main())
