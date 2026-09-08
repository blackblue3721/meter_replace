#!/usr/bin/env python3
"""真实 CR5 的只读运动前检查。

只允许发送四个查询：RobotMode、GetAngle、GetPose、GetErrorID。程序不会调用
EnableRobot、ClearError、SpeedFactor、MovJ、MovL、JointMovJ 或 ServoJ。
"""

from __future__ import annotations

import re
import socket
import sys
from pathlib import Path

import numpy as np

from read_cr5_state import (
    DEFAULT_CONFIG,
    FEEDBACK_FRAME_BYTES,
    FEEDBACK_MAGIC,
    ROBOT_MODE_NAMES,
    load_hardware_config,
    load_official_feedback_dtype,
    recv_exact,
    scalar,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_SDK_DIR = PROJECT_ROOT / "third_party" / "TCP-IP-Python-V3"


def parse_first_number_list(reply: str) -> np.ndarray:
    """提取越疆查询响应中第一层花括号内的数字。"""
    match = re.search(r"\{([^{}]*)\}", reply)
    if match is None:
        raise ValueError(f"响应中没有可解析的数据：{reply!r}")
    values = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", match.group(1))
    if not values:
        raise ValueError(f"响应数据为空：{reply!r}")
    return np.asarray([float(value) for value in values], dtype=float)


def response_code(reply: str) -> int:
    """读取厂商响应开头的返回码，例如 ``0`` 或 ``-2``。"""
    match = re.match(r"\s*(-?\d+)\s*,", reply)
    if match is None:
        raise ValueError(f"无法解析厂商响应返回码：{reply!r}")
    return int(match.group(1))


def read_one_feedback(config: dict) -> np.void:
    """从 30004 读取一帧经过魔数校验的反馈。"""
    dtype = load_official_feedback_dtype()
    address = (str(config["ip"]), int(config["feedback_port"]))
    with socket.create_connection(
        address, timeout=float(config.get("connection_timeout_s", 2.0))
    ) as sock:
        sock.settimeout(float(config.get("feedback_timeout_s", 2.0)))
        for _ in range(5):
            record = np.frombuffer(
                recv_exact(sock, FEEDBACK_FRAME_BYTES), dtype=dtype, count=1
            )[0]
            if int(scalar(record, "test_value")) == FEEDBACK_MAGIC:
                return record
    raise RuntimeError("连续 5 帧 CR5 反馈校验失败")


def main() -> int:
    config = load_hardware_config(DEFAULT_CONFIG)
    sys.path.insert(0, str(OFFICIAL_SDK_DIR))
    from dobot_api import DobotApiDashboard  # pylint: disable=import-outside-toplevel

    print("CR5 READ-ONLY PREFLIGHT / 只读运动前检查")
    print("允许的查询：RobotMode, GetAngle, GetPose, GetErrorID")
    print("禁止且不会调用：Enable/ClearError/Speed/MovJ/MovL/ServoJ\n")

    dashboard = DobotApiDashboard(str(config["ip"]), int(config["dashboard_port"]))
    dashboard.socket_dobot.settimeout(float(config.get("feedback_timeout_s", 2.0)))
    try:
        mode_reply = dashboard.RobotMode()
        angle_reply = dashboard.GetAngle()
        pose_reply = dashboard.GetPose()
        error_reply = dashboard.GetErrorID()
    finally:
        dashboard.close()

    print("\n--- 查询原始响应（留作协议审计）---")
    print(f"RobotMode: {mode_reply}")
    print(f"GetAngle : {angle_reply}")
    print(f"GetPose  : {pose_reply}")
    print(f"GetErrorID: {error_reply}")

    replies = {
        "RobotMode": mode_reply,
        "GetAngle": angle_reply,
        "GetPose": pose_reply,
        "GetErrorID": error_reply,
    }
    rejected = {
        name: (response_code(reply), reply)
        for name, reply in replies.items()
        if response_code(reply) != 0
    }
    if rejected:
        print("\nPREFLIGHT RESULT: BLOCKED (NO MOTION SENT)")
        for name, (code, reply) in rejected.items():
            print(f"- {name}: return_code={code}, response={reply}")
        print(
            "请保持机器人未使能，在 DobotStudio Pro 中将当前模式切换为 "
            "'TCP/IP二次开发'，应用后重新运行本检查。"
        )
        return 2

    feedback = read_one_feedback(config)
    dashboard_mode = int(parse_first_number_list(mode_reply)[0])
    dashboard_angles_deg = parse_first_number_list(angle_reply)
    if dashboard_angles_deg.size != 6:
        raise RuntimeError(f"GetAngle 未返回六轴：{angle_reply!r}")
    feedback_angles_deg = np.asarray(feedback["q_actual"], dtype=float)
    angle_error_deg = np.abs(dashboard_angles_deg - feedback_angles_deg)

    feedback_mode = int(scalar(feedback, "robot_mode"))
    enabled = bool(scalar(feedback, "enable_status"))
    running = bool(scalar(feedback, "running_status"))
    error = bool(scalar(feedback, "error_status"))

    print("\n--- 交叉检查 ---")
    print(
        f"mode: dashboard={dashboard_mode}, feedback={feedback_mode} "
        f"({ROBOT_MODE_NAMES.get(feedback_mode, 'UNKNOWN')})"
    )
    print(f"enabled={enabled}, running={running}, error={error}")
    print(f"maximum GetAngle/q_actual difference: {angle_error_deg.max():.6f} deg")

    failures: list[str] = []
    if dashboard_mode != feedback_mode:
        failures.append("29999 与 30004 的机器人模式不一致")
    if angle_error_deg.max() > 0.05:
        failures.append("GetAngle 与 q_actual 最大差值超过 0.05°")
    if running:
        failures.append("机器人当前正在运行")
    if error:
        failures.append("机器人当前存在报警")

    if failures:
        print("\nPREFLIGHT RESULT: FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nPREFLIGHT RESULT: READ-ONLY CHECK PASSED")
    print("注意：这不等于获得运动许可；仍需现场安全确认和首次小角度测试。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
