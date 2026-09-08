#!/usr/bin/env python3
"""只读获取真实越疆 CR5 的 30004 反馈状态。

安全边界：
- 只建立到反馈端口 30004 的 TCP 连接；
- 不连接 29999/30003；
- 不发送任何字符串、使能、清错或运动指令；
- 关节角按越疆反馈原值（degree）打印，暂不转换为 ROS 关节值。
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "hardware.yaml"
OFFICIAL_SDK_DIR = PROJECT_ROOT / "third_party" / "TCP-IP-Python-V3"
FEEDBACK_FRAME_BYTES = 1440
FEEDBACK_MAGIC = 0x0123456789ABCDEF

ROBOT_MODE_NAMES = {
    1: "INIT（初始化）",
    2: "BRAKE_OPEN（抱闸松开）",
    3: "RESERVED（保留）",
    4: "DISABLED（未使能）",
    5: "ENABLE（已使能/静止）",
    6: "BACKDRIVE（拖拽）",
    7: "RUNNING（运行）",
    8: "RECORDING（轨迹录制）",
    9: "ERROR（报警）",
    10: "PAUSE（暂停）",
    11: "JOG（点动）",
}


def load_hardware_config(path: Path) -> dict[str, Any]:
    """读取真机网络配置；调用本函数的程序自身必须保持只读。"""
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)

    real_cr5 = document.get("real_cr5", {})
    if real_cr5.get("access_mode") not in {"read_only", "commissioning"}:
        raise RuntimeError(
            "安全拒绝：real_cr5.access_mode 必须是 read_only 或 commissioning"
        )
    if int(real_cr5.get("feedback_port", 0)) != 30004:
        raise RuntimeError("安全拒绝：只读诊断只允许连接 CR5 反馈端口 30004")
    return real_cr5


def load_official_feedback_dtype() -> np.dtype:
    """复用已固定提交版本的越疆官方 Python V3 反馈数据结构。"""
    if not (OFFICIAL_SDK_DIR / "dobot_api.py").is_file():
        raise FileNotFoundError(
            f"未找到越疆 TCP/IP Python V3 SDK：{OFFICIAL_SDK_DIR}"
        )
    sys.path.insert(0, str(OFFICIAL_SDK_DIR))
    from dobot_api import MyType  # pylint: disable=import-outside-toplevel

    if MyType.itemsize != FEEDBACK_FRAME_BYTES:
        raise RuntimeError(
            f"SDK 反馈结构长度为 {MyType.itemsize}，预期 {FEEDBACK_FRAME_BYTES}"
        )
    return MyType


def recv_exact(sock: socket.socket, byte_count: int) -> bytes:
    """按 TCP 字节流语义收满一帧，避免把半帧误当成完整状态。"""
    chunks: list[bytes] = []
    remaining = byte_count
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("CR5 在完整反馈帧到达前关闭了连接")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def scalar(record: np.void, field: str) -> int | float:
    """把 NumPy 标量转换为便于打印的 Python 标量。"""
    return record[field].item()


def print_feedback(record: np.void, sample_index: int) -> None:
    """打印尚未做 ROS 零位映射的厂商原始状态。"""
    robot_mode = int(scalar(record, "robot_mode"))
    q_actual_deg = np.asarray(record["q_actual"], dtype=float)
    tcp_actual = np.asarray(record["tool_vector_actual"], dtype=float)

    print(f"\n--- CR5 只读反馈样本 {sample_index} ---")
    print(f"robot_mode : {robot_mode} / {ROBOT_MODE_NAMES.get(robot_mode, 'UNKNOWN')}")
    print(f"enabled    : {bool(scalar(record, 'enable_status'))}")
    print(f"running    : {bool(scalar(record, 'running_status'))}")
    print(f"error      : {bool(scalar(record, 'error_status'))}")
    print(
        "motion_queue_running : "
        f"{bool(scalar(record, 'run_queued_cmd'))}"
    )
    print("q_actual_deg (厂商 J1..J6，尚未映射到 ROS):")
    print("  " + ", ".join(f"J{i + 1}={value:.6f}" for i, value in enumerate(q_actual_deg)))
    print("tool_vector_actual (厂商原始 X,Y,Z,Rx,Ry,Rz):")
    print("  " + ", ".join(f"{value:.6f}" for value in tcp_actual))


def read_feedback(config: dict[str, Any], samples: int) -> None:
    """连接反馈端口并读取指定数量的有效状态帧。"""
    feedback_dtype = load_official_feedback_dtype()
    address = (str(config["ip"]), int(config["feedback_port"]))
    connection_timeout = float(config.get("connection_timeout_s", 2.0))
    feedback_timeout = float(config.get("feedback_timeout_s", 2.0))

    print("CR5 READ-ONLY DIAGNOSTIC / 只读诊断")
    print(f"仅连接 {address[0]}:{address[1]}；不会发送任何控制命令。")

    with socket.create_connection(address, timeout=connection_timeout) as sock:
        sock.settimeout(feedback_timeout)
        valid_samples = 0
        discarded_frames = 0
        while valid_samples < samples:
            frame = recv_exact(sock, FEEDBACK_FRAME_BYTES)
            record = np.frombuffer(frame, dtype=feedback_dtype, count=1)[0]
            if int(scalar(record, "test_value")) != FEEDBACK_MAGIC:
                discarded_frames += 1
                if discarded_frames >= 5:
                    raise RuntimeError(
                        "连续 5 帧校验失败；请检查控制器/SDK 协议版本是否匹配"
                    )
                continue
            valid_samples += 1
            print_feedback(record, valid_samples)

    print("\n只读诊断完成：连接已关闭，机器人未被使能、清错或运动。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"硬件配置文件（默认：{DEFAULT_CONFIG}）",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="读取有效反馈样本数量（默认：1）",
    )
    args = parser.parse_args()
    if args.samples < 1 or args.samples > 20:
        parser.error("--samples 必须在 1..20 之间")
    return args


def main() -> int:
    args = parse_args()
    config = load_hardware_config(args.config.resolve())
    read_feedback(config, args.samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
