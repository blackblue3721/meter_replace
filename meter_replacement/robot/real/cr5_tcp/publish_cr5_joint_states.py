#!/usr/bin/env python3
"""将真实 CR5 的只读反馈发布为 ROS 2 ``/real_joint_states``。

该节点只连接 30004，不连接控制/运动端口，也不发送任何 TCP 数据。当前关节
映射仍处于外观验证阶段，只可用于在 RViz 中对照真实机械臂，禁止据此运动。
"""

from __future__ import annotations

import argparse
import fcntl
import socket
import threading
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from read_cr5_state import (
    DEFAULT_CONFIG,
    FEEDBACK_FRAME_BYTES,
    FEEDBACK_MAGIC,
    load_hardware_config,
    load_official_feedback_dtype,
    recv_exact,
    scalar,
)


class CR5ReadOnlyJointStatePublisher(Node):
    """持续读取厂商反馈，并以 ROS 弧度关节状态发布最新有效帧。"""

    def __init__(self, config_path: Path, topic: str = "/real_joint_states") -> None:
        super().__init__("cr5_read_only_joint_state_publisher")
        self._config = load_hardware_config(config_path)
        mapping = self._config["joint_mapping"]
        if mapping.get("status") not in {
            "visual_validation_pending",
            "visual_validation_passed",
        }:
            raise RuntimeError("关节映射状态无效，拒绝启动")

        self._ros_joint_names = list(mapping["ros_joint_names"])
        self._sign = np.asarray(mapping["sign"], dtype=float)
        self._offset_deg = np.asarray(mapping["offset_deg"], dtype=float)
        if not (
            len(self._ros_joint_names) == len(self._sign) == len(self._offset_deg) == 6
        ):
            raise RuntimeError("joint_mapping 必须完整定义六个关节")

        self._virtual_gripper_position = float(
            self._config["end_effector"].get("visual_open_position_m", 0.020)
        )
        self._feedback_dtype = load_official_feedback_dtype()
        self._latest_q_deg: np.ndarray | None = None
        self._latest_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._reader_error: Exception | None = None
        self._error_reported = False

        self._publisher = self.create_publisher(JointState, topic, 10)
        self._timer = self.create_timer(0.05, self._publish_latest)
        self._reader = threading.Thread(
            target=self._feedback_loop,
            name="cr5-read-only-feedback",
            daemon=True,
        )
        self._reader.start()

        ip = self._config["ip"]
        port = self._config["feedback_port"]
        self.get_logger().warn(
            f"READ ONLY: connecting only to {ip}:{port}; no command can be sent"
        )
        self.get_logger().warn(
            f"Joint mapping status: {mapping['status']}; this node remains read-only"
        )
        self.get_logger().info(f"Publishing real CR5 feedback on: {topic}")

    def _feedback_loop(self) -> None:
        """后台收取完整 1440 字节帧，主线程只负责 ROS 发布。"""
        address = (str(self._config["ip"]), int(self._config["feedback_port"]))
        connect_timeout = float(self._config.get("connection_timeout_s", 2.0))
        feedback_timeout = float(self._config.get("feedback_timeout_s", 2.0))
        try:
            with socket.create_connection(address, timeout=connect_timeout) as sock:
                sock.settimeout(feedback_timeout)
                while not self._stop_event.is_set():
                    frame = recv_exact(sock, FEEDBACK_FRAME_BYTES)
                    record = np.frombuffer(
                        frame, dtype=self._feedback_dtype, count=1
                    )[0]
                    if int(scalar(record, "test_value")) != FEEDBACK_MAGIC:
                        continue
                    with self._latest_lock:
                        self._latest_q_deg = np.asarray(
                            record["q_actual"], dtype=float
                        ).copy()
        except Exception as exc:  # ROS timer reports it once in the main thread.
            self._reader_error = exc

    def _publish_latest(self) -> None:
        """将配置化的厂商角度映射成 ROS 弧度并发布。"""
        if self._reader_error is not None and not self._error_reported:
            self.get_logger().error(f"CR5 feedback stopped: {self._reader_error}")
            self._error_reported = True

        with self._latest_lock:
            if self._latest_q_deg is None:
                return
            q_vendor_deg = self._latest_q_deg.copy()

        q_ros_rad = np.deg2rad(self._sign * q_vendor_deg + self._offset_deg)
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "base_link"
        message.name = self._ros_joint_names + ["gripper_finger_joint"]
        message.position = q_ros_rad.tolist() + [self._virtual_gripper_position]
        self._publisher.publish(message)

    def destroy_node(self) -> bool:
        self._stop_event.set()
        return super().destroy_node()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--topic", default="/real_joint_states")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lock_file = Path("/tmp/meter_replacement_cr5_joint_state_publisher.lock").open(
        "w"
    )
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(
            "已有一个 CR5 只读关节反馈发布器在运行，拒绝重复启动"
        ) from exc
    rclpy.init()
    node = CR5ReadOnlyJointStatePublisher(args.config.resolve(), str(args.topic))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
