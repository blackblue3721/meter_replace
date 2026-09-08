#!/usr/bin/env python3
"""Generate one continuous eight-stage install bundle in the mock controller.

This tool never opens a Dobot TCP port.  It refuses to run while the real CR5
joint-state publisher is present and requires the mock ros2_control nodes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import JointState


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "logs/real_cr5/full_install_bundle"
HOME_PROFILE = "real_commissioning_cabinet_return"
CONFIRMATION = "GENERATE_CONTINUOUS_MOCK_INSTALL_BUNDLE"
STAGES = (
    ("pregrasp", "joint_target", None),
    ("grasp_candidate", "servoj", None),
    ("lift", "servoj", "virtual_attach"),
    ("semantic_flip", "joint_target", None),
    ("slot_transfer", "servoj", None),
    ("slot_insert", "servoj", None),
    ("slot_retreat", "servoj", "virtual_detach"),
    ("return_home", "servoj", None),
)


def read_joint_state_once(timeout_s: float = 5.0) -> JointState:
    """Read one typed ROS message without parsing noisy CLI/YAML output."""
    rclpy.init()
    node = Node("mock_bundle_joint_state_check")
    received: list[JointState] = []
    subscription = node.create_subscription(
        JointState, "/joint_states", received.append, 10
    )
    deadline = time.monotonic() + timeout_s
    try:
        while not received and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if not received:
            raise RuntimeError("5秒内未收到/joint_states")
        return received[-1]
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def require_mock_scene() -> None:
    result = subprocess.run(
        ["ros2", "node", "list"],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    nodes = set(result.stdout.splitlines())
    if "/cr5_read_only_joint_state_publisher" in nodes:
        raise RuntimeError(
            "安全拒绝：检测到真实CR5关节反馈发布器；请先停止真机同步终端"
        )
    required = {"/controller_manager", "/cr5_group_controller", "/move_group"}
    missing = sorted(required - nodes)
    if missing:
        raise RuntimeError(f"mock场景未完整启动，缺少节点：{missing}")

    state = read_joint_state_once()
    actual = dict(zip(state.name, state.position))
    poses = yaml.safe_load(
        (PROJECT_ROOT / "config/poses.yaml").read_text(encoding="utf-8")
    )
    expected = poses["home_profiles"][HOME_PROFILE]["joint_positions_rad"]
    error_deg = max(
        abs(float(actual[name]) - float(value))
        for name, value in expected.items()
    ) * 180.0 / np.pi
    if error_deg > 0.15:
        raise RuntimeError(
            "mock起点不是real_commissioning_cabinet_return；"
            f"最大误差={error_deg:.6f}°。请启动bundle_generation_scene.launch.py"
        )
    semantic = poses["single_arm"]["meter_semantic_flip"]
    profile_targets = semantic["target_position_rad_by_home_profile"]
    if HOME_PROFILE not in profile_targets:
        raise RuntimeError(
            f"配置缺少{HOME_PROFILE}的语义翻转方向，禁止生成轨迹"
        )
    semantic_target = float(profile_targets[HOME_PROFILE])
    if not -3.10 < semantic_target < -3.00:
        raise RuntimeError(
            "真实安装语义翻转必须保持已验证的J5负向分支；"
            f"当前目标={semantic_target:.6f} rad"
        )


def execute_stage(stage: str, output: Path) -> None:
    run(
        [
            "ros2",
            "run",
            "meter_grasp",
            "run_pick_sequence",
            "--ros-args",
            "-p",
            f"home_profile:={HOME_PROFILE}",
            "-p",
            f"target_stage:={stage}",
            "-p",
            "target_slot:=slot_r1_c2",
            "-p",
            "execute:=true",
            "-p",
            "export_executed_trajectory:=true",
            "-p",
            f"trajectory_output_path:={output}",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_mock_scene()
    if not args.execute:
        print("MOCK BUNDLE DRY-RUN PASSED")
        print("检测到mock控制器；未执行仿真动作，也未写入轨迹。")
        return 0
    if args.confirmation != CONFIRMATION:
        raise RuntimeError(f"必须准确传入{CONFIRMATION}")
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise RuntimeError(
            f"输出目录已包含文件，拒绝覆盖：{OUTPUT_ROOT}"
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "workflow": "desktop_meter_to_slot_r1_c2",
        "home_profile": HOME_PROFILE,
        "status": "mock_generated_operator_review_pending",
        "stages": [],
    }
    for stage, executor, before in STAGES:
        if before == "virtual_attach":
            run(["ros2", "run", "meter_grasp", "attach_meter"])
        elif before == "virtual_detach":
            run(["ros2", "run", "meter_grasp", "detach_meter"])
        output = OUTPUT_ROOT / f"{stage}.json"
        execute_stage(stage, output)
        payload = json.loads(output.read_text(encoding="utf-8"))
        if stage == "semantic_flip":
            joint5_index = payload["joint_names"].index("joint5")
            actual_target = float(
                payload["points"][-1]["positions_rad"][joint5_index]
            )
            expected_target = float(
                yaml.safe_load(
                    (PROJECT_ROOT / "config/poses.yaml").read_text(
                        encoding="utf-8"
                    )
                )["single_arm"]["meter_semantic_flip"]
                ["target_position_rad_by_home_profile"][HOME_PROFILE]
            )
            if abs(actual_target - expected_target) > 0.02:
                raise RuntimeError(
                    "语义翻转终点未落在已验证J5负向分支，停止生成："
                    f"actual={actual_target:.6f}, expected={expected_target:.6f}"
                )
        entry = {
            "name": stage,
            "trajectory": str(output.relative_to(PROJECT_ROOT)),
            "sha256": payload["sha256"],
            "executor": executor,
        }
        if before:
            entry["before"] = before
        manifest["stages"].append(entry)

    manifest_path = OUTPUT_ROOT / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\nMOCK INSTALL BUNDLE GENERATED")
    print(f"manifest={manifest_path}")
    print("状态仍为operator_review_pending；绝不会自动授权真机执行。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
