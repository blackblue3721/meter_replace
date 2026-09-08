#!/usr/bin/env python3
"""只读检查一个 CR5 笛卡尔目标并计算厂商逆运动学解。

本程序仅连接 Dashboard 端口 29999，读取当前状态并调用 InverseSolution。
它不会连接运动端口 30003，也不会使能机器人或发送运动命令。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from commission_j6_step import load_config, read_dashboard_state, require_success
from preflight_cr5 import parse_first_number_list, response_code
from read_cr5_state import DEFAULT_CONFIG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--pose",
        type=float,
        nargs=6,
        required=True,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="目标笛卡尔位姿，位置单位 mm、姿态单位 degree",
    )
    parser.add_argument("--user", type=int, required=True, help="CR5 用户坐标系索引")
    parser.add_argument("--tool", type=int, required=True, help="CR5 工具坐标系索引")
    return parser.parse_args()


def inverse_solution_near(dashboard, pose: np.ndarray, user: int, tool: int,
                          joint_near: np.ndarray) -> str:
    """按越疆协议发送带 JointNear 的逆解查询。

    官方 V3 Python SDK 的 ``InverseSolution(..., *dynParams)`` 没有正确补逗号，
    因此这里仅对 29999 端口直接构造官方协议字符串。该指令只计算逆解，
    不会进入运动队列。
    """
    pose_text = ",".join(f"{value:.6f}" for value in pose)
    joints_text = ",".join(f"{value:.6f}" for value in joint_near)
    command = (
        f"InverseSolution({pose_text},{user:d},{tool:d},"
        f"1,{{{joints_text}}})"
    )
    return dashboard.sendRecvMsg(command)


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())
    target_pose = np.asarray(args.pose, dtype=float)

    print("CR5 CARTESIAN TARGET DIAGNOSTIC / 笛卡尔目标只读检查")
    print("仅调用 PositiveSolution/InverseSolution；不连接 30003；绝不发送运动。")

    dashboard, mode, current_deg = read_dashboard_state(config)
    try:
        if mode not in (4, 5):
            raise RuntimeError(f"只读诊断要求机器人静止，当前 RobotMode={mode}")
        if not 0 <= args.user <= 9 or not 0 <= args.tool <= 9:
            raise ValueError("--user 和 --tool 必须在 0..9 范围内")

        current_pose_reply = dashboard.PositiveSolution(
            *current_deg.tolist(), int(args.user), int(args.tool)
        )
        require_success("PositiveSolution", current_pose_reply)
        current_pose = parse_first_number_list(current_pose_reply)
        if current_pose.size != 6:
            raise RuntimeError(
                f"PositiveSolution 未返回六维位姿：{current_pose_reply!r}"
            )

        current_base_reply = dashboard.PositiveSolution(
            *current_deg.tolist(), 0, int(args.tool)
        )
        require_success("PositiveSolution(base)", current_base_reply)
        current_base_pose = parse_first_number_list(current_base_reply)
        if current_base_pose.size != 6:
            raise RuntimeError(
                f"基座坐标正解未返回六维位姿：{current_base_reply!r}"
            )

        print(f"coordinate_system: User={args.user}, Tool={args.tool}")
        print("current_pose: " + ", ".join(f"{value:.6f}" for value in current_pose))
        print("target_pose : " + ", ".join(f"{value:.6f}" for value in target_pose))
        print("current_base_pose (User=0): " + ", ".join(
            f"{value:.6f}" for value in current_base_pose
        ))

        reply = dashboard.InverseSolution(
            *target_pose.tolist(), int(args.user), int(args.tool)
        )
        if response_code(reply) != 0:
            print(f"default_inverse_reply: {reply}")
            print("默认逆解未选中分支，改用当前六轴角作为 JointNear 参考继续只读求解。")
            reply = inverse_solution_near(
                dashboard,
                target_pose,
                int(args.user),
                int(args.tool),
                current_deg,
            )
            if response_code(reply) != 0:
                print(f"\nTARGET IK UNRESOLVED: 带 JointNear 的逆解仍失败：{reply}")
                print("这不等同于目标不可达；若人工到达过，应在该点记录六轴角作为种子。")
                print("未连接 30003，机器人没有运动，也没有运动命令进入队列。")
                return 2
        target_deg = parse_first_number_list(reply)
        if target_deg.size != 6:
            raise RuntimeError(f"InverseSolution 未返回六轴角度：{reply!r}")

        delta_deg = target_deg - current_deg
        target_base_reply = dashboard.PositiveSolution(
            *target_deg.tolist(), 0, int(args.tool)
        )
        require_success("PositiveSolution(target, base)", target_base_reply)
        target_base_pose = parse_first_number_list(target_base_reply)
        if target_base_pose.size != 6:
            raise RuntimeError(
                f"目标基座坐标正解未返回六维位姿：{target_base_reply!r}"
            )

        print("current_deg: " + ", ".join(f"{value:.6f}" for value in current_deg))
        print("target_deg : " + ", ".join(f"{value:.6f}" for value in target_deg))
        print("delta_deg  : " + ", ".join(f"{value:+.6f}" for value in delta_deg))
        print(f"maximum_delta_deg: {np.max(np.abs(delta_deg)):.6f}")
        print("target_base_pose (User=0): " + ", ".join(
            f"{value:.6f}" for value in target_base_pose
        ))
        print("\nREAD-ONLY IK PASSED: 目标存在厂商逆解；尚未验证 MoveIt 碰撞与路径。")
        print("机器人没有运动，也没有运动命令进入队列。")
        return 0
    finally:
        dashboard.close()


if __name__ == "__main__":
    raise SystemExit(main())
