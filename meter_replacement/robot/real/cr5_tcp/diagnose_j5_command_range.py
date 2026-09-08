#!/usr/bin/env python3
"""只读诊断 CR5 控制器当前接受的 J5 指令参数范围。

程序仅连接 Dashboard 端口 29999，调用 GetAngle、RobotMode、GetErrorID 和
PositiveSolution（正运动学计算）。它不会连接 30003，不会改变速度、使能状态，
也不会发送任何运动命令。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from commission_j6_step import load_config, read_dashboard_state, require_success
from preflight_cr5 import response_code
from read_cr5_state import DEFAULT_CONFIG


DEFAULT_CANDIDATES_DEG = (-179.993014, -179.0, -178.0, -175.0, -170.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--candidate-deg",
        type=float,
        nargs="+",
        default=DEFAULT_CANDIDATES_DEG,
        help="要交给正运动学接口检查的 J5 候选角（不会运动）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())
    print("CR5 J5 COMMAND-RANGE DIAGNOSTIC / 只读参数范围诊断")
    print("仅调用 PositiveSolution；不连接 30003；绝不发送运动。")

    dashboard, mode, current_deg = read_dashboard_state(config)
    try:
        if mode not in (4, 5):
            raise RuntimeError(f"只读诊断要求机器人静止，当前 RobotMode={mode}")
        print("current_deg: " + ", ".join(f"{v:.6f}" for v in current_deg))
        print("\nJ5 candidate results:")
        accepted = []
        for candidate in args.candidate_deg:
            target = np.asarray(current_deg, dtype=float).copy()
            target[4] = float(candidate)
            reply = dashboard.PositiveSolution(*target.tolist(), 0, 0)
            code = response_code(reply)
            result = "ACCEPTED" if code == 0 else f"REJECTED (code={code})"
            print(f"  J5={candidate:11.6f} deg -> {result}")
            if code == 0:
                accepted.append(float(candidate))

        require_success("GetErrorID", dashboard.GetErrorID())
        if not accepted:
            print("\nRESULT: 所有候选均被拒绝；不要生成或执行新的翻转轨迹。")
            return 2
        print(
            "\nRESULT: READ-ONLY CHECK COMPLETE; accepted candidates="
            + ", ".join(f"{value:.6f}" for value in accepted)
        )
        print("机器人没有运动，也没有运动命令进入队列。")
        return 0
    finally:
        dashboard.close()


if __name__ == "__main__":
    raise SystemExit(main())
