#!/usr/bin/env python3
"""Safely resume an idle CR5 TCP motion queue; dry-run by default."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from commission_j6_step import load_config, read_dashboard_state, require_success
from commission_moveit_pregrasp_segment import verify_static_feedback
from preflight_cr5 import read_one_feedback
from read_cr5_state import DEFAULT_CONFIG, scalar


CONFIRMATION = "RESUME_REAL_CR5_TCP_MOTION_SESSION"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())
    authorization = config.get("commissioning_authorization", {})
    if config.get("access_mode") != "commissioning":
        raise RuntimeError("安全拒绝：当前不处于commissioning模式")
    scope = str(authorization.get("scope", ""))
    if not (scope.startswith("moveit_servoj_") and scope.endswith("_only")):
        raise RuntimeError("安全拒绝：当前授权范围不允许恢复TCP运动会话")
    if not authorization.get("operator_safety_check_confirmed", False):
        raise RuntimeError("安全拒绝：没有现场安全确认记录")

    dashboard, dashboard_mode, dashboard_deg = read_dashboard_state(config)
    try:
        feedback = read_one_feedback(config)
        actual_deg = np.asarray(feedback["q_actual"], dtype=float)
        feedback_mode = int(scalar(feedback, "robot_mode"))
        running = bool(scalar(feedback, "running_status"))
        error = bool(scalar(feedback, "error_status"))
        queue_running = bool(scalar(feedback, "run_queued_cmd"))
        if dashboard_mode != 5 or feedback_mode != 5 or running or error:
            raise RuntimeError(
                "安全拒绝：恢复队列前要求mode=5、静止且无报警；"
                f"mode={dashboard_mode}/{feedback_mode}, "
                f"running={running}, error={error}"
            )
        if float(np.max(np.abs(dashboard_deg - actual_deg))) > 0.05:
            raise RuntimeError("安全拒绝：29999与30004关节反馈差异超过0.05°")

        print("CR5 TCP MOTION QUEUE RESUME")
        print(f"mode={'EXECUTE' if args.execute else 'DRY-RUN'}")
        print("current_deg: " + ", ".join(f"{v:.6f}" for v in actual_deg))
        print(f"queue_running={queue_running}")
        print("motion port 30003 will not be opened")
        if queue_running:
            print("QUEUE CHECK PASSED: 队列已经运行，无需发送continue()。")
            return 0
        if not args.execute:
            print("DRY-RUN PASSED: 候选命令为continue()；尚未发送。")
            return 0
        if args.confirmation != CONFIRMATION:
            raise RuntimeError(f"安全拒绝：必须准确传入{CONFIRMATION}")
        reply = dashboard.Continue()
        if reply:
            require_success("continue", reply)
    finally:
        dashboard.close()

    # The controller may close the command socket after continue(); verify on
    # a fresh connection rather than treating that expected close as failure.
    time.sleep(0.5)
    resumed, resumed_mode, resumed_deg = read_dashboard_state(config)
    try:
        verify_static_feedback(config, resumed_mode, resumed_deg)
    finally:
        resumed.close()
    print("QUEUE RESUME PASSED: TCP运动队列已恢复；未发送任何运动目标。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
