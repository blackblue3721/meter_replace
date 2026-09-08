#!/usr/bin/env python3
"""真实 CR5 的 J1 +15° 分段往返调试；默认只做 dry-run。

去程与返回必须分别执行并分别确认，中间留给操作员观察。程序不会自动使能；
返回阶段会读取去程记录并验证当前六轴确实位于去程终点附近。
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from commission_j6_step import (
    load_config,
    read_dashboard_state,
    require_success,
)
from preflight_cr5 import OFFICIAL_SDK_DIR, read_one_feedback, response_code
from read_cr5_state import DEFAULT_CONFIG, scalar


OUTBOUND_CONFIRMATION = "MOVE_REAL_CR5_J1_OUTBOUND_15"
RETURN_CONFIRMATION = "RETURN_REAL_CR5_J1_TO_ORIGIN"
RECEIPT = Path(__file__).resolve().parents[3] / "logs/commissioning/cr5_j1_roundtrip.json"


def wait_for_feedback_target(
    config: dict, target_deg: np.ndarray, timeout_s: float = 120.0
) -> None:
    """在命令应答不明确时，用独立实时反馈确认是否实际到达。"""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        feedback = read_one_feedback(config)
        actual_deg = np.asarray(feedback["q_actual"], dtype=float)
        running = bool(scalar(feedback, "running_status"))
        error = bool(scalar(feedback, "error_status"))
        if error:
            raise RuntimeError("控制器在等待终点期间报告错误")
        maximum_error = float(np.max(np.abs(actual_deg - target_deg)))
        if not running and maximum_error <= 0.10:
            print(
                "Feedback verification passed after ambiguous Sync response: "
                f"maximum error={maximum_error:.6f}°"
            )
            return
        time.sleep(0.2)
    raise RuntimeError(
        f"{timeout_s:g} 秒内未能通过实时反馈确认终点；禁止继续后续阶段"
    )


def execute_joint_move(config: dict, dashboard, target_deg: np.ndarray) -> None:
    """低速发送一条 JointMovJ，并等待控制器完成。"""
    import sys

    sys.path.insert(0, str(OFFICIAL_SDK_DIR))
    from dobot_api import DobotApiMove  # pylint: disable=import-outside-toplevel

    speed = int(config["commissioning_limits"]["speed_factor_percent"])
    require_success("SpeedFactor", dashboard.SpeedFactor(speed))
    require_success("SpeedJ", dashboard.SpeedJ(speed))
    require_success("AccJ", dashboard.AccJ(speed))

    move = DobotApiMove(str(config["ip"]), int(config["motion_port"]))
    move.socket_dobot.settimeout(30.0)
    try:
        reply = move.JointMovJ(*target_deg.tolist())
        require_success("JointMovJ", reply)
        sync_reply = move.Sync()
        if not sync_reply:
            print(
                "WARNING: Sync acknowledgement timed out; the motion will NOT "
                "be resent. Verifying target through port 30004."
            )
            wait_for_feedback_target(config, target_deg)
        elif response_code(sync_reply) != 0:
            reset_reply = dashboard.ResetRobot()
            raise RuntimeError(
                "Sync failed; ResetRobot recovery was requested. "
                f"Sync={sync_reply!r}, ResetRobot={reset_reply!r}"
            )
    finally:
        move.close()


def verify_static_state(config: dict, dashboard_mode: int, current_deg: np.ndarray) -> None:
    feedback = read_one_feedback(config)
    feedback_deg = np.asarray(feedback["q_actual"], dtype=float)
    feedback_mode = int(scalar(feedback, "robot_mode"))
    running = bool(scalar(feedback, "running_status"))
    error = bool(scalar(feedback, "error_status"))
    if dashboard_mode != 5 or feedback_mode != 5:
        raise RuntimeError(
            f"安全拒绝：机器人必须已通过 TCP 使能并静止，mode={dashboard_mode}/{feedback_mode}"
        )
    if running or error:
        raise RuntimeError(f"安全拒绝：running={running}, error={error}")
    if np.max(np.abs(current_deg - feedback_deg)) > 0.05:
        raise RuntimeError("安全拒绝：29999/30004 关节反馈差异超过 0.05°")


def save_receipt(
    origin_deg: np.ndarray, outbound_deg: np.ndarray, status: str
) -> None:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": status,
        "origin_deg": origin_deg.tolist(),
        "outbound_deg": outbound_deg.tolist(),
    }
    RECEIPT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_receipt() -> dict:
    if not RECEIPT.exists():
        raise RuntimeError(f"找不到去程记录：{RECEIPT}")
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def mark_returned(receipt: dict, final_deg: np.ndarray) -> None:
    receipt["status"] = "returned"
    receipt["returned_at"] = datetime.now().astimezone().isoformat()
    receipt["final_deg"] = final_deg.tolist()
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("outbound", "return"), default="outbound")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())
    if config.get("access_mode") != "commissioning":
        raise RuntimeError("安全拒绝：当前不处于 commissioning 模式")
    authorization = config.get("commissioning_authorization", {})
    if authorization.get("scope") != "j1_positive_15deg_roundtrip_only":
        raise RuntimeError("安全拒绝：当前调试授权范围不允许 J1 往返测试")
    if not authorization.get("operator_safety_check_confirmed", False):
        raise RuntimeError("安全拒绝：尚未记录现场操作员安全确认")

    delta = float(config["commissioning_limits"]["j1_roundtrip_delta_deg"])
    dashboard, mode, current_deg = read_dashboard_state(config)
    try:
        verify_static_state(config, mode, current_deg)
        receipt = None
        if args.phase == "outbound":
            origin_deg = current_deg.copy()
            target_deg = current_deg.copy()
            target_deg[0] += delta
            required_confirmation = OUTBOUND_CONFIRMATION
        else:
            receipt = load_receipt()
            if receipt.get("status") != "outbound_complete":
                raise RuntimeError("安全拒绝：去程记录不是待返回状态")
            expected_outbound = np.asarray(receipt["outbound_deg"], dtype=float)
            if np.max(np.abs(current_deg - expected_outbound)) > 0.10:
                raise RuntimeError("安全拒绝：当前六轴姿态与已记录去程终点不一致")
            origin_deg = np.asarray(receipt["origin_deg"], dtype=float)
            target_deg = origin_deg.copy()
            required_confirmation = RETURN_CONFIRMATION

        if not -359.8 <= target_deg[0] <= 359.8:
            raise RuntimeError(f"J1 目标 {target_deg[0]:.3f}° 超出保守软件限位")

        print("CR5 J1 ROUNDTRIP COMMISSIONING")
        print(f"phase={args.phase}, mode={'EXECUTE' if args.execute else 'DRY-RUN'}")
        print("current_deg: " + ", ".join(f"{v:.6f}" for v in current_deg))
        print("target_deg : " + ", ".join(f"{v:.6f}" for v in target_deg))
        print(f"J1 change : {target_deg[0] - current_deg[0]:+.6f}°")
        print("J2..J6    : 保持目标值不变")

        if not args.execute:
            print("DRY-RUN PASSED: 未连接 30003，未发送运动。")
            return 0
        if args.confirmation != required_confirmation:
            raise RuntimeError(f"安全拒绝：必须准确传入 {required_confirmation}")

        # 去程发令前先保存原始姿态，即使应答超时也不会丢失安全返回依据。
        if args.phase == "outbound":
            save_receipt(origin_deg, target_deg, "outbound_command_pending")

        execute_joint_move(config, dashboard, target_deg)
        final_feedback = read_one_feedback(config)
        final_deg = np.asarray(final_feedback["q_actual"], dtype=float)
        maximum_error = float(np.max(np.abs(final_deg - target_deg)))
        print("actual_deg : " + ", ".join(f"{v:.6f}" for v in final_deg))
        print(f"maximum target error: {maximum_error:.6f}°")
        if maximum_error > 0.10:
            raise RuntimeError("执行终点误差超过 0.10°，禁止继续")

        if args.phase == "outbound":
            save_receipt(origin_deg, target_deg, "outbound_complete")
            print(f"OUTBOUND PASSED: J1 +{delta:.1f}°；等待人工观察后才能返回。")
            print(f"receipt={RECEIPT}")
        else:
            mark_returned(receipt, final_deg)
            print("RETURN PASSED: 已回到去程前记录的六轴姿态。")
        return 0
    finally:
        dashboard.close()


if __name__ == "__main__":
    raise SystemExit(main())
