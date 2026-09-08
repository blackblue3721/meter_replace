#!/usr/bin/env python3
"""停止并清空首次调试遗留的 CR5 TCP 运动队列，不使能、不运动。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

from preflight_cr5 import OFFICIAL_SDK_DIR, parse_first_number_list, response_code
from read_cr5_state import DEFAULT_CONFIG


CONFIRMATION = "CLEAR_PENDING_TCP_QUEUE"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MOVEIT_SEGMENT_RECEIPT = (
    PROJECT_ROOT / "logs/commissioning/cr5_moveit_pregrasp_segment.json"
)
QUEUE_RESET_RECEIPT = PROJECT_ROOT / "logs/commissioning/cr5_queue_reset.json"


def mark_moveit_receipt_recovered() -> None:
    """Record that ResetRobot cleared a previously ambiguous MoveIt command."""
    if not MOVEIT_SEGMENT_RECEIPT.exists():
        return
    receipt = json.loads(MOVEIT_SEGMENT_RECEIPT.read_text(encoding="utf-8"))
    if receipt.get("status") != "command_pending":
        return
    receipt["status"] = "queue_cleared_after_sync_failure"
    receipt["recovered_at"] = datetime.now().astimezone().isoformat()
    MOVEIT_SEGMENT_RECEIPT.write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Marked recovered MoveIt receipt: {MOVEIT_SEGMENT_RECEIPT}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", required=True)
    args = parser.parse_args()
    if args.confirmation != CONFIRMATION:
        raise RuntimeError("安全拒绝：清理队列确认字符串不正确")

    with DEFAULT_CONFIG.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)["real_cr5"]
    if config.get("access_mode") != "commissioning":
        raise RuntimeError("安全拒绝：当前不处于 commissioning 模式")

    sys.path.insert(0, str(OFFICIAL_SDK_DIR))
    from dobot_api import DobotApiDashboard  # pylint: disable=import-outside-toplevel

    dashboard = DobotApiDashboard(str(config["ip"]), int(config["dashboard_port"]))
    dashboard.socket_dobot.settimeout(float(config.get("feedback_timeout_s", 2.0)))
    try:
        before_reply = dashboard.RobotMode()
        if response_code(before_reply) != 0:
            raise RuntimeError(f"RobotMode 查询失败：{before_reply}")
        before_mode = int(parse_first_number_list(before_reply)[0])
        angle_reply = dashboard.GetAngle()
        if response_code(angle_reply) != 0:
            raise RuntimeError(f"GetAngle 查询失败：{angle_reply}")
        before_deg = parse_first_number_list(angle_reply)
        if before_mode in {2, 6, 7, 8, 10, 11}:
            raise RuntimeError(
                f"安全拒绝：机器人处于 mode={before_mode}，请先人工停止/急停"
            )

        print(f"mode_before={before_mode}")
        print("Calling ResetRobot(): stop robot and clear planned command queue")
        reset_reply = dashboard.ResetRobot()
        if response_code(reset_reply) != 0:
            raise RuntimeError(f"ResetRobot 被控制器拒绝：{reset_reply}")

        after_reply = dashboard.RobotMode()
        if response_code(after_reply) != 0:
            raise RuntimeError(f"清理后 RobotMode 查询失败：{after_reply}")
        after_mode = int(parse_first_number_list(after_reply)[0])
        error_reply = dashboard.GetErrorID()
        if response_code(error_reply) != 0:
            raise RuntimeError(f"GetErrorID 查询失败：{error_reply}")

        print(f"mode_after={after_mode}")
        print(f"errors={error_reply}")
        QUEUE_RESET_RECEIPT.parent.mkdir(parents=True, exist_ok=True)
        QUEUE_RESET_RECEIPT.write_text(
            json.dumps(
                {
                    "created_at": datetime.now().astimezone().isoformat(),
                    "status": "queue_reset_complete",
                    "mode_before": before_mode,
                    "mode_after": after_mode,
                    "joint_deg": list(map(float, before_deg)),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"queue_reset_receipt={QUEUE_RESET_RECEIPT}")
        mark_moveit_receipt_recovered()
        print("QUEUE RECOVERY PASSED: pending TCP motion queue was stopped and cleared")
        print("The script did not issue EnableRobot and sent no motion target.")
        return 0
    finally:
        dashboard.close()


if __name__ == "__main__":
    raise SystemExit(main())
