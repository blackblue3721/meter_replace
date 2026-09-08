#!/usr/bin/env python3
"""Commission the reviewed-waypoint ServoJ executor with a J6 round trip.

The default is dry-run.  ``outbound`` moves only J6 by +2 degrees and records
the exact origin.  ``return`` uses that receipt to return to the origin.  This
isolates dynamic-following communication from MoveIt geometry before another
multi-joint trajectory is attempted.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from commission_j1_roundtrip import wait_for_feedback_target
from commission_j6_step import load_config, read_dashboard_state
from commission_moveit_pregrasp_segment import PROJECT_ROOT, verify_static_feedback
from execute_moveit_grasp_descent import execute_servoj_path, resample_servoj_path
from read_cr5_state import DEFAULT_CONFIG


RECEIPT = PROJECT_ROOT / "logs/commissioning/cr5_servoj_j6_probe.json"
DELTA_DEG = 2.0
OUTBOUND_CONFIRMATION = "MOVE_REAL_CR5_SERVOJ_J6_OUTBOUND_2"
RETURN_CONFIRMATION = "RETURN_REAL_CR5_SERVOJ_J6_TO_ORIGIN"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("outbound", "return"), required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    return parser.parse_args()


def save_receipt(payload: dict) -> None:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())
    authorization = config.get("commissioning_authorization", {})
    if config.get("access_mode") != "commissioning":
        raise RuntimeError("安全拒绝：当前不处于commissioning模式")
    scope = str(authorization.get("scope", ""))
    if not (scope.startswith("moveit_servoj_") and scope.endswith("_only")):
        raise RuntimeError("安全拒绝：当前授权范围不允许ServoJ链路验证")
    if not authorization.get("operator_safety_check_confirmed", False):
        raise RuntimeError("安全拒绝：没有现场安全确认记录")

    dashboard, dashboard_mode, dashboard_deg = read_dashboard_state(config)
    try:
        actual_deg = verify_static_feedback(config, dashboard_mode, dashboard_deg)
        if args.phase == "outbound":
            origin_deg = actual_deg.copy()
            target_deg = origin_deg.copy()
            target_deg[5] += DELTA_DEG
            confirmation = OUTBOUND_CONFIRMATION
        else:
            if not RECEIPT.exists():
                raise RuntimeError("安全拒绝：缺少ServoJ去程记录")
            prior = json.loads(RECEIPT.read_text(encoding="utf-8"))
            if prior.get("status") != "outbound_complete":
                raise RuntimeError("安全拒绝：ServoJ去程尚未完成")
            origin_deg = np.asarray(prior["origin_deg"], dtype=float)
            outbound_deg = np.asarray(prior["outbound_deg"], dtype=float)
            if float(np.max(np.abs(actual_deg - outbound_deg))) > 0.20:
                raise RuntimeError("安全拒绝：当前姿态与ServoJ去程终点不一致")
            target_deg = origin_deg
            confirmation = RETURN_CONFIRMATION

        points_deg = np.vstack((actual_deg, target_deg))
        times_s = np.asarray((0.0, 0.5), dtype=float)
        limits = config["commissioning_limits"]
        preview_points, preview_times = resample_servoj_path(
            points_deg,
            times_s,
            time_scale=float(limits["moveit_servoj_time_scale"]),
            cycle_s=float(limits["moveit_servoj_cycle_s"]),
        )

        print("CR5 SERVOJ REVIEWED-WAYPOINT PROBE")
        print(f"phase={args.phase}, mode={'EXECUTE' if args.execute else 'DRY-RUN'}")
        print(f"commands={len(preview_points)}, duration_s={preview_times[-1]:.3f}")
        print("current_deg: " + ", ".join(f"{v:.6f}" for v in actual_deg))
        print("target_deg : " + ", ".join(f"{v:.6f}" for v in target_deg))
        print("delta_deg  : " + ", ".join(f"{v:+.6f}" for v in target_deg - actual_deg))

        if not args.execute:
            print("DRY-RUN PASSED: 未连接30003，未发送运动。")
            return 0
        if args.confirmation != confirmation:
            raise RuntimeError(f"安全拒绝：必须准确传入{confirmation}")

        execute_servoj_path(config, dashboard, points_deg, times_s)
        wait_for_feedback_target(config, target_deg, timeout_s=10.0)

        if args.phase == "outbound":
            save_receipt(
                {
                    "created_at": datetime.now().astimezone().isoformat(),
                    "status": "outbound_complete",
                    "origin_deg": origin_deg.tolist(),
                    "outbound_deg": target_deg.tolist(),
                }
            )
            print("SERVOJ OUTBOUND PASSED: J6已通过审核点动态跟随移动+2°；尚未回程。")
        else:
            prior = json.loads(RECEIPT.read_text(encoding="utf-8"))
            prior["status"] = "roundtrip_complete"
            prior["returned_at"] = datetime.now().astimezone().isoformat()
            save_receipt(prior)
            print("SERVOJ RETURN PASSED: J6已回到验证前姿态。")
        return 0
    finally:
        dashboard.close()


if __name__ == "__main__":
    raise SystemExit(main())
