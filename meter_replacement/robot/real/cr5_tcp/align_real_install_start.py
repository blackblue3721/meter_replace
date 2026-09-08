#!/usr/bin/env python3
"""Guardedly align the real CR5 to the reviewed full-install bundle start.

The command is dry-run by default.  It derives the exact six-axis target from
the hash-validated workflow manifest, permits only a very small JointMovJ, and
never continues into the installation workflow.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from commission_j6_step import load_config, read_dashboard_state
from commission_moveit_pregrasp_segment import (
    PROJECT_ROOT,
    execute_joint_target,
    verify_static_feedback,
)
from preflight_cr5 import read_one_feedback
from read_cr5_state import DEFAULT_CONFIG
from real_install_workflow import validate_manifest


DEFAULT_MANIFEST = PROJECT_ROOT / "config/real_cr5_install_manifest.json"
RECEIPT = PROJECT_ROOT / "logs/commissioning/real_install_start_alignment.json"
CONFIRMATION = "ALIGN_REAL_CR5_FULL_INSTALL_START"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    return parser.parse_args()


def save_receipt(payload: dict) -> None:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    temporary = RECEIPT.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(RECEIPT)


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())
    limits = config["commissioning_limits"]
    manifest, stages = validate_manifest(
        args.manifest.resolve(),
        project_root=PROJECT_ROOT,
        joint_mapping=config["joint_mapping"],
        continuity_tolerance_deg=float(
            limits["moveit_trajectory_start_tolerance_deg"]
        ),
    )
    if manifest.get("status") != "validated_for_real_execution":
        raise RuntimeError("安全拒绝：完整任务轨迹包尚未审核")
    target_deg = stages[0].start_deg

    dashboard, dashboard_mode, dashboard_deg = read_dashboard_state(config)
    try:
        actual_deg = verify_static_feedback(config, dashboard_mode, dashboard_deg)
        delta_deg = target_deg - actual_deg
        maximum_delta = float(np.max(np.abs(delta_deg)))
        maximum_allowed = float(
            limits["real_install_start_alignment_max_delta_deg"]
        )

        print("CR5 FULL-INSTALL START ALIGNMENT")
        print(f"mode={'EXECUTE' if args.execute else 'DRY-RUN (NO MOTION)'}")
        print(f"workflow={manifest['workflow']}")
        print("current_deg: " + ", ".join(f"{v:.6f}" for v in actual_deg))
        print("target_deg : " + ", ".join(f"{v:.6f}" for v in target_deg))
        print("delta_deg  : " + ", ".join(f"{v:+.6f}" for v in delta_deg))
        print(f"maximum_delta_deg: {maximum_delta:.6f}")
        if maximum_delta > maximum_allowed:
            raise RuntimeError(
                f"安全拒绝：起点归一化变化{maximum_delta:.6f}° > "
                f"{maximum_allowed:.3f}°"
            )

        if not args.execute:
            print("ALIGNMENT DRY-RUN PASSED: 未连接30003，未发送运动。")
            return 0

        authorization = config.get("commissioning_authorization", {})
        if config.get("access_mode") != "commissioning":
            raise RuntimeError("安全拒绝：当前不处于commissioning模式")
        if authorization.get("scope") != "real_install_start_alignment_only":
            raise RuntimeError("安全拒绝：配置未授权起点归一化")
        if authorization.get("full_workflow_authorized", False):
            raise RuntimeError("安全拒绝：起点归一化时不得开启完整流程授权")
        if not authorization.get("operator_safety_check_confirmed", False):
            raise RuntimeError("安全拒绝：缺少现场操作员安全确认")
        if args.confirmation != CONFIRMATION:
            raise RuntimeError(f"安全拒绝：必须准确传入{CONFIRMATION}")

        speed = int(limits["speed_factor_percent"])
        execute_joint_target(config, dashboard, target_deg, speed)
        final_deg = np.asarray(read_one_feedback(config)["q_actual"], dtype=float)
        final_error = float(np.max(np.abs(final_deg - target_deg)))
        print("actual_deg : " + ", ".join(f"{v:.6f}" for v in final_deg))
        print(f"maximum target error: {final_error:.6f}°")
        if final_error > float(limits["moveit_waypoint_target_tolerance_deg"]):
            raise RuntimeError("起点归一化终点反馈误差超限；禁止进入完整流程")
        save_receipt(
            {
                "created_at": datetime.now().astimezone().isoformat(),
                "status": "aligned",
                "workflow": manifest["workflow"],
                "origin_deg": actual_deg.tolist(),
                "target_deg": target_deg.tolist(),
                "actual_deg": final_deg.tolist(),
                "maximum_target_error_deg": final_error,
            }
        )
        print("START ALIGNMENT PASSED: 真机已精确对齐完整安装轨迹起点；未执行工艺动作。")
        print(f"receipt={RECEIPT}")
        return 0
    finally:
        dashboard.close()


if __name__ == "__main__":
    raise SystemExit(main())
