#!/usr/bin/env python3
"""Validate and eventually execute one complete real-CR5 meter installation.

The default is a read-only dry-run.  Execution remains deliberately locked
until the continuous eight-stage bundle has passed validation and the project
configuration explicitly authorizes the complete workflow.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

from commission_j6_step import load_config, read_dashboard_state
from commission_moveit_pregrasp_segment import (
    PROJECT_ROOT,
    execute_joint_target,
    verify_static_feedback,
)
from execute_moveit_grasp_descent import execute_servoj_path
from preflight_cr5 import read_one_feedback
from read_cr5_state import DEFAULT_CONFIG
from real_install_workflow import validate_manifest


DEFAULT_MANIFEST = PROJECT_ROOT / "config/real_cr5_install_manifest.json"
CONFIRMATION = "RUN_REAL_CR5_FULL_METER_INSTALL"
RUNS_ROOT = PROJECT_ROOT / "logs/commissioning/runs"

MAXIMUM_DELTA_KEYS = {
    "pregrasp": "moveit_pregrasp_full_max_joint_delta_deg",
    "grasp_candidate": "moveit_lift_max_joint_delta_deg",
    "lift": "moveit_lift_max_joint_delta_deg",
    "semantic_flip": "moveit_semantic_flip_max_joint_delta_deg",
    "slot_transfer": "moveit_slot_transfer_max_joint_delta_deg",
    "slot_insert": "moveit_slot_insert_max_joint_delta_deg",
    "slot_retreat": "moveit_slot_retreat_max_joint_delta_deg",
    "return_home": "moveit_return_home_max_joint_delta_deg",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    return parser.parse_args()


def save_run_receipt(path: Path, payload: dict) -> None:
    """Replace one run receipt without touching earlier commissioning records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_stage_limits(stages, limits: dict) -> None:
    for stage in stages:
        maximum_delta = float(np.max(np.abs(stage.target_deg - stage.start_deg)))
        maximum_allowed = float(limits[MAXIMUM_DELTA_KEYS[stage.name]])
        if maximum_delta > maximum_allowed:
            raise RuntimeError(
                f"{stage.name}：最大关节变化{maximum_delta:.6f}° > "
                f"{maximum_allowed:.3f}°"
            )
        if stage.executor == "joint_target":
            line = stage.points_deg[-1] - stage.points_deg[0]
            norm = float(line @ line)
            if norm <= 1.0e-12:
                raise RuntimeError(f"{stage.name}：关节目标起终点相同")
            progress = ((stage.points_deg - stage.points_deg[0]) @ line) / norm
            projected = stage.points_deg[0] + progress[:, None] * line
            deviation = float(np.max(np.abs(stage.points_deg - projected)))
            deviation_limit = float(
                limits["moveit_joint_line_deviation_tolerance_deg"]
            )
            if np.any(np.diff(progress) < -1.0e-10) or deviation > deviation_limit:
                raise RuntimeError(f"{stage.name}：不是已验证的单调JointMovJ轨迹")

    flip = next(stage for stage in stages if stage.name == "semantic_flip")
    j5_min, j5_max = limits["moveit_semantic_flip_j5_guard_deg"]
    if not float(j5_min) <= float(flip.target_deg[4]) <= float(j5_max):
        raise RuntimeError("semantic_flip：J5目标超出实机保护范围")


def planning_scene_handshake(action: str) -> None:
    executable = "attach_meter" if action == "virtual_attach" else "detach_meter"
    print(f"PLANNING SCENE HANDSHAKE: {action}")
    subprocess.run(
        ["ros2", "run", "meter_grasp", executable],
        check=True,
        cwd=PROJECT_ROOT,
    )


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())
    limits = config["commissioning_limits"]
    tolerance = float(limits["moveit_trajectory_start_tolerance_deg"])
    manifest, stages = validate_manifest(
        args.manifest.resolve(),
        project_root=PROJECT_ROOT,
        joint_mapping=config["joint_mapping"],
        continuity_tolerance_deg=tolerance,
    )
    verify_stage_limits(stages, limits)

    receipt_path = RUNS_ROOT / args.run_id / "meter_install.json"
    resume_index = 0
    receipt = None
    if args.resume:
        if not receipt_path.exists():
            raise RuntimeError(f"安全拒绝：续跑回执不存在：{args.run_id}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        completed = receipt.get("completed_stages", [])
        stage_names = [stage.name for stage in stages]
        expected_hashes = {stage.name: stage.digest for stage in stages}
        if receipt.get("status") != "failed":
            raise RuntimeError("安全拒绝：只允许续跑状态为failed的流程")
        if receipt.get("workflow") != manifest["workflow"]:
            raise RuntimeError("安全拒绝：续跑回执workflow不匹配")
        if receipt.get("trajectory_sha256") != expected_hashes:
            raise RuntimeError("安全拒绝：续跑轨迹哈希与原回执不一致")
        if completed != stage_names[: len(completed)] or len(completed) >= len(stages):
            raise RuntimeError("安全拒绝：已完成阶段不是合法前缀")
        resume_index = len(completed)

    dashboard, dashboard_mode, dashboard_deg = read_dashboard_state(config)
    try:
        actual_deg = verify_static_feedback(config, dashboard_mode, dashboard_deg)
    finally:
        dashboard.close()
    start_error = float(
        np.max(np.abs(actual_deg - stages[resume_index].start_deg))
    )
    if start_error > tolerance:
        raise RuntimeError(
            f"安全拒绝：真机与完整任务起点误差{start_error:.6f}° > "
            f"{tolerance:.3f}°"
        )

    print("CR5 COMPLETE METER INSTALL WORKFLOW")
    print(f"mode={'EXECUTE' if args.execute else 'DRY-RUN (NO MOTION)'}")
    print(f"workflow={manifest['workflow']}")
    print(f"home_profile={manifest['home_profile']}")
    if args.resume:
        print(f"resume_from={stages[resume_index].name}")
    for index, stage in enumerate(stages, start=1):
        delta = float(np.max(np.abs(stage.target_deg - stage.start_deg)))
        print(
            f"{index}. {stage.name}: executor={stage.executor}, "
            f"points={len(stage.points_deg)}, max_delta={delta:.3f}°, "
            f"sha256={stage.digest}"
        )

    if not args.execute:
        print("FULL WORKFLOW DRY-RUN PASSED: 轨迹链、哈希和真机起点均已核验；未发送运动。")
        return 0

    authorization = config.get("commissioning_authorization", {})
    if authorization.get("scope") != "real_meter_install_full_workflow":
        raise RuntimeError("安全拒绝：配置尚未授权完整安装流程")
    if not authorization.get("full_workflow_authorized", False):
        raise RuntimeError("安全拒绝：full_workflow_authorized尚未开启")
    if args.confirmation != CONFIRMATION:
        raise RuntimeError(f"安全拒绝：必须准确传入{CONFIRMATION}")
    if not args.run_id:
        raise RuntimeError("安全拒绝：实机整流程必须指定唯一--run-id")
    authorized_run_id = str(authorization.get("authorized_run_id", ""))
    if not authorized_run_id or args.run_id != authorized_run_id:
        raise RuntimeError(
            "安全拒绝：run-id与本次单次授权不一致："
            f"requested={args.run_id!r}, authorized={authorized_run_id!r}"
        )
    if manifest.get("status") != "validated_for_real_execution":
        raise RuntimeError("安全拒绝：任务清单尚未标记为实机整流程验证通过")
    if not args.run_id.replace("-", "").replace("_", "").isalnum():
        raise RuntimeError("安全拒绝：run-id只能包含字母、数字、横线和下划线")

    if args.resume:
        receipt.setdefault("resume_history", []).append(
            {
                "resumed_at": datetime.now().astimezone().isoformat(),
                "previous_error": receipt.get("error", ""),
            }
        )
        receipt.pop("failed_at", None)
        receipt.pop("error", None)
        receipt["status"] = "running"
    else:
        if receipt_path.exists():
            raise RuntimeError(f"安全拒绝：run-id已存在：{args.run_id}")
        receipt = {
            "run_id": args.run_id,
            "workflow": manifest["workflow"],
            "created_at": datetime.now().astimezone().isoformat(),
            "status": "running",
            "completed_stages": [],
            "trajectory_sha256": {stage.name: stage.digest for stage in stages},
        }
    save_run_receipt(receipt_path, receipt)

    speed = int(limits["moveit_first_segment_speed_percent"])
    target_tolerance = float(limits["moveit_servoj_target_tolerance_deg"])
    try:
        for stage in stages[resume_index:]:
            dashboard, dashboard_mode, dashboard_deg = read_dashboard_state(config)
            try:
                actual_deg = verify_static_feedback(
                    config, dashboard_mode, dashboard_deg
                )
                live_error = float(np.max(np.abs(actual_deg - stage.start_deg)))
                if live_error > tolerance:
                    raise RuntimeError(
                        f"{stage.name}：实时起点误差{live_error:.6f}° > "
                        f"{tolerance:.3f}°"
                    )
                if stage.before:
                    planning_scene_handshake(stage.before)
                print(f"STATE -> EXECUTING_{stage.name.upper()}")
                if stage.executor == "joint_target":
                    execute_joint_target(
                        config, dashboard, stage.target_deg, speed
                    )
                else:
                    execute_servoj_path(
                        config, dashboard, stage.points_deg, stage.times_s
                    )
            finally:
                dashboard.close()
            actual_deg = np.asarray(read_one_feedback(config)["q_actual"], dtype=float)
            final_error = float(np.max(np.abs(actual_deg - stage.target_deg)))
            print(f"{stage.name}: maximum target error={final_error:.6f}°")
            if final_error > target_tolerance:
                raise RuntimeError(
                    f"{stage.name}：终点反馈误差超限；完整流程立即停止"
                )
            receipt["completed_stages"].append(stage.name)
            receipt["last_actual_deg"] = actual_deg.tolist()
            save_run_receipt(receipt_path, receipt)
            print(f"STATE -> {stage.name.upper()}_COMPLETE")
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["failed_at"] = datetime.now().astimezone().isoformat()
        receipt["error"] = str(exc)
        save_run_receipt(receipt_path, receipt)
        raise
    receipt["status"] = "complete"
    receipt["completed_at"] = datetime.now().astimezone().isoformat()
    save_run_receipt(receipt_path, receipt)
    print("FULL METER INSTALL PASSED: 真机完整空载安装流程执行成功。")
    print(f"receipt={receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
