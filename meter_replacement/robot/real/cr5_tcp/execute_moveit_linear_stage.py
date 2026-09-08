#!/usr/bin/env python3
"""Guarded ServoJ executor for reviewed MoveIt trajectory stages.

The stage profile defines its artifact, predecessor receipt, authorization,
confirmation phrase and joint bounds.  Adding later LIN stages reuses this
single implementation instead of duplicating hardware motion code.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from commission_j6_step import load_config, read_dashboard_state
from execute_moveit_grasp_descent import execute_servoj_path
from commission_moveit_pregrasp_segment import (
    PROJECT_ROOT,
    ros_to_vendor_deg,
    verify_digest,
    verify_static_feedback,
)
from preflight_cr5 import read_one_feedback
from read_cr5_state import DEFAULT_CONFIG
from clear_pending_queue import QUEUE_RESET_RECEIPT


STAGE_PROFILES = {
    "vision_safe_transition": {
        "artifact": (
            PROJECT_ROOT / "logs/vision/first_meter_safe_transition_candidate.json"
        ),
        "expected_target_stage": "return_home",
        "authorization_scope": "moveit_servoj_vision_two_stage_approach_only",
        "predecessor_receipt": None,
        "predecessor_status": None,
        "receipt": (
            PROJECT_ROOT / "logs/commissioning/cr5_vision_safe_transition.json"
        ),
        "completion_status": "vision_safe_transition_complete",
        "confirmation": "MOVE_REAL_CR5_TO_VISION_SAFE_TRANSITION",
        "maximum_delta_key": "moveit_vision_safe_transition_max_joint_delta_deg",
        "maximum_step_key": "moveit_slot_transfer_servoj_max_step_deg",
        "success_message": (
            "VISION SAFE TRANSITION PASSED: 真机已到视觉任务安全过渡位。"
        ),
    },
    "vision_meter_approach": {
        "artifact": (
            PROJECT_ROOT / "logs/vision/first_meter_dynamic_approach_candidate.json"
        ),
        "expected_target_stage": "vision_meter_approach",
        "authorization_scope": "moveit_servoj_vision_two_stage_approach_only",
        "predecessor_receipt": (
            PROJECT_ROOT / "logs/commissioning/cr5_vision_safe_transition.json"
        ),
        "predecessor_status": "vision_safe_transition_complete",
        "receipt": (
            PROJECT_ROOT / "logs/commissioning/cr5_vision_meter_approach.json"
        ),
        "completion_status": "vision_meter_approach_complete",
        "confirmation": "MOVE_REAL_CR5_TO_VISION_METER_APPROACH",
        "maximum_delta_key": "moveit_vision_approach_max_joint_delta_deg",
        "maximum_step_key": "moveit_slot_transfer_servoj_max_step_deg",
        "success_message": (
            "VISION APPROACH PASSED: 真机已到D435检测电表的渐进位；"
            "不会自动下降或夹取。"
        ),
    },
    "install_start_transition": {
        "artifact": (
            PROJECT_ROOT / "logs/real_cr5/eight_stage_start_candidate.json"
        ),
        "expected_target_stage": "return_home",
        "authorization_scope": "moveit_servoj_install_start_transition_only",
        "predecessor_receipt": None,
        "predecessor_status": None,
        "receipt": (
            PROJECT_ROOT
            / "logs/commissioning/cr5_moveit_install_start_transition.json"
        ),
        "completion_status": "install_start_transition_complete",
        "confirmation": "MOVE_REAL_CR5_TO_EIGHT_STAGE_START",
        "maximum_delta_key": "moveit_install_start_transition_max_joint_delta_deg",
        "maximum_step_key": "moveit_slot_transfer_servoj_max_step_deg",
        "success_message": (
            "START TRANSITION PASSED: 真机已沿MoveIt避碰轨迹到达8动作起点。"
        ),
    },
    "custom_flange_pose": {
        "artifact": (
            PROJECT_ROOT / "logs/real_cr5/custom_target_candidate_current.json"
        ),
        "expected_target_stage": "custom_flange_pose",
        "authorization_scope": "moveit_servoj_custom_flange_pose_only",
        "predecessor_receipt": None,
        "predecessor_status": None,
        "receipt": (
            PROJECT_ROOT
            / "logs/commissioning/cr5_moveit_custom_flange_pose.json"
        ),
        "completion_status": "custom_flange_pose_complete",
        "confirmation": "MOVE_REAL_CR5_TO_CUSTOM_FLANGE_POSE",
        "maximum_delta_key": "moveit_custom_pose_max_joint_delta_deg",
        "maximum_step_key": "moveit_slot_transfer_servoj_max_step_deg",
        "success_message": (
            "CUSTOM TARGET PASSED: 真机已沿审核后的MoveIt轨迹到达目标位。"
        ),
    },
    "slot_pick_approach": {
        "artifact": (
            PROJECT_ROOT / "logs/vision/first_meter_pregrasp_candidate.json"
        ),
        "expected_target_stage": "slot_pick_approach",
        "authorization_scope": "moveit_servoj_slot_pick_approach_only",
        "predecessor_receipt": None,
        "predecessor_status": None,
        "receipt": (
            PROJECT_ROOT
            / "logs/commissioning/cr5_moveit_slot_pick_approach.json"
        ),
        "completion_status": "slot_pick_approach_complete",
        "confirmation": "MOVE_REAL_CR5_TO_FIRST_METER_APPROACH",
        "maximum_delta_key": "moveit_slot_pick_approach_max_joint_delta_deg",
        "maximum_step_key": "moveit_slot_transfer_servoj_max_step_deg",
        "success_message": (
            "METER APPROACH PASSED: 真机已到第一个电表的渐进位；"
            "不会自动下降或夹取。"
        ),
    },
    "lift": {
        "artifact": PROJECT_ROOT / "logs/real_cr5/lift_candidate.json",
        "expected_target_stage": "lift",
        "authorization_scope": "moveit_linear_lift_only",
        "predecessor_receipt": (
            PROJECT_ROOT / "logs/commissioning/cr5_moveit_grasp_descent.json"
        ),
        "predecessor_status": "grasp_descent_complete",
        "receipt": PROJECT_ROOT / "logs/commissioning/cr5_moveit_lift.json",
        "completion_status": "lift_complete",
        "confirmation": "MOVE_REAL_CR5_VERTICAL_LIFT",
        "maximum_delta_key": "moveit_lift_max_joint_delta_deg",
        "maximum_step_key": "moveit_servoj_max_step_deg",
        "success_message": (
            "LIFT PASSED: 真机已沿MoveIt直线轨迹抬升；不会自动转移到电表箱。"
        ),
    },
    "slot_transfer": {
        "artifact": PROJECT_ROOT / "logs/real_cr5/slot_transfer_candidate.json",
        "expected_target_stage": "slot_transfer",
        "authorization_scope": "moveit_servoj_slot_transfer_only",
        "predecessor_receipt": (
            PROJECT_ROOT / "logs/commissioning/cr5_moveit_semantic_flip.json"
        ),
        "predecessor_status": "semantic_flip_complete",
        "receipt": PROJECT_ROOT / "logs/commissioning/cr5_moveit_slot_transfer.json",
        "completion_status": "slot_transfer_complete",
        "confirmation": "MOVE_REAL_CR5_SLOT_TRANSFER",
        "maximum_delta_key": "moveit_slot_transfer_max_joint_delta_deg",
        "maximum_step_key": "moveit_slot_transfer_servoj_max_step_deg",
        "success_message": (
            "SLOT TRANSFER PASSED: 真机已沿MoveIt避碰轨迹到达安装位前方；"
            "不会自动插入电表。"
        ),
    },
    "slot_insert": {
        "artifact": PROJECT_ROOT / "logs/real_cr5/slot_insert_candidate.json",
        "expected_target_stage": "slot_insert",
        "authorization_scope": "moveit_servoj_slot_insert_only",
        "predecessor_receipt": (
            PROJECT_ROOT / "logs/commissioning/cr5_moveit_slot_transfer.json"
        ),
        "predecessor_status": "slot_transfer_complete",
        "receipt": PROJECT_ROOT / "logs/commissioning/cr5_moveit_slot_insert.json",
        "completion_status": "slot_insert_complete",
        "confirmation": "MOVE_REAL_CR5_SLOT_INSERT",
        "maximum_delta_key": "moveit_slot_insert_max_joint_delta_deg",
        "maximum_step_key": "moveit_servoj_max_step_deg",
        "success_message": (
            "SLOT INSERT PASSED: 真机已沿MoveIt直线轨迹到达虚拟安装位；"
            "未发送夹爪松开命令。"
        ),
    },
    "slot_retreat": {
        "artifact": PROJECT_ROOT / "logs/real_cr5/slot_retreat_candidate.json",
        "expected_target_stage": "slot_retreat",
        "authorization_scope": "moveit_servoj_slot_retreat_only",
        "predecessor_receipt": (
            PROJECT_ROOT / "logs/commissioning/cr5_moveit_slot_insert.json"
        ),
        "predecessor_status": "slot_insert_complete",
        "receipt": PROJECT_ROOT / "logs/commissioning/cr5_moveit_slot_retreat.json",
        "completion_status": "slot_retreat_complete",
        "confirmation": "MOVE_REAL_CR5_SLOT_RETREAT",
        "maximum_delta_key": "moveit_slot_retreat_max_joint_delta_deg",
        "maximum_step_key": "moveit_servoj_max_step_deg",
        "success_message": (
            "SLOT RETREAT PASSED: 真机已沿MoveIt直线轨迹退出虚拟安装位；"
            "不会自动返回Home。"
        ),
    },
    "return_home": {
        "artifact": PROJECT_ROOT / "logs/real_cr5/return_home_safe_candidate.json",
        "expected_target_stage": "return_home",
        "authorization_scope": "moveit_servoj_return_home_only",
        "predecessor_receipt": (
            PROJECT_ROOT / "logs/commissioning/cr5_moveit_slot_retreat.json"
        ),
        "predecessor_status": "slot_retreat_complete",
        "receipt": PROJECT_ROOT / "logs/commissioning/cr5_moveit_return_home.json",
        "completion_status": "return_home_complete",
        "confirmation": "MOVE_REAL_CR5_RETURN_HOME",
        "maximum_delta_key": "moveit_return_home_max_joint_delta_deg",
        "maximum_step_key": "moveit_slot_transfer_servoj_max_step_deg",
        "success_message": (
            "RETURN HOME PASSED: 真机已回到最近腕部分支的实机Home。"
        ),
    },
}


def merge_short_terminal_interval(
    points_rad: np.ndarray,
    times_s: np.ndarray,
    *,
    minimum_interval_s: float = 0.02,
    maximum_deviation_deg: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Merge one near-collinear penultimate point when MoveIt emits a tiny tail.

    MoveIt's time parameterization can leave a final remainder shorter than the
    ServoJ command interval.  We only accept that special final-tail case and
    remove the penultimate point after proving that it lies nearly on the
    time-interpolated segment between its neighbors.
    """
    intervals = np.diff(times_s)
    short_indices = np.flatnonzero(intervals < minimum_interval_s)
    if len(short_indices) == 0:
        return points_rad, times_s, None
    if len(short_indices) != 1 or int(short_indices[0]) != len(intervals) - 1:
        raise RuntimeError(f"安全拒绝：存在非末端ServoJ短时间间隔：{intervals}")
    if len(points_rad) < 3:
        raise RuntimeError("安全拒绝：轨迹点不足，无法合并末端短时间间隔")

    combined_interval = float(times_s[-1] - times_s[-3])
    if combined_interval < minimum_interval_s:
        raise RuntimeError("安全拒绝：合并后的末端ServoJ时间间隔仍然过短")
    alpha = float((times_s[-2] - times_s[-3]) / combined_interval)
    interpolated = points_rad[-3] + alpha * (points_rad[-1] - points_rad[-3])
    deviation_deg = float(
        np.max(np.abs(np.rad2deg(points_rad[-2] - interpolated)))
    )
    if deviation_deg > maximum_deviation_deg:
        raise RuntimeError(
            "安全拒绝：末端短时间点不满足近共线合并条件，"
            f"偏差={deviation_deg:.6f}°"
        )
    return (
        np.delete(points_rad, -2, axis=0),
        np.delete(times_s, -2),
        deviation_deg,
    )


def save_receipt(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def archive_completed_receipt(path: Path, prior: dict, digest: str) -> None:
    """Preserve an older completed run before a newly planned trajectory."""
    if prior.get("status") == "command_pending":
        raise RuntimeError("安全拒绝：历史命令状态不明确，禁止开始新轨迹")
    if prior.get("trajectory_sha256") == digest:
        return
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f")
    archived = path.with_name(f"{path.stem}.{stamp}{path.suffix}")
    path.replace(archived)
    print(f"Archived previous stage receipt: {archived}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=sorted(STAGE_PROFILES), required=True)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--trajectory-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--speed-percent", type=int, choices=range(1, 101))
    parser.add_argument(
        "--retry-no-motion",
        action="store_true",
        help="复用一次已确认实际关节仍停在原起点的同SHA失败轨迹",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = STAGE_PROFILES[args.stage]
    config = load_config(args.config.resolve())
    if args.speed_percent is not None:
        config["commissioning_limits"]["moveit_servoj_speed_percent"] = (
            args.speed_percent
        )
        # ServoJ follows the host-provided t schedule, so SpeedFactor alone
        # does not retime a MoveIt trajectory. Keep the controller percentage
        # aligned with DobotStudio and scale the host schedule by the same knob.
        config["commissioning_limits"]["moveit_servoj_time_scale"] = (
            100.0 / args.speed_percent
        )
    authorization = config.get("commissioning_authorization", {})
    if config.get("access_mode") != "commissioning":
        raise RuntimeError("安全拒绝：当前不处于commissioning模式")
    if authorization.get("scope") != profile["authorization_scope"]:
        raise RuntimeError("安全拒绝：当前授权范围不允许此线性阶段")
    if not authorization.get("operator_safety_check_confirmed", False):
        raise RuntimeError("安全拒绝：没有现场安全确认记录")
    if authorization.get("full_workflow_authorized", False):
        raise RuntimeError("配置异常：完整工艺仍不应被授权")

    artifact_path = (
        args.trajectory.resolve()
        if args.trajectory is not None
        else Path(profile["artifact"]).resolve()
    )
    trajectory = json.loads(artifact_path.read_text(encoding="utf-8"))
    digest = verify_digest(trajectory)
    if args.trajectory_sha256 != digest:
        raise RuntimeError("安全拒绝：命令行SHA256与轨迹文件不一致")
    if trajectory.get("target_stage") != profile["expected_target_stage"]:
        raise RuntimeError("安全拒绝：轨迹阶段与执行配置不一致")

    predecessor_receipt = profile["predecessor_receipt"]
    if predecessor_receipt is not None:
        predecessor_path = Path(predecessor_receipt)
        if not predecessor_path.exists():
            raise RuntimeError("安全拒绝：缺少前序实机阶段记录")
        predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
        if predecessor.get("status") != profile["predecessor_status"]:
            raise RuntimeError("安全拒绝：前序实机阶段尚未成功")

    joint_names = list(trajectory["joint_names"])
    points_rad = np.asarray(
        [point["positions_rad"] for point in trajectory["points"]], dtype=float
    )
    times_s = np.asarray(
        [point["time_from_start_s"] for point in trajectory["points"]], dtype=float
    )
    if len(points_rad) < 2 or len(points_rad) != len(times_s):
        raise RuntimeError("安全拒绝：轨迹点或时间数据无效")
    points_rad, times_s, merged_tail_deviation_deg = merge_short_terminal_interval(
        points_rad, times_s
    )
    intervals = np.diff(times_s)
    if np.any(intervals < 0.02) or np.any(intervals > 0.25):
        raise RuntimeError(f"安全拒绝：ServoJ原始时间间隔异常：{intervals}")

    vendor_points_deg = np.asarray(
        [
            ros_to_vendor_deg(point, joint_names, config["joint_mapping"])
            for point in points_rad
        ]
    )
    limits = config["commissioning_limits"]
    j5_lower, j5_upper = map(
        float, limits["moveit_semantic_flip_j5_guard_deg"]
    )
    j5_values = vendor_points_deg[:, 4]
    if np.any(j5_values < j5_lower) or np.any(j5_values > j5_upper):
        raise RuntimeError(
            "安全拒绝：轨迹J5超出真机命令范围"
            f"[{j5_lower:.1f}, {j5_upper:.1f}]°："
            f"[{float(np.min(j5_values)):.3f}, "
            f"{float(np.max(j5_values)):.3f}]°"
        )
    maximum_step = float(np.max(np.abs(np.diff(vendor_points_deg, axis=0))))
    step_limit = float(limits[profile["maximum_step_key"]])
    if maximum_step > step_limit:
        raise RuntimeError(
            f"安全拒绝：相邻轨迹点最大变化{maximum_step:.6f}° > {step_limit:.3f}°"
        )

    dashboard, dashboard_mode, dashboard_deg = read_dashboard_state(config)
    try:
        actual_deg = verify_static_feedback(config, dashboard_mode, dashboard_deg)
        start_error = float(np.max(np.abs(actual_deg - vendor_points_deg[0])))
        start_tolerance = float(limits["moveit_trajectory_start_tolerance_deg"])
        if start_error > start_tolerance:
            raise RuntimeError(
                f"安全拒绝：真机与轨迹起点误差{start_error:.6f}°"
            )

        target_deg = vendor_points_deg[-1]
        total_delta = target_deg - actual_deg
        maximum_delta = float(np.max(np.abs(total_delta)))
        delta_limit = float(limits[profile["maximum_delta_key"]])
        if maximum_delta > delta_limit:
            raise RuntimeError(
                f"安全拒绝：最大关节变化{maximum_delta:.6f}° > {delta_limit:.3f}°"
            )
        duration = float(times_s[-1]) * float(limits["moveit_servoj_time_scale"])

        print(f"CR5 MOVEIT SERVOJ STAGE: {args.stage}")
        print(f"mode={'EXECUTE' if args.execute else 'DRY-RUN (NO MOTION)'}")
        print(f"trajectory_sha256: {digest}")
        print(f"points: {len(points_rad)}")
        print(f"scaled_duration_s: {duration:.3f}")
        print(
            "dobot_global_speed_factor_percent: "
            f"{int(limits['moveit_servoj_speed_percent'])}"
        )
        print(f"servoj_time_scale: {float(limits['moveit_servoj_time_scale']):.3f}")
        print(f"maximum_adjacent_step_deg: {maximum_step:.6f}")
        if merged_tail_deviation_deg is not None:
            print(
                "merged_short_terminal_interval: true "
                f"(deviation={merged_tail_deviation_deg:.6f} deg)"
            )
        print("current_deg: " + ", ".join(f"{v:.6f}" for v in actual_deg))
        print("target_deg : " + ", ".join(f"{v:.6f}" for v in target_deg))
        print("delta_deg  : " + ", ".join(f"{v:+.6f}" for v in total_delta))
        print(f"maximum_delta_deg: {maximum_delta:.6f}")

        if not args.execute:
            print("DRY-RUN PASSED: 未连接30003，未发送运动。")
            return 0
        if args.confirmation != profile["confirmation"]:
            raise RuntimeError(
                f"安全拒绝：必须准确传入{profile['confirmation']}"
            )

        receipt_path = Path(profile["receipt"])
        if receipt_path.exists():
            prior = json.loads(receipt_path.read_text(encoding="utf-8"))
            archive_completed_receipt(receipt_path, prior, digest)
        if receipt_path.exists():
            prior = json.loads(receipt_path.read_text(encoding="utf-8"))
            if prior.get("status") == profile["completion_status"]:
                raise RuntimeError("安全拒绝：此阶段已有成功记录，禁止重复执行")
            if prior.get("status") == "command_pending":
                raise RuntimeError("安全拒绝：历史命令状态不明确，禁止重发")
            if (
                prior.get("trajectory_sha256") == digest
                and prior.get("status") == "motion_interrupted_recovery_requested"
            ):
                raise RuntimeError("安全拒绝：运动中断状态不明确，必须按实时状态重规划")
            if (
                prior.get("trajectory_sha256") == digest
                and prior.get("status") == "terminal_verification_failed"
            ):
                if not args.retry_no_motion:
                    raise RuntimeError(
                        "安全拒绝：同SHA上次终点核验失败；若确认真机零运动，"
                        "请加 --retry-no-motion"
                    )
                prior_origin = np.asarray(prior.get("origin_deg", []), dtype=float)
                prior_actual = np.asarray(prior.get("actual_deg", []), dtype=float)
                if prior_origin.shape != (6,) or prior_actual.shape != (6,):
                    raise RuntimeError("安全拒绝：旧收据缺少可核验的六轴反馈")
                prior_no_motion_error = float(
                    np.max(np.abs(prior_actual - prior_origin))
                )
                if prior_no_motion_error > start_tolerance:
                    raise RuntimeError(
                        "安全拒绝：旧收据不能证明零运动，必须按实时状态重规划"
                    )
                print(
                    "RETRY AUTHORIZED: 同一SHA已由旧收据和当前反馈共同证明零运动；"
                    "无需重新规划。"
                )
                if not QUEUE_RESET_RECEIPT.exists():
                    raise RuntimeError(
                        "安全拒绝：缺少 ResetRobot 清队列凭证"
                    )
                reset = json.loads(QUEUE_RESET_RECEIPT.read_text(encoding="utf-8"))
                reset_time = datetime.fromisoformat(reset["created_at"])
                failure_time = datetime.fromisoformat(
                    prior.get("failed_at", prior["created_at"])
                )
                if (
                    reset.get("status") != "queue_reset_complete"
                    or reset_time <= failure_time
                ):
                    raise RuntimeError("安全拒绝：清队列凭证早于本次失败命令")

        receipt = {
            "created_at": datetime.now().astimezone().isoformat(),
            "status": "command_pending",
            "stage": args.stage,
            "trajectory_sha256": digest,
            "dobot_global_speed_factor_percent": int(
                limits["moveit_servoj_speed_percent"]
            ),
            "servoj_time_scale": float(limits["moveit_servoj_time_scale"]),
            "origin_deg": actual_deg.tolist(),
            "target_deg": target_deg.tolist(),
        }
        save_receipt(receipt_path, receipt)
        try:
            execute_servoj_path(config, dashboard, vendor_points_deg, times_s)
        except Exception as exc:
            # Streaming may have completed a prefix of the path before recovery.
            # Record the ambiguity and require a fresh live-state plan instead
            # of allowing the same artifact to be sent again automatically.
            receipt["status"] = "motion_interrupted_recovery_requested"
            receipt["failed_at"] = datetime.now().astimezone().isoformat()
            receipt["error"] = str(exc)
            save_receipt(receipt_path, receipt)
            raise

        final_deg = np.asarray(read_one_feedback(config)["q_actual"], dtype=float)
        final_error = float(np.max(np.abs(final_deg - target_deg)))
        tolerance = float(limits["moveit_servoj_target_tolerance_deg"])
        print("actual_deg : " + ", ".join(f"{v:.6f}" for v in final_deg))
        print(f"maximum target error: {final_error:.6f}°")
        if final_error > tolerance:
            receipt["status"] = "terminal_verification_failed"
            receipt["failed_at"] = datetime.now().astimezone().isoformat()
            receipt["actual_deg"] = final_deg.tolist()
            receipt["maximum_target_error_deg"] = final_error
            save_receipt(receipt_path, receipt)
            raise RuntimeError("ServoJ阶段终点反馈误差超限；禁止继续")
        receipt["status"] = profile["completion_status"]
        receipt["completed_at"] = datetime.now().astimezone().isoformat()
        receipt["actual_deg"] = final_deg.tolist()
        save_receipt(receipt_path, receipt)
        print(profile["success_message"])
        print(f"receipt={receipt_path}")
        return 0
    finally:
        dashboard.close()


if __name__ == "__main__":
    raise SystemExit(main())
