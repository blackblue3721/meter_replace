#!/usr/bin/env python3
"""Register OpenClaw cube stages with the existing guarded CR5 executor."""
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METER_EXECUTOR = ROOT.parent / "meter_replacement/robot/real/cr5_tcp"
sys.path.insert(0, str(METER_EXECUTOR))

import execute_moveit_linear_stage as guarded


def _runtime_run_dir():
    if "--run-dir" not in sys.argv:
        return None
    index = sys.argv.index("--run-dir")
    run = Path(sys.argv[index + 1]).resolve()
    del sys.argv[index:index + 2]
    return run


RUN_DIR = _runtime_run_dir()


guarded.STAGE_PROFILES["cube_front_pregrasp"] = {
    "artifact": ROOT / "logs/grasp/unused.json",
    "expected_target_stage": "vision_meter_approach",
    "authorization_scope": "moveit_servoj_vision_two_stage_approach_only",
    "predecessor_receipt": (
        ROOT.parent
        / "meter_replacement/logs/commissioning/cr5_vision_safe_transition.json"
    ),
    "predecessor_status": "vision_safe_transition_complete",
    "receipt": ROOT / "logs/grasp/cr5_cube_front_pregrasp.json",
    "completion_status": "cube_front_pregrasp_complete",
    "confirmation": "MOVE_REAL_CR5_TO_CUBE_FRONT_PREGRASP",
    "maximum_delta_key": "moveit_vision_approach_max_joint_delta_deg",
    "maximum_step_key": "moveit_slot_transfer_servoj_max_step_deg",
    "success_message": "CUBE PREGRASP PASSED: 真机已到方块前方150 mm；未接触或夹取。",
}

guarded.STAGE_PROFILES["cube_horizontal_wrist"] = {
    "artifact": ROOT / "logs/grasp/unused.json",
    "expected_target_stage": "pregrasp",
    "authorization_scope": "moveit_servoj_vision_two_stage_approach_only",
    "predecessor_receipt": ROOT / "logs/grasp/cr5_cube_front_pregrasp.json",
    "predecessor_status": "cube_front_pregrasp_complete",
    "receipt": ROOT / "logs/grasp/cr5_cube_horizontal_wrist.json",
    "completion_status": "cube_horizontal_wrist_complete",
    "confirmation": "ROTATE_REAL_CR5_J6_FOR_HORIZONTAL_GRASP",
    "maximum_delta_key": "moveit_vision_approach_max_joint_delta_deg",
    "maximum_step_key": "moveit_slot_transfer_servoj_max_step_deg",
    "success_message": "CUBE WRIST PASSED: J6已旋转90度，夹爪处于横向姿态。",
}

guarded.STAGE_PROFILES["cube_horizontal_contact"] = {
    "artifact": ROOT / "logs/grasp/unused.json",
    "expected_target_stage": "pregrasp",
    "authorization_scope": "moveit_servoj_vision_two_stage_approach_only",
    "predecessor_receipt": ROOT / "logs/grasp/cr5_cube_horizontal_wrist.json",
    "predecessor_status": "cube_horizontal_wrist_complete",
    "receipt": ROOT / "logs/grasp/cr5_cube_horizontal_contact.json",
    "completion_status": "cube_horizontal_contact_complete",
    "confirmation": "MOVE_REAL_CR5_HORIZONTAL_TO_CUBE",
    "maximum_delta_key": "moveit_slot_insert_max_joint_delta_deg",
    "maximum_step_key": "moveit_slot_transfer_servoj_max_step_deg",
    "success_message": "CUBE CONTACT PASSED: 真机已沿审核直线轨迹到达方块中心。",
}

guarded.STAGE_PROFILES["cube_manual_forward"] = {
    "artifact": ROOT / "logs/grasp/unused.json",
    "expected_target_stage": "pregrasp",
    "authorization_scope": "moveit_servoj_vision_two_stage_approach_only",
    "predecessor_receipt": ROOT / "logs/grasp/cr5_cube_horizontal_contact.json",
    "predecessor_status": "cube_horizontal_contact_complete",
    "receipt": ROOT / "logs/grasp/cr5_cube_manual_forward.json",
    "completion_status": "cube_manual_forward_complete",
    "confirmation": "MOVE_REAL_CR5_CUBE_FORWARD_70MM",
    "maximum_delta_key": "moveit_slot_insert_max_joint_delta_deg",
    "maximum_step_key": "moveit_slot_transfer_servoj_max_step_deg",
    "success_message": "CUBE FORWARD PASSED: 真机已沿审核方向继续前进70 mm。",
}

guarded.STAGE_PROFILES["cube_manual_down"] = {
    "artifact": ROOT / "logs/grasp/unused.json",
    "expected_target_stage": "pregrasp",
    "authorization_scope": "moveit_servoj_vision_two_stage_approach_only",
    "predecessor_receipt": ROOT / "logs/grasp/cr5_cube_manual_forward.json",
    "predecessor_status": "cube_manual_forward_complete",
    "receipt": ROOT / "logs/grasp/cr5_cube_manual_down.json",
    "completion_status": "cube_manual_down_complete",
    "confirmation": "MOVE_REAL_CR5_CUBE_DOWN_20MM",
    "maximum_delta_key": "moveit_slot_insert_max_joint_delta_deg",
    "maximum_step_key": "moveit_slot_transfer_servoj_max_step_deg",
    "success_message": "CUBE DOWN PASSED: 真机已沿基座-Z下降20 mm。",
}

guarded.STAGE_PROFILES["cube_manual_up"] = {
    "artifact": ROOT / "logs/grasp/unused.json",
    "expected_target_stage": "pregrasp",
    "authorization_scope": "moveit_servoj_vision_two_stage_approach_only",
    "predecessor_receipt": ROOT / "logs/grasp/cr5_cube_manual_down.json",
    "predecessor_status": "cube_manual_down_complete",
    "receipt": ROOT / "logs/grasp/cr5_cube_manual_up.json",
    "completion_status": "cube_manual_up_complete",
    "confirmation": "MOVE_REAL_CR5_CUBE_UP_20MM",
    "maximum_delta_key": "moveit_slot_retreat_max_joint_delta_deg",
    "maximum_step_key": "moveit_slot_transfer_servoj_max_step_deg",
    "success_message": "CUBE UP PASSED: 真机已沿基座+Z上升20 mm。",
}

guarded.STAGE_PROFILES["cube_manual_back"] = {
    "artifact": ROOT / "logs/grasp/unused.json",
    "expected_target_stage": "pregrasp",
    "authorization_scope": "moveit_servoj_vision_two_stage_approach_only",
    "predecessor_receipt": ROOT / "logs/grasp/cr5_cube_manual_up.json",
    "predecessor_status": "cube_manual_up_complete",
    "receipt": ROOT / "logs/grasp/cr5_cube_manual_back.json",
    "completion_status": "cube_manual_back_complete",
    "confirmation": "MOVE_REAL_CR5_CUBE_BACK_70MM",
    "maximum_delta_key": "moveit_slot_retreat_max_joint_delta_deg",
    "maximum_step_key": "moveit_slot_transfer_servoj_max_step_deg",
    "success_message": "CUBE BACK PASSED: 真机已沿审核方向后退70 mm。",
}

guarded.STAGE_PROFILES["cube_visual_safe_pregrasp"] = {
    "artifact": ROOT / "logs/grasp/unused.json",
    "expected_target_stage": "pregrasp",
    "authorization_scope": "moveit_servoj_vision_two_stage_approach_only",
    "predecessor_receipt": ROOT / "logs/grasp/cr5_cube_manual_back.json",
    "predecessor_status": "cube_manual_back_complete",
    "receipt": ROOT / "logs/grasp/cr5_cube_visual_safe_pregrasp.json",
    "completion_status": "cube_visual_safe_pregrasp_complete",
    "confirmation": "MOVE_REAL_CR5_TO_VISUAL_SAFE_PREGRASP",
    "maximum_delta_key": "moveit_slot_retreat_max_joint_delta_deg",
    "maximum_step_key": "moveit_slot_transfer_servoj_max_step_deg",
    "success_message": "VISUAL SAFE PREGRASP PASSED: 真机已回视觉预抓取安全位。",
}

if RUN_DIR is not None:
    chain = (
        ("cube_cycle_pregrasp", "pregrasp.json", "cube_cycle_pregrasp", None),
        ("cube_cycle_contact", "contact.json", "cube_cycle_contact", "cube_cycle_pregrasp"),
        ("cube_cycle_retreat", "retreat.json", "cube_cycle_retreat", "cube_cycle_contact"),
        ("cube_cycle_safe_return", "safe_return.json", "cube_cycle_safe_return", "cube_cycle_retreat"),
    )
    for stage, artifact, target_name, predecessor in chain:
        guarded.STAGE_PROFILES[stage] = {
            "artifact": RUN_DIR / artifact,
            "expected_target_stage": "pregrasp",
            "authorization_scope": "moveit_servoj_vision_two_stage_approach_only",
            "predecessor_receipt": RUN_DIR / f"{predecessor}.receipt.json" if predecessor else None,
            "predecessor_status": f"{predecessor}_complete" if predecessor else None,
            "receipt": RUN_DIR / f"{stage}.receipt.json",
            "completion_status": f"{stage}_complete",
            "confirmation": f"EXECUTE_{stage.upper()}",
            "maximum_delta_key": (
                "moveit_vision_approach_max_joint_delta_deg"
                if stage in {"cube_cycle_pregrasp", "cube_cycle_safe_return"}
                else "moveit_slot_retreat_max_joint_delta_deg"
            ),
            "maximum_step_key": "moveit_slot_transfer_servoj_max_step_deg",
            "success_message": f"{stage.upper()} PASSED",
        }
    guarded.STAGE_PROFILES["cube_cycle_operator_correction"] = {
        "artifact": RUN_DIR / "operator_up10_back10.json",
        "expected_target_stage": "pregrasp",
        "authorization_scope": "moveit_servoj_vision_two_stage_approach_only",
        "predecessor_receipt": RUN_DIR / "cube_cycle_contact.receipt.json",
        "predecessor_status": "terminal_verification_failed",
        "receipt": RUN_DIR / "cube_cycle_operator_correction.receipt.json",
        "completion_status": "cube_cycle_operator_correction_complete",
        "confirmation": "EXECUTE_CUBE_CYCLE_OPERATOR_CORRECTION",
        "maximum_delta_key": "moveit_slot_retreat_max_joint_delta_deg",
        "maximum_step_key": "moveit_slot_transfer_servoj_max_step_deg",
        "success_message": "CUBE OPERATOR CORRECTION PASSED",
    }
    guarded.STAGE_PROFILES["cube_direct_safe_return"] = {
        "artifact": RUN_DIR / "direct_safe_return.json",
        "expected_target_stage": "pregrasp",
        "authorization_scope": "moveit_servoj_vision_two_stage_approach_only",
        "predecessor_receipt": None,
        "predecessor_status": None,
        "receipt": RUN_DIR / "cube_direct_safe_return.receipt.json",
        "completion_status": "cube_direct_safe_return_complete",
        "confirmation": "EXECUTE_CUBE_DIRECT_SAFE_RETURN",
        "maximum_delta_key": "moveit_vision_approach_max_joint_delta_deg",
        "maximum_step_key": "moveit_slot_transfer_servoj_max_step_deg",
        "success_message": "CUBE DIRECT SAFE RETURN PASSED",
    }


if __name__ == "__main__":
    if "--help" in sys.argv:
        raise SystemExit(guarded.main())
    trajectory = Path(sys.argv[sys.argv.index("--trajectory") + 1])
    stage = sys.argv[sys.argv.index("--stage") + 1]
    expected = {
        "cube_front_pregrasp": "cube_front_pregrasp",
        "cube_horizontal_wrist": "cube_horizontal_wrist_turn",
        "cube_horizontal_contact": "cube_horizontal_contact",
        "cube_manual_forward": "cube_manual_forward_70mm",
        "cube_manual_down": "cube_manual_down_20mm",
        "cube_manual_up": "cube_manual_up_20mm",
        "cube_manual_back": "cube_manual_back_70mm",
        "cube_visual_safe_pregrasp": "cube_visual_safe_pregrasp",
        "cube_cycle_pregrasp": "cube_cycle_pregrasp",
        "cube_cycle_contact": "cube_cycle_contact",
        "cube_cycle_retreat": "cube_cycle_retreat",
        "cube_cycle_safe_return": "cube_cycle_safe_return",
        "cube_cycle_operator_correction": "cube_operator_up10_back10",
        "cube_direct_safe_return": "cube_direct_safe_return",
    }[stage]
    if json.loads(trajectory.read_text(encoding="utf-8")).get("target_name") != expected:
        raise RuntimeError(f"安全拒绝：不是 {expected} 轨迹")
    raise SystemExit(guarded.main())
