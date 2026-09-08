#!/usr/bin/env python3
"""Validate the EPG40-100 actuator model using MuJoCo contact dynamics.

This is deliberately a component test rather than a replacement for MoveIt:

* MoveIt remains responsible for collision-aware path planning.
* MuJoCo checks actuator/jaw-block motion and contact under gravity.
* The meter starts face-up on the worktable, matching the RViz pickup scene.
* The jaws close from above and the arm then performs a 0.15 m vertical lift.
* Gravity remains enabled for the complete experiment.

The body and moving jaw blocks mirror the supplied EPG40-100 CAD. The custom
meter-contact fingers are absent, so meter-holding results are not valid until
those fingers are modeled and calibrated.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from view_single_arm_scene import (
    DEFAULT_CONFIG,
    _vector,
    load_config,
)
from gripper_dynamics_model import (
    ARM_JOINT_NAMES,
    LEFT_ACTUATOR_NAME,
    LEFT_GEOM_NAME,
    LEFT_JOINT_NAME,
    METER_GEOM_NAME,
    METER_JOINT_NAME,
    RIGHT_ACTUATOR_NAME,
    RIGHT_GEOM_NAME,
    RIGHT_JOINT_NAME,
    TCP_SITE_NAME,
    build_grasp_contact_model,
)
from mujoco_kinematics import solve_site_pose_ik


@dataclass(frozen=True)
class ExperimentResult:
    """Measurements collected after gravity is applied."""

    sliding_friction: float
    relative_displacement_m: float
    world_vertical_displacement_m: float
    left_mean_normal_force_n: float
    right_mean_normal_force_n: float
    left_peak_normal_force_n: float
    right_peak_normal_force_n: float
    left_final_normal_force_n: float
    right_final_normal_force_n: float
    left_actuator_force_n: float
    right_actuator_force_n: float

    @property
    def has_final_two_sided_contact(self) -> bool:
        return (
            self.left_final_normal_force_n > 0.1
            and self.right_final_normal_force_n > 0.1
        )


@dataclass(frozen=True)
class ExperimentHandles:
    """Compiled model indices needed by the experiment loop."""

    meter_qpos_address: int
    tcp_site_id: int
    meter_geom_id: int
    left_geom_id: int
    right_geom_id: int
    left_actuator_id: int
    right_actuator_id: int
    grasp_joint_positions: np.ndarray
    lift_joint_positions: np.ndarray


def _object_id(model: mujoco.MjModel, kind: Any, name: str) -> int:
    object_id = mujoco.mj_name2id(model, kind, name)
    if object_id < 0:
        raise ValueError(f"MuJoCo object does not exist: {name}")
    return object_id


def _relative_meter_position(
    data: mujoco.MjData,
    *,
    site_id: int,
    meter_qpos_address: int,
) -> np.ndarray:
    site_rotation = data.site_xmat[site_id].reshape(3, 3)
    world_offset = (
        data.qpos[meter_qpos_address : meter_qpos_address + 3]
        - data.site_xpos[site_id]
    )
    return site_rotation.T @ world_offset


def initialize_experiment(
    model: mujoco.MjModel,
    config: dict[str, Any],
) -> tuple[mujoco.MjData, ExperimentHandles]:
    """Place the arm over the face-up meter with both jaws open.

    The Cartesian targets mirror the MoveIt pickup and lift poses.  They are
    solved against the MuJoCo model instead of copying joint values blindly,
    because the two model import paths can have small frame differences.
    """
    data = mujoco.MjData(model)
    test_config = config["simulation"]["grasp_contact_test"]
    seed_positions = _vector(
        test_config,
        "ik_seed_joint_positions_rad",
        len(ARM_JOINT_NAMES),
    )
    grasp_position = np.asarray(
        _vector(test_config, "grasp_tcp_position_m", 3)
    )
    grasp_quaternion = _vector(
        test_config,
        "grasp_tcp_quaternion_wxyz",
        4,
    )
    lift_position = grasp_position + np.asarray(
        _vector(test_config, "lift_offset_world_m", 3)
    )
    grasp_solution = solve_site_pose_ik(
        model,
        site_name=TCP_SITE_NAME,
        joint_names=ARM_JOINT_NAMES,
        seed_positions=seed_positions,
        target_position_m=grasp_position,
        target_quaternion_wxyz=grasp_quaternion,
    )
    lift_solution = solve_site_pose_ik(
        model,
        site_name=TCP_SITE_NAME,
        joint_names=ARM_JOINT_NAMES,
        seed_positions=grasp_solution.joint_positions,
        target_position_m=lift_position,
        target_quaternion_wxyz=grasp_quaternion,
    )
    print(
        "ik: tabletop grasp solved; "
        f"position_error={grasp_solution.position_error_m:.2e} m, "
        f"orientation_error={grasp_solution.orientation_error_rad:.2e} rad"
    )
    print(
        "ik: vertical lift solved; "
        f"position_error={lift_solution.position_error_m:.2e} m, "
        f"orientation_error={lift_solution.orientation_error_rad:.2e} rad"
    )

    for actuator_index, (joint_name, position) in enumerate(
        zip(ARM_JOINT_NAMES, grasp_solution.joint_positions)
    ):
        joint_id = _object_id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        data.qpos[int(model.jnt_qposadr[joint_id])] = position
        data.ctrl[actuator_index] = position

    left_joint_id = _object_id(
        model, mujoco.mjtObj.mjOBJ_JOINT, LEFT_JOINT_NAME
    )
    right_joint_id = _object_id(
        model, mujoco.mjtObj.mjOBJ_JOINT, RIGHT_JOINT_NAME
    )
    left_qpos_address = int(model.jnt_qposadr[left_joint_id])
    right_qpos_address = int(model.jnt_qposadr[right_joint_id])
    half_stroke = (
        float(config["simulation"]["epg40_100_gripper"]["stroke_m"])
        / 2.0
    )
    data.qpos[left_qpos_address] = half_stroke
    data.qpos[right_qpos_address] = -half_stroke
    data.ctrl[6] = half_stroke
    data.ctrl[7] = -half_stroke
    mujoco.mj_forward(model, data)

    site_id = _object_id(
        model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE_NAME
    )

    meter_joint_id = _object_id(
        model, mujoco.mjtObj.mjOBJ_JOINT, METER_JOINT_NAME
    )
    meter_qpos_address = int(model.jnt_qposadr[meter_joint_id])
    meter_dof_address = int(model.jnt_dofadr[meter_joint_id])
    # Keep the free object's position and face-up orientation from scene.json.
    # No weld, attachment or pose reset is used to create the grasp.
    data.qvel[meter_dof_address : meter_dof_address + 6] = 0.0
    mujoco.mj_forward(model, data)

    handles = ExperimentHandles(
        meter_qpos_address=meter_qpos_address,
        tcp_site_id=site_id,
        meter_geom_id=_object_id(
            model, mujoco.mjtObj.mjOBJ_GEOM, METER_GEOM_NAME
        ),
        left_geom_id=_object_id(
            model, mujoco.mjtObj.mjOBJ_GEOM, LEFT_GEOM_NAME
        ),
        right_geom_id=_object_id(
            model, mujoco.mjtObj.mjOBJ_GEOM, RIGHT_GEOM_NAME
        ),
        left_actuator_id=_object_id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            LEFT_ACTUATOR_NAME,
        ),
        right_actuator_id=_object_id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            RIGHT_ACTUATOR_NAME,
        ),
        grasp_joint_positions=grasp_solution.joint_positions,
        lift_joint_positions=lift_solution.joint_positions,
    )
    return data, handles


def _finger_normal_forces(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: ExperimentHandles,
) -> tuple[float, float]:
    totals = [0.0, 0.0]
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if handles.meter_geom_id not in pair:
            continue
        side_index = (
            0
            if handles.left_geom_id in pair
            else 1
            if handles.right_geom_id in pair
            else None
        )
        if side_index is None:
            continue
        contact_force = np.zeros(6)
        mujoco.mj_contactForce(
            model,
            data,
            contact_index,
            contact_force,
        )
        totals[side_index] += max(0.0, float(contact_force[0]))
    return totals[0], totals[1]


def run_experiment(
    model: mujoco.MjModel,
    config: dict[str, Any],
    *,
    sliding_friction: float,
    data: mujoco.MjData | None = None,
    handles: ExperimentHandles | None = None,
    sync_callback: Any | None = None,
    real_time: bool = False,
) -> ExperimentResult:
    """Settle the meter, close the jaws, lift, then measure hold or drop."""
    if data is None or handles is None:
        data, handles = initialize_experiment(model, config)
    simulation = config["simulation"]
    test_config = simulation["grasp_contact_test"]
    half_stroke = (
        float(simulation["epg40_100_gripper"]["stroke_m"]) / 2.0
    )
    target = float(test_config["finger_position_target_m"])
    table_settle_duration = float(
        test_config["table_settle_duration_s"]
    )
    close_duration = float(test_config["close_duration_s"])
    closed_settle_duration = float(
        test_config["closed_settle_duration_s"]
    )
    lift_duration = float(test_config["lift_duration_s"])
    loaded_duration = float(test_config["loaded_duration_s"])
    if not 0.0 <= target < half_stroke:
        raise ValueError("finger_position_target_m must be within the jaw stroke")
    durations = (
        table_settle_duration,
        close_duration,
        closed_settle_duration,
        lift_duration,
        loaded_duration,
    )
    if any(duration <= 0.0 for duration in durations):
        raise ValueError("All grasp-contact phase durations must be positive")

    close_start = table_settle_duration
    closed_settle_start = close_start + close_duration
    lift_start = closed_settle_start + closed_settle_duration
    loaded_start = lift_start + lift_duration
    total_duration = loaded_start + loaded_duration
    left_samples: list[float] = []
    right_samples: list[float] = []
    final_forces = (0.0, 0.0)
    peak_forces = [0.0, 0.0]
    baseline_relative_position: np.ndarray | None = None
    baseline_world_position: np.ndarray | None = None
    current_phase = ""
    while data.time < total_duration:
        step_started = time.monotonic()
        if data.time < close_start:
            phase = "TABLE_SETTLING"
            finger_command = half_stroke
            arm_command = handles.grasp_joint_positions
        elif data.time < closed_settle_start:
            phase = "JAW_CLOSING"
            alpha = (data.time - close_start) / close_duration
            finger_command = half_stroke + alpha * (
                target - half_stroke
            )
            arm_command = handles.grasp_joint_positions
        elif data.time < lift_start:
            phase = "GRASP_SETTLING"
            finger_command = target
            arm_command = handles.grasp_joint_positions
        elif data.time < loaded_start:
            phase = "VERTICAL_LIFT"
            if baseline_relative_position is None:
                baseline_relative_position = _relative_meter_position(
                    data,
                    site_id=handles.tcp_site_id,
                    meter_qpos_address=handles.meter_qpos_address,
                )
                baseline_world_position = data.qpos[
                    handles.meter_qpos_address
                    : handles.meter_qpos_address + 3
                ].copy()
            alpha = (data.time - lift_start) / lift_duration
            arm_command = (
                (1.0 - alpha) * handles.grasp_joint_positions
                + alpha * handles.lift_joint_positions
            )
            finger_command = target
        else:
            phase = "LOADED_HOLD"
            finger_command = target
            arm_command = handles.lift_joint_positions

        if phase != current_phase:
            print(f"phase: {phase}")
            current_phase = phase
        data.ctrl[:6] = arm_command
        data.ctrl[6] = finger_command
        data.ctrl[7] = -finger_command

        mujoco.mj_step(model, data)
        if data.time >= loaded_start:
            final_forces = _finger_normal_forces(
                model, data, handles
            )
            left_samples.append(final_forces[0])
            right_samples.append(final_forces[1])
            peak_forces[0] = max(peak_forces[0], final_forces[0])
            peak_forces[1] = max(peak_forces[1], final_forces[1])

        if sync_callback is not None:
            sync_callback()
        if real_time:
            remaining = model.opt.timestep - (
                time.monotonic() - step_started
            )
            if remaining > 0.0:
                time.sleep(remaining)

    mujoco.mj_forward(model, data)
    if (
        baseline_relative_position is None
        or baseline_world_position is None
    ):
        raise RuntimeError("Lift baseline was not recorded")
    final_relative_position = _relative_meter_position(
        data,
        site_id=handles.tcp_site_id,
        meter_qpos_address=handles.meter_qpos_address,
    )
    final_world_position = data.qpos[
        handles.meter_qpos_address : handles.meter_qpos_address + 3
    ]
    return ExperimentResult(
        sliding_friction=sliding_friction,
        relative_displacement_m=float(
            np.linalg.norm(
                final_relative_position
                - baseline_relative_position
            )
        ),
        world_vertical_displacement_m=float(
            final_world_position[2]
            - baseline_world_position[2]
        ),
        left_mean_normal_force_n=float(np.mean(left_samples)),
        right_mean_normal_force_n=float(np.mean(right_samples)),
        left_peak_normal_force_n=peak_forces[0],
        right_peak_normal_force_n=peak_forces[1],
        left_final_normal_force_n=final_forces[0],
        right_final_normal_force_n=final_forces[1],
        left_actuator_force_n=float(
            data.actuator_force[handles.left_actuator_id]
        ),
        right_actuator_force_n=float(
            data.actuator_force[handles.right_actuator_id]
        ),
    )


def print_result(label: str, result: ExperimentResult) -> None:
    print(f"[{label}] sliding_friction={result.sliding_friction:.3f}")
    print(
        f"[{label}] relative_displacement_m="
        f"{result.relative_displacement_m:.5f}"
    )
    print(
        f"[{label}] world_vertical_displacement_m="
        f"{result.world_vertical_displacement_m:.5f}"
    )
    print(
        f"[{label}] mean_normal_force_n="
        f"left:{result.left_mean_normal_force_n:.2f}, "
        f"right:{result.right_mean_normal_force_n:.2f}"
    )
    print(
        f"[{label}] final_normal_force_n="
        f"left:{result.left_final_normal_force_n:.2f}, "
        f"right:{result.right_final_normal_force_n:.2f}"
    )
    print(
        f"[{label}] actuator_force_n="
        f"left:{result.left_actuator_force_n:.2f}, "
        f"right:{result.right_actuator_force_n:.2f}"
    )


def run_automatic_check(config_path: Path) -> None:
    config = load_config(config_path)
    test_config = config["simulation"]["grasp_contact_test"]
    nominal_friction = float(test_config["nominal_sliding_friction"])
    low_friction = float(test_config["low_sliding_friction"])

    nominal_model, nominal_config = build_grasp_contact_model(
        config_path,
        sliding_friction=nominal_friction,
    )
    nominal = run_experiment(
        nominal_model,
        nominal_config,
        sliding_friction=nominal_friction,
    )
    print_result("NOMINAL", nominal)

    low_model, low_config = build_grasp_contact_model(
        config_path,
        sliding_friction=low_friction,
    )
    low = run_experiment(
        low_model,
        low_config,
        sliding_friction=low_friction,
    )
    print_result("LOW_FRICTION", low)

    maximum_slip = float(
        test_config["maximum_held_relative_slip_m"]
    )
    minimum_drop = float(
        test_config["minimum_dropped_relative_displacement_m"]
    )
    minimum_lift = float(
        test_config["minimum_held_vertical_lift_m"]
    )
    nominal_held = (
        nominal.relative_displacement_m <= maximum_slip
        and nominal.world_vertical_displacement_m >= minimum_lift
        and nominal.has_final_two_sided_contact
    )
    low_friction_dropped = (
        low.relative_displacement_m >= minimum_drop
    )
    print(f"nominal_hold_check: {'PASS' if nominal_held else 'FAIL'}")
    print(
        "low_friction_drop_check: "
        f"{'PASS' if low_friction_dropped else 'FAIL'}"
    )
    if not (nominal_held and low_friction_dropped):
        raise RuntimeError("MuJoCo grasp contact regression check failed")
    print("grasp_contact_test: PASS")


def run_viewer(config_path: Path, sliding_friction: float) -> None:
    import mujoco.viewer

    model, config = build_grasp_contact_model(
        config_path,
        sliding_friction=sliding_friction,
    )
    data, handles = initialize_experiment(model, config)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "workcell_camera"
        )
        if camera_id >= 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = camera_id
        result = run_experiment(
            model,
            config,
            sliding_friction=sliding_friction,
            data=data,
            handles=handles,
            sync_callback=viewer.sync,
            real_time=True,
        )
        print_result("VIEWER", result)
        print("viewer: experiment complete; close the window or press Ctrl+C")
        while viewer.is_running():
            step_started = time.monotonic()
            mujoco.mj_step(model, data)
            viewer.sync()
            remaining = model.opt.timestep - (
                time.monotonic() - step_started
            )
            if remaining > 0.0:
                time.sleep(remaining)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--check",
        action="store_true",
        help="run nominal and low-friction cases without opening a window",
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=None,
        help="override sliding friction for the visible single experiment",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check:
        run_automatic_check(args.config)
        return

    config = load_config(args.config)
    configured_friction = float(
        config["simulation"]["grasp_contact_test"][
            "nominal_sliding_friction"
        ]
    )
    friction = (
        configured_friction
        if args.friction is None
        else float(args.friction)
    )
    run_viewer(args.config, friction)


if __name__ == "__main__":
    main()
