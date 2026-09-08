"""Build the CAD-backed EPG40-100 MuJoCo gripper actuator model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco

from view_single_arm_scene import (
    PROJECT_ROOT,
    _vector,
    build_scene_spec,
    load_config,
)


ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 7))
METER_JOINT_NAME = "electric_meter_free_joint"
METER_GEOM_NAME = "electric_meter_collision"
LEFT_JOINT_NAME = "gripper_finger_joint"
RIGHT_JOINT_NAME = "gripper_right_finger_joint"
LEFT_GEOM_NAME = "gripper_left_finger_collision"
RIGHT_GEOM_NAME = "gripper_right_finger_collision"
GRASP_SITE_NAME = "gripper_grasp_center"
TCP_SITE_NAME = "gripper_tcp"
LEFT_ACTUATOR_NAME = "gripper_left_position_actuator"
RIGHT_ACTUATOR_NAME = "gripper_right_position_actuator"


def _full_size(config: dict[str, Any], key: str) -> list[float]:
    values = _vector(config, key, 3)
    if any(value <= 0.0 for value in values):
        raise ValueError(f"{key} values must all be positive")
    return values


def _half_size(config: dict[str, Any], key: str) -> list[float]:
    return [value / 2.0 for value in _full_size(config, key)]


def _find_body(spec: mujoco.MjSpec, name: str) -> Any:
    for body in spec.bodies:
        if body.name == name:
            return body
    raise ValueError(f"MuJoCo body does not exist: {name}")


def _configure_arm_fixture(
    spec: mujoco.MjSpec,
    test_config: dict[str, Any],
) -> None:
    """Stiffen the arm only for this isolated gripper component test."""
    gain = float(test_config["arm_fixture_position_gain"])
    damping = float(test_config["arm_fixture_joint_damping"])
    if gain <= 0.0 or damping < 0.0:
        raise ValueError("Arm fixture gain must be positive and damping non-negative")

    arm_actuators = {item.name: item for item in spec.actuators}
    arm_joints = {item.name: item for item in spec.joints}
    for joint_name in ARM_JOINT_NAMES:
        actuator_name = f"{joint_name}_motor"
        actuator = arm_actuators.get(actuator_name)
        joint = arm_joints.get(joint_name)
        if actuator is None:
            raise ValueError(f"Missing arm actuator: {actuator_name}")
        if joint is None:
            raise ValueError(f"Missing arm joint: {joint_name}")
        actuator.gainprm[0] = gain
        actuator.biasprm[1] = -gain
        joint.damping[0] = damping


def _add_position_actuator(
    spec: mujoco.MjSpec,
    *,
    name: str,
    joint_name: str,
    joint_range: list[float],
    gain: float,
    force_limit_n: float,
) -> None:
    """Add the general-actuator equivalent of MuJoCo's position shortcut."""
    spec.add_actuator(
        name=name,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        gainprm=[gain] + [0.0] * 9,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        biasprm=[0.0, -gain] + [0.0] * 8,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        target=joint_name,
        ctrllimited=True,
        ctrlrange=joint_range,
        forcelimited=True,
        forcerange=[-force_limit_n, force_limit_n],
    )


def _add_epg40_100_gripper(
    spec: mujoco.MjSpec,
    simulation_config: dict[str, Any],
    *,
    sliding_friction: float,
) -> None:
    """Add supplier CAD visuals and simplified actuator/jaw collisions."""
    gripper = simulation_config["epg40_100_gripper"]
    contact = simulation_config["contact_solver"]
    if sliding_friction <= 0.0:
        raise ValueError("Sliding friction must be positive")

    friction = [
        sliding_friction,
        float(gripper["torsional_friction"]),
        float(gripper["rolling_friction"]),
    ]
    link6 = _find_body(spec, "Link6")
    base = link6.add_body(
        name="gripper_base_link",
        pos=_vector(gripper, "mount_position_m", 3),
    )
    base.add_geom(
        name="gripper_base_collision",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=_vector(gripper, "base_center_position_m", 3),
        size=_half_size(gripper, "base_size_m"),
        mass=float(gripper["base_mass_kg"]),
        rgba=[0.0, 0.0, 0.0, 0.0],
        # Keep the unverified base collision visual-only in this component test.
        contype=0,
        conaffinity=0,
    )
    mesh_scale = _vector(gripper, "mesh_scale", 3)
    body_mesh_name = "epg40_100_body_visual_mesh"
    spec.add_mesh(
        name=body_mesh_name,
        file=str(PROJECT_ROOT / str(gripper["body_visual_mesh"])),
        scale=mesh_scale,
    )
    base.add_geom(
        name="epg40_100_body_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=body_mesh_name,
        rgba=[0.34, 0.36, 0.40, 1.0],
        contype=0,
        conaffinity=0,
        mass=0.0,
        group=2,
    )
    test_config = simulation_config["grasp_contact_test"]
    base.add_site(
        name=GRASP_SITE_NAME,
        pos=_vector(test_config, "grasp_center_from_mount_m", 3),
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.005],
        rgba=[0.1, 1.0, 0.1, 0.7],
    )
    base.add_site(
        name=TCP_SITE_NAME,
        pos=_vector(test_config, "tcp_from_mount_m", 3),
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.004],
        rgba=[1.0, 0.2, 0.1, 0.8],
    )

    half_stroke = float(gripper["stroke_m"]) / 2.0
    jaw_axis = _vector(gripper, "jaw_axis", 3)
    definitions = (
        (
            "left", LEFT_JOINT_NAME, [0.0, half_stroke],
            "left_jaw_center_position_m", "left_jaw_size_m",
            "left_jaw_visual_mesh",
        ),
        (
            "right", RIGHT_JOINT_NAME, [-half_stroke, 0.0],
            "right_jaw_center_position_m", "right_jaw_size_m",
            "right_jaw_visual_mesh",
        ),
    )
    for side, joint_name, joint_range, center_key, size_key, mesh_key in definitions:
        finger = base.add_body(
            name=f"gripper_{side}_finger_link",
        )
        finger.add_joint(
            name=joint_name,
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=jaw_axis,
            limited=True,
            range=joint_range,
            damping=float(gripper["joint_damping_n_s_per_m"]),
        )
        finger.add_geom(
            name=LEFT_GEOM_NAME if side == "left" else RIGHT_GEOM_NAME,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=_vector(gripper, center_key, 3),
            size=_half_size(gripper, size_key),
            mass=float(gripper["finger_mass_kg"]),
            friction=friction,
            solref=_vector(contact, "solref", 2),
            solimp=_vector(contact, "solimp", 5),
            condim=4,
            rgba=[0.0, 0.0, 0.0, 0.0],
            contype=1,
            conaffinity=1,
        )
        visual_mesh_name = f"epg40_100_{side}_jaw_visual_mesh"
        spec.add_mesh(
            name=visual_mesh_name,
            file=str(PROJECT_ROOT / str(gripper[mesh_key])),
            scale=mesh_scale,
        )
        finger.add_geom(
            name=f"epg40_100_{side}_jaw_visual",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=visual_mesh_name,
            rgba=[0.20, 0.48, 0.78, 1.0],
            contype=0,
            conaffinity=0,
            mass=0.0,
            group=2,
        )

    gain = float(gripper["position_gain_n_per_m"])
    force_limit = float(gripper["maximum_total_grip_force_n"]) / 2.0
    _add_position_actuator(
        spec,
        name=LEFT_ACTUATOR_NAME,
        joint_name=LEFT_JOINT_NAME,
        joint_range=[0.0, half_stroke],
        gain=gain,
        force_limit_n=force_limit,
    )
    _add_position_actuator(
        spec,
        name=RIGHT_ACTUATOR_NAME,
        joint_name=RIGHT_JOINT_NAME,
        joint_range=[-half_stroke, 0.0],
        gain=gain,
        force_limit_n=force_limit,
    )

    # MuJoCo equivalent of the Xacro mimic joint; this does not attach the meter.
    spec.add_equality(
        name="gripper_symmetric_coupling",
        type=mujoco.mjtEq.mjEQ_JOINT,
        name1=LEFT_JOINT_NAME,
        name2=RIGHT_JOINT_NAME,
        data=[0.0, -1.0, 0.0, 0.0, 0.0] + [0.0] * 6,
        active=True,
        solref=[0.005, 1.0],
        solimp=[0.95, 0.99, 0.001, 0.5, 2.0],
    )


def build_grasp_contact_model(
    config_path: Path,
    *,
    sliding_friction: float,
) -> tuple[mujoco.MjModel, dict[str, Any]]:
    """Build one independently configurable contact experiment."""
    config = load_config(config_path)
    simulation = config["simulation"]
    spec = build_scene_spec(
        config_path,
        robot_collisions_enabled=False,
        include_end_effector_visual=False,
    )
    _configure_arm_fixture(spec, simulation["grasp_contact_test"])
    _add_epg40_100_gripper(
        spec,
        simulation,
        sliding_friction=sliding_friction,
    )

    solver = simulation["contact_solver"]
    if str(solver["cone"]).lower() != "elliptic":
        raise ValueError("The grasp contact test currently requires cone=elliptic")
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.iterations = int(solver["iterations"])
    spec.option.noslip_iterations = int(solver["noslip_iterations"])
    spec.option.impratio = float(
        solver["friction_to_normal_impedance_ratio"]
    )
    model = spec.compile()

    # Override both contact materials for a controlled friction comparison.
    for geom_name in (LEFT_GEOM_NAME, RIGHT_GEOM_NAME, METER_GEOM_NAME):
        geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
        )
        if geom_id < 0:
            raise ValueError(f"Required contact geom is missing: {geom_name}")
        model.geom_friction[geom_id, 0] = sliding_friction
    return model, config
