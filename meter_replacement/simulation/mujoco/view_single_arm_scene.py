#!/usr/bin/env python3
"""Build and view the configurable single-CR5 meter-picking scene.

The CR5 stays in ``cr5_robot.xml``.  Workcell objects are composed at runtime
with MuJoCo's MjSpec API so object poses do not become part of the robot model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "scene.json"
ROBOT_MODEL = Path(__file__).with_name("cr5_robot.xml")


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _vector(config: dict[str, Any], key: str, length: int) -> list[float]:
    value = config[key]
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{key} must contain exactly {length} numbers")
    return [float(item) for item in value]


def _configured_collision_objects(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return enabled scene objects, accepting the original v1 schema too."""
    if "collision_objects" in config:
        objects = config["collision_objects"]
        if not isinstance(objects, list) or not objects:
            raise ValueError("collision_objects must be a non-empty list")
        enabled_objects = [
            item for item in objects if isinstance(item, dict) and item.get("enabled", True)
        ]
    else:
        enabled_objects = [config["floor"], config["meter"]]

    names = [str(item.get("name", "")) for item in enabled_objects]
    if any(not name for name in names):
        raise ValueError("Every collision object must have a non-empty name")
    if len(names) != len(set(names)):
        raise ValueError("Collision object names must be unique")
    return enabled_objects


def _add_meter_cad_visuals(
    spec: mujoco.MjSpec,
    body: Any,
    visual_model: dict[str, Any],
) -> None:
    """Add detailed CAD meshes without changing contact or body inertia."""
    mesh_scale = _vector(visual_model, "mesh_scale", 3)
    mesh_pose = visual_model["mesh_pose_in_meter_frame"]
    mesh_position = _vector(mesh_pose, "position_m", 3)
    mesh_quaternion = _vector(mesh_pose, "quaternion_wxyz", 4)
    meshes = visual_model.get("meshes")
    if not isinstance(meshes, list) or not meshes:
        raise ValueError("meter visual_model.meshes must be a non-empty list")

    for mesh_config in meshes:
        mesh_name = str(mesh_config["name"])
        mesh_path = PROJECT_ROOT / str(mesh_config["project_file"])
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Meter CAD mesh does not exist: {mesh_path}")
        spec.add_mesh(
            name=mesh_name,
            file=str(mesh_path),
            scale=mesh_scale,
        )
        body.add_geom(
            name=mesh_name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name,
            pos=mesh_position,
            quat=mesh_quaternion,
            rgba=_vector(mesh_config, "rgba", 4),
            contype=0,
            conaffinity=0,
            mass=0.0,
            group=2,
        )


def _add_collision_object(
    spec: mujoco.MjSpec,
    item: dict[str, Any],
    meter_visual_model: dict[str, Any] | None = None,
) -> None:
    name = str(item["name"])
    body = spec.worldbody.add_body(
        name=f"{name}_body",
        pos=_vector(item, "position_m", 3),
        quat=_vector(item, "quaternion_wxyz", 4),
    )
    is_dynamic = bool(item.get("dynamic", False))
    if is_dynamic:
        body.add_freejoint(name=f"{name}_free_joint")

    geometry = str(item.get("geometry", "box"))
    geom_parameters: dict[str, Any] = {
        "name": (
            "electric_meter_collision" if name == "electric_meter" else name
        ),
        "rgba": _vector(item, "rgba", 4),
        "friction": [
            float(value)
            for value in item.get("friction", [0.9, 0.01, 0.001])
        ],
    }
    if geometry == "box":
        size = _vector(item, "size_m", 3)
        if any(dimension <= 0 for dimension in size):
            raise ValueError(f"{name}.size_m values must all be positive")
        geom_parameters["type"] = mujoco.mjtGeom.mjGEOM_BOX
        # MuJoCo box sizes are half-extents.
        geom_parameters["size"] = [dimension / 2.0 for dimension in size]
    elif geometry == "cylinder":
        height = float(item["height_m"])
        radius = float(item["radius_m"])
        if height <= 0.0 or radius <= 0.0:
            raise ValueError(f"{name} cylinder height_m/radius_m must be positive")
        geom_parameters["type"] = mujoco.mjtGeom.mjGEOM_CYLINDER
        # MuJoCo cylinder size is [radius, half-height, unused].
        geom_parameters["size"] = [radius, height / 2.0, 0.0]
    else:
        raise ValueError(f"Unsupported collision geometry: {geometry}")
    if len(geom_parameters["friction"]) != 3:
        raise ValueError(f"{name}.friction must contain exactly 3 numbers")
    if "solref" in item:
        geom_parameters["solref"] = _vector(item, "solref", 2)
    if "solimp" in item:
        geom_parameters["solimp"] = _vector(item, "solimp", 5)
    if "condim" in item:
        geom_parameters["condim"] = int(item["condim"])
    if is_dynamic:
        if "mass_kg" not in item:
            raise ValueError(f"{name}.mass_kg is required for dynamic objects")
        geom_parameters["mass"] = float(item["mass_kg"])
    body.add_geom(**geom_parameters)
    if name == "electric_meter" and meter_visual_model is not None:
        _add_meter_cad_visuals(spec, body, meter_visual_model)


def _compose_pose(
    parent_position: list[float],
    parent_quaternion: list[float],
    child_position: list[float],
    child_quaternion: list[float],
) -> tuple[list[float], list[float]]:
    """Compose two poses whose quaternions use MuJoCo's wxyz convention."""
    w, x, y, z = parent_quaternion
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )
    position = np.asarray(parent_position) + rotation @ np.asarray(child_position)

    child_w, child_x, child_y, child_z = child_quaternion
    quaternion = [
        w * child_w - x * child_x - y * child_y - z * child_z,
        w * child_x + x * child_w + y * child_z - z * child_y,
        w * child_y - x * child_z + y * child_w + z * child_x,
        w * child_z + x * child_y - y * child_x + z * child_w,
    ]
    return position.tolist(), quaternion


def _add_slot_visuals(spec: mujoco.MjSpec, config: dict[str, Any]) -> None:
    """Add translucent slot envelopes that cannot generate contacts."""
    meter_box = config["frames"]["meter_box"]
    if str(meter_box["parent_frame"]) != "world":
        raise ValueError("MuJoCo scene currently requires meter_box parent_frame=world")
    parent_position = _vector(meter_box, "position_m", 3)
    parent_quaternion = _vector(meter_box, "quaternion_wxyz", 4)
    size = _vector(config["meter_template"], "installed_envelope_size_m", 3)

    for slot in config["meter_slots"]:
        position, quaternion = _compose_pose(
            parent_position,
            parent_quaternion,
            _vector(slot, "position_m", 3),
            _vector(slot, "quaternion_wxyz", 4),
        )
        slot_id = str(slot["id"])
        body = spec.worldbody.add_body(
            name=f"{slot_id}_visual_body",
            pos=position,
            quat=quaternion,
        )
        body.add_geom(
            name=f"{slot_id}_visual",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[dimension / 2.0 for dimension in size],
            rgba=_vector(slot, "marker_rgba", 4),
            contype=0,
            conaffinity=0,
        )


def _find_body(spec: mujoco.MjSpec, name: str) -> Any:
    """Return a named body from an editable MuJoCo specification."""
    for body in spec.bodies:
        if body.name == name:
            return body
    raise ValueError(f"MuJoCo body does not exist: {name}")


def _add_epg40_100_static_visual(
    spec: mujoco.MjSpec,
    simulation: dict[str, Any],
) -> None:
    """Display the purchased EPG40-100 in the normal workcell viewer.

    This viewer representation is fixed at the CAD assembly pose.  The
    dedicated grasp-contact experiment adds the two prismatic jaw joints and
    actuators separately.
    """
    gripper = simulation["epg40_100_gripper"]
    link6 = _find_body(spec, "Link6")
    base = link6.add_body(
        name="epg40_100_static_visual_body",
        pos=_vector(gripper, "mount_position_m", 3),
    )
    mesh_scale = _vector(gripper, "mesh_scale", 3)
    mesh_definitions = (
        ("body", "body_visual_mesh", [0.34, 0.36, 0.40, 1.0]),
        ("left_jaw", "left_jaw_visual_mesh", [0.20, 0.48, 0.78, 1.0]),
        ("right_jaw", "right_jaw_visual_mesh", [0.20, 0.48, 0.78, 1.0]),
    )
    for part_name, path_key, rgba in mesh_definitions:
        mesh_name = f"epg40_100_static_{part_name}_mesh"
        mesh_path = PROJECT_ROOT / str(gripper[path_key])
        if not mesh_path.is_file():
            raise FileNotFoundError(f"EPG40-100 CAD mesh does not exist: {mesh_path}")
        spec.add_mesh(name=mesh_name, file=str(mesh_path), scale=mesh_scale)
        base.add_geom(
            name=f"epg40_100_static_{part_name}_visual",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name,
            rgba=rgba,
            contype=0,
            conaffinity=0,
            mass=0.0,
            group=2,
        )


def build_scene_spec(
    config_path: Path = DEFAULT_CONFIG,
    *,
    robot_collisions_enabled: bool = True,
    include_end_effector_visual: bool = True,
) -> mujoco.MjSpec:
    """Compose an editable CR5 workcell specification.

    The separate specification builder lets focused dynamics experiments add
    an end effector or sensors before MuJoCo compiles the final model.
    """
    config = load_config(config_path)
    simulation = config["simulation"]
    objects = _configured_collision_objects(config)

    spec = mujoco.MjSpec.from_file(str(ROBOT_MODEL))
    if not robot_collisions_enabled:
        # The imported CR5 meshes are adequate for display but have not been
        # simplified or validated as stable dynamics collision geometry.  A
        # gripper-only fixture test therefore makes them visual-only.
        for geom in spec.geoms:
            geom.contype = 0
            geom.conaffinity = 0
    spec.option.gravity = _vector(simulation, "gravity_m_s2", 3)
    spec.option.timestep = float(simulation["timestep_s"])

    meter_visual_model = config["meter_template"].get("visual_model")
    for item in objects:
        _add_collision_object(spec, item, meter_visual_model)
    _add_slot_visuals(spec, config)
    if include_end_effector_visual:
        _add_epg40_100_static_visual(spec, simulation)

    spec.worldbody.add_camera(
        name="workcell_camera",
        pos=[1.35, 1.10, 1.25],
        # Camera looks along local -Z toward the CR5/table work area.
        xyaxes=[
            -0.83646113,
            0.54802626,
            0.0,
            -0.26337556,
            -0.40199428,
            0.87694578,
        ],
        fovy=45.0,
    )
    return spec


def build_scene(config_path: Path = DEFAULT_CONFIG) -> mujoco.MjModel:
    """Compose the CR5 and configured workcell boxes, then compile the model."""
    spec = build_scene_spec(config_path)
    return spec.compile()


def describe_scene(model: mujoco.MjModel) -> None:
    meter_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "electric_meter_collision"
    )
    full_size = 2.0 * model.geom_size[meter_geom_id]
    slot_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if (
            (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id))
            and name.startswith("slot_")
            and name.endswith("_visual")
        )
    ]
    if not slot_geom_ids or any(
        model.geom_contype[geom_id] != 0
        or model.geom_conaffinity[geom_id] != 0
        for geom_id in slot_geom_ids
    ):
        raise RuntimeError("Meter slot visuals must exist and remain non-collidable")
    meter_visual_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in (
            "electric_meter_body_visual",
            "electric_meter_cover_visual",
        )
    ]
    if any(geom_id < 0 for geom_id in meter_visual_ids) or any(
        model.geom_contype[geom_id] != 0
        or model.geom_conaffinity[geom_id] != 0
        for geom_id in meter_visual_ids
    ):
        raise RuntimeError("Meter CAD visuals must exist and remain non-collidable")
    gripper_visual_ids = [
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"epg40_100_static_{part_name}_visual",
        )
        for part_name in ("body", "left_jaw", "right_jaw")
    ]
    if any(geom_id < 0 for geom_id in gripper_visual_ids) or any(
        model.geom_contype[geom_id] != 0
        or model.geom_conaffinity[geom_id] != 0
        for geom_id in gripper_visual_ids
    ):
        raise RuntimeError("EPG40-100 CAD visuals must exist and remain non-collidable")
    print(f"model: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print(f"configured workcell geoms: {model.ngeom}")
    print(f"non-collision meter-slot visuals: {len(slot_geom_ids)}")
    print(f"non-collision meter CAD visuals: {len(meter_visual_ids)}")
    print(f"non-collision EPG40-100 CAD visuals: {len(gripper_visual_ids)}")
    print(f"meter size (m): {np.round(full_size, 4).tolist()}")
    print("scene check: OK")


def run_viewer(model: mujoco.MjModel, duration_s: float | None) -> None:
    import mujoco.viewer

    data = mujoco.MjData(model)
    deadline = None if duration_s is None else time.monotonic() + duration_s
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "workcell_camera"
        )
        while viewer.is_running() and (deadline is None or time.monotonic() < deadline):
            step_started = time.monotonic()
            mujoco.mj_step(model, data)
            viewer.sync()
            remaining = model.opt.timestep - (time.monotonic() - step_started)
            if remaining > 0:
                time.sleep(remaining)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compile and validate the scene without opening a window",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="close the viewer automatically after this many seconds",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = build_scene(args.config)
    describe_scene(model)
    if not args.check:
        run_viewer(model, args.duration)


if __name__ == "__main__":
    main()
