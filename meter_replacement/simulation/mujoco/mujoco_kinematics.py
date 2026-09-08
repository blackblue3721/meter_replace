"""Small MuJoCo kinematics helpers shared by dynamics experiments.

MoveIt remains the project's path planner.  This module only converts an
already-defined Cartesian test pose into the matching MuJoCo joint state so a
dynamics experiment starts from the same semantic pose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mujoco
import numpy as np


@dataclass(frozen=True)
class IkSolution:
    """One converged joint solution and its residual errors."""

    joint_positions: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    iterations: int


def _object_id(
    model: mujoco.MjModel,
    kind: mujoco.mjtObj,
    name: str,
) -> int:
    object_id = mujoco.mj_name2id(model, kind, name)
    if object_id < 0:
        raise ValueError(f"MuJoCo object does not exist: {name}")
    return object_id


def _quaternion_error_world(
    target_wxyz: np.ndarray,
    current_wxyz: np.ndarray,
) -> np.ndarray:
    """Return the shortest world-frame rotation vector current -> target."""
    current_inverse = current_wxyz.copy()
    current_inverse[1:] *= -1.0
    error_quaternion = np.empty(4)
    mujoco.mju_mulQuat(
        error_quaternion,
        target_wxyz,
        current_inverse,
    )
    if error_quaternion[0] < 0.0:
        error_quaternion *= -1.0

    vector_norm = float(np.linalg.norm(error_quaternion[1:]))
    if vector_norm < 1e-12:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(vector_norm, error_quaternion[0])
    return error_quaternion[1:] * (angle / vector_norm)


def solve_site_pose_ik(
    model: mujoco.MjModel,
    *,
    site_name: str,
    joint_names: Sequence[str],
    seed_positions: Sequence[float],
    target_position_m: Sequence[float],
    target_quaternion_wxyz: Sequence[float],
    position_tolerance_m: float = 1e-6,
    orientation_tolerance_rad: float = 1e-6,
    maximum_iterations: int = 500,
) -> IkSolution:
    """Solve a site pose with damped least-squares inverse kinematics.

    This intentionally stays narrow: it finds a deterministic configuration
    near the supplied MoveIt seed and does not perform collision-aware path
    planning.
    """
    if len(joint_names) != len(seed_positions):
        raise ValueError("joint_names and seed_positions must have equal length")

    data = mujoco.MjData(model)
    site_id = _object_id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    qpos_addresses: list[int] = []
    dof_addresses: list[int] = []
    joint_ranges: list[np.ndarray] = []
    for joint_name in joint_names:
        joint_id = _object_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        dof_addresses.append(int(model.jnt_dofadr[joint_id]))
        joint_ranges.append(model.jnt_range[joint_id].copy())

    target_position = np.asarray(target_position_m, dtype=float)
    target_quaternion = np.asarray(
        target_quaternion_wxyz,
        dtype=float,
    )
    if target_position.shape != (3,) or target_quaternion.shape != (4,):
        raise ValueError("IK target must contain a 3D position and quaternion")
    quaternion_norm = float(np.linalg.norm(target_quaternion))
    if quaternion_norm < 1e-12:
        raise ValueError("IK target quaternion cannot be zero")
    target_quaternion /= quaternion_norm

    joint_positions = np.asarray(seed_positions, dtype=float).copy()
    position_error = float("inf")
    orientation_error = float("inf")
    for iteration in range(1, maximum_iterations + 1):
        data.qpos[qpos_addresses] = joint_positions
        mujoco.mj_forward(model, data)

        current_quaternion = np.empty(4)
        mujoco.mju_mat2Quat(
            current_quaternion,
            data.site_xmat[site_id],
        )
        position_vector = target_position - data.site_xpos[site_id]
        orientation_vector = _quaternion_error_world(
            target_quaternion,
            current_quaternion,
        )
        position_error = float(np.linalg.norm(position_vector))
        orientation_error = float(np.linalg.norm(orientation_vector))
        if (
            position_error <= position_tolerance_m
            and orientation_error <= orientation_tolerance_rad
        ):
            return IkSolution(
                joint_positions=joint_positions.copy(),
                position_error_m=position_error,
                orientation_error_rad=orientation_error,
                iterations=iteration,
            )

        position_jacobian = np.zeros((3, model.nv))
        rotation_jacobian = np.zeros((3, model.nv))
        mujoco.mj_jacSite(
            model,
            data,
            position_jacobian,
            rotation_jacobian,
            site_id,
        )
        jacobian = np.vstack(
            (
                position_jacobian[:, dof_addresses],
                rotation_jacobian[:, dof_addresses],
            )
        )
        error = np.concatenate((position_vector, orientation_vector))
        damping = 2e-3
        joint_step = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T
            + damping * damping * np.eye(6),
            error,
        )
        largest_step = float(np.max(np.abs(joint_step)))
        if largest_step > 0.08:
            joint_step *= 0.08 / largest_step
        joint_positions += 0.6 * joint_step

        for index, limits in enumerate(joint_ranges):
            if model.jnt_limited[
                _object_id(
                    model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    joint_names[index],
                )
            ]:
                joint_positions[index] = np.clip(
                    joint_positions[index],
                    limits[0] + 1e-6,
                    limits[1] - 1e-6,
                )

    raise RuntimeError(
        "MuJoCo IK did not converge: "
        f"position_error={position_error:.6f} m, "
        f"orientation_error={orientation_error:.6f} rad"
    )
