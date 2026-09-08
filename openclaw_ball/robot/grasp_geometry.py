import math
import numpy as np


def nearest_equivalent(angle, current, lower, upper, margin=0.0):
    candidates = [
        float(angle) + 2.0 * math.pi * turn
        for turn in range(-2, 3)
        if lower + margin <= float(angle) + 2.0 * math.pi * turn <= upper - margin
    ]
    if not candidates:
        raise ValueError("no equivalent angle inside joint limits")
    return min(candidates, key=lambda value: abs(value - float(current)))


def front_grasp_points(surface_xyz_mm, camera_to_robot_rotation, cube_depth_mm=25.0,
                       standoff_mm=150.0):
    """Return frozen cube center and pregrasp points in robot base coordinates."""
    surface = np.asarray(surface_xyz_mm, dtype=float)
    direction = np.asarray(camera_to_robot_rotation, dtype=float)[:, 2]
    direction /= np.linalg.norm(direction)
    center = surface + direction * (cube_depth_mm / 2.0)
    pregrasp = center - direction * standoff_mm
    return tuple(map(float, center)), tuple(map(float, pregrasp)), tuple(map(float, direction))


def offset_along(point_xyz_mm, direction, distance_mm):
    return tuple(map(float, np.asarray(point_xyz_mm) + np.asarray(direction) * distance_mm))
