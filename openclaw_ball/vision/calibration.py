import json
from pathlib import Path

import numpy as np


class CameraToRobot:
    def __init__(self, rotation, translation_mm):
        self.rotation = np.asarray(rotation, dtype=float)
        self.translation_mm = np.asarray(translation_mm, dtype=float)
        if self.rotation.shape != (3, 3) or self.translation_mm.shape != (3,):
            raise ValueError("invalid camera-to-robot transform shape")

    @classmethod
    def load(cls, path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["rotation_camera_to_robot"], data["translation_mm"])

    def transform_m_to_mm(self, camera_xyz_m) -> tuple[float, float, float]:
        result = self.rotation @ (np.asarray(camera_xyz_m, dtype=float) * 1000.0)
        result += self.translation_mm
        return tuple(float(value) for value in result)


def deproject(pixel_x: int, pixel_y: int, depth_m: float, intrinsics) -> tuple[float, float, float]:
    """Pinhole deprojection; intrinsics must expose fx, fy, ppx and ppy."""
    if depth_m <= 0:
        raise ValueError("depth must be positive")
    x = (pixel_x - intrinsics.ppx) / intrinsics.fx * depth_m
    y = (pixel_y - intrinsics.ppy) / intrinsics.fy * depth_m
    return float(x), float(y), float(depth_m)

