from dataclasses import asdict, dataclass
from typing import Optional


COLORS = frozenset({"red", "yellow", "green", "blue"})
MODES = frozenset({"detect", "plan", "execute"})


@dataclass(frozen=True)
class BallDetection:
    color: str
    pixel_x: int
    pixel_y: int
    radius_px: float
    area_px: float
    circularity: float
    depth_m: Optional[float] = None
    camera_xyz_m: Optional[tuple[float, float, float]] = None
    robot_xyz_mm: Optional[tuple[float, float, float]] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TaskRequest:
    color: str
    mode: str = "detect"
    destination: str = "box"

    def __post_init__(self) -> None:
        if self.color not in COLORS:
            raise ValueError(f"unsupported color: {self.color}")
        if self.mode not in MODES:
            raise ValueError(f"unsupported mode: {self.mode}")
        if self.destination != "box":
            raise ValueError("only destination 'box' is supported")

