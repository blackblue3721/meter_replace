from pathlib import Path


ROOT = Path(__file__).resolve().parent
CALIBRATION_FILE = ROOT / "calibration/camera_to_cr5_corrected.json"
DEFAULT_SAFE_POSE_FILE = ROOT / "config/default_safe_pose.json"

# OpenCV HSV ranges. Red wraps around hue=0, so it has two ranges.
HSV_RANGES = {
    "red": (((0, 100, 80), (10, 255, 255)), ((170, 100, 80), (179, 255, 255))),
    "yellow": (((18, 90, 90), (36, 255, 255)),),
    "green": (((38, 60, 50), (85, 255, 255)),),
    "blue": (((90, 80, 50), (135, 255, 255)),),
}

MIN_AREA_PX = 300
MIN_CIRCULARITY = 0.65
