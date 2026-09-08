from dataclasses import replace
import threading

from config import CALIBRATION_FILE
from task_types import TaskRequest
from vision.calibration import CameraToRobot, deproject
from vision.color_detector import ColorBallDetector


def median_depth(depth_frame, pixel_x: int, pixel_y: int, radius: int = 3):
    """Ignore noisy/empty D435 pixels by sampling a small neighborhood."""
    values = []
    for y in range(max(0, pixel_y - radius), min(depth_frame.get_height(), pixel_y + radius + 1)):
        for x in range(max(0, pixel_x - radius), min(depth_frame.get_width(), pixel_x + radius + 1)):
            value = float(depth_frame.get_distance(x, y))
            if value > 0:
                values.append(value)
    if not values:
        return None
    values.sort()
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


class BallTaskExecutor:
    def __init__(self, allow_motion=False):
        self.detector = ColorBallDetector()
        self.transform = CameraToRobot.load(CALIBRATION_FILE)
        self.allow_motion = allow_motion
        self.camera = None
        self.grasp_cycle = None
        self.camera_lock = threading.Lock()

    def start(self):
        from vision.camera import RealSenseCamera
        self.camera = RealSenseCamera()
        if self.allow_motion:
            from robot.grasp_cycle import GraspCycle
            self.grasp_cycle = GraspCycle()
            self.grasp_cycle.start()

    def close(self):
        if self.camera:
            self.camera.close()
            self.camera = None
        if self.grasp_cycle:
            self.grasp_cycle.close()
            self.grasp_cycle = None

    def detect(self, request: TaskRequest, image, depth_frame=None, intrinsics=None):
        detections = self.detector.detect(image, request.color)
        if depth_frame is None or intrinsics is None:
            return detections
        enriched = []
        for item in detections:
            depth_m = median_depth(depth_frame, item.pixel_x, item.pixel_y)
            if depth_m is None:
                enriched.append(item)
                continue
            camera_xyz = deproject(item.pixel_x, item.pixel_y, depth_m, intrinsics)
            enriched.append(replace(
                item,
                depth_m=depth_m,
                camera_xyz_m=camera_xyz,
                robot_xyz_mm=self.transform.transform_m_to_mm(camera_xyz),
            ))
        return enriched

    def run(self, request: TaskRequest, image, depth_frame=None, intrinsics=None) -> dict:
        detections = self.detect(request, image, depth_frame, intrinsics)
        if request.mode == "execute":
            if not self.allow_motion:
                raise PermissionError("hardware motion is locked; start server with --allow-motion")
            if len(detections) != 1:
                raise RuntimeError(
                    f"expected exactly one {request.color} cube, found {len(detections)}"
                )
            if self.grasp_cycle is None:
                raise RuntimeError("motion backend is not started")
            return self.grasp_cycle.run(request.color, detections[0].to_dict())
        if request.mode != "detect":
            raise PermissionError("plan mode is not exposed; use detect or execute")
        return {"status": "ok", "mode": request.mode, "detections": [d.to_dict() for d in detections]}

    def run_camera(self, request: TaskRequest) -> dict:
        if request.mode == "execute" and self.grasp_cycle is not None:
            with self.grasp_cycle.lock:
                return self._capture_and_run(request)
        return self._capture_and_run(request)

    def _capture_and_run(self, request: TaskRequest) -> dict:
        with self.camera_lock:
            if self.camera is None:
                from vision.camera import RealSenseCamera
                with RealSenseCamera() as camera:
                    image, depth, intrinsics = camera.capture()
            else:
                image, depth, intrinsics = self.camera.capture()
        return self.run(request, image, depth, intrinsics)
