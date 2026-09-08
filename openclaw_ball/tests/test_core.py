import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task_types import TaskRequest
from executor import median_depth
from vision.calibration import CameraToRobot, deproject
from vision.color_detector import ColorBallDetector
from robot.gripper import crc16, parse_status
from robot.grasp_geometry import front_grasp_points, nearest_equivalent, offset_along
from config import CALIBRATION_FILE, DEFAULT_SAFE_POSE_FILE
from executor import BallTaskExecutor


class CoreTest(unittest.TestCase):
    def test_detects_red_circle_not_blue_circle(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.circle(image, (80, 120), 30, (0, 0, 255), -1)
        cv2.circle(image, (240, 120), 30, (255, 0, 0), -1)
        found = ColorBallDetector().detect(image, "red")
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0].pixel_x, 80, delta=1)

    def test_deproject_and_transform_units(self):
        intrinsics = SimpleNamespace(fx=100, fy=100, ppx=50, ppy=50)
        camera = deproject(60, 70, 1.0, intrinsics)
        self.assertEqual(camera, (0.1, 0.2, 1.0))
        transform = CameraToRobot(np.eye(3), [1, 2, 3])
        self.assertEqual(transform.transform_m_to_mm(camera), (101.0, 202.0, 1003.0))

    def test_hardware_modes_are_validated_but_not_silently_downgraded(self):
        self.assertEqual(TaskRequest("green", mode="execute").mode, "execute")
        with self.assertRaises(ValueError):
            TaskRequest("purple")

    def test_depth_uses_neighborhood_median_and_ignores_zeros(self):
        class Depth:
            def get_width(self): return 3
            def get_height(self): return 3
            def get_distance(self, x, y): return [0, 0.9, 1.0, 1.1][(x + y) % 4]

        self.assertEqual(median_depth(Depth(), 1, 1, radius=1), 1.0)

    def test_gripper_crc_and_status_parser(self):
        body = bytes.fromhex("09 04 08 00 f5 00 00 00 00 2c 18")
        status = parse_status(body + crc16(body))
        self.assertTrue(status.enabled)
        self.assertEqual((status.position, status.fault, status.bus_voltage_v), (0, 0, 24))

    def test_front_grasp_freezes_surface_center_and_pregrasp(self):
        center, pregrasp, direction = front_grasp_points(
            [100, 200, 300], np.eye(3), cube_depth_mm=25, standoff_mm=150
        )
        self.assertEqual(direction, (0.0, 0.0, 1.0))
        self.assertEqual(center, (100.0, 200.0, 312.5))
        self.assertEqual(pregrasp, (100.0, 200.0, 162.5))
        self.assertEqual(offset_along(center, direction, 15), (100.0, 200.0, 327.5))

    def test_operator_translation_correction_is_traceable(self):
        import json
        data = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
        original = np.asarray(data["original_translation_mm"])
        correction = np.asarray(data["operator_correction_mm"])
        self.assertTrue(np.allclose(original + correction, data["translation_mm"]))
        predicted = np.asarray(data["reference"]["original_predicted_cube_center_base_mm"])
        confirmed = np.asarray(data["reference"]["operator_confirmed_gripper_tcp_base_mm"])
        self.assertTrue(np.allclose(predicted + correction, confirmed))

    def test_default_safe_pose_matches_verified_receipt(self):
        import json
        pose = json.loads(DEFAULT_SAFE_POSE_FILE.read_text(encoding="utf-8"))
        receipt = json.loads(Path(pose["source_receipt"]).read_text(encoding="utf-8"))
        self.assertEqual(pose["validation_status"], receipt["status"])
        self.assertTrue(np.allclose(pose["joint_positions_deg"], receipt["actual_deg"]))
        self.assertTrue(np.allclose(pose["joint_positions_rad"], receipt["actual_rad"]))

    def test_execute_is_locked_without_explicit_server_motion_flag(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        with self.assertRaises(PermissionError):
            BallTaskExecutor().run(TaskRequest("red", mode="execute"), image)

    def test_camera_uses_bounded_frame_waits(self):
        source = (Path(__file__).parents[1] / "vision/camera.py").read_text()
        self.assertIn("wait_for_frames(timeout_ms=5000)", source)

    def test_equivalent_wrist_goal_can_be_near_positive_current_angle(self):
        result = nearest_equivalent(-np.pi, 3.40, -6.28, 6.28, 0.174533)
        self.assertAlmostEqual(result, np.pi)


if __name__ == "__main__":
    unittest.main()
