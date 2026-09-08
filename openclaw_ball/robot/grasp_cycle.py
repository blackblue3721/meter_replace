#!/usr/bin/env python3
"""One deterministic detect-plan-pinch-release-retreat workflow."""
import json
import os
import shlex
import signal
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

from config import CALIBRATION_FILE, DEFAULT_SAFE_POSE_FILE, ROOT
from robot.grasp_geometry import front_grasp_points, offset_along
from robot.gripper import EPG40100


METER = ROOT.parent / "meter_replacement"
ROS_SETUP = Path("/opt/ros/humble/setup.bash")
OVERLAY_SETUP = METER / "robot/install/setup.bash"
QUEUE_RESUME = METER / "robot/real/cr5_tcp/resume_cr5_tcp_motion.py"
INTER_STAGE_SETTLE_S = 1.0


def ros_command(parts):
    setup = f"source {shlex.quote(str(ROS_SETUP))} && source {shlex.quote(str(OVERLAY_SETUP))}"
    return ["/bin/bash", "-lc", setup + " && " + " ".join(shlex.quote(str(p)) for p in parts)]


class GraspCycle:
    def __init__(self):
        self.scene = None
        self.lock = threading.RLock()

    def start(self):
        if self.scene and self.scene.poll() is None:
            return
        subprocess.run(
            ["/usr/bin/python3", str(METER / "tools/start_vision_guided_grasp.py"), "--check"],
            cwd=METER, check=True,
        )
        self.scene = subprocess.Popen(
            ros_command(["ros2", "launch", "meter_grasp", "vision_guided_scene.launch.py",
                         f"project_root:={METER}", "start_vision:=false", "execute_motion:=false"]),
            cwd=METER, start_new_session=True,
        )
        time.sleep(6)
        if self.scene.poll() is not None:
            raise RuntimeError("MoveIt scene failed to start")

    def close(self):
        if self.scene and self.scene.poll() is None:
            os.killpg(self.scene.pid, signal.SIGTERM)
            self.scene.wait(timeout=15)
        self.scene = None

    def _plan(self, detected, run):
        calibration = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
        center, pregrasp, direction = front_grasp_points(
            detected["robot_xyz_mm"], calibration["rotation_camera_to_robot"]
        )
        contact = offset_along(center, direction, 15.0)
        target = {
            "status": "frozen_execute_target", "detected": detected,
            "cube_center_base_mm": center, "pregrasp_base_mm": pregrasp,
            "contact_base_mm": contact, "contact_overtravel_mm": 15.0,
            "approach_direction_base": direction,
        }
        (run / "target.json").write_text(
            json.dumps(target, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        subprocess.run(ros_command([
            "/usr/bin/python3", str(ROOT / "robot/front_grasp_planner.py"), "--ros-args",
            "-p", "home_profile:=vision_safe_transition", "-p", "target_stage:=pregrasp",
            "-p", "execute:=false", "-p", "complete_cycle:=true",
            "-p", f"cube_center_base_mm:={list(center)}",
            "-p", f"contact_base_mm:={list(contact)}",
            "-p", f"pregrasp_base_mm:={list(pregrasp)}",
            "-p", f"default_safe_pose_path:={DEFAULT_SAFE_POSE_FILE}",
            "-p", f"pregrasp_output_path:={run / 'pregrasp.json'}",
            "-p", f"contact_output_path:={run / 'contact.json'}",
            "-p", f"retreat_output_path:={run / 'retreat.json'}",
            "-p", f"safe_output_path:={run / 'safe_return.json'}",
        ]), cwd=ROOT, check=True)
        return target

    @staticmethod
    def _execute(run, stage, artifact):
        trajectory = json.loads((run / artifact).read_text(encoding="utf-8"))
        subprocess.run([
            "/usr/bin/python3", str(ROOT / "robot/execute_front_grasp_stage.py"),
            "--run-dir", str(run), "--stage", stage,
            "--trajectory", str(run / artifact),
            "--trajectory-sha256", trajectory["sha256"], "--execute",
            "--confirmation", f"EXECUTE_{stage.upper()}",
        ], cwd=ROOT, check=True)
        time.sleep(INTER_STAGE_SETTLE_S)

    @staticmethod
    def _resume_motion_queue():
        subprocess.run([
            "/usr/bin/python3", str(QUEUE_RESUME), "--execute",
            "--confirmation", "RESUME_REAL_CR5_TCP_MOTION_SESSION",
        ], cwd=METER, check=True)

    def run(self, color, detected):
        if not detected.get("robot_xyz_mm") or not detected.get("depth_m"):
            raise RuntimeError(f"{color} cube has no valid depth")
        with self.lock:
            self.start()
            run = ROOT / "logs/grasp" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            run.mkdir(parents=True)
            try:
                target = self._plan(detected, run)
                with EPG40100() as gripper:
                    enabled = gripper.enable().to_dict()
                    opened = gripper.open().to_dict()
                    self._resume_motion_queue()
                    self._execute(run, "cube_cycle_pregrasp", "pregrasp.json")
                    self._execute(run, "cube_cycle_contact", "contact.json")
                    closed = gripper.close_gripper().to_dict()
                    self._execute(run, "cube_cycle_retreat", "retreat.json")
                    self._execute(run, "cube_cycle_safe_return", "safe_return.json")
                    reopened = gripper.open().to_dict()
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"grasp workflow command failed with exit code {exc.returncode}") from exc
            result = {
                "status": "completed", "mode": "execute", "color": color,
                "action": "pinch_and_release", "target": target,
                "gripper": {"enabled": enabled, "opened": opened,
                            "closed": closed, "reopened": reopened},
                "run_dir": str(run), "returned_to_default_safe_pose": True,
            }
            (run / "result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            return result
