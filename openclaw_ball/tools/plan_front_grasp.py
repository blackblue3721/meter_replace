#!/usr/bin/env python3
"""Freeze one D435 cube target and launch an RViz front-grasp plan-only review."""
import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CALIBRATION_FILE, ROOT
from robot.grasp_geometry import front_grasp_points
from task_types import COLORS


METER = ROOT.parent / "meter_replacement"
ROS_SETUP = Path("/opt/ros/humble/setup.bash")
OVERLAY_SETUP = METER / "robot/install/setup.bash"
CONDA = Path("/home/haoran/miniconda3/bin/conda")


def ros_command(parts):
    setup = f"source {shlex.quote(str(ROS_SETUP))} && source {shlex.quote(str(OVERLAY_SETUP))}"
    return ["/bin/bash", "-lc", setup + " && " + " ".join(shlex.quote(str(p)) for p in parts)]


def capture(color):
    result = subprocess.run(
        [str(CONDA), "run", "--no-capture-output", "-n", "arm", "python", str(ROOT / "cli.py"), color, "--camera"],
        cwd=ROOT, check=True, text=True, capture_output=True,
    )
    payload = json.loads(result.stdout)
    if not payload["detections"]:
        raise RuntimeError(f"{color} cube was not detected")
    return payload["detections"][0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("color", choices=sorted(COLORS))
    args = parser.parse_args()

    subprocess.run(["/usr/bin/python3", str(METER / "tools/start_vision_guided_grasp.py"), "--check"], cwd=METER, check=True)
    detected = capture(args.color)
    calibration = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    center, pregrasp, direction = front_grasp_points(
        detected["robot_xyz_mm"], calibration["rotation_camera_to_robot"]
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = ROOT / "logs/grasp" / stamp
    output.mkdir(parents=True, exist_ok=False)
    snapshot = {
        "status": "frozen_plan_only_target", "color": args.color,
        "detected": detected, "cube_center_base_mm": center,
        "pregrasp_base_mm": pregrasp, "approach_direction_base": direction,
    }
    (output / "target.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"FROZEN TARGET: {output / 'target.json'}")

    launch = subprocess.Popen(
        ros_command(["ros2", "launch", "meter_grasp", "vision_guided_scene.launch.py",
                     f"project_root:={METER}", "start_vision:=false", "execute_motion:=false"]),
        cwd=METER, start_new_session=True,
    )
    try:
        time.sleep(6)
        transition = output / "safe_transition.json"
        subprocess.run(ros_command([
            "ros2", "run", "meter_grasp", "run_pick_sequence", "--ros-args",
            "-p", "home_profile:=vision_safe_transition", "-p", "target_stage:=return_home",
            "-p", "execute:=false", "-p", f"trajectory_output_path:={transition}",
        ]), cwd=METER, check=True)
        subprocess.run(ros_command([
            "/usr/bin/python3", str(ROOT / "robot/front_grasp_planner.py"), "--ros-args",
            "-p", "home_profile:=vision_safe_transition", "-p", "target_stage:=vision_meter_approach",
            "-p", "execute:=false", "-p", f"planning_start_trajectory_path:={transition}",
            "-p", f"vision_target_base_mm:={list(detected['robot_xyz_mm'])}",
            "-p", f"cube_center_base_mm:={list(center)}", "-p", f"pregrasp_base_mm:={list(pregrasp)}",
            "-p", f"pregrasp_output_path:={output / 'pregrasp.json'}",
            "-p", f"contact_output_path:={output / 'contact.json'}",
        ]), cwd=ROOT, check=True)
        print(f"PLAN-ONLY PASSED: {output}")
        input("请在RViz检查红色方块、预抓取和水平推进轨迹；按Enter关闭场景...")
    finally:
        os.killpg(launch.pid, signal.SIGTERM)
        launch.wait(timeout=15)


if __name__ == "__main__":
    main()
