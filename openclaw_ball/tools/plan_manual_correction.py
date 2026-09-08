#!/usr/bin/env python3
"""Review the operator-requested 70 mm forward and 20 mm down correction."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.plan_front_grasp import METER, ROOT, ros_command


def main():
    run = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "logs/grasp/20260907_150141").resolve()
    target = json.loads((run / "target.json").read_text(encoding="utf-8"))
    launch = subprocess.Popen(
        ros_command(["ros2", "launch", "meter_grasp", "vision_guided_scene.launch.py",
                     f"project_root:={METER}", "start_vision:=false", "execute_motion:=false"]),
        cwd=METER, start_new_session=True,
    )
    try:
        time.sleep(6)
        subprocess.run(ros_command([
            "/usr/bin/python3", str(ROOT / "robot/front_grasp_planner.py"), "--ros-args",
            "-p", "home_profile:=vision_safe_transition", "-p", "execute:=false",
            "-p", "manual_correction:=true",
            "-p", f"planning_start_trajectory_path:={run / 'horizontal_contact.json'}",
            "-p", f"cube_center_base_mm:={list(target['cube_center_base_mm'])}",
            "-p", f"approach_direction_base:={list(target['approach_direction_base'])}",
            "-p", f"forward_output_path:={run / 'manual_forward_70mm.json'}",
            "-p", f"down_output_path:={run / 'manual_down_20mm.json'}",
        ]), cwd=ROOT, check=True)
        print(f"MANUAL CORRECTION PLAN-ONLY PASSED: {run}")
        input("请在RViz检查前进70 mm和下降20 mm轨迹；按Enter关闭场景...")
    finally:
        os.killpg(launch.pid, signal.SIGTERM)
        launch.wait(timeout=15)


if __name__ == "__main__":
    main()
