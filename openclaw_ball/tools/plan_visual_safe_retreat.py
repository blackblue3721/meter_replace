#!/usr/bin/env python3
"""Plan a straight retreat to the visual pregrasp safe point."""
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
            "-p", "visual_safe_retreat:=true",
            "-p", f"planning_start_trajectory_path:={run / 'manual_back_70mm.json'}",
            "-p", f"pregrasp_base_mm:={list(target['pregrasp_base_mm'])}",
            "-p", f"pregrasp_output_path:={run / 'visual_safe_pregrasp.json'}",
        ]), cwd=ROOT, check=True)
    finally:
        os.killpg(launch.pid, signal.SIGTERM)
        launch.wait(timeout=15)


if __name__ == "__main__":
    main()
