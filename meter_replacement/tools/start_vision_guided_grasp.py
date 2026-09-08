#!/usr/bin/env python3
"""One-command D435-guided planning or guarded two-stage CR5 approach."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONDA_ROOT = Path("/home/haoran/miniconda3")
ROS_SETUP = Path("/opt/ros/humble/setup.bash")
OVERLAY_SETUP = PROJECT_ROOT / "robot" / "install" / "setup.bash"
LAUNCH_FILE = (
    PROJECT_ROOT
    / "simulation"
    / "ros2"
    / "meter_grasp"
    / "launch"
    / "vision_guided_scene.launch.py"
)
VISION_SCRIPT = (
    PROJECT_ROOT
    / "vision"
    / "D435_Project_new"
    / "realtime_multi_meter_d435.py"
)

# 同时设置Dobot全局SpeedFactor和ServoJ轨迹节拍：100为原速，50为半速。
MOTION_SPEED_PERCENT = 100


def clean_ros_environment() -> dict[str, str]:
    """Keep Conda out of ROS; the launch file starts YOLO with conda run."""
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith("CONDA_") or key in {"_CE_CONDA", "_CE_M", "PYTHONHOME"}:
            env.pop(key, None)
    for key in ("PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "CMAKE_PREFIX_PATH"):
        if key not in env:
            continue
        env[key] = os.pathsep.join(
            entry
            for entry in env[key].split(os.pathsep)
            if entry and not Path(entry).is_relative_to(CONDA_ROOT)
        )
    return env


def running_conflicts() -> list[str]:
    """Return project processes that would duplicate RViz or real feedback."""
    markers = (
        "rviz2",
        "/move_group",
        "publish_cr5_joint_states.py",
        "real_arm_plan_only_scene.launch.py",
        "vision_guided_scene.launch.py",
        "realsense2_camera_node",
        "realtime_multi_meter_d435.py",
    )
    conflicts: list[str] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, UnicodeDecodeError):
            continue
        if command and any(marker in command for marker in markers):
            conflicts.append(f"PID {entry.name}: {command.strip()}")
    return conflicts


def validate_files() -> None:
    missing = [
        path
        for path in (ROS_SETUP, OVERLAY_SETUP, LAUNCH_FILE, VISION_SCRIPT)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("缺少启动文件：" + ", ".join(map(str, missing)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-only",
        action="store_true",
        help="只启动真机反馈和 MoveIt/RViz，不占用 D435",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="只检查文件和重复进程，不启动任何程序",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="只规划两段轨迹，不向真机发送运动",
    )
    args = parser.parse_args()

    if not 1 <= MOTION_SPEED_PERCENT <= 100:
        parser.error("MOTION_SPEED_PERCENT 必须在 1～100 之间")

    if args.scene_only and not args.plan_only:
        parser.error("--scene-only 必须与 --plan-only 同时使用")

    validate_files()
    conflicts = running_conflicts()
    if conflicts:
        print("安全拒绝：检测到旧 RViz/MoveIt 或真机反馈进程，请先关闭：", file=sys.stderr)
        print("\n".join(conflicts), file=sys.stderr)
        return 2

    if args.check:
        clean_env = clean_ros_environment()
        assert not any(
            Path(entry).is_relative_to(CONDA_ROOT)
            for entry in clean_env.get("PATH", "").split(os.pathsep)
            if entry
        )
        print("STARTUP CHECK PASSED: 文件完整，未检测到重复进程。")
        print("环境隔离正常：ROS 使用系统环境；YOLO 子进程单独使用 Conda arm。")
        return 0

    start_vision = "false" if args.scene_only else "true"
    command = " && ".join(
        [
            f"source {shlex.quote(str(ROS_SETUP))}",
            f"source {shlex.quote(str(OVERLAY_SETUP))}",
            "exec ros2 launch meter_grasp vision_guided_scene.launch.py "
            f"project_root:={shlex.quote(str(PROJECT_ROOT))} "
            f"start_vision:={start_vision} "
            f"execute_motion:={'false' if args.plan_only else 'true'} "
            f"motion_speed_percent:={MOTION_SPEED_PERCENT}",
        ]
    )
    action = "仅生成两段候选轨迹" if args.plan_only else "规划并执行两段真机轨迹"
    print(f"启动真机反馈、MoveIt/RViz 与 D435 检测；{action}。Ctrl+C 可统一退出。")
    return subprocess.call(["/bin/bash", "-lc", command], env=clean_ros_environment())


if __name__ == "__main__":
    raise SystemExit(main())
