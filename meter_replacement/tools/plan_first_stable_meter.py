#!/usr/bin/env python3
"""Plan and optionally execute the two-stage D435 meter approach."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
VISION = ROOT / "vision" / "D435_Project_new"
READY = VISION / "logs" / "robot_ready_targets.csv"
OUTPUT = ROOT / "logs" / "vision"
EXECUTOR = ROOT / "robot" / "real" / "cr5_tcp" / "execute_moveit_linear_stage.py"
QUEUE_RESUME = ROOT / "robot" / "real" / "cr5_tcp" / "resume_cr5_tcp_motion.py"
INTER_STAGE_SETTLE_S = 1.0


def read_target(path: Path, target_id: str) -> dict[str, str] | None:
    try:
        with path.open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
    except (FileNotFoundError, OSError, csv.Error):
        return None
    return next(
        (row for row in rows if row.get("target_id") == target_id and row.get("status") == "ready"),
        None,
    )


def write_snapshot(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def execute_stage(
    stage: str, trajectory: Path, *, execute: bool, speed_percent: int
) -> None:
    digest = json.loads(trajectory.read_text(encoding="utf-8"))["sha256"]
    command = [
        "/usr/bin/python3", str(EXECUTOR),
        "--stage", stage,
        "--trajectory", str(trajectory),
        "--trajectory-sha256", digest,
        "--speed-percent", str(speed_percent),
    ]
    if execute:
        resume_motion_queue()
    subprocess.run(command, cwd=ROOT, check=True)
    if execute:
        confirmations = {
            "vision_safe_transition": "MOVE_REAL_CR5_TO_VISION_SAFE_TRANSITION",
            "vision_meter_approach": "MOVE_REAL_CR5_TO_VISION_METER_APPROACH",
        }
        subprocess.run(
            command + ["--execute", "--confirmation", confirmations[stage]],
            cwd=ROOT,
            check=True,
        )
        time.sleep(INTER_STAGE_SETTLE_S)


def resume_motion_queue() -> None:
    """Open an idle TCP queue without sending a motion target."""
    subprocess.run(
        [
            "/usr/bin/python3", str(QUEUE_RESUME),
            "--execute",
            "--confirmation", "RESUME_REAL_CR5_TCP_MOTION_SESSION",
        ],
        cwd=ROOT,
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-id", default="1")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--execute-mode", choices=("false", "true"), default="true")
    parser.add_argument("--speed-percent", type=int, choices=range(1, 101), default=10)
    args = parser.parse_args()
    execute = args.execute_mode == "true" and not args.plan_only and not args.check

    if args.check:
        row = read_target(READY, args.target_id)
        print("VISION COORDINATOR CHECK PASSED" if row else "No ready target yet")
        return 0 if row else 1

    started = time.time()
    deadline = time.monotonic() + args.timeout_s
    row = None
    while time.monotonic() < deadline:
        if READY.exists() and READY.stat().st_mtime >= started:
            row = read_target(READY, args.target_id)
            if row:
                break
        time.sleep(0.2)
    if row is None:
        raise TimeoutError(f"No fresh stable D435 target ID{args.target_id} within {args.timeout_s:.0f}s")

    snapshot = OUTPUT / "first_ready_target.csv"
    converted = OUTPUT / "first_ready_target_user0.csv"
    converted_user2 = OUTPUT / "first_ready_target_user2.csv"
    transition = OUTPUT / "first_meter_safe_transition_candidate.json"
    trajectory = OUTPUT / "first_meter_dynamic_approach_candidate.json"
    write_snapshot(snapshot, row)

    subprocess.run(
        [
            "/usr/bin/python3", str(VISION / "convert_ready_targets_to_cr5.py"),
            "--ready-targets", str(snapshot), "--output", str(converted),
        ],
        check=True,
    )
    subprocess.run(
        [
            "/usr/bin/python3", str(VISION / "convert_cr5_targets_user0_to_user2.py"),
            "--input", str(converted), "--output", str(converted_user2),
        ],
        check=True,
    )
    with converted.open(encoding="utf-8-sig") as stream:
        target = next(csv.DictReader(stream))
    xyz = [target[f"cr5_{axis}_mm"] for axis in "xyz"]

    subprocess.run(
        [
            "ros2", "run", "meter_grasp", "run_pick_sequence", "--ros-args",
            "-p", "home_profile:=vision_safe_transition",
            "-p", "target_stage:=return_home",
            "-p", "execute:=false",
            "-p", f"trajectory_output_path:={transition}",
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            "ros2", "run", "meter_grasp", "run_pick_sequence", "--ros-args",
            "-p", "home_profile:=vision_safe_transition",
            "-p", "target_stage:=vision_meter_approach",
            "-p", "target_slot:=slot_r1_c1",
            "-p", "execute:=false",
            "-p", f"vision_target_base_mm:=[{','.join(xyz)}]",
            "-p", f"planning_start_trajectory_path:={transition}",
            "-p", f"trajectory_output_path:={trajectory}",
        ],
        cwd=ROOT,
        check=True,
    )
    print(f"SAFE TRANSITION CANDIDATE READY: {transition}")
    print(f"VISION APPROACH CANDIDATE READY: {trajectory}")
    if not execute:
        print("No real-robot motion command was sent.")
        return 0

    execute_stage(
        "vision_safe_transition", transition,
        execute=True, speed_percent=args.speed_percent,
    )
    execute_stage(
        "vision_meter_approach", trajectory,
        execute=True, speed_percent=args.speed_percent,
    )
    print("VISION TWO-STAGE WORKFLOW PASSED: 真机已到第一个电表的视觉渐进位。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
