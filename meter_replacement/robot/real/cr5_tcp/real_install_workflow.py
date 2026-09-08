#!/usr/bin/env python3
"""Offline validation helpers for a run-scoped real-CR5 install workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from commission_moveit_pregrasp_segment import ros_to_vendor_deg, verify_digest


EXPECTED_STAGES = (
    "pregrasp",
    "grasp_candidate",
    "lift",
    "semantic_flip",
    "slot_transfer",
    "slot_insert",
    "slot_retreat",
    "return_home",
)


@dataclass(frozen=True)
class ValidatedStage:
    """One digest-checked trajectory with vendor-order joint data."""

    name: str
    path: Path
    digest: str
    executor: str
    before: str | None
    joint_names: list[str]
    points_deg: np.ndarray
    times_s: np.ndarray

    @property
    def start_deg(self) -> np.ndarray:
        return self.points_deg[0]

    @property
    def target_deg(self) -> np.ndarray:
        return self.points_deg[-1]


def load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise RuntimeError("任务清单版本不受支持")
    return payload


def validate_manifest(
    manifest_path: Path,
    *,
    project_root: Path,
    joint_mapping: dict,
    continuity_tolerance_deg: float,
) -> tuple[dict, list[ValidatedStage]]:
    """Validate digests, stage order and every stage boundary without motion."""
    manifest = load_manifest(manifest_path)
    entries = manifest.get("stages", [])
    names = tuple(str(entry.get("name")) for entry in entries)
    if names != EXPECTED_STAGES:
        raise RuntimeError(f"阶段顺序错误：{names!r}")

    stages: list[ValidatedStage] = []
    for entry in entries:
        name = str(entry["name"])
        path = (project_root / str(entry["trajectory"])).resolve()
        try:
            path.relative_to(project_root.resolve())
        except ValueError as exc:
            raise RuntimeError(f"轨迹文件越出项目目录：{path}") from exc
        trajectory = json.loads(path.read_text(encoding="utf-8"))
        digest = verify_digest(trajectory)
        if digest != str(entry["sha256"]):
            raise RuntimeError(f"{name}：任务清单SHA256与文件不一致")
        if trajectory.get("target_stage") != name:
            raise RuntimeError(f"{name}：轨迹内部阶段名称不一致")
        if trajectory.get("home_profile") != manifest.get("home_profile"):
            raise RuntimeError(f"{name}：轨迹Home profile与任务清单不一致")

        joint_names = list(trajectory["joint_names"])
        points_rad = np.asarray(
            [point["positions_rad"] for point in trajectory["points"]], dtype=float
        )
        times_s = np.asarray(
            [point["time_from_start_s"] for point in trajectory["points"]],
            dtype=float,
        )
        if points_rad.ndim != 2 or points_rad.shape[0] < 2:
            raise RuntimeError(f"{name}：轨迹点数量无效")
        if len(times_s) != len(points_rad) or np.any(np.diff(times_s) <= 0.0):
            raise RuntimeError(f"{name}：轨迹时间必须严格递增")
        points_deg = np.asarray(
            [ros_to_vendor_deg(point, joint_names, joint_mapping) for point in points_rad]
        )
        executor = str(entry.get("executor", ""))
        if executor not in {"joint_target", "servoj"}:
            raise RuntimeError(f"{name}：未知执行器{executor!r}")
        before = entry.get("before")
        if before not in {None, "virtual_attach", "virtual_detach"}:
            raise RuntimeError(f"{name}：未知阶段握手{before!r}")
        stages.append(
            ValidatedStage(
                name=name,
                path=path,
                digest=digest,
                executor=executor,
                before=before,
                joint_names=joint_names,
                points_deg=points_deg,
                times_s=times_s,
            )
        )

    discontinuities: list[str] = []
    for previous, current in zip(stages, stages[1:]):
        error = float(np.max(np.abs(previous.target_deg - current.start_deg)))
        if error > continuity_tolerance_deg:
            discontinuities.append(
                f"{previous.name}->{current.name}: {error:.6f}°"
            )
    if discontinuities:
        details = "\n  ".join(discontinuities)
        raise RuntimeError("轨迹链不连续：\n  " + details)
    return manifest, stages
