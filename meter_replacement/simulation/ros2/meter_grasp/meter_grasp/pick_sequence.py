#!/usr/bin/env python3
"""Plan and optionally execute one configured meter-grasp motion stage."""

from __future__ import annotations

import math
import hashlib
import json
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    Constraints,
    DisplayTrajectory,
    JointConstraint,
    MoveItErrorCodes,
    OrientationConstraint,
    PlanningSceneComponents,
    PositionConstraint,
    RobotTrajectory,
)
from moveit_msgs.srv import GetPlanningScene, GetPositionIK, GetStateValidity
from rclpy.action import ActionClient
from rclpy.time import Time
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive
from tf2_geometry_msgs import do_transform_pose
from tf2_ros import Buffer, TransformException, TransformListener


PACKAGE_NAME = "meter_grasp"


def _number(mapping: dict[str, Any], key: str) -> float:
    value = float(mapping[key])
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


class PickSequence(Node):
    """Use MoveGroup action handshakes for collision-aware motion stages."""

    def __init__(
        self,
        node_name: str = "single_arm_pick_sequence",
        *,
        announce_target_stage: bool = True,
    ) -> None:
        super().__init__(node_name)
        default_config = (
            Path(get_package_share_directory(PACKAGE_NAME)) / "config" / "poses.yaml"
        )
        self.declare_parameter("poses_config", str(default_config))

        config_path = Path(str(self.get_parameter("poses_config").value))
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        self._config = config

        self._frames = config["frames"]
        self._planning = config["planning"]
        self.declare_parameter("home_profile", "simulation")
        self._home_profile = str(self.get_parameter("home_profile").value)
        home_profiles = config.get("home_profiles", {})
        if home_profiles:
            if self._home_profile not in home_profiles:
                raise ValueError(
                    f"Unknown home_profile '{self._home_profile}'; available: "
                    f"{sorted(home_profiles)}"
                )
            selected_home = home_profiles[self._home_profile]
            home_positions = selected_home["joint_positions_rad"]
            compensate_home_mount_yaw = bool(
                selected_home.get("apply_mount_yaw_compensation", False)
            )
        else:
            # Backward-compatible fallback for an older poses.yaml.
            home_positions = config["home_joint_positions_rad"]
            compensate_home_mount_yaw = True
        self._home_joint_positions = {
            str(name): float(value)
            for name, value in home_positions.items()
        }
        compensation = self._planning.get("mount_yaw_compensation", {})
        self._mount_yaw_compensation_enabled = bool(
            compensation.get("enabled", False)
        )
        self._mount_yaw_joint = str(compensation.get("joint_name", "joint1"))
        self._mount_yaw_joint_offset = (
            float(compensation.get("reference_mount_yaw_rad", 0.0))
            - float(compensation.get("actual_mount_yaw_rad", 0.0))
        )
        if (
            self._mount_yaw_compensation_enabled
            and compensate_home_mount_yaw
        ):
            if self._mount_yaw_joint not in self._home_joint_positions:
                raise ValueError(
                    "mount_yaw_compensation joint is missing from home pose"
                )
            self._home_joint_positions[self._mount_yaw_joint] += (
                self._mount_yaw_joint_offset
            )
            self.get_logger().info(
                "Applied base-yaw compensation to legacy references: "
                f"{self._mount_yaw_joint}{self._mount_yaw_joint_offset:+.6f} rad"
            )
        self.get_logger().info(
            f"Selected home profile: {self._home_profile}"
        )
        self._attachment = config["attachment"]
        self._candidate = config["single_arm"]["meter_top_grasp_candidate"]
        self._place_candidate = config["single_arm"]["meter_place_candidate"]
        self._slot_candidate = config["single_arm"]["meter_slot_install_candidate"]
        self._semantic_flip = config["single_arm"]["meter_semantic_flip"]
        self.declare_parameter(
            "target_slot", str(self._slot_candidate["target_slot_id"])
        )
        self._target_slot = str(self.get_parameter("target_slot").value)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self.declare_parameter("execute", bool(self._planning["execute"]))
        self.declare_parameter("target_stage", str(self._planning["target_stage"]))
        self.declare_parameter("trajectory_output_path", "")
        self.declare_parameter("planning_start_trajectory_path", "")
        self.declare_parameter("export_executed_trajectory", False)
        # Generic commissioning target expressed as the physical CR5 flange
        # (controller Tool0) in the robot base frame.  Keeping this input as a
        # runtime parameter lets calibrated/vision targets reuse the existing
        # MoveIt IK, collision checking, trajectory review and export path.
        self.declare_parameter(
            "custom_flange_pose_base_mm_deg", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        self.declare_parameter("custom_base_frame", "base_link")
        self.declare_parameter("custom_flange_frame", "Link6")
        self.declare_parameter("vision_target_base_mm", [0.0, 0.0, 0.0])
        self.declare_parameter("vision_max_slot_correction_m", 0.20)
        self._execute = bool(self.get_parameter("execute").value)
        self._target_stage = str(self.get_parameter("target_stage").value)
        self._trajectory_output_path = str(
            self.get_parameter("trajectory_output_path").value
        ).strip()
        self._planning_start_trajectory_path = str(
            self.get_parameter("planning_start_trajectory_path").value
        ).strip()
        self._export_executed_trajectory = bool(
            self.get_parameter("export_executed_trajectory").value
        )
        if self._export_executed_trajectory and not self._execute:
            raise ValueError(
                "export_executed_trajectory requires execute:=true"
            )
        if self._planning_start_trajectory_path and self._execute:
            raise ValueError(
                "planning_start_trajectory_path is plan-only; chained hardware "
                "execution must verify the first segment before the second"
            )
        valid_stages = {
            "pregrasp",
            "grasp_candidate",
            "realign_and_descend",
            "lift",
            "place_transfer",
            "place_descend",
            "place_retreat",
            "slot_transfer",
            "slot_pick_approach",
            "vision_meter_approach",
            "slot_pick_insert",
            "slot_extract",
            "slot_uninstall_transfer",
            "semantic_flip",
            "semantic_unflip",
            "desktop_place_descend",
            "desktop_place_retreat",
            "slot_insert",
            "slot_retreat",
            "return_home",
            "custom_flange_pose",
        }
        if self._target_stage not in valid_stages:
            raise ValueError(
                "target_stage must be 'pregrasp', 'grasp_candidate', "
                "'realign_and_descend', 'lift', 'place_transfer', "
                "'place_descend', 'place_retreat', 'slot_transfer', "
                "'slot_pick_approach', 'vision_meter_approach', 'slot_pick_insert', "
                "'slot_extract', 'slot_uninstall_transfer', "
                "'semantic_flip', 'semantic_unflip', "
                "'desktop_place_descend', 'desktop_place_retreat', "
                "'slot_insert', 'slot_retreat', 'return_home', or "
                "'custom_flange_pose'"
            )

        if self._target_stage in {"custom_flange_pose", "vision_meter_approach"} and self._execute:
            raise ValueError(
                f"{self._target_stage} is commissioning plan-only; use execute:=false"
            )

        action_name = str(self._planning["move_group_action"])
        self._move_group = ActionClient(self, MoveGroup, action_name)
        self._execute_trajectory = ActionClient(
            self,
            ExecuteTrajectory,
            str(self._planning["execute_trajectory_action"]),
        )
        self._scene_client = self.create_client(GetPlanningScene, "/get_planning_scene")
        self._ik_client = self.create_client(
            GetPositionIK, str(self._planning["compute_ik_service"])
        )
        self._state_validity_client = self.create_client(
            GetStateValidity,
            str(self._planning.get("state_validity_service", "/check_state_validity")),
        )
        self._display_publisher = self.create_publisher(
            DisplayTrajectory, "/display_planned_path", 10
        )
        self.get_logger().info(f"Loaded motion configuration: {config_path}")
        self.get_logger().info(
            "Execution is %s" % ("ENABLED" if self._execute else "DISABLED (plan only)")
        )
        if announce_target_stage:
            self.get_logger().info(
                f"Requested motion stage: {self._target_stage}"
            )
        self.get_logger().info(f"Selected meter slot: {self._target_slot}")

    def _configured_pose(
        self,
        candidate: dict[str, Any],
        offset_name: str | None = None,
    ) -> Pose:
        configured_pose = candidate["pose"]
        position = configured_pose["position_m"]
        orientation = configured_pose["quaternion_xyzw"]
        offset = (
            candidate[offset_name]["offset_m"] if offset_name is not None else {}
        )

        pose = Pose()
        pose.position.x = _number(position, "x") + float(offset.get("x", 0.0))
        pose.position.y = _number(position, "y") + float(offset.get("y", 0.0))
        pose.position.z = _number(position, "z") + float(offset.get("z", 0.0))

        values = [_number(orientation, axis) for axis in ("x", "y", "z", "w")]
        norm = math.sqrt(sum(value * value for value in values))
        if norm < 1.0e-9:
            raise ValueError("Configured quaternion has zero length")
        (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ) = [value / norm for value in values]
        return pose

    def _pose(self, offset_name: str | None = None) -> Pose:
        return self._configured_pose(self._candidate, offset_name)

    def _place_pose(self, offset_name: str | None = None) -> Pose:
        return self._configured_pose(self._place_candidate, offset_name)

    def _slot_frame(self) -> str:
        prefix = "slot_"
        if not self._target_slot.startswith(prefix):
            raise ValueError("target_slot must use an id such as 'slot_r1_c2'")
        return f"meter_{self._target_slot}"

    def _slot_pose(self, offset_name: str | None = None) -> Pose | None:
        local_pose = self._configured_pose(self._slot_candidate, offset_name)
        planning_frame = str(self._frames["planning_frame"])
        slot_frame = self._slot_frame()
        timeout = float(
            self._config["state_machine"]["slot_transform_timeout_s"]
        )
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            if self._tf_buffer.can_transform(planning_frame, slot_frame, Time()):
                break
            rclpy.spin_once(self, timeout_sec=0.05)
        try:
            transform = self._tf_buffer.lookup_transform(
                planning_frame, slot_frame, Time()
            )
        except TransformException as error:
            self.get_logger().error(
                f"Cannot transform selected slot '{self._target_slot}' "
                f"from '{slot_frame}' to '{planning_frame}': {error}"
            )
            return None
        return do_transform_pose(local_pose, transform)

    def _vision_meter_approach_pose(self) -> Pose | None:
        """Shift the validated slot approach by the D435 target correction."""
        raw = list(self.get_parameter("vision_target_base_mm").value)
        if len(raw) != 3 or any(not math.isfinite(float(value)) for value in raw):
            self.get_logger().error("vision_target_base_mm must contain three finite values")
            return None

        planning_frame = str(self._frames["planning_frame"])
        base_frame = str(self.get_parameter("custom_base_frame").value)
        base_to_planning = self._wait_for_transform(planning_frame, base_frame)
        slot_to_planning = self._wait_for_transform(planning_frame, self._slot_frame())
        approach = self._slot_pose("approach")
        if base_to_planning is None or slot_to_planning is None or approach is None:
            return None

        detected_in_base = Pose()
        detected_in_base.position.x = float(raw[0]) / 1000.0
        detected_in_base.position.y = float(raw[1]) / 1000.0
        detected_in_base.position.z = float(raw[2]) / 1000.0
        detected_in_base.orientation.w = 1.0
        detected = do_transform_pose(detected_in_base, base_to_planning)
        slot = slot_to_planning.transform.translation
        correction = (
            detected.position.x - slot.x,
            detected.position.y - slot.y,
            detected.position.z - slot.z,
        )
        correction_norm = math.sqrt(sum(value * value for value in correction))
        maximum = float(self.get_parameter("vision_max_slot_correction_m").value)
        if correction_norm > maximum:
            self.get_logger().error(
                f"Vision correction {correction_norm:.3f} m exceeds {maximum:.3f} m"
            )
            return None

        approach.position.x += correction[0]
        approach.position.y += correction[1]
        approach.position.z += correction[2]
        self.get_logger().info(
            "D435 target correction in planning frame: "
            f"dx={correction[0]:+.4f}, dy={correction[1]:+.4f}, "
            f"dz={correction[2]:+.4f} m"
        )
        return approach

    @staticmethod
    def _quaternion_from_rpy(
        roll: float, pitch: float, yaw: float
    ) -> tuple[float, float, float, float]:
        """Return xyzw quaternion for the controller's fixed-XYZ RPY angles."""
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        return (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )

    @staticmethod
    def _quaternion_multiply(
        left: tuple[float, float, float, float],
        right: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        lx, ly, lz, lw = left
        rx, ry, rz, rw = right
        return (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        )

    @staticmethod
    def _rotate_vector(
        quaternion: tuple[float, float, float, float],
        vector: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        qx, qy, qz, qw = quaternion
        vx, vy, vz = vector
        # Efficient q * v * conjugate(q), for a unit quaternion.
        tx = 2.0 * (qy * vz - qz * vy)
        ty = 2.0 * (qz * vx - qx * vz)
        tz = 2.0 * (qx * vy - qy * vx)
        return (
            vx + qw * tx + qy * tz - qz * ty,
            vy + qw * ty + qz * tx - qx * tz,
            vz + qw * tz + qx * ty - qy * tx,
        )

    def _wait_for_transform(self, target: str, source: str):
        timeout = float(self._planning["server_timeout_s"])
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            if self._tf_buffer.can_transform(target, source, Time()):
                break
            rclpy.spin_once(self, timeout_sec=0.05)
        try:
            return self._tf_buffer.lookup_transform(target, source, Time())
        except TransformException as error:
            self.get_logger().error(
                f"Cannot transform '{source}' to '{target}': {error}"
            )
            return None

    def _custom_tool_pose(self) -> Pose | None:
        """Convert a base-frame Tool0/flange target to MoveIt's tool-frame pose."""
        raw = list(
            self.get_parameter("custom_flange_pose_base_mm_deg").value
        )
        if len(raw) != 6 or any(not math.isfinite(float(value)) for value in raw):
            self.get_logger().error(
                "custom_flange_pose_base_mm_deg must contain six finite values"
            )
            return None
        x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg = map(float, raw)
        base_frame = str(self.get_parameter("custom_base_frame").value)
        flange_frame = str(self.get_parameter("custom_flange_frame").value)
        tool_frame = str(self._frames["tool_frame"])
        planning_frame = str(self._frames["planning_frame"])

        flange_to_tool = self._wait_for_transform(flange_frame, tool_frame)
        base_to_planning = self._wait_for_transform(planning_frame, base_frame)
        if flange_to_tool is None or base_to_planning is None:
            return None

        flange_q = self._quaternion_from_rpy(
            math.radians(roll_deg),
            math.radians(pitch_deg),
            math.radians(yaw_deg),
        )
        relative = flange_to_tool.transform
        relative_q = (
            relative.rotation.x,
            relative.rotation.y,
            relative.rotation.z,
            relative.rotation.w,
        )
        rotated_offset = self._rotate_vector(
            flange_q,
            (
                relative.translation.x,
                relative.translation.y,
                relative.translation.z,
            ),
        )
        tool_q = self._quaternion_multiply(flange_q, relative_q)

        tool_in_base = Pose()
        tool_in_base.position.x = x_mm / 1000.0 + rotated_offset[0]
        tool_in_base.position.y = y_mm / 1000.0 + rotated_offset[1]
        tool_in_base.position.z = z_mm / 1000.0 + rotated_offset[2]
        (
            tool_in_base.orientation.x,
            tool_in_base.orientation.y,
            tool_in_base.orientation.z,
            tool_in_base.orientation.w,
        ) = tool_q
        tool_in_planning = do_transform_pose(tool_in_base, base_to_planning)
        self.get_logger().info(
            "Custom Tool0 target in base frame: "
            f"xyz_mm=({x_mm:.3f}, {y_mm:.3f}, {z_mm:.3f}), "
            f"rpy_deg=({roll_deg:.3f}, {pitch_deg:.3f}, {yaw_deg:.3f})"
        )
        self.get_logger().info(
            f"Converted modeled '{tool_frame}' target in '{planning_frame}': "
            f"xyz_m=({tool_in_planning.position.x:.6f}, "
            f"{tool_in_planning.position.y:.6f}, "
            f"{tool_in_planning.position.z:.6f})"
        )
        return tool_in_planning

    def _planning_seed(self) -> dict[str, float]:
        configured = self._candidate["planning_seed_joint_positions_rad"]
        seed = {str(name): float(value) for name, value in configured.items()}
        if self._mount_yaw_compensation_enabled:
            if self._mount_yaw_joint not in seed:
                raise ValueError(
                    "mount_yaw_compensation joint is missing from planning seed"
                )
            seed[self._mount_yaw_joint] += self._mount_yaw_joint_offset
        return seed

    def _semantic_flip_goal(
        self, current_positions: dict[str, float]
    ) -> dict[str, float] | None:
        arm_joints = self._planning_seed().keys()
        missing = [name for name in arm_joints if name not in current_positions]
        if missing:
            self.get_logger().error(
                f"Semantic-flip current state is missing arm joints: {missing}"
            )
            return None
        goal = {name: float(current_positions[name]) for name in arm_joints}
        joint_name = str(self._semantic_flip["joint_name"])
        if joint_name not in goal:
            self.get_logger().error(
                f"Semantic-flip joint '{joint_name}' is not in the arm group"
            )
            return None
        profile_targets = self._semantic_flip.get(
            "target_position_rad_by_home_profile", {}
        )
        goal[joint_name] = float(
            profile_targets.get(
                self._home_profile,
                self._semantic_flip["target_position_rad"],
            )
        )
        self.get_logger().info(
            f"Semantic-flip target for home profile '{self._home_profile}': "
            f"{joint_name}={goal[joint_name]:.6f} rad"
        )
        return goal

    @staticmethod
    def _closest_equivalent(
        angle: float,
        reference: float,
        lower: float,
        upper: float,
    ) -> float:
        """Choose the most centered equivalent angle inside the soft limits."""
        two_pi = 2.0 * math.pi
        nearest_turn = round((reference - angle) / two_pi)
        candidates = [
            angle + two_pi * turn
            for turn in range(nearest_turn - 2, nearest_turn + 3)
            if lower <= angle + two_pi * turn <= upper
        ]
        if not candidates:
            raise ValueError(
                f"No equivalent of {angle:.3f} rad lies inside "
                f"[{lower:.3f}, {upper:.3f}]"
            )
        return min(
            candidates,
            key=lambda value: (
                -min(value - lower, upper - value),
                abs(value - reference),
            ),
        )

    def _solve_ik(
        self,
        pose: Pose,
        seed_positions: dict[str, float] | None = None,
        *,
        log_result: bool = True,
    ) -> dict[str, float] | None:
        timeout = float(self._planning["server_timeout_s"])
        if not self._ik_client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error("GetPositionIK service was not available")
            return None

        configured_seed = self._planning_seed()
        if seed_positions is None:
            seed = configured_seed
            seed_source = "recorded branch"
        else:
            missing = [name for name in configured_seed if name not in seed_positions]
            if missing:
                self.get_logger().error(
                    f"Current-state IK seed is missing arm joints: {missing}"
                )
                return None
            seed = {
                name: float(seed_positions[name])
                for name in configured_seed
            }
            seed_source = "current robot state"
        request = GetPositionIK.Request()
        request.ik_request.group_name = str(self._planning["group_name"])
        request.ik_request.ik_link_name = str(self._frames["tool_frame"])
        request.ik_request.avoid_collisions = True
        request.ik_request.robot_state.joint_state.name = list(seed)
        request.ik_request.robot_state.joint_state.position = list(seed.values())
        request.ik_request.pose_stamped.header.frame_id = str(
            self._frames["planning_frame"]
        )
        request.ik_request.pose_stamped.pose = pose

        ik_timeout = float(self._planning["ik_timeout_s"])
        request.ik_request.timeout.sec = int(ik_timeout)
        request.ik_request.timeout.nanosec = int((ik_timeout % 1.0) * 1.0e9)

        if log_result:
            self.get_logger().info(
                f"Solving collision-aware IK from {seed_source}"
            )
        future = self._ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done() or future.result() is None:
            self.get_logger().error("IK request timed out")
            return None
        response = future.result()
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f"IK failed with MoveIt error {response.error_code.val}"
            )
            return None

        returned = dict(
            zip(
                response.solution.joint_state.name,
                response.solution.joint_state.position,
            )
        )
        if any(name not in returned for name in seed):
            self.get_logger().error("IK response did not contain every arm joint")
            return None

        configured_limits = self._planning["joint_position_limits_rad"]
        limit_margin = float(self._planning["joint_limit_margin_rad"])
        try:
            solution = {
                name: self._closest_equivalent(
                    float(returned[name]),
                    reference,
                    float(configured_limits[name][0]) + limit_margin,
                    float(configured_limits[name][1]) - limit_margin,
                )
                for name, reference in seed.items()
            }
        except (KeyError, IndexError, TypeError, ValueError) as error:
            self.get_logger().error(f"IK joint-limit normalization failed: {error}")
            return None
        if log_result:
            formatted = ", ".join(
                f"{name}={value:.3f}" for name, value in solution.items()
            )
            self.get_logger().info(f"Selected IK branch: {formatted}")
        return solution

    def _solve_best_ik(
        self,
        pose: Pose,
        current_positions: dict[str, float],
        *,
        ranking: str = "ergonomic",
        maximum_allowed: float | None = None,
    ) -> dict[str, float] | None:
        """Search and rank collision-free IK branches near the current state."""
        if ranking not in {"ergonomic", "nearest"}:
            raise ValueError(f"Unsupported IK ranking mode: {ranking}")
        arm_joints = list(self._planning_seed())
        missing = [name for name in arm_joints if name not in current_positions]
        if missing:
            self.get_logger().error(
                f"Multi-seed IK current state is missing arm joints: {missing}"
            )
            return None

        configured_offsets = self._planning[
            "slot_transfer_ik_seed_offsets_rad"
        ]
        offset_names = list(configured_offsets)
        unknown = [name for name in offset_names if name not in arm_joints]
        if unknown:
            self.get_logger().error(
                f"Multi-seed IK configuration contains unknown joints: {unknown}"
            )
            return None

        solutions: dict[tuple[float, ...], dict[str, float]] = {}
        offset_sets = [
            [float(value) for value in configured_offsets[name]]
            for name in offset_names
        ]
        preferred = {
            str(name): float(value)
            for name, value in self._planning[
                "slot_transfer_preferred_joint_positions_rad"
            ].items()
        }
        if self._mount_yaw_compensation_enabled:
            if self._mount_yaw_joint not in preferred:
                self.get_logger().error(
                    "Mount-yaw joint is missing from slot-transfer preferred posture"
                )
                return None
            preferred[self._mount_yaw_joint] += self._mount_yaw_joint_offset
        preferred_missing = [
            name for name in arm_joints if name not in preferred
        ]
        if preferred_missing:
            self.get_logger().error(
                "Slot-transfer preferred posture is missing arm joints: "
                f"{preferred_missing}"
            )
            return None

        self.get_logger().info(
            "Searching collision-aware IK branches around the current state"
        )
        seeds: list[dict[str, float]] = []
        for offsets in product(*offset_sets):
            seed = {
                name: float(current_positions[name])
                for name in arm_joints
            }
            for name, offset in zip(offset_names, offsets):
                seed[name] += offset
            seeds.append(seed)
        # A task-specific nominal posture is a standard way to guide a
        # redundant/multi-branch IK search without hard-coding the final pose.
        seeds.append(dict(preferred))

        for seed in seeds:
            solution = self._solve_ik(pose, seed, log_result=False)
            if solution is None:
                continue
            key = tuple(round(solution[name], 6) for name in arm_joints)
            solutions[key] = solution

        if not solutions:
            self.get_logger().error(
                "No collision-free IK branch was found by the configured seeds"
            )
            return None

        if maximum_allowed is None:
            maximum_allowed = float(
                self._planning["slot_transfer_max_joint_travel_rad"]
            )
        eligible: list[dict[str, float]] = []
        for solution in solutions.values():
            maximum = max(
                abs(solution[name] - float(current_positions[name]))
                for name in arm_joints
            )
            if maximum <= maximum_allowed:
                eligible.append(solution)
        if not eligible:
            self.get_logger().error(
                "Every IK branch exceeds the configured slot-transfer "
                f"single-joint limit of {maximum_allowed:.2f} rad"
            )
            return None

        def score(solution: dict[str, float]) -> tuple[float, float, float]:
            travel = [
                abs(solution[name] - float(current_positions[name]))
                for name in arm_joints
            ]
            ergonomic_cost = sum(
                (solution[name] - preferred[name]) ** 2
                for name in arm_joints
            )
            if ranking == "nearest":
                # For the first motion from a measured real-arm posture, stay
                # on the closest feasible kinematic branch.  Ergonomics is
                # only a tie-breaker; it must not pull the arm across an elbow
                # or wrist branch merely to resemble a legacy nominal pose.
                return max(travel), sum(travel), ergonomic_cost
            return ergonomic_cost, max(travel), sum(travel)

        selected = min(eligible, key=score)
        travel = [
            abs(selected[name] - float(current_positions[name]))
            for name in arm_joints
        ]
        maximum_travel = max(travel)
        total_travel = sum(travel)
        ergonomic_cost = sum(
            (selected[name] - preferred[name]) ** 2
            for name in arm_joints
        )
        formatted = ", ".join(
            f"{name}={value:.3f}" for name, value in selected.items()
        )
        self.get_logger().info(
            f"Selected best of {len(solutions)} unique IK branches: "
            f"{formatted}; max_joint_delta={maximum_travel:.3f}, "
            f"total_joint_delta={total_travel:.3f}, "
            f"ergonomic_cost={ergonomic_cost:.3f}, ranking={ranking}"
        )
        return selected

    def _joint_goal_constraints(
        self, joint_positions: dict[str, float], name: str
    ) -> Constraints:
        tolerance = float(self._planning["joint_tolerance_rad"])
        constraints = Constraints()
        constraints.name = name
        for joint_name, target in joint_positions.items():
            joint = JointConstraint()
            joint.joint_name = joint_name
            joint.position = target
            joint.tolerance_above = tolerance
            joint.tolerance_below = tolerance
            joint.weight = 1.0
            constraints.joint_constraints.append(joint)
        return constraints

    def _state_is_valid(
        self, joint_positions: dict[str, float], label: str
    ) -> bool:
        """Ask MoveIt for bounds/collision validity and report contact pairs."""
        timeout = float(self._planning["server_timeout_s"])
        if not self._state_validity_client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error("GetStateValidity service was not available")
            return False

        merged = dict(self._scene_joint_positions)
        merged.update({name: float(value) for name, value in joint_positions.items()})
        request = GetStateValidity.Request()
        request.group_name = str(self._planning["group_name"])
        request.robot_state.joint_state.name = list(merged)
        request.robot_state.joint_state.position = list(merged.values())
        request.robot_state.is_diff = False
        future = self._state_validity_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done() or future.result() is None:
            self.get_logger().error(f"State-validity query timed out for '{label}'")
            return False

        response = future.result()
        if response.valid:
            self.get_logger().info(f"State-validity check passed: '{label}'")
            return True

        if response.contacts:
            pairs = sorted(
                {
                    (contact.contact_body_1, contact.contact_body_2)
                    for contact in response.contacts
                }
            )
            formatted = ", ".join(f"{first} <-> {second}" for first, second in pairs)
            self.get_logger().error(
                f"State-validity check failed for '{label}'; contacts: {formatted}"
            )
        else:
            self.get_logger().error(
                f"State-validity check failed for '{label}' without contact data; "
                "check joint bounds and constraints"
            )
        return False

    def _pose_goal_constraints(self, pose: Pose, name: str) -> Constraints:
        frame = str(self._frames["planning_frame"])
        tool = str(self._frames["tool_frame"])

        region = SolidPrimitive()
        region.type = SolidPrimitive.SPHERE
        region.dimensions = [float(self._planning["position_tolerance_m"])]

        position = PositionConstraint()
        position.header.frame_id = frame
        position.link_name = tool
        position.constraint_region.primitives.append(region)
        position.constraint_region.primitive_poses.append(pose)
        position.weight = 1.0

        angular_tolerance = float(self._planning["orientation_tolerance_rad"])
        orientation = OrientationConstraint()
        orientation.header.frame_id = frame
        orientation.link_name = tool
        orientation.orientation = pose.orientation
        orientation.absolute_x_axis_tolerance = angular_tolerance
        orientation.absolute_y_axis_tolerance = angular_tolerance
        orientation.absolute_z_axis_tolerance = angular_tolerance
        orientation.weight = 1.0

        constraints = Constraints()
        constraints.name = name
        constraints.position_constraints.append(position)
        constraints.orientation_constraints.append(orientation)
        return constraints

    def _scene_contains_meter(self) -> bool:
        timeout = float(self._planning["server_timeout_s"])
        self.get_logger().info("Waiting for MoveIt planning scene...")
        if not self._scene_client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error("GetPlanningScene service was not available")
            return False

        request = GetPlanningScene.Request()
        request.components.components = (
            PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
            | PlanningSceneComponents.ROBOT_STATE
            | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
        )
        future = self._scene_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done() or future.result() is None:
            self.get_logger().error("Planning-scene query timed out")
            return False

        expected = [str(item) for item in self._planning["required_collision_objects"]]
        scene = future.result().scene
        self._scene_joint_positions = dict(
            zip(
                scene.robot_state.joint_state.name,
                scene.robot_state.joint_state.position,
            )
        )
        if self._planning_start_trajectory_path:
            if not self._apply_planning_start_trajectory():
                return False
        self._world_object_poses = {
            item.id: item.pose
            for item in scene.world.collision_objects
        }
        world_ids = [item.id for item in scene.world.collision_objects]
        attached_ids = [
            item.object.id for item in scene.robot_state.attached_collision_objects
        ]
        self._attached_object_ids = set(attached_ids)
        available_ids = set(world_ids) | self._attached_object_ids
        missing = [item for item in expected if item not in available_ids]
        if missing:
            self.get_logger().error(
                f"Required collision objects are missing: {missing}; "
                f"world={world_ids}, attached={attached_ids}"
            )
            return False
        self.get_logger().info(
            f"Verified collision objects: world={world_ids}, attached={attached_ids}"
        )
        return True

    def _apply_planning_start_trajectory(self) -> bool:
        """Use a validated preceding plan's endpoint as this plan's start."""
        path = Path(self._planning_start_trajectory_path).expanduser().resolve()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            recorded_hash = str(payload.pop("sha256"))
            canonical = json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            if hashlib.sha256(canonical).hexdigest() != recorded_hash:
                raise ValueError("SHA256 mismatch")
            if payload["status"] != "moveit_plan_only_operator_review_pending":
                raise ValueError("preceding artifact is not a reviewed plan-only candidate")
            if payload["robot_model"] != str(self._planning["robot_model_name"]):
                raise ValueError("robot model mismatch")
            joint_names = [str(name) for name in payload["joint_names"]]
            positions = [float(value) for value in payload["points"][-1]["positions_rad"]]
            if len(joint_names) != len(positions):
                raise ValueError("joint/position length mismatch")
        except (OSError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.get_logger().error(f"Invalid planning-start trajectory '{path}': {error}")
            return False

        endpoint = dict(zip(joint_names, positions))
        if not self._state_is_valid(endpoint, "preceding trajectory endpoint"):
            return False
        self._scene_joint_positions.update(endpoint)
        self.get_logger().info(
            "Planning from validated preceding trajectory endpoint: "
            f"{path} (sha256={recorded_hash})"
        )
        return True

    def _plan_constraints(
        self,
        constraints: Constraints,
        name: str,
        pipeline_id: str,
        planner_id: str,
    ) -> RobotTrajectory | None:
        timeout = float(self._planning["server_timeout_s"])
        if not self._move_group.wait_for_server(timeout_sec=timeout):
            self.get_logger().error("MoveGroup action server was not available")
            return None

        goal = MoveGroup.Goal()
        goal.request.group_name = str(self._planning["group_name"])
        goal.request.pipeline_id = pipeline_id
        goal.request.planner_id = planner_id
        goal.request.num_planning_attempts = int(self._planning["planning_attempts"])
        goal.request.allowed_planning_time = float(
            self._planning["allowed_planning_time_s"]
        )
        goal.request.max_velocity_scaling_factor = float(
            self._planning["velocity_scaling"]
        )
        goal.request.max_acceleration_scaling_factor = float(
            self._planning["acceleration_scaling"]
        )
        # Pin the request to the robot state captured from the planning-scene
        # query immediately before planning.  Leaving an empty differential
        # start state lets MoveGroup choose a cached/default state, which is
        # unsafe for real-arm plan-only commissioning and makes RViz animate
        # from a posture different from the physical CR5.
        start_positions = {
            str(name): float(value)
            for name, value in self._scene_joint_positions.items()
        }
        goal.request.start_state.joint_state.name = list(start_positions)
        goal.request.start_state.joint_state.position = list(
            start_positions.values()
        )
        goal.request.start_state.is_diff = False
        goal.request.goal_constraints.append(constraints)
        goal.planning_options.plan_only = True
        goal.planning_options.look_around = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 2
        goal.planning_options.replan_delay = 0.2
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        self.get_logger().info(
            f"Planning '{name}' with pipeline='{pipeline_id}', planner='{planner_id}'"
        )
        send_future = self._move_group.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=timeout)
        if not send_future.done() or send_future.result() is None:
            self.get_logger().error(f"'{name}' goal request timed out")
            return None

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f"'{name}' goal was rejected")
            return None

        result_future = goal_handle.get_result_async()
        result_timeout = timeout + float(self._planning["allowed_planning_time_s"])
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=result_timeout)
        if not result_future.done() or result_future.result() is None:
            self.get_logger().error(f"'{name}' result timed out")
            return None

        result = result_future.result().result
        error_code = result.error_code.val
        if error_code != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f"'{name}' failed with MoveIt error {error_code}")
            return None

        should_display = (
            not self._execute
            or bool(self._planning["display_trajectory_in_execute_mode"])
        )
        if should_display:
            display = DisplayTrajectory()
            display.model_id = str(self._planning["robot_model_name"])
            display.trajectory_start = result.trajectory_start
            display.trajectory.append(result.planned_trajectory)
            self._display_publisher.publish(display)
            rclpy.spin_once(self, timeout_sec=0.2)
        self.get_logger().info(f"'{name}' succeeded")
        return result.planned_trajectory

    def _plan_joint_goal(
        self, joint_positions: dict[str, float], name: str
    ) -> RobotTrajectory | None:
        return self._plan_constraints(
            self._joint_goal_constraints(joint_positions, name),
            name,
            str(self._planning["pipeline_id"]),
            str(self._planning["planner_id"]),
        )

    def _plan_ptp_joint_goal(
        self, joint_positions: dict[str, float], name: str
    ) -> RobotTrajectory | None:
        """Plan a deterministic, collision-checked industrial PTP motion."""
        return self._plan_constraints(
            self._joint_goal_constraints(joint_positions, name),
            name,
            str(self._planning["linear_pipeline_id"]),
            "PTP",
        )

    def _plan_linear_pose_goal(self, pose: Pose, name: str) -> RobotTrajectory | None:
        return self._plan_constraints(
            self._pose_goal_constraints(pose, name),
            name,
            str(self._planning["linear_pipeline_id"]),
            str(self._planning["linear_planner_id"]),
        )

    def _plan_pose_goal(self, pose: Pose, name: str) -> RobotTrajectory | None:
        """Plan a collision-aware free-space pose motion with MoveIt's OMPL pipeline."""
        return self._plan_constraints(
            self._pose_goal_constraints(pose, name),
            name,
            str(self._planning["pipeline_id"]),
            str(self._planning["planner_id"]),
        )

    def _trajectory_is_safe(
        self,
        trajectory: RobotTrajectory,
        maximum_travel: float | None = None,
    ) -> bool:
        joint_trajectory = trajectory.joint_trajectory
        if len(joint_trajectory.points) < 2:
            self.get_logger().error("Planned trajectory contains fewer than two points")
            return False

        configured_limits = self._planning["joint_position_limits_rad"]
        limit_margin = float(self._planning["joint_limit_margin_rad"])
        for point_index, point in enumerate(joint_trajectory.points):
            for joint_index, name in enumerate(joint_trajectory.joint_names):
                if name not in configured_limits:
                    continue
                lower = float(configured_limits[name][0]) + limit_margin
                upper = float(configured_limits[name][1]) - limit_margin
                position = float(point.positions[joint_index])
                if not lower <= position <= upper:
                    self.get_logger().error(
                        "Trajectory rejected: waypoint "
                        f"{point_index} puts {name}={position:.3f} rad outside "
                        f"the soft range [{lower:.3f}, {upper:.3f}]"
                    )
                    return False

        maximum = (
            float(maximum_travel)
            if maximum_travel is not None
            else float(self._planning["max_joint_travel_rad"])
        )
        totals = {name: 0.0 for name in joint_trajectory.joint_names}
        previous = joint_trajectory.points[0].positions
        for point in joint_trajectory.points[1:]:
            for index, name in enumerate(joint_trajectory.joint_names):
                totals[name] += abs(float(point.positions[index]) - float(previous[index]))
            previous = point.positions

        first = joint_trajectory.points[0].positions
        last = joint_trajectory.points[-1].positions
        net_travel = {
            name: abs(float(last[index]) - float(first[index]))
            for index, name in enumerate(joint_trajectory.joint_names)
        }
        extra_travel = {
            name: max(0.0, totals[name] - net_travel[name])
            for name in joint_trajectory.joint_names
        }
        formatted = ", ".join(
            f"{name}={travel:.3f}" for name, travel in totals.items()
        )
        self.get_logger().info(f"Planned cumulative joint travel: {formatted}")
        excessive = {name: travel for name, travel in totals.items() if travel > maximum}
        if excessive:
            self.get_logger().error(
                f"Trajectory rejected: joint travel exceeds {maximum:.2f} rad: {excessive}"
            )
            return False
        maximum_extra = float(self._planning["max_extra_joint_travel_rad"])
        detours = {
            name: travel
            for name, travel in extra_travel.items()
            if travel > maximum_extra
        }
        if detours:
            self.get_logger().error(
                "Trajectory rejected: cumulative travel exceeds net travel by "
                f"more than {maximum_extra:.2f} rad: {detours}"
            )
            return False
        return True

    def _export_validated_trajectory(
        self,
        trajectory: RobotTrajectory,
        target_name: str,
    ) -> bool:
        """Write an auditable plan-only artifact for a later hardware gate."""
        if not self._trajectory_output_path:
            return True
        if self._execute and not self._export_executed_trajectory:
            self.get_logger().error(
                "trajectory_output_path with execute:=true requires "
                "export_executed_trajectory:=true"
            )
            return False

        joint_trajectory = trajectory.joint_trajectory
        payload: dict[str, Any] = {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "status": (
                "moveit_mock_execution_export"
                if self._execute
                else "moveit_plan_only_operator_review_pending"
            ),
            "robot_model": str(self._planning["robot_model_name"]),
            "planning_group": str(self._planning["group_name"]),
            "home_profile": self._home_profile,
            "target_stage": self._target_stage,
            "target_name": target_name,
            "planning_frame": str(self._frames["planning_frame"]),
            "joint_names": list(joint_trajectory.joint_names),
            "live_start_joint_positions_rad": {
                name: float(self._scene_joint_positions[name])
                for name in joint_trajectory.joint_names
            },
            "points": [
                {
                    "positions_rad": [float(value) for value in point.positions],
                    "velocities_rad_s": [float(value) for value in point.velocities],
                    "accelerations_rad_s2": [
                        float(value) for value in point.accelerations
                    ],
                    "time_from_start_s": (
                        float(point.time_from_start.sec)
                        + float(point.time_from_start.nanosec) * 1.0e-9
                    ),
                }
                for point in joint_trajectory.points
            ],
        }
        canonical = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        payload["sha256"] = hashlib.sha256(canonical).hexdigest()

        output = Path(self._trajectory_output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.get_logger().info(
            f"Exported validated plan-only trajectory: {output}"
        )
        self.get_logger().info(f"Trajectory SHA256: {payload['sha256']}")
        return True

    def _execute_planned_trajectory(
        self, trajectory: RobotTrajectory, name: str
    ) -> bool:
        timeout = float(self._planning["server_timeout_s"])
        if not self._execute_trajectory.wait_for_server(timeout_sec=timeout):
            self.get_logger().error("ExecuteTrajectory action server was not available")
            return False

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        self.get_logger().info(f"Executing previously validated trajectory: '{name}'")
        send_future = self._execute_trajectory.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=timeout)
        if not send_future.done() or send_future.result() is None:
            self.get_logger().error(f"'{name}' execution request timed out")
            return False

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f"'{name}' execution was rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout)
        if not result_future.done() or result_future.result() is None:
            self.get_logger().error(f"'{name}' execution result timed out")
            return False

        error_code = result_future.result().result.error_code.val
        if error_code != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f"'{name}' execution failed with MoveIt error {error_code}"
            )
            return False
        self.get_logger().info(f"'{name}' execution succeeded")
        return True

    def run(self) -> bool:
        if not self._scene_contains_meter():
            return False

        if self._target_stage == "realign_and_descend":
            if not self._execute:
                self.get_logger().error(
                    "'realign_and_descend' requires execute:=true because the "
                    "second plan must start from the executed first trajectory"
                )
                return False

            realign_name = "meter_realign_pregrasp"
            realign_trajectory = self._plan_linear_pose_goal(
                self._pose("approach"), realign_name
            )
            if realign_trajectory is None:
                return False
            if not self._trajectory_is_safe(realign_trajectory):
                return False
            if not self._execute_planned_trajectory(
                realign_trajectory, realign_name
            ):
                return False

            descend_name = "meter_centered_grasp_candidate"
            descend_trajectory = self._plan_linear_pose_goal(
                self._pose(), descend_name
            )
            if descend_trajectory is None:
                return False
            if not self._trajectory_is_safe(descend_trajectory):
                return False
            return self._execute_planned_trajectory(
                descend_trajectory, descend_name
            )

        if self._target_stage in {
            "lift",
            "place_transfer",
            "place_descend",
            "slot_transfer",
            "semantic_flip",
            "slot_insert",
            "slot_extract",
            "slot_uninstall_transfer",
            "semantic_unflip",
            "desktop_place_descend",
        }:
            object_id = str(self._attachment["object_id"])
            if object_id not in self._attached_object_ids:
                self.get_logger().error(
                    f"Cannot run '{self._target_stage}': "
                    f"'{object_id}' is not attached to the robot"
                )
                return False

        if self._target_stage == "custom_flange_pose":
            target_name = "custom_flange_pose"
            target_pose = self._custom_tool_pose()
            custom_maximum_travel = float(
                self._planning["custom_pose_max_joint_travel_rad"]
            )
            # First let MoveIt sample a feasible IK branch while planning the
            # complete collision-free path.  This avoids locking the request
            # to the first valid IK solution, which can lie on the wrong side
            # of a support post even though another branch reaches the same
            # flange pose safely.
            trajectory = None
            if target_pose is not None:
                for attempt in range(1, 11):
                    candidate = self._plan_pose_goal(
                        target_pose,
                        f"{target_name}_pose_attempt_{attempt}",
                    )
                    if candidate is None:
                        continue
                    if self._trajectory_is_safe(
                        candidate, custom_maximum_travel
                    ):
                        trajectory = candidate
                        break
                    self.get_logger().warn(
                        "Discarding unsafe custom-pose candidate and "
                        "continuing the search"
                    )
            if trajectory is None and target_pose is not None:
                self.get_logger().warn(
                    "Pose-goal planning failed; retrying with the nearest "
                    "collision-aware IK branch"
                )
                target_joints = self._solve_best_ik(
                    target_pose,
                    self._scene_joint_positions,
                    ranking="nearest",
                    maximum_allowed=custom_maximum_travel,
                )
                if target_joints is not None:
                    candidate = self._plan_ptp_joint_goal(
                        target_joints,
                        f"{target_name}_ptp",
                    )
                    if candidate is not None and self._trajectory_is_safe(
                        candidate, custom_maximum_travel
                    ):
                        trajectory = candidate
                    for attempt in range(1, 11):
                        if trajectory is not None:
                            break
                        candidate = self._plan_joint_goal(
                            target_joints,
                            f"{target_name}_joint_attempt_{attempt}",
                        )
                        if candidate is None:
                            continue
                        if self._trajectory_is_safe(
                            candidate, custom_maximum_travel
                        ):
                            trajectory = candidate
                            break
                        self.get_logger().warn(
                            "Discarding unsafe custom-joint candidate and "
                            "continuing the search"
                        )
        elif self._target_stage == "return_home":
            object_id = str(self._attachment["object_id"])
            if object_id in self._attached_object_ids:
                self.get_logger().error(
                    f"Cannot return home: '{object_id}' is still attached"
                )
                return False
            target_name = "return_home"
            trajectory = self._plan_joint_goal(
                self._home_joint_positions, target_name
            )
        elif self._target_stage in {"slot_pick_approach", "vision_meter_approach"}:
            object_id = str(self._attachment["object_id"])
            if object_id in self._attached_object_ids:
                self.get_logger().error(
                    "Cannot approach an installed meter while "
                    f"'{object_id}' is already attached"
                )
                return False
            target_name = (
                "vision_meter_approach"
                if self._target_stage == "vision_meter_approach"
                else "meter_slot_pick_approach"
            )
            target_pose = (
                self._vision_meter_approach_pose()
                if self._target_stage == "vision_meter_approach"
                else self._slot_pose("approach")
            )
            maximum_travel = float(
                self._planning[
                    "vision_approach_max_joint_travel_rad"
                    if self._target_stage == "vision_meter_approach"
                    else "slot_transfer_max_joint_travel_rad"
                ]
            )
            target_joints = (
                self._solve_best_ik(
                    target_pose,
                    self._scene_joint_positions,
                    ranking="nearest",
                    maximum_allowed=maximum_travel,
                )
                if target_pose is not None
                else None
            )
            trajectory = None
            if target_joints is not None:
                candidate = self._plan_ptp_joint_goal(
                    target_joints,
                    f"{target_name}_ptp",
                )
                if candidate is not None and self._trajectory_is_safe(
                    candidate, maximum_travel
                ):
                    trajectory = candidate
                for attempt in range(1, 11):
                    if trajectory is not None:
                        break
                    candidate = self._plan_joint_goal(
                        target_joints,
                        f"{target_name}_attempt_{attempt}",
                    )
                    if candidate is None:
                        continue
                    if self._trajectory_is_safe(candidate, maximum_travel):
                        trajectory = candidate
                        break
                    self.get_logger().warn(
                        "Discarding unsafe slot-approach candidate and "
                        "continuing the search"
                    )
        elif self._target_stage == "slot_pick_insert":
            object_id = str(self._attachment["object_id"])
            if object_id in self._attached_object_ids:
                self.get_logger().error(
                    "Cannot enter the installed-meter grasp pose while "
                    f"'{object_id}' is already attached"
                )
                return False
            target_name = "meter_slot_pick_insert"
            target_pose = self._slot_pose()
            trajectory = (
                self._plan_linear_pose_goal(target_pose, target_name)
                if target_pose is not None
                else None
            )
        elif self._target_stage == "slot_extract":
            target_name = "meter_slot_extract"
            target_pose = self._slot_pose("approach")
            trajectory = (
                self._plan_linear_pose_goal(target_pose, target_name)
                if target_pose is not None
                else None
            )
        elif self._target_stage == "slot_uninstall_transfer":
            target_name = "meter_slot_uninstall_transfer"
            flat_lift_joints = self._solve_ik(
                self._pose("lift"),
                self._planning_seed(),
            )
            target_joints = (
                self._semantic_flip_goal(flat_lift_joints)
                if flat_lift_joints is not None
                else None
            )
            trajectory = (
                self._plan_joint_goal(target_joints, target_name)
                if target_joints is not None
                else None
            )
        elif self._target_stage == "semantic_unflip":
            target_name = "meter_semantic_unflip"
            flat_lift_joints = self._solve_ik(
                self._pose("lift"),
                self._planning_seed(),
            )
            target_joints = self._semantic_flip_goal(
                self._scene_joint_positions
            )
            if (
                flat_lift_joints is not None
                and target_joints is not None
            ):
                semantic_joint = str(self._semantic_flip["joint_name"])
                target_joints[semantic_joint] = flat_lift_joints[
                    semantic_joint
                ]
            else:
                target_joints = None
            trajectory = (
                self._plan_joint_goal(target_joints, target_name)
                if target_joints is not None
                else None
            )
        elif self._target_stage == "desktop_place_descend":
            target_name = "meter_desktop_place_descend"
            trajectory = self._plan_linear_pose_goal(
                self._pose(),
                target_name,
            )
        elif self._target_stage in {
            "place_retreat",
            "slot_retreat",
            "desktop_place_retreat",
        }:
            object_id = str(self._attachment["object_id"])
            if object_id in self._attached_object_ids:
                self.get_logger().error(
                    f"Cannot retreat: '{object_id}' is still attached"
                )
                return False
            if self._target_stage == "slot_retreat":
                target_name = "meter_slot_retreat"
                target_pose = self._slot_pose("approach")
            elif self._target_stage == "desktop_place_retreat":
                target_name = "meter_desktop_place_retreat"
                target_pose = self._pose("approach")
            else:
                target_name = "meter_place_retreat"
                target_pose = self._place_pose("approach")
            trajectory = (
                self._plan_linear_pose_goal(target_pose, target_name)
                if target_pose is not None
                else None
            )
        elif self._target_stage == "slot_insert":
            target_name = "meter_slot_insert"
            target_pose = self._slot_pose()
            trajectory = (
                self._plan_linear_pose_goal(target_pose, target_name)
                if target_pose is not None
                else None
            )
        elif self._target_stage == "semantic_flip":
            target_name = "meter_semantic_flip"
            target_joints = self._semantic_flip_goal(
                self._scene_joint_positions
            )
            trajectory = (
                self._plan_joint_goal(target_joints, target_name)
                if target_joints is not None
                else None
            )
        elif self._target_stage == "slot_transfer":
            target_name = "meter_slot_transfer"
            target_pose = self._slot_pose("approach")
            target_joints = (
                self._solve_best_ik(target_pose, self._scene_joint_positions)
                if target_pose is not None
                else None
            )
            trajectory = (
                self._plan_joint_goal(target_joints, target_name)
                if target_joints is not None
                else None
            )
        elif self._target_stage == "place_descend":
            target_name = "meter_place_descend"
            trajectory = self._plan_linear_pose_goal(
                self._place_pose(), target_name
            )
        elif self._target_stage == "place_transfer":
            target_name = "meter_place_transfer"
            trajectory = self._plan_linear_pose_goal(
                self._place_pose("approach"), target_name
            )
        elif self._target_stage == "lift":
            target_name = "meter_vertical_lift"
            trajectory = self._plan_linear_pose_goal(
                self._pose("lift"), target_name
            )
        elif self._target_stage == "pregrasp":
            target_name = "meter_pregrasp"
            target_joints = self._solve_best_ik(
                self._pose("approach"),
                self._scene_joint_positions,
                ranking="nearest",
                maximum_allowed=float(
                    self._planning["pregrasp_max_joint_travel_rad"]
                ),
            )
            if target_joints is None:
                return False
            trajectory = self._plan_joint_goal(target_joints, target_name)
        else:
            target_name = "meter_top_grasp_candidate"
            trajectory = self._plan_linear_pose_goal(
                self._pose(), target_name
            )

        if trajectory is None:
            return False
        if self._target_stage == "custom_flange_pose":
            maximum_travel = float(
                self._planning["custom_pose_max_joint_travel_rad"]
            )
        elif self._target_stage == "return_home":
            maximum_travel = float(
                self._planning["return_home_max_joint_travel_rad"]
            )
        elif self._target_stage == "pregrasp":
            maximum_travel = float(
                self._planning["pregrasp_max_joint_travel_rad"]
            )
        elif self._target_stage == "vision_meter_approach":
            maximum_travel = float(
                self._planning["vision_approach_max_joint_travel_rad"]
            )
        elif self._target_stage in {
            "slot_transfer",
            "slot_pick_approach",
            "slot_uninstall_transfer",
        }:
            maximum_travel = float(
                self._planning["slot_transfer_max_joint_travel_rad"]
            )
        elif self._target_stage in {
            "semantic_flip",
            "semantic_unflip",
        }:
            maximum_travel = float(
                self._planning["semantic_flip_max_joint_travel_rad"]
            )
        else:
            maximum_travel = None
        if not self._trajectory_is_safe(trajectory, maximum_travel):
            return False
        if self._trajectory_output_path:
            if not self._export_validated_trajectory(trajectory, target_name):
                return False
        if not self._execute:
            self.get_logger().info(f"Plan-only safety check complete: '{target_name}'")
            return True
        return self._execute_planned_trajectory(trajectory, target_name)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PickSequence()
    success = False
    try:
        success = node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    if not success:
        raise RuntimeError("Single-arm pick sequence failed")
