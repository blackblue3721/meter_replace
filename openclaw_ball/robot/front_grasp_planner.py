#!/usr/bin/env python3
"""MoveIt plan-only front approach for one frozen D435 cube target."""
from copy import deepcopy
import json
import math
from pathlib import Path

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive
from tf2_geometry_msgs import do_transform_pose
from visualization_msgs.msg import Marker

from meter_grasp.pick_sequence import PickSequence
from grasp_geometry import nearest_equivalent


class FrontGraspPlanner(PickSequence):
    def __init__(self):
        super().__init__("cube_front_grasp_planner", announce_target_stage=False)
        self.declare_parameter("cube_center_base_mm", [0.0, 0.0, 0.0])
        self.declare_parameter("contact_base_mm", [0.0, 0.0, 0.0])
        self.declare_parameter("pregrasp_base_mm", [0.0, 0.0, 0.0])
        self.declare_parameter("cube_size_m", 0.025)
        self.declare_parameter("pregrasp_output_path", "")
        self.declare_parameter("contact_output_path", "")
        self.declare_parameter("wrist_output_path", "")
        self.declare_parameter("horizontal_continuation", False)
        self.declare_parameter("manual_correction", False)
        self.declare_parameter("manual_retreat", False)
        self.declare_parameter("visual_safe_retreat", False)
        self.declare_parameter("complete_cycle", False)
        self.declare_parameter("safe_only", False)
        self.declare_parameter("operator_contact_correction", False)
        self.declare_parameter("operator_correction_output_path", "")
        self.declare_parameter("retreat_output_path", "")
        self.declare_parameter("safe_output_path", "")
        self.declare_parameter("default_safe_pose_path", "")
        self.declare_parameter("approach_direction_base", [0.0, 0.0, 1.0])
        self.declare_parameter("forward_correction_mm", 70.0)
        self.declare_parameter("down_correction_mm", 20.0)
        self.declare_parameter("forward_output_path", "")
        self.declare_parameter("down_output_path", "")
        self.declare_parameter("up_output_path", "")
        self.declare_parameter("back_output_path", "")
        self._apply_scene = self.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self._marker = self.create_publisher(Marker, "/cube_grasp_target", 1)

    def _pose_in_planning_frame(self, xyz_mm, orientation):
        transform = self._wait_for_transform(
            str(self._frames["planning_frame"]),
            str(self.get_parameter("custom_base_frame").value),
        )
        if transform is None:
            return None
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = [float(v) / 1000.0 for v in xyz_mm]
        pose.orientation.w = 1.0
        result = do_transform_pose(pose, transform)
        result.orientation = deepcopy(orientation)
        return result

    def _apply_cube(self, pose, operation):
        if not self._apply_scene.wait_for_service(timeout_sec=10):
            raise RuntimeError("/apply_planning_scene unavailable")
        obj = CollisionObject()
        obj.header.frame_id = str(self._frames["planning_frame"])
        obj.id = "openclaw_target_cube"
        obj.operation = operation
        if operation == CollisionObject.ADD:
            primitive = SolidPrimitive()
            primitive.type = SolidPrimitive.BOX
            size = float(self.get_parameter("cube_size_m").value)
            primitive.dimensions = [size, size, size]
            obj.primitives = [primitive]
            obj.primitive_poses = [pose]
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [obj]
        future = self._apply_scene.call_async(ApplyPlanningScene.Request(scene=scene))
        rclpy.spin_until_future_complete(self, future, timeout_sec=10)
        if not future.done() or not future.result() or not future.result().success:
            raise RuntimeError("failed to update cube collision object")

    def _publish_marker(self, pose):
        marker = Marker()
        marker.header.frame_id = str(self._frames["planning_frame"])
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns, marker.id, marker.type, marker.action = "openclaw_cube", 0, Marker.CUBE, Marker.ADD
        marker.pose = pose
        size = float(self.get_parameter("cube_size_m").value)
        marker.scale.x = marker.scale.y = marker.scale.z = size
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.1, 0.1, 0.85
        marker.lifetime.sec = 30
        self._marker.publish(marker)

    @staticmethod
    def _endpoint(trajectory):
        names = trajectory.joint_trajectory.joint_names
        values = trajectory.joint_trajectory.points[-1].positions
        return dict(zip(names, values))

    def _unwrap_goal_near_current(self, goal):
        """Choose each revolute-joint equivalent nearest the live start state."""
        limits = self._planning["joint_position_limits_rad"]
        margin = float(self._planning["joint_limit_margin_rad"])
        result = {}
        for name, angle in goal.items():
            lower, upper = map(float, limits[name])
            try:
                result[name] = nearest_equivalent(
                    angle, self._scene_joint_positions[name], lower, upper, margin
                )
            except ValueError as exc:
                raise RuntimeError(f"no in-limit equivalent angle for {name}") from exc
        return result

    def _nearest_front_orientation(self, pose, reference):
        """Use the gripper's 180-degree symmetric pose without flipping J1."""
        base = (
            reference.x, reference.y, reference.z, reference.w,
        )
        orientations = [base, self._quaternion_multiply(base, (0.0, 0.0, 1.0, 0.0))]
        best = None
        for orientation in orientations:
            candidate = deepcopy(pose)
            (
                candidate.orientation.x,
                candidate.orientation.y,
                candidate.orientation.z,
                candidate.orientation.w,
            ) = orientation
            joints = self._solve_best_ik(
                candidate,
                self._scene_joint_positions,
                ranking="nearest",
                maximum_allowed=float(
                    self._planning["slot_transfer_max_joint_travel_rad"]
                ),
            )
            if joints is None:
                continue
            deltas = {
                name: abs(value - self._scene_joint_positions[name])
                for name, value in joints.items()
            }
            # The old 3.5 rad vision allowance exists only for a known J6
            # wrist change. Never use it to permit a base/elbow branch flip.
            if deltas["joint1"] > 1.0:
                continue
            score = (max(deltas.values()), sum(deltas.values()))
            if best is None or score < best[0]:
                best = score, candidate, joints
        if best is None:
            return None, None
        return best[1], best[2]

    def plan(self):
        if self._execute:
            raise RuntimeError("front grasp planner is plan-only")
        if not self._scene_contains_meter():
            raise RuntimeError("planning scene validation failed")

        # Reuse only the already validated cabinet-front tool orientation;
        # cube positions are independent of the old meter-slot correction gate.
        reference = self._slot_pose("approach")
        if reference is None:
            raise RuntimeError("could not resolve cabinet-front orientation")
        center = list(self.get_parameter("cube_center_base_mm").value)
        pregrasp = list(self.get_parameter("pregrasp_base_mm").value)
        center_pose = self._pose_in_planning_frame(center, reference.orientation)
        pregrasp_pose = self._pose_in_planning_frame(pregrasp, reference.orientation)
        if center_pose is None or pregrasp_pose is None:
            raise RuntimeError("could not transform cube poses")

        self._publish_marker(center_pose)
        self._apply_cube(center_pose, CollisionObject.ADD)
        pregrasp_pose, joints = self._nearest_front_orientation(
            pregrasp_pose, reference.orientation
        )
        if joints is None:
            raise RuntimeError("no same-side collision-free pregrasp IK")
        center_pose.orientation = deepcopy(pregrasp_pose.orientation)
        pregrasp_trajectory = self._plan_ptp_joint_goal(joints, "cube_front_pregrasp")
        if pregrasp_trajectory is None or not self._trajectory_is_safe(pregrasp_trajectory):
            raise RuntimeError("pregrasp planning failed")
        self._trajectory_output_path = str(self.get_parameter("pregrasp_output_path").value)
        if not self._export_validated_trajectory(pregrasp_trajectory, "cube_front_pregrasp"):
            raise RuntimeError("pregrasp export failed")

        self._scene_joint_positions.update(self._endpoint(pregrasp_trajectory))
        # Intentional contact segment: keep the RViz marker, remove collision geometry.
        self._apply_cube(center_pose, CollisionObject.REMOVE)
        contact_trajectory = self._plan_linear_pose_goal(center_pose, "cube_front_contact")
        if contact_trajectory is None or not self._trajectory_is_safe(contact_trajectory):
            raise RuntimeError("linear contact planning failed")
        self._trajectory_output_path = str(self.get_parameter("contact_output_path").value)
        if not self._export_validated_trajectory(contact_trajectory, "cube_front_contact"):
            raise RuntimeError("contact export failed")
        self.get_logger().info("CUBE FRONT GRASP PLAN-ONLY PASSED")

    def plan_horizontal_continuation(self):
        """From the reviewed pregrasp, turn J6 90 degrees then approach."""
        if self._execute:
            raise RuntimeError("front grasp planner is plan-only")
        if not self._scene_contains_meter():
            raise RuntimeError("planning scene validation failed")
        reference = self._slot_pose("approach")
        if reference is None:
            raise RuntimeError("could not resolve horizontal grasp orientation")
        center = list(self.get_parameter("cube_center_base_mm").value)
        center_pose = self._pose_in_planning_frame(center, reference.orientation)
        if center_pose is None:
            raise RuntimeError("could not resolve horizontal grasp pose")

        base = (reference.orientation.x, reference.orientation.y,
                reference.orientation.z, reference.orientation.w)
        # The selected pregrasp used the symmetric 180-degree tool roll.
        horizontal = self._quaternion_multiply(
            base, (0.0, 0.0, math.sin(3.0 * math.pi / 4.0),
                   math.cos(3.0 * math.pi / 4.0))
        )
        (center_pose.orientation.x, center_pose.orientation.y,
         center_pose.orientation.z, center_pose.orientation.w) = horizontal
        self._publish_marker(center_pose)
        self._apply_cube(center_pose, CollisionObject.ADD)

        wrist_goal = {
            name: self._scene_joint_positions[name]
            for name in self._planning_seed()
        }
        wrist_goal["joint6"] += math.pi / 2.0
        wrist = self._plan_ptp_joint_goal(wrist_goal, "cube_horizontal_wrist_turn")
        maximum = float(self._planning["semantic_flip_max_joint_travel_rad"])
        if wrist is None or not self._trajectory_is_safe(wrist, maximum):
            raise RuntimeError("horizontal wrist-turn planning failed")
        self._trajectory_output_path = str(self.get_parameter("wrist_output_path").value)
        if not self._export_validated_trajectory(wrist, "cube_horizontal_wrist_turn"):
            raise RuntimeError("wrist-turn export failed")

        self._scene_joint_positions.update(self._endpoint(wrist))
        self._apply_cube(center_pose, CollisionObject.REMOVE)
        contact = self._plan_linear_pose_goal(center_pose, "cube_horizontal_contact")
        if contact is None or not self._trajectory_is_safe(contact):
            raise RuntimeError("horizontal contact planning failed")
        self._trajectory_output_path = str(self.get_parameter("contact_output_path").value)
        if not self._export_validated_trajectory(contact, "cube_horizontal_contact"):
            raise RuntimeError("horizontal contact export failed")
        self.get_logger().info("HORIZONTAL CUBE PLAN-ONLY PASSED")

    def plan_manual_correction(self):
        """Plan operator-measured forward then base-down Cartesian corrections."""
        if self._execute or not self._scene_contains_meter():
            raise RuntimeError("manual correction requires a valid plan-only scene")
        reference = self._slot_pose("approach")
        if reference is None:
            raise RuntimeError("could not resolve horizontal grasp orientation")
        center = list(map(float, self.get_parameter("cube_center_base_mm").value))
        direction = list(map(float, self.get_parameter("approach_direction_base").value))
        forward_mm = float(self.get_parameter("forward_correction_mm").value)
        down_mm = float(self.get_parameter("down_correction_mm").value)
        forward = [center[i] + direction[i] * forward_mm for i in range(3)]
        down = [forward[0], forward[1], forward[2] - down_mm]
        base = (reference.orientation.x, reference.orientation.y,
                reference.orientation.z, reference.orientation.w)
        horizontal = self._quaternion_multiply(
            base, (0.0, 0.0, math.sin(3.0 * math.pi / 4.0),
                   math.cos(3.0 * math.pi / 4.0))
        )
        orientation = deepcopy(reference.orientation)
        orientation.x, orientation.y, orientation.z, orientation.w = horizontal
        forward_pose = self._pose_in_planning_frame(forward, orientation)
        down_pose = self._pose_in_planning_frame(down, orientation)
        if forward_pose is None or down_pose is None:
            raise RuntimeError("could not transform manual correction poses")
        self._publish_marker(down_pose)

        forward_trajectory = self._plan_linear_pose_goal(
            forward_pose, "cube_manual_forward_70mm"
        )
        if forward_trajectory is None or not self._trajectory_is_safe(forward_trajectory):
            raise RuntimeError("manual forward planning failed")
        self._trajectory_output_path = str(self.get_parameter("forward_output_path").value)
        if not self._export_validated_trajectory(forward_trajectory, "cube_manual_forward_70mm"):
            raise RuntimeError("manual forward export failed")
        self._scene_joint_positions.update(self._endpoint(forward_trajectory))

        down_trajectory = self._plan_linear_pose_goal(down_pose, "cube_manual_down_20mm")
        if down_trajectory is None or not self._trajectory_is_safe(down_trajectory):
            raise RuntimeError("manual down planning failed")
        self._trajectory_output_path = str(self.get_parameter("down_output_path").value)
        if not self._export_validated_trajectory(down_trajectory, "cube_manual_down_20mm"):
            raise RuntimeError("manual down export failed")
        self.get_logger().info("MANUAL CUBE CORRECTION PLAN-ONLY PASSED")

    def plan_manual_retreat(self):
        """Retrace the measured correction with new Cartesian plans."""
        if self._execute or not self._scene_contains_meter():
            raise RuntimeError("manual retreat requires a valid plan-only scene")
        reference = self._slot_pose("approach")
        if reference is None:
            raise RuntimeError("could not resolve horizontal retreat orientation")
        center = list(map(float, self.get_parameter("cube_center_base_mm").value))
        direction = list(map(float, self.get_parameter("approach_direction_base").value))
        forward_mm = float(self.get_parameter("forward_correction_mm").value)
        up = [center[i] + direction[i] * forward_mm for i in range(3)]
        base = (reference.orientation.x, reference.orientation.y,
                reference.orientation.z, reference.orientation.w)
        horizontal = self._quaternion_multiply(
            base, (0.0, 0.0, math.sin(3.0 * math.pi / 4.0),
                   math.cos(3.0 * math.pi / 4.0))
        )
        orientation = deepcopy(reference.orientation)
        orientation.x, orientation.y, orientation.z, orientation.w = horizontal
        up_pose = self._pose_in_planning_frame(up, orientation)
        back_pose = self._pose_in_planning_frame(center, orientation)
        if up_pose is None or back_pose is None:
            raise RuntimeError("could not transform manual retreat poses")

        up_trajectory = self._plan_linear_pose_goal(up_pose, "cube_manual_up_20mm")
        if up_trajectory is None or not self._trajectory_is_safe(up_trajectory):
            raise RuntimeError("manual up planning failed")
        self._trajectory_output_path = str(self.get_parameter("up_output_path").value)
        if not self._export_validated_trajectory(up_trajectory, "cube_manual_up_20mm"):
            raise RuntimeError("manual up export failed")
        self._scene_joint_positions.update(self._endpoint(up_trajectory))

        back_trajectory = self._plan_linear_pose_goal(back_pose, "cube_manual_back_70mm")
        if back_trajectory is None or not self._trajectory_is_safe(back_trajectory):
            raise RuntimeError("manual back planning failed")
        self._trajectory_output_path = str(self.get_parameter("back_output_path").value)
        if not self._export_validated_trajectory(back_trajectory, "cube_manual_back_70mm"):
            raise RuntimeError("manual back export failed")
        self.get_logger().info("MANUAL CUBE RETREAT PLAN-ONLY PASSED")

    def plan_visual_safe_retreat(self):
        """Move straight back to the frozen visual pregrasp point."""
        if self._execute or not self._scene_contains_meter():
            raise RuntimeError("visual-safe retreat requires a valid plan-only scene")
        reference = self._slot_pose("approach")
        if reference is None:
            raise RuntimeError("could not resolve retreat orientation")
        pregrasp = list(map(float, self.get_parameter("pregrasp_base_mm").value))
        base = (reference.orientation.x, reference.orientation.y,
                reference.orientation.z, reference.orientation.w)
        horizontal = self._quaternion_multiply(
            base, (0.0, 0.0, math.sin(3.0 * math.pi / 4.0),
                   math.cos(3.0 * math.pi / 4.0))
        )
        orientation = deepcopy(reference.orientation)
        orientation.x, orientation.y, orientation.z, orientation.w = horizontal
        pose = self._pose_in_planning_frame(pregrasp, orientation)
        trajectory = self._plan_linear_pose_goal(pose, "cube_visual_safe_pregrasp")
        if trajectory is None or not self._trajectory_is_safe(trajectory):
            raise RuntimeError("visual-safe retreat planning failed")
        self._trajectory_output_path = str(self.get_parameter("pregrasp_output_path").value)
        if not self._export_validated_trajectory(trajectory, "cube_visual_safe_pregrasp"):
            raise RuntimeError("visual-safe retreat export failed")
        self.get_logger().info("VISUAL SAFE PREGRASP PLAN-ONLY PASSED")

    def plan_complete_cycle(self):
        """Plan a horizontal approach/contact/retreat/safe-return in one scene."""
        if self._execute or not self._scene_contains_meter():
            raise RuntimeError("complete cycle requires a valid plan-only scene")
        safe_path = Path(str(self.get_parameter("default_safe_pose_path").value))
        safe = json.loads(safe_path.read_text(encoding="utf-8"))
        safe_joints = dict(zip(safe["joint_names"], safe["joint_positions_rad"]))

        reference = self._slot_pose("approach")
        if reference is None:
            raise RuntimeError("could not resolve horizontal grasp orientation")
        base = (reference.orientation.x, reference.orientation.y,
                reference.orientation.z, reference.orientation.w)
        horizontal = self._quaternion_multiply(
            base, (0.0, 0.0, math.sin(3.0 * math.pi / 4.0),
                   math.cos(3.0 * math.pi / 4.0))
        )
        center = list(map(float, self.get_parameter("cube_center_base_mm").value))
        contact_point = list(map(float, self.get_parameter("contact_base_mm").value))
        pregrasp = list(map(float, self.get_parameter("pregrasp_base_mm").value))
        orientations = [
            horizontal,
            self._quaternion_multiply(horizontal, (0.0, 0.0, 1.0, 0.0)),
        ]
        candidates = []
        for value in orientations:
            orientation = deepcopy(reference.orientation)
            orientation.x, orientation.y, orientation.z, orientation.w = value
            candidate_pose = self._pose_in_planning_frame(pregrasp, orientation)
            if candidate_pose is None:
                continue
            candidate_joints = self._solve_best_ik(
                candidate_pose, self._scene_joint_positions, ranking="nearest",
                maximum_allowed=4.0 * math.pi,
            )
            if candidate_joints is None:
                continue
            candidate_joints = self._unwrap_goal_near_current(candidate_joints)
            deltas = [
                abs(candidate_joints[name] - float(self._scene_joint_positions[name]))
                for name in candidate_joints
            ]
            candidates.append((max(deltas), sum(deltas), candidate_pose, candidate_joints))
        if not candidates:
            raise RuntimeError("no collision-free horizontal pregrasp IK")
        maximum_delta, _, pregrasp_pose, joints = min(candidates, key=lambda item: item[:2])
        center_pose = self._pose_in_planning_frame(center, pregrasp_pose.orientation)
        contact_pose = self._pose_in_planning_frame(contact_point, pregrasp_pose.orientation)
        if center_pose is None or contact_pose is None or pregrasp_pose is None:
            raise RuntimeError("could not transform cube poses")

        self._publish_marker(center_pose)
        self._apply_cube(center_pose, CollisionObject.ADD)
        allowed = float(self._planning["slot_transfer_max_joint_travel_rad"])
        if maximum_delta > allowed:
            raise RuntimeError(
                f"nearest horizontal pregrasp IK exceeds {allowed:.2f} rad "
                f"({maximum_delta:.3f} rad)"
            )
        pregrasp_trajectory = self._plan_ptp_joint_goal(joints, "cube_cycle_pregrasp")
        if pregrasp_trajectory is None or not self._trajectory_is_safe(pregrasp_trajectory):
            raise RuntimeError("cycle pregrasp planning failed")
        self._trajectory_output_path = str(self.get_parameter("pregrasp_output_path").value)
        if not self._export_validated_trajectory(pregrasp_trajectory, "cube_cycle_pregrasp"):
            raise RuntimeError("cycle pregrasp export failed")
        self._scene_joint_positions.update(self._endpoint(pregrasp_trajectory))

        self._apply_cube(center_pose, CollisionObject.REMOVE)
        contact = self._plan_linear_pose_goal(contact_pose, "cube_cycle_contact")
        if contact is None or not self._trajectory_is_safe(contact):
            raise RuntimeError("cycle contact planning failed")
        self._trajectory_output_path = str(self.get_parameter("contact_output_path").value)
        if not self._export_validated_trajectory(contact, "cube_cycle_contact"):
            raise RuntimeError("cycle contact export failed")
        self._scene_joint_positions.update(self._endpoint(contact))

        retreat = self._plan_linear_pose_goal(pregrasp_pose, "cube_cycle_retreat")
        if retreat is None or not self._trajectory_is_safe(retreat):
            raise RuntimeError("cycle retreat planning failed")
        self._trajectory_output_path = str(self.get_parameter("retreat_output_path").value)
        if not self._export_validated_trajectory(retreat, "cube_cycle_retreat"):
            raise RuntimeError("cycle retreat export failed")
        self._scene_joint_positions.update(self._endpoint(retreat))

        safe_return = self._plan_ptp_joint_goal(safe_joints, "cube_cycle_safe_return")
        if safe_return is None or not self._trajectory_is_safe(safe_return):
            raise RuntimeError("cycle safe-return planning failed")
        self._trajectory_output_path = str(self.get_parameter("safe_output_path").value)
        if not self._export_validated_trajectory(safe_return, "cube_cycle_safe_return"):
            raise RuntimeError("cycle safe-return export failed")
        self.get_logger().info("COMPLETE CUBE CYCLE PLAN-ONLY PASSED")

    def plan_safe_only(self):
        """Plan directly from the live joint state to the saved visual-safe pose."""
        if self._execute or not self._scene_contains_meter():
            raise RuntimeError("safe-only return requires a valid plan-only scene")
        safe = json.loads(Path(str(self.get_parameter("default_safe_pose_path").value)).read_text(encoding="utf-8"))
        goal = dict(zip(safe["joint_names"], safe["joint_positions_rad"]))
        trajectory = self._plan_ptp_joint_goal(goal, "cube_direct_safe_return")
        if trajectory is None or not self._trajectory_is_safe(trajectory):
            raise RuntimeError("direct safe-return planning failed")
        self._trajectory_output_path = str(self.get_parameter("safe_output_path").value)
        if not self._export_validated_trajectory(trajectory, "cube_direct_safe_return"):
            raise RuntimeError("direct safe-return export failed")
        self.get_logger().info("DIRECT SAFE RETURN PLAN-ONLY PASSED")

    def plan_operator_contact_correction(self):
        """Plan the operator-requested 10 mm up and 10 mm horizontal retreat."""
        if self._execute or not self._scene_contains_meter():
            raise RuntimeError("operator correction requires a valid plan-only scene")
        center = list(map(float, self.get_parameter("cube_center_base_mm").value))
        direction = list(map(float, self.get_parameter("approach_direction_base").value))
        horizontal_norm = math.hypot(direction[0], direction[1])
        if horizontal_norm < 1.0e-9:
            raise RuntimeError("approach direction has no horizontal component")
        corrected = [
            center[0] - 10.0 * direction[0] / horizontal_norm,
            center[1] - 10.0 * direction[1] / horizontal_norm,
            center[2] + 10.0,
        ]
        reference = self._slot_pose("approach")
        if reference is None:
            raise RuntimeError("could not resolve correction orientation")
        base = (reference.orientation.x, reference.orientation.y,
                reference.orientation.z, reference.orientation.w)
        horizontal = self._quaternion_multiply(
            base, (0.0, 0.0, math.sin(3.0 * math.pi / 4.0),
                   math.cos(3.0 * math.pi / 4.0))
        )
        orientation = deepcopy(reference.orientation)
        orientation.x, orientation.y, orientation.z, orientation.w = horizontal
        pose = self._pose_in_planning_frame(corrected, orientation)
        if pose is None:
            raise RuntimeError("could not transform operator correction pose")
        trajectory = self._plan_linear_pose_goal(pose, "cube_operator_up10_back10")
        if trajectory is None or not self._trajectory_is_safe(trajectory):
            raise RuntimeError("operator correction planning failed")
        self._trajectory_output_path = str(
            self.get_parameter("operator_correction_output_path").value
        )
        if not self._export_validated_trajectory(trajectory, "cube_operator_up10_back10"):
            raise RuntimeError("operator correction export failed")
        self.get_logger().info(
            f"OPERATOR CORRECTION PLAN-ONLY PASSED: base_mm={corrected}"
        )


def main():
    rclpy.init()
    node = FrontGraspPlanner()
    try:
        if bool(node.get_parameter("safe_only").value):
            node.plan_safe_only()
        elif bool(node.get_parameter("operator_contact_correction").value):
            node.plan_operator_contact_correction()
        elif bool(node.get_parameter("complete_cycle").value):
            node.plan_complete_cycle()
        elif bool(node.get_parameter("visual_safe_retreat").value):
            node.plan_visual_safe_retreat()
        elif bool(node.get_parameter("manual_retreat").value):
            node.plan_manual_retreat()
        elif bool(node.get_parameter("manual_correction").value):
            node.plan_manual_correction()
        elif bool(node.get_parameter("horizontal_continuation").value):
            node.plan_horizontal_continuation()
        else:
            node.plan()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
