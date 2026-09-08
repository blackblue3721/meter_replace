#!/usr/bin/env python3
"""Attach the configured meter collision object to the simulated gripper."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    PlanningScene,
    PlanningSceneComponents,
)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from rclpy.node import Node


PACKAGE_NAME = "meter_grasp"


class MeterAttachment(Node):
    """Move a world collision object into the robot's attached-body state."""

    def __init__(self) -> None:
        super().__init__("meter_attachment")
        default_config = (
            Path(get_package_share_directory(PACKAGE_NAME)) / "config" / "poses.yaml"
        )
        self.declare_parameter("poses_config", str(default_config))
        self.declare_parameter("get_scene_service", "/get_planning_scene")
        self.declare_parameter("apply_scene_service", "/apply_planning_scene")

        config_path = Path(str(self.get_parameter("poses_config").value))
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        self._attachment = config["attachment"]
        self._timeout = float(self._attachment["service_timeout_s"])
        self._already_attached = False

        self._get_scene = self.create_client(
            GetPlanningScene,
            str(self.get_parameter("get_scene_service").value),
        )
        self._apply_scene = self.create_client(
            ApplyPlanningScene,
            str(self.get_parameter("apply_scene_service").value),
        )
        self.get_logger().info(f"Loaded attachment configuration: {config_path}")

    def _world_object(self) -> CollisionObject | None:
        if not self._get_scene.wait_for_service(timeout_sec=self._timeout):
            self.get_logger().error("GetPlanningScene service was not available")
            return None

        request = GetPlanningScene.Request()
        request.components.components = (
            PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
            | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
        )
        future = self._get_scene.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self._timeout)
        if not future.done() or future.result() is None:
            self.get_logger().error("Planning-scene query timed out")
            return None

        object_id = str(self._attachment["object_id"])
        for attached_object in (
            future.result().scene.robot_state.attached_collision_objects
        ):
            if attached_object.object.id == object_id:
                self._already_attached = True
                self.get_logger().info(
                    f"'{object_id}' is already attached to "
                    f"'{attached_object.link_name}'"
                )
                return None
        for collision_object in future.result().scene.world.collision_objects:
            if collision_object.id == object_id:
                return deepcopy(collision_object)
        self.get_logger().error(f"World collision object '{object_id}' was not found")
        return None

    def attach(self) -> bool:
        collision_object = self._world_object()
        if collision_object is None:
            return self._already_attached
        if not self._apply_scene.wait_for_service(timeout_sec=self._timeout):
            self.get_logger().error("ApplyPlanningScene service was not available")
            return False

        object_id = str(self._attachment["object_id"])
        attached = AttachedCollisionObject()
        attached.link_name = str(self._attachment["link_name"])
        attached.object = collision_object
        attached.object.operation = CollisionObject.ADD
        attached.touch_links = [
            str(link_name) for link_name in self._attachment["touch_links"]
        ]

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects.append(attached)

        request = ApplyPlanningScene.Request()
        request.scene = scene
        future = self._apply_scene.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self._timeout)
        if not future.done() or future.result() is None:
            self.get_logger().error("Attach planning-scene update timed out")
            return False
        if not future.result().success:
            self.get_logger().error("MoveIt rejected the attachment update")
            return False

        self.get_logger().info(
            f"Attached '{object_id}' to '{attached.link_name}' with touch links "
            f"{attached.touch_links}"
        )
        return True

    def detach(self) -> bool:
        if not self._get_scene.wait_for_service(timeout_sec=self._timeout):
            self.get_logger().error("GetPlanningScene service was not available")
            return False

        request = GetPlanningScene.Request()
        request.components.components = (
            PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
            | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
        )
        future = self._get_scene.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self._timeout)
        if not future.done() or future.result() is None:
            self.get_logger().error("Planning-scene query timed out")
            return False

        object_id = str(self._attachment["object_id"])
        scene_state = future.result().scene
        attached_object = next(
            (
                item
                for item in scene_state.robot_state.attached_collision_objects
                if item.object.id == object_id
            ),
            None,
        )
        if attached_object is None:
            world_ids = [
                item.id for item in scene_state.world.collision_objects
            ]
            if object_id in world_ids:
                self.get_logger().info(f"'{object_id}' is already detached")
                return True
            self.get_logger().error(
                f"'{object_id}' is neither attached nor present in the world"
            )
            return False

        if not self._apply_scene.wait_for_service(timeout_sec=self._timeout):
            self.get_logger().error("ApplyPlanningScene service was not available")
            return False

        remove_attachment = AttachedCollisionObject()
        remove_attachment.link_name = attached_object.link_name
        remove_attachment.object.id = object_id
        remove_attachment.object.operation = CollisionObject.REMOVE

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects.append(remove_attachment)

        apply_request = ApplyPlanningScene.Request()
        apply_request.scene = scene
        apply_future = self._apply_scene.call_async(apply_request)
        rclpy.spin_until_future_complete(
            self, apply_future, timeout_sec=self._timeout
        )
        if not apply_future.done() or apply_future.result() is None:
            self.get_logger().error("Detach planning-scene update timed out")
            return False
        if not apply_future.result().success:
            self.get_logger().error("MoveIt rejected the detach update")
            return False

        self.get_logger().info(
            f"Detached '{object_id}' from '{attached_object.link_name}'"
        )
        return True


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MeterAttachment()
    success = False
    try:
        success = node.attach()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    if not success:
        raise RuntimeError("Failed to attach meter collision object")


def detach_main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MeterAttachment()
    success = False
    try:
        success = node.detach()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    if not success:
        raise RuntimeError("Failed to detach meter collision object")
