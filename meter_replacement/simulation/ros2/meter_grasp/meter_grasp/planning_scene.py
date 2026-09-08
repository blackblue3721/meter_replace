#!/usr/bin/env python3
"""Add configured workcell geometry to the MoveIt planning scene."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, ObjectColor, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive


PACKAGE_NAME = "meter_grasp"


def _vector(config: dict[str, Any], key: str, length: int) -> list[float]:
    value = config.get(key)
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{key} must contain exactly {length} numbers")
    return [float(item) for item in value]


def _collision_object(
    config: dict[str, Any], frame_id: str
) -> tuple[CollisionObject, ObjectColor]:
    collision_object = CollisionObject()
    collision_object.header.frame_id = frame_id
    collision_object.id = str(config["name"])
    collision_object.operation = CollisionObject.ADD

    geometry = str(config.get("geometry", "box"))
    primitive = SolidPrimitive()
    if geometry == "box":
        size = _vector(config, "size_m", 3)
        if any(dimension <= 0.0 for dimension in size):
            raise ValueError(f"{config['name']}.size_m values must all be positive")
        primitive.type = SolidPrimitive.BOX
        # SolidPrimitive box dimensions are full X/Y/Z lengths in metres.
        primitive.dimensions = size
    elif geometry == "cylinder":
        height = float(config["height_m"])
        radius = float(config["radius_m"])
        if height <= 0.0 or radius <= 0.0:
            raise ValueError(
                f"{config['name']} cylinder height_m/radius_m must be positive"
            )
        primitive.type = SolidPrimitive.CYLINDER
        # MoveIt cylinder dimensions are [height, radius], with its axis on Z.
        primitive.dimensions = [height, radius]
    else:
        raise ValueError(f"Unsupported collision geometry: {geometry}")
    collision_object.primitives.append(primitive)

    position = _vector(config, "position_m", 3)
    quaternion = _vector(config, "quaternion_wxyz", 4)
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = position
    pose.orientation.w = quaternion[0]
    pose.orientation.x = quaternion[1]
    pose.orientation.y = quaternion[2]
    pose.orientation.z = quaternion[3]
    collision_object.primitive_poses.append(pose)

    rgba = _vector(config, "rgba", 4)
    color = ObjectColor()
    color.id = collision_object.id
    color.color.r, color.color.g, color.color.b, color.color.a = rgba
    return collision_object, color


def _configured_collision_objects(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return enabled scene objects, accepting the original v1 schema too."""
    if "collision_objects" in config:
        objects = config["collision_objects"]
        if not isinstance(objects, list) or not objects:
            raise ValueError("collision_objects must be a non-empty list")
        enabled_objects = [
            item for item in objects if isinstance(item, dict) and item.get("enabled", True)
        ]
    else:
        enabled_objects = [config["floor"], config["meter"]]

    names = [str(item.get("name", "")) for item in enabled_objects]
    if any(not name for name in names):
        raise ValueError("Every collision object must have a non-empty name")
    if len(names) != len(set(names)):
        raise ValueError("Collision object names must be unique")
    return enabled_objects


class MeterPlanningScene(Node):
    """Apply shared workcell geometry to MoveIt's monitored planning scene."""

    def __init__(self) -> None:
        super().__init__("meter_planning_scene")
        default_config = (
            Path(get_package_share_directory(PACKAGE_NAME)) / "config" / "scene.json"
        )
        self.declare_parameter("scene_config", str(default_config))
        self.declare_parameter("planning_frame", "world")
        self.declare_parameter("apply_service", "/apply_planning_scene")

        config_path = Path(self.get_parameter("scene_config").value)
        with config_path.open(encoding="utf-8") as stream:
            self._config = json.load(stream)

        service_name = str(self.get_parameter("apply_service").value)
        self._client = self.create_client(ApplyPlanningScene, service_name)
        self.get_logger().info(f"Loaded scene configuration: {config_path}")

    def apply_scene(self) -> bool:
        frame_id = str(self.get_parameter("planning_frame").value)
        objects_and_colors = [
            _collision_object(item, frame_id)
            for item in _configured_collision_objects(self._config)
        ]

        scene = PlanningScene()
        scene.name = str(
            self._config.get("scene_name", "single_arm_workcell_scene")
        )
        scene.is_diff = True
        scene.robot_state.is_diff = True
        for collision_object, color in objects_and_colors:
            scene.world.collision_objects.append(collision_object)
            scene.object_colors.append(color)

        self.get_logger().info("Waiting for MoveIt planning-scene service...")
        if not self._client.wait_for_service(timeout_sec=30.0):
            self.get_logger().error("ApplyPlanningScene service was not available")
            return False

        request = ApplyPlanningScene.Request()
        request.scene = scene
        future = self._client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if not future.done() or future.result() is None:
            self.get_logger().error("ApplyPlanningScene request timed out")
            return False
        if not future.result().success:
            self.get_logger().error("MoveIt rejected the planning-scene update")
            return False

        object_ids = [item.id for item, _ in objects_and_colors]
        self.get_logger().info(f"Added collision objects in {frame_id}: {object_ids}")
        return True


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MeterPlanningScene()
    try:
        if not node.apply_scene():
            raise RuntimeError("Failed to add workcell geometry to MoveIt planning scene")
    finally:
        node.destroy_node()
        rclpy.shutdown()
