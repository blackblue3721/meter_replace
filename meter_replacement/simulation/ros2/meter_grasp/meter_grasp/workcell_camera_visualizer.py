#!/usr/bin/env python3
"""Publish the fixed workcell D435 CAD and its provisional TF frame.

The detailed mesh is visual-only. MoveIt receives the camera body, round
vertical post and horizontal rail as simplified collision objects from
``config/scene.json`` through ``planning_scene.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray


PACKAGE_NAME = "meter_grasp"


def _vector(config: dict[str, Any], key: str, length: int) -> list[float]:
    value = config.get(key)
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{key} must contain exactly {length} numbers")
    return [float(item) for item in value]


class WorkcellCameraVisualizer(Node):
    """Publish one fixed D435 mesh and world-to-camera mounting transform."""

    def __init__(self) -> None:
        super().__init__("workcell_camera_visualizer")
        default_config = (
            Path(get_package_share_directory(PACKAGE_NAME))
            / "config"
            / "scene.json"
        )
        self.declare_parameter("scene_config", str(default_config))

        config_path = Path(str(self.get_parameter("scene_config").value))
        with config_path.open(encoding="utf-8") as stream:
            config = json.load(stream)

        frames = config["frames"]
        self._camera_frame = frames["workcell_d435_link"]
        self._visual = config["workcell_camera_visual"]
        self._mesh_pose = self._visual["mesh_pose_in_camera_frame"]

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._publisher = self.create_publisher(
            MarkerArray,
            str(self._visual["marker_topic"]),
            qos,
        )
        self._tf_broadcaster = StaticTransformBroadcaster(self)
        self._publish_transform()
        self._publish_marker()
        # Re-publish briefly for RViz instances that start after this node.
        self._timer = self.create_timer(1.0, self._publish_marker)
        self.get_logger().info(
            "D435 CAD and provisional fixed frame loaded from "
            f"{config_path}; MoveIt collision uses simplified boxes"
        )

    def _publish_transform(self) -> None:
        position = _vector(self._camera_frame, "position_m", 3)
        quaternion = _vector(self._camera_frame, "quaternion_wxyz", 4)
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = str(self._camera_frame["parent_frame"])
        transform.child_frame_id = str(self._visual["frame_id"])
        transform.transform.translation.x = position[0]
        transform.transform.translation.y = position[1]
        transform.transform.translation.z = position[2]
        transform.transform.rotation.w = quaternion[0]
        transform.transform.rotation.x = quaternion[1]
        transform.transform.rotation.y = quaternion[2]
        transform.transform.rotation.z = quaternion[3]
        self._tf_broadcaster.sendTransform(transform)

    def _publish_marker(self) -> None:
        position = _vector(self._mesh_pose, "position_m", 3)
        quaternion = _vector(self._mesh_pose, "quaternion_wxyz", 4)
        scale = _vector(self._visual, "mesh_scale", 3)
        rgba = _vector(self._visual, "rgba", 4)

        marker = Marker()
        marker.header.frame_id = str(self._visual["frame_id"])
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "workcell_d435_cad"
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        marker.pose.orientation.w = quaternion[0]
        marker.pose.orientation.x = quaternion[1]
        marker.pose.orientation.y = quaternion[2]
        marker.pose.orientation.z = quaternion[3]
        marker.scale.x = scale[0]
        marker.scale.y = scale[1]
        marker.scale.z = scale[2]
        marker.color.r = rgba[0]
        marker.color.g = rgba[1]
        marker.color.b = rgba[2]
        marker.color.a = rgba[3]
        marker.mesh_resource = str(self._visual["mesh_resource"])
        marker.mesh_use_embedded_materials = True
        marker.frame_locked = True

        message = MarkerArray()
        message.markers.append(marker)
        self._publisher.publish(message)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = WorkcellCameraVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
