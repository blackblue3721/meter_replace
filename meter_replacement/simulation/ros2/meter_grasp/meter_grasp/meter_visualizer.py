#!/usr/bin/env python3
"""Display the detailed electric-meter CAD while MoveIt keeps a box collision.

The detailed SolidWorks STL meshes are visualization-only.  Their pose is
derived from the authoritative MoveIt world/attached collision object, so the
CAD follows the meter when it is attached to the gripper and after it is
placed back into the world.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose
from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import GetPlanningScene
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray


PACKAGE_NAME = "meter_grasp"


def _vector(config: dict[str, Any], key: str, length: int) -> list[float]:
    value = config.get(key)
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{key} must contain exactly {length} numbers")
    return [float(item) for item in value]


def _quaternion_multiply(
    left: list[float], right: list[float]
) -> list[float]:
    """Multiply two quaternions stored in wxyz order."""
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return [
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ]


def _rotate_vector(quaternion: list[float], vector: list[float]) -> list[float]:
    """Rotate one xyz vector by a wxyz quaternion."""
    w, x, y, z = quaternion
    vx, vy, vz = vector
    return [
        (1.0 - 2.0 * (y * y + z * z)) * vx
        + 2.0 * (x * y - z * w) * vy
        + 2.0 * (x * z + y * w) * vz,
        2.0 * (x * y + z * w) * vx
        + (1.0 - 2.0 * (x * x + z * z)) * vy
        + 2.0 * (y * z - x * w) * vz,
        2.0 * (x * z - y * w) * vx
        + 2.0 * (y * z + x * w) * vy
        + (1.0 - 2.0 * (x * x + y * y)) * vz,
    ]


def _compose_ros_poses(parent: Pose, child: Pose) -> Pose:
    """Compose two ROS poses, preserving their parent-to-child order."""
    parent_position = [parent.position.x, parent.position.y, parent.position.z]
    parent_quaternion = [
        parent.orientation.w,
        parent.orientation.x,
        parent.orientation.y,
        parent.orientation.z,
    ]
    child_position = [child.position.x, child.position.y, child.position.z]
    child_quaternion = [
        child.orientation.w,
        child.orientation.x,
        child.orientation.y,
        child.orientation.z,
    ]
    rotated_child = _rotate_vector(parent_quaternion, child_position)
    composed_quaternion = _quaternion_multiply(
        parent_quaternion, child_quaternion
    )

    result = Pose()
    result.position.x = parent_position[0] + rotated_child[0]
    result.position.y = parent_position[1] + rotated_child[1]
    result.position.z = parent_position[2] + rotated_child[2]
    result.orientation.w = composed_quaternion[0]
    result.orientation.x = composed_quaternion[1]
    result.orientation.y = composed_quaternion[2]
    result.orientation.z = composed_quaternion[3]
    return result


def _compose_pose(parent: Pose, child_config: dict[str, Any]) -> Pose:
    """Compose a ROS Pose with a child pose configured in wxyz order."""
    child = Pose()
    child_position = _vector(child_config, "position_m", 3)
    child_quaternion = _vector(child_config, "quaternion_wxyz", 4)
    child.position.x, child.position.y, child.position.z = child_position
    child.orientation.w = child_quaternion[0]
    child.orientation.x = child_quaternion[1]
    child.orientation.y = child_quaternion[2]
    child.orientation.z = child_quaternion[3]
    return _compose_ros_poses(parent, child)


class MeterCadVisualizer(Node):
    """Track the MoveIt meter object and publish its detailed CAD markers."""

    def __init__(self) -> None:
        super().__init__("meter_cad_visualizer")
        default_config = (
            Path(get_package_share_directory(PACKAGE_NAME))
            / "config"
            / "scene.json"
        )
        self.declare_parameter("scene_config", str(default_config))
        self.declare_parameter("marker_topic", "/electric_meter_visual")
        self.declare_parameter("get_scene_service", "/get_planning_scene")
        self.declare_parameter("refresh_period_s", 0.25)

        config_path = Path(str(self.get_parameter("scene_config").value))
        with config_path.open(encoding="utf-8") as stream:
            config = json.load(stream)
        self._meter_id = "electric_meter"
        self._visual = config["meter_template"]["visual_model"]
        self._mesh_pose = self._visual["mesh_pose_in_meter_frame"]
        self._mesh_scale = _vector(self._visual, "mesh_scale", 3)
        self._meshes = self._validated_meshes()

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._publisher = self.create_publisher(
            MarkerArray,
            str(self.get_parameter("marker_topic").value),
            qos,
        )
        self._scene_client = self.create_client(
            GetPlanningScene,
            str(self.get_parameter("get_scene_service").value),
        )
        self._request_future = None
        self._last_location = ""
        self._timer = self.create_timer(
            float(self.get_parameter("refresh_period_s").value),
            self._request_meter_pose,
        )
        self.get_logger().info(
            f"Loaded {len(self._meshes)} CAD meshes from {config_path}"
        )

    def _validated_meshes(self) -> list[dict[str, Any]]:
        meshes = self._visual.get("meshes")
        if not isinstance(meshes, list) or not meshes:
            raise ValueError("meter_template.visual_model.meshes must not be empty")
        for mesh in meshes:
            if not str(mesh.get("name", "")):
                raise ValueError("Every meter visual mesh requires a name")
            resource = str(mesh.get("ros_resource", ""))
            if not resource.startswith("package://"):
                raise ValueError("Every ROS mesh must use a package:// resource")
            _vector(mesh, "rgba", 4)
        return meshes

    def _request_meter_pose(self) -> None:
        if self._request_future is not None and not self._request_future.done():
            return
        if not self._scene_client.service_is_ready():
            return

        request = GetPlanningScene.Request()
        request.components.components = (
            PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
            | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
        )
        self._request_future = self._scene_client.call_async(request)
        self._request_future.add_done_callback(self._handle_scene)

    @staticmethod
    def _geometry_pose(collision_object) -> Pose | None:
        if collision_object.primitive_poses:
            return _compose_ros_poses(
                collision_object.pose,
                collision_object.primitive_poses[0],
            )
        if collision_object.mesh_poses:
            return _compose_ros_poses(
                collision_object.pose,
                collision_object.mesh_poses[0],
            )
        return None

    def _handle_scene(self, future) -> None:
        try:
            response = future.result()
        except Exception as error:  # pragma: no cover - ROS middleware failure
            self.get_logger().warning(f"Planning-scene query failed: {error}")
            return
        if response is None:
            return

        scene = response.scene
        for attached in scene.robot_state.attached_collision_objects:
            if attached.object.id != self._meter_id:
                continue
            pose = self._geometry_pose(attached.object)
            if pose is None:
                return
            frame_id = attached.object.header.frame_id or attached.link_name
            self._publish_meshes(frame_id, pose, f"attached:{attached.link_name}")
            return

        for collision_object in scene.world.collision_objects:
            if collision_object.id != self._meter_id:
                continue
            pose = self._geometry_pose(collision_object)
            if pose is None:
                return
            frame_id = collision_object.header.frame_id or "world"
            self._publish_meshes(frame_id, pose, "world")
            return

    def _publish_meshes(
        self, frame_id: str, meter_pose: Pose, location: str
    ) -> None:
        stamp = self.get_clock().now().to_msg()
        cad_pose = _compose_pose(meter_pose, self._mesh_pose)
        marker_array = MarkerArray()
        for index, mesh in enumerate(self._meshes):
            rgba = _vector(mesh, "rgba", 4)
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = stamp
            marker.ns = "electric_meter_cad"
            marker.id = index
            marker.type = Marker.MESH_RESOURCE
            marker.action = Marker.ADD
            marker.pose = cad_pose
            marker.scale.x = self._mesh_scale[0]
            marker.scale.y = self._mesh_scale[1]
            marker.scale.z = self._mesh_scale[2]
            marker.color.r = rgba[0]
            marker.color.g = rgba[1]
            marker.color.b = rgba[2]
            marker.color.a = rgba[3]
            marker.mesh_resource = str(mesh["ros_resource"])
            marker.mesh_use_embedded_materials = False
            marker.frame_locked = True
            marker_array.markers.append(marker)
        self._publisher.publish(marker_array)

        if location != self._last_location:
            self.get_logger().info(
                f"Displaying CAD meter in frame '{frame_id}' ({location})"
            )
            self._last_location = location


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MeterCadVisualizer()
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
