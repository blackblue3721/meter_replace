#!/usr/bin/env python3
"""Publish non-collision meter-slot frames and RViz markers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import TransformStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray


PACKAGE_NAME = "meter_grasp"
VALID_STATES = {"EMPTY", "OCCUPIED", "RESERVED", "UNKNOWN"}


def slot_state_parameter_name(slot_id: str) -> str:
    """Return the standard ROS parameter used as a slot-state handshake."""
    return f"slot_states.{slot_id}"


def _vector(config: dict[str, Any], key: str, length: int) -> list[float]:
    value = config.get(key)
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{key} must contain exactly {length} numbers")
    return [float(item) for item in value]


class MeterSlotManager(Node):
    """Display configured empty slots without adding them to MoveIt."""

    def __init__(self) -> None:
        super().__init__("meter_slot_manager")
        default_config = (
            Path(get_package_share_directory(PACKAGE_NAME)) / "config" / "scene.json"
        )
        self.declare_parameter("scene_config", str(default_config))
        self.declare_parameter("marker_topic", "/meter_slots")
        self.declare_parameter("republish_period_s", 2.0)

        config_path = Path(str(self.get_parameter("scene_config").value))
        with config_path.open(encoding="utf-8") as stream:
            self._config = json.load(stream)
        self._slots = self._validated_slots()
        configured_colors = self._config["meter_slot_state_rgba"]
        self._state_colors = {
            state: _vector(configured_colors, state, 4)
            for state in VALID_STATES
        }
        self._slot_states = {
            str(slot["id"]): str(slot["initial_state"])
            for slot in self._slots
        }
        self._state_parameters = {
            slot_state_parameter_name(slot_id): slot_id
            for slot_id in self._slot_states
        }
        for parameter_name, slot_id in self._state_parameters.items():
            self.declare_parameter(
                parameter_name,
                self._slot_states[slot_id],
            )
        self.add_on_set_parameters_callback(self._on_set_parameters)
        self._meter_size = _vector(
            self._config["meter_template"], "installed_envelope_size_m", 3
        )

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._marker_publisher = self.create_publisher(
            MarkerArray,
            str(self.get_parameter("marker_topic").value),
            qos,
        )
        self._static_broadcaster = StaticTransformBroadcaster(self)
        self._publish_transforms()
        self._publish_markers()
        self._timer = self.create_timer(
            float(self.get_parameter("republish_period_s").value),
            self._publish_markers,
        )
        self.get_logger().info(
            f"Published {len(self._slots)} non-collision meter slots from "
            f"{config_path}; initial states="
            f"{[slot['initial_state'] for slot in self._slots]}"
        )

    def _on_set_parameters(
        self, parameters: list[Parameter]
    ) -> SetParametersResult:
        updates: dict[str, str] = {}
        for parameter in parameters:
            slot_id = self._state_parameters.get(parameter.name)
            if slot_id is None:
                continue
            if parameter.type_ != Parameter.Type.STRING:
                return SetParametersResult(
                    successful=False,
                    reason=f"{parameter.name} must be a string",
                )
            state = str(parameter.value).upper()
            if state not in VALID_STATES:
                return SetParametersResult(
                    successful=False,
                    reason=(
                        f"{parameter.name} must be one of "
                        f"{sorted(VALID_STATES)}"
                    ),
                )
            updates[slot_id] = state

        for slot_id, state in updates.items():
            previous = self._slot_states[slot_id]
            self._slot_states[slot_id] = state
            self.get_logger().info(
                f"SLOT {slot_id}: {previous} -> {state}"
            )
        if updates:
            self._publish_markers()
        return SetParametersResult(successful=True)

    def _validated_slots(self) -> list[dict[str, Any]]:
        slots = self._config.get("meter_slots")
        if not isinstance(slots, list) or not slots:
            raise ValueError("meter_slots must be a non-empty list")
        slot_ids = [str(slot.get("id", "")) for slot in slots]
        frame_ids = [str(slot.get("frame_id", "")) for slot in slots]
        if any(not item for item in slot_ids + frame_ids):
            raise ValueError("Every meter slot requires id and frame_id")
        if len(slot_ids) != len(set(slot_ids)):
            raise ValueError("Meter slot ids must be unique")
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("Meter slot frame ids must be unique")
        for slot in slots:
            state = str(slot.get("initial_state", "UNKNOWN")).upper()
            if state not in VALID_STATES:
                raise ValueError(
                    f"{slot['id']}.initial_state must be one of {sorted(VALID_STATES)}"
                )
            slot["initial_state"] = state
            _vector(slot, "position_m", 3)
            _vector(slot, "quaternion_wxyz", 4)
            _vector(slot, "marker_rgba", 4)
        return slots

    @staticmethod
    def _transform(
        parent_frame: str,
        child_frame: str,
        position: list[float],
        quaternion_wxyz: list[float],
        stamp,
    ) -> TransformStamped:
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame
        transform.transform.translation.x = position[0]
        transform.transform.translation.y = position[1]
        transform.transform.translation.z = position[2]
        transform.transform.rotation.w = quaternion_wxyz[0]
        transform.transform.rotation.x = quaternion_wxyz[1]
        transform.transform.rotation.y = quaternion_wxyz[2]
        transform.transform.rotation.z = quaternion_wxyz[3]
        return transform

    def _publish_transforms(self) -> None:
        meter_box = self._config["frames"]["meter_box"]
        stamp = self.get_clock().now().to_msg()
        transforms = [
            self._transform(
                str(meter_box["parent_frame"]),
                "meter_box",
                _vector(meter_box, "position_m", 3),
                _vector(meter_box, "quaternion_wxyz", 4),
                stamp,
            )
        ]
        transforms.extend(
            self._transform(
                str(slot["parent_frame"]),
                str(slot["frame_id"]),
                _vector(slot, "position_m", 3),
                _vector(slot, "quaternion_wxyz", 4),
                stamp,
            )
            for slot in self._slots
        )
        self._static_broadcaster.sendTransform(transforms)

    def _publish_markers(self) -> None:
        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()
        for index, slot in enumerate(self._slots):
            state = self._slot_states[str(slot["id"])]
            rgba = self._state_colors[state]

            envelope = Marker()
            envelope.header.frame_id = str(slot["frame_id"])
            envelope.header.stamp = stamp
            envelope.ns = "meter_slot_envelopes"
            envelope.id = index
            envelope.type = Marker.CUBE
            envelope.action = Marker.ADD
            envelope.pose.orientation.w = 1.0
            envelope.scale.x = self._meter_size[0]
            envelope.scale.y = self._meter_size[1]
            envelope.scale.z = self._meter_size[2]
            envelope.color.r = rgba[0]
            envelope.color.g = rgba[1]
            envelope.color.b = rgba[2]
            envelope.color.a = rgba[3]
            envelope.frame_locked = True
            markers.markers.append(envelope)

            label = Marker()
            label.header.frame_id = str(slot["frame_id"])
            label.header.stamp = stamp
            label.ns = "meter_slot_labels"
            label.id = index
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.z = self._meter_size[2] / 2.0 + 0.035
            label.pose.orientation.w = 1.0
            label.scale.z = 0.025
            label.color.r = rgba[0]
            label.color.g = rgba[1]
            label.color.b = rgba[2]
            label.color.a = 1.0
            label.text = f"{slot['id']}  {state}"
            label.frame_locked = True
            markers.markers.append(label)

        self._marker_publisher.publish(markers)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MeterSlotManager()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
