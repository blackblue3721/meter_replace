"""MoveIt mock demo whose FakeSystem starts at the validated real-arm Home."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_demo_launch


def generate_launch_description():
    package_share = Path(get_package_share_directory("cr5_moveit_config"))
    initial_positions = (
        package_share / "config" / "real_commissioning_initial_positions.yaml"
    )
    moveit_config = (
        MoveItConfigsBuilder(
            "cr5_with_gripper", package_name="cr5_moveit_config"
        )
        .robot_description(
            mappings={"initial_positions_file": str(initial_positions)}
        )
        .to_moveit_configs()
    )
    return generate_demo_launch(moveit_config)
