"""Start MoveIt/RViz against real CR5 joint feedback, with no motion controller.

The real feedback publisher must already provide ``/real_joint_states``.  This
launch deliberately omits ros2_control and controller spawners, so it can plan
and visualize trajectories but cannot execute them on either mock or hardware.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetRemap
from launch_ros.substitutions import FindPackageShare


def _moveit_launch(filename: str, launch_arguments=None):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("cr5_moveit_config"), "launch", filename]
            )
        ),
        launch_arguments=(launch_arguments or {}).items(),
    )


def generate_launch_description() -> LaunchDescription:
    scene_config = LaunchConfiguration("scene_config")
    rviz_config = LaunchConfiguration("rviz_config")

    add_meter = Node(
        package="meter_grasp",
        executable="add_meter_collision",
        name="meter_planning_scene",
        output="screen",
        parameters=[{"scene_config": scene_config}],
    )
    show_slots = Node(
        package="meter_grasp",
        executable="show_meter_slots",
        name="meter_slot_manager",
        output="screen",
        parameters=[{"scene_config": scene_config}],
    )
    show_meter_cad = Node(
        package="meter_grasp",
        executable="show_meter_cad",
        name="meter_cad_visualizer",
        output="screen",
        parameters=[{"scene_config": scene_config}],
    )

    real_feedback_moveit = GroupAction(
        actions=[
            # Isolate hardware feedback from any stale/mock ros2_control
            # publisher that may still be using the conventional topic.
            SetRemap(src="/joint_states", dst="/real_joint_states"),
            _moveit_launch("static_virtual_joint_tfs.launch.py"),
            _moveit_launch("rsp.launch.py"),
            _moveit_launch("move_group.launch.py"),
            TimerAction(
                period=2.5,
                actions=[
                    _moveit_launch(
                        "moveit_rviz.launch.py", {"rviz_config": rviz_config}
                    )
                ],
            ),
        ]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "scene_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("meter_grasp"), "config", "scene.json"]
                ),
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("cr5_moveit_config"), "config", "moveit.rviz"]
                ),
            ),
            real_feedback_moveit,
            add_meter,
            show_slots,
            show_meter_cad,
        ]
    )
