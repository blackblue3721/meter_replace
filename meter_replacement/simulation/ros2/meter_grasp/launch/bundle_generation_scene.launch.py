"""Pure mock scene initialized at the real commissioning return-Home branch."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    scene_config = LaunchConfiguration("scene_config")
    rviz_config = LaunchConfiguration("rviz_config")
    moveit_demo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("cr5_moveit_config"),
                    "launch",
                    "demo_real_commissioning_home.launch.py",
                ]
            )
        ),
        launch_arguments={"rviz_config": rviz_config}.items(),
    )
    nodes = [
        Node(
            package="meter_grasp",
            executable="add_meter_collision",
            name="meter_planning_scene",
            output="screen",
            parameters=[{"scene_config": scene_config}],
        ),
        Node(
            package="meter_grasp",
            executable="show_meter_slots",
            name="meter_slot_manager",
            output="screen",
            parameters=[{"scene_config": scene_config}],
        ),
        Node(
            package="meter_grasp",
            executable="show_meter_cad",
            name="meter_cad_visualizer",
            output="screen",
            parameters=[{"scene_config": scene_config}],
        ),
    ]
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
            moveit_demo,
            *nodes,
        ]
    )
