"""Start the complete mock single-arm planning scene.

CLI:
    ros2 launch meter_grasp single_arm_scene.launch.py

This is the infrastructure entry point. It reuses the generated MoveIt demo
launch, then adds this project's workcell collision objects and meter-slot
manager. Starting this file does not run the meter-grasp state machine.
"""

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
                [FindPackageShare("cr5_moveit_config"), "launch", "demo.launch.py"]
            )
        ),
        launch_arguments={"rviz_config": rviz_config}.items(),
    )

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

    show_workcell_camera = Node(
        package="meter_grasp",
        executable="show_workcell_camera",
        name="workcell_camera_visualizer",
        output="screen",
        parameters=[{"scene_config": scene_config}],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "scene_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("meter_grasp"), "config", "scene.json"]
                ),
                description="Shared meter-grasp scene configuration",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("cr5_moveit_config"), "config", "moveit.rviz"]
                ),
                description="RViz configuration including the meter-slot display",
            ),
            moveit_demo,
            add_meter,
            show_slots,
            show_meter_cad,
            show_workcell_camera,
        ]
    )
