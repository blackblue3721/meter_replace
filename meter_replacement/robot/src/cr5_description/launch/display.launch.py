from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    package_dir = get_package_share_directory("cr5_description")
    model_path = os.path.join(package_dir, "urdf", "cr5_with_gripper.urdf.xacro")
    rviz_path = os.path.join(package_dir, "rviz", "cr5.rviz")

    use_gui = LaunchConfiguration("use_gui")
    use_rviz = LaunchConfiguration("use_rviz")
    external_joint_states = LaunchConfiguration("external_joint_states")
    robot_description = ParameterValue(
        Command(["xacro ", model_path]), value_type=str
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_gui", default_value="true"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument(
            "external_joint_states",
            default_value="false",
            description="Do not start a local joint-state publisher; use /joint_states from an external read-only source.",
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
        ),
        GroupAction(
            condition=UnlessCondition(external_joint_states),
            actions=[
                Node(
                    package="joint_state_publisher_gui",
                    executable="joint_state_publisher_gui",
                    condition=IfCondition(use_gui),
                ),
                Node(
                    package="joint_state_publisher",
                    executable="joint_state_publisher",
                    condition=UnlessCondition(use_gui),
                ),
            ],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", rviz_path],
            condition=IfCondition(use_rviz),
        ),
    ])
