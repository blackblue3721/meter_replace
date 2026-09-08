"""Start live CR5 state, MoveIt/RViz, D435 and the guarded coordinator."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    project_root = LaunchConfiguration("project_root")
    start_vision = LaunchConfiguration("start_vision")
    conda_executable = LaunchConfiguration("conda_executable")
    vision_env = LaunchConfiguration("vision_env")
    execute_motion = LaunchConfiguration("execute_motion")
    motion_speed_percent = LaunchConfiguration("motion_speed_percent")

    real_feedback = ExecuteProcess(
        cmd=[
            "/usr/bin/python3",
            PathJoinSubstitution(
                [project_root, "robot", "real", "cr5_tcp", "publish_cr5_joint_states.py"]
            ),
        ],
        output="screen",
    )

    real_scene = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("meter_grasp"), "launch", "real_arm_plan_only_scene.launch.py"]
            )
        )
    )

    detector = ExecuteProcess(
        cmd=[
            conda_executable,
            "run",
            "--no-capture-output",
            "-n",
            vision_env,
            "python",
            PathJoinSubstitution(
                [project_root, "vision", "D435_Project_new", "realtime_multi_meter_d435.py"]
            ),
        ],
        output="both",
        condition=IfCondition(start_vision),
    )

    planner = ExecuteProcess(
        cmd=[
            "/usr/bin/python3",
            "-u",
            PathJoinSubstitution([project_root, "tools", "plan_first_stable_meter.py"]),
            "--execute-mode",
            execute_motion,
            "--speed-percent",
            motion_speed_percent,
        ],
        output="both",
        condition=IfCondition(start_vision),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "project_root", default_value="/home/haoran/Arm/meter_replacement"
            ),
            DeclareLaunchArgument("start_vision", default_value="true"),
            DeclareLaunchArgument(
                "conda_executable", default_value="/home/haoran/miniconda3/bin/conda"
            ),
            DeclareLaunchArgument("vision_env", default_value="arm"),
            DeclareLaunchArgument("execute_motion", default_value="true"),
            DeclareLaunchArgument("motion_speed_percent", default_value="10"),
            real_feedback,
            real_scene,
            detector,
            planner,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=planner,
                    on_exit=[
                        EmitEvent(
                            event=Shutdown(
                                reason="vision-guided coordinator finished"
                            )
                        )
                    ],
                ),
                condition=IfCondition(start_vision),
            ),
        ]
    )
