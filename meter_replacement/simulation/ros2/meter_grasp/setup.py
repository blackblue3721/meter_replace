from setuptools import find_packages, setup


PACKAGE_NAME = "meter_grasp"
# Keep the project-level file as the single source of truth.  colcon requires
# data-file sources to be relative to this package's setup.py directory.
SCENE_CONFIG = "../../../config/scene.json"
POSES_CONFIG = "../../../config/poses.yaml"
METER_MESHES = [
    "meshes/electric_meter_body.stl",
    "meshes/electric_meter_cover.stl",
]
LAUNCH_FILES = [
    "launch/single_arm_scene.launch.py",
    "launch/real_arm_plan_only_scene.launch.py",
    "launch/bundle_generation_scene.launch.py",
    "launch/vision_guided_scene.launch.py",
]


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml"]),
        (f"share/{PACKAGE_NAME}/launch", LAUNCH_FILES),
        (f"share/{PACKAGE_NAME}/config", [SCENE_CONFIG, POSES_CONFIG]),
        (f"share/{PACKAGE_NAME}/meshes", METER_MESHES),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Haoran",
    maintainer_email="2274899897@qq.com",
    description="MoveIt planning and execution for the single-arm meter grasp.",
    license="Apache-2.0",
    entry_points={
        # ROS 2 CLI -> Python entry-point map:
        #   ros2 run meter_grasp add_meter_collision -> planning_scene.py:main
        #   ros2 run meter_grasp attach_meter        -> object_attachment.py:main
        #   ros2 run meter_grasp detach_meter        -> object_attachment.py:detach_main
        #   ros2 run meter_grasp run_meter_grasp     -> meter_grasp_state_machine.py:main
        #   ros2 run meter_grasp run_meter_remove    -> meter_remove_state_machine.py:main
        #   ros2 run meter_grasp run_pick_sequence   -> pick_sequence.py:main
        #   ros2 run meter_grasp show_meter_slots    -> slot_manager.py:main
        #   ros2 run meter_grasp show_meter_cad      -> meter_visualizer.py:main
        #   ros2 run meter_grasp show_workcell_camera -> workcell_camera_visualizer.py:main
        #
        # The --ros-args -p ... values used at the terminal are runtime
        # parameter overrides; the motion implementation remains in these
        # Python modules and the project-level YAML/JSON configuration files.
        "console_scripts": [
            "add_meter_collision = meter_grasp.planning_scene:main",
            "attach_meter = meter_grasp.object_attachment:main",
            "detach_meter = meter_grasp.object_attachment:detach_main",
            "run_meter_grasp = meter_grasp.meter_grasp_state_machine:main",
            "run_meter_remove = meter_grasp.meter_remove_state_machine:main",
            "run_pick_sequence = meter_grasp.pick_sequence:main",
            "show_meter_slots = meter_grasp.slot_manager:main",
            "show_meter_cad = meter_grasp.meter_visualizer:main",
            "show_workcell_camera = meter_grasp.workcell_camera_visualizer:main",
        ],
    },
)
