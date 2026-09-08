#!/usr/bin/env python3
"""Execute the validated single-arm meter removal workflow."""

from __future__ import annotations

import math
from enum import Enum, auto

import rclpy

from .meter_grasp_state_machine import MeterGraspStateMachine


# 电表拆取流程的离散状态。
#
# 它与“桌面取表并安装”的状态分开定义，避免把安装、拆取两个方向不同的
# 工艺强行塞进同一个枚举。日志中的 STATE -> XXX 只会在相应动作和反馈
# 检查都成功后出现。
class RemovalState(Enum):
    START = auto()
    SLOT_RESERVED = auto()
    GRIPPER_OPEN = auto()
    AT_SLOT_APPROACH = auto()
    AT_SLOT = auto()
    GRIPPED = auto()
    ATTACHED = auto()
    EXTRACTED = auto()
    SLOT_EMPTY = auto()
    AT_UNINSTALL_WAYPOINT = auto()
    SEMANTIC_UNFLIPPED = auto()
    AT_DESKTOP = auto()
    DETACHED = auto()
    RELEASED = auto()
    RETREATED = auto()
    HOME = auto()
    FAILED = auto()


class MeterRemoveStateMachine(MeterGraspStateMachine):
    """Remove one installed meter and place it at the desktop pickup pose.

    本类复用安装状态机已经建立的 ROS 通信和安全执行能力：

    - PickSequence 提供 MoveIt、IK、OMPL、Pilz LIN 和轨迹安全检查；
    - MeterGraspStateMachine 提供夹爪 Action、槽位状态握手、关节反馈
      到位检查以及 PlanningScene 电表附着接口；
    - 本类只定义拆取方向特有的预检、动作顺序和失败恢复规则。

    当前流程代表“螺丝、透明盖和导线均已处理完毕”之后的机械搬运子流程，
    不是完整的真实拆表工艺。
    """

    def __init__(self) -> None:
        super().__init__(
            "meter_remove_state_machine",
            "installed-meter removal to desktop",
        )
        self._state = RemovalState.START
        self._meter_was_extracted = False
        self._slot_is_empty = False

    # 只有动作完成且反馈验证通过后，才调用本函数推进流程状态。
    def _transition(self, state: RemovalState) -> None:
        self._state = state
        self.get_logger().info(f"STATE -> {state.name}")

    # 检查机械臂是否位于配置的 Home 姿态。
    #
    # 这里读取 /joint_states，而不是相信 RViz 画面或规划器中的目标状态。
    def _robot_is_at_home(self) -> bool:
        joint_state = self._wait_for_joint_state()
        if joint_state is None:
            return False
        current = dict(zip(joint_state.name, joint_state.position))
        tolerance = float(self._state_machine["home_tolerance_rad"])
        missing = [
            name for name in self._home_joint_positions if name not in current
        ]
        errors = {
            name: abs(float(current[name]) - target)
            for name, target in self._home_joint_positions.items()
            if name in current
            and abs(float(current[name]) - target) > tolerance
        }
        if missing or errors:
            self.get_logger().error(
                "Removal preflight failed: robot is not at home; "
                f"missing={missing}, errors={errors}"
            )
            return False
        return True

    # 拆取流程的运动前预检。
    #
    # 与安装预检的方向相反：
    # 1. 电表必须是世界碰撞物，不能已经附着在夹爪上；
    # 2. 目标表位必须是 OCCUPIED；
    # 3. 电表位置必须与目标表位的安装位接近；
    # 4. 机械臂必须处于 Home。
    def _removal_preflight(self) -> bool:
        if not self._scene_contains_meter():
            return False

        object_id = str(self._attachment["object_id"])
        if object_id in self._attached_object_ids:
            self.get_logger().error(
                f"Removal preflight failed: '{object_id}' is already attached"
            )
            return False

        slot_state = self._get_target_slot_state()
        if slot_state != "OCCUPIED":
            self.get_logger().error(
                f"Removal preflight failed: target slot '{self._target_slot}' "
                f"is {slot_state or 'UNAVAILABLE'}, expected OCCUPIED"
            )
            return False

        installed_pose = self._slot_pose()
        world_pose = self._world_object_poses.get(object_id)
        if installed_pose is None or world_pose is None:
            self.get_logger().error(
                "Removal preflight failed: installed meter pose is unavailable"
            )
            return False
        position_error = math.sqrt(
            (world_pose.position.x - installed_pose.position.x) ** 2
            + (world_pose.position.y - installed_pose.position.y) ** 2
            + (world_pose.position.z - installed_pose.position.z) ** 2
        )
        position_tolerance = float(
            self._state_machine["installed_position_tolerance_m"]
        )
        if position_error > position_tolerance:
            self.get_logger().error(
                "Removal preflight failed: meter is not at the selected slot; "
                f"position error={position_error:.4f} m"
            )
            return False

        if not self._robot_is_at_home():
            return False
        self.get_logger().info(
            "Removal preflight passed: home, occupied slot, installed meter, "
            "and scene are valid"
        )
        return True

    # 取出电表后的安全中转关节目标。
    #
    # 复用此前逐段验证的桌面 lift IK 分支，并先保持 joint5 的“竖直电表”
    # 语义姿态。这样先把电表带离箱体，再在空旷区域执行姿态还原。
    def _uninstall_waypoint_goal(self) -> dict[str, float] | None:
        flat_lift_joints = self._solve_ik(
            self._pose("lift"),
            self._planning_seed(),
        )
        if flat_lift_joints is None:
            return None
        return self._semantic_flip_goal(flat_lift_joints)

    # 把竖直安装姿态还原为桌面放置姿态，只改变语义关节 joint5。
    def _semantic_unflip_goal(self) -> dict[str, float] | None:
        if self._latest_joint_state is None:
            self.get_logger().error(
                "Cannot build semantic-unflip goal: no current joint state"
            )
            return None
        current = dict(
            zip(
                self._latest_joint_state.name,
                self._latest_joint_state.position,
            )
        )
        goal = self._semantic_flip_goal(current)
        flat_lift_joints = self._solve_ik(
            self._pose("lift"),
            self._planning_seed(),
        )
        if goal is None or flat_lift_joints is None:
            return None
        semantic_joint = str(self._semantic_flip["joint_name"])
        goal[semantic_joint] = flat_lift_joints[semantic_joint]
        return goal

    # 拆取失败时采用保守恢复，不自动追加机械臂动作。
    #
    # - 尚未附着电表：把 RESERVED 恢复成 OCCUPIED；
    # - 已附着但未完成取出：保留 RESERVED 并保持夹紧；
    # - 已取出：表位保持 EMPTY；若仍附着则继续安全夹持；
    # - 已放到桌面：不再移动机械臂，等待操作员检查。
    def _fail(self, reason: str) -> bool:
        previous_state = self._state
        self._transition(RemovalState.FAILED)

        if self._slot_is_reserved and not self._meter_is_attached:
            if self._set_target_slot_state("OCCUPIED"):
                self._slot_is_reserved = False
                self.get_logger().info(
                    "Restored target slot to OCCUPIED after pre-grasp failure"
                )
            else:
                self.get_logger().error(
                    "Target slot remains RESERVED; operator reset is required"
                )

        if self._meter_is_attached:
            location = (
                "outside the cabinet"
                if self._meter_was_extracted
                else "at the cabinet slot"
            )
            self.get_logger().error(
                f"{reason}. Safe hold active: meter remains attached {location}; "
                "the gripper will not be opened automatically."
            )
        elif previous_state in {
            RemovalState.DETACHED,
            RemovalState.RELEASED,
            RemovalState.RETREATED,
            RemovalState.HOME,
        }:
            self.get_logger().error(
                f"{reason}. Meter is detached on the desktop; no further arm "
                "motion will be attempted."
            )
        else:
            self.get_logger().error(
                f"{reason}. Motion stopped before a verified attachment; "
                "operator inspection is required."
            )
        return False

    # 完整的“电表箱取表并放回桌面”流程。
    #
    # 这是已经逐段验证过的动作序列：
    # Home -> 箱前接近 -> 直线插入 -> 夹紧 -> 附着 -> 直线抽出
    # -> 安全中转 -> 姿态还原 -> 直线放到桌面 -> 解除附着并张开
    # -> 直线撤离 -> Home。
    def run_process(self) -> bool:
        # 阶段 0：只做检查。execute=false 是默认安全模式，不发运动命令。
        if not self._removal_preflight():
            return self._fail("Removal preflight failed")
        if not self._execute:
            self.get_logger().info(
                "Removal preflight-only check complete (execute is disabled)"
            )
            return True

        # 阶段 1：把 OCCUPIED 改成 RESERVED，锁住该表位。
        if not self._set_target_slot_state("RESERVED"):
            return self._fail("Failed to reserve occupied target slot")
        self._slot_is_reserved = True
        self._transition(RemovalState.SLOT_RESERVED)

        # 阶段 2：张开夹爪，准备从箱体正前方包住电表。
        if not self._command_gripper(
            float(self._gripper_config["open_position_m"]), "open for removal"
        ):
            return self._fail("Failed to open gripper")
        self._transition(RemovalState.GRIPPER_OPEN)

        # 阶段 3：搜索并选择已验证的柜前 IK 分支，关节空间移动到接近位。
        slot_approach = self._slot_pose("approach")
        current_positions = (
            dict(
                zip(
                    self._latest_joint_state.name,
                    self._latest_joint_state.position,
                )
            )
            if self._latest_joint_state is not None
            else {}
        )
        approach_joints = (
            self._solve_best_ik(slot_approach, current_positions)
            if slot_approach is not None
            else None
        )
        if approach_joints is None or not self._run_motion(
            "meter_slot_pick_approach",
            lambda: self._plan_joint_goal(
                approach_joints, "meter_slot_pick_approach"
            ),
            float(self._planning["slot_transfer_max_joint_travel_rad"]),
        ):
            return self._fail("Slot-pick approach failed")
        self._transition(RemovalState.AT_SLOT_APPROACH)

        # 阶段 4：保持电表安装姿态，使用 Pilz LIN 直线进入抓取位。
        slot_pick = self._slot_pose()
        if slot_pick is None or not self._run_motion(
            "meter_slot_pick_insert",
            lambda: self._plan_linear_pose_goal(
                slot_pick, "meter_slot_pick_insert"
            ),
        ):
            return self._fail("Slot-pick insertion failed")
        self._transition(RemovalState.AT_SLOT)

        # 阶段 5：夹紧并等待夹爪控制器确认到位。
        if not self._command_gripper(
            float(self._gripper_config["meter_grasp_position_m"]),
            "grasp installed meter",
        ):
            return self._fail("Failed to grasp installed meter")
        self._transition(RemovalState.GRIPPED)

        # 阶段 6：把电表附着到夹爪碰撞模型，后续规划会携带它验碰。
        if not self._attachment_node.attach():
            return self._fail("Failed to attach installed meter")
        self._meter_is_attached = True
        self._transition(RemovalState.ATTACHED)

        # 阶段 7：沿插入路径反向直线抽出 0.10 m。
        if not self._run_motion(
            "meter_slot_extract",
            lambda: self._plan_linear_pose_goal(
                slot_approach, "meter_slot_extract"
            ),
        ):
            return self._fail("Meter extraction failed")
        self._meter_was_extracted = True
        self._transition(RemovalState.EXTRACTED)

        # 电表已经离开箱体，通过参数设置 + 回读把槽位正式标记 EMPTY。
        if not self._set_target_slot_state("EMPTY"):
            return self._fail("Failed to mark extracted slot empty")
        self._slot_is_reserved = False
        self._slot_is_occupied = False
        self._slot_is_empty = True
        self._transition(RemovalState.SLOT_EMPTY)

        # 阶段 8：关节空间移动到柜外、桌面抓取位上方的竖直安全中转姿态。
        uninstall_goal = self._uninstall_waypoint_goal()
        if uninstall_goal is None or not self._run_motion(
            "meter_slot_uninstall_transfer",
            lambda: self._plan_joint_goal(
                uninstall_goal, "meter_slot_uninstall_transfer"
            ),
            float(self._planning["slot_transfer_max_joint_travel_rad"]),
        ):
            return self._fail("Uninstall transfer failed")
        self._transition(RemovalState.AT_UNINSTALL_WAYPOINT)

        # 阶段 9：只转 joint5，把竖直电表还原为正面朝上的桌面姿态。
        semantic_unflip_goal = self._semantic_unflip_goal()
        if semantic_unflip_goal is None or not self._run_motion(
            "meter_semantic_unflip",
            lambda: self._plan_joint_goal(
                semantic_unflip_goal, "meter_semantic_unflip"
            ),
            float(self._planning["semantic_flip_max_joint_travel_rad"]),
        ):
            return self._fail("Semantic meter unflip failed")
        self._transition(RemovalState.SEMANTIC_UNFLIPPED)

        # 阶段 10：使用 Pilz LIN 直线下降到原桌面抓取位。
        if not self._run_motion(
            "meter_desktop_place_descend",
            lambda: self._plan_linear_pose_goal(
                self._pose(), "meter_desktop_place_descend"
            ),
        ):
            return self._fail("Desktop placement descent failed")
        self._transition(RemovalState.AT_DESKTOP)

        # 阶段 11：解除 PlanningScene 附着，电表重新成为桌面世界物体。
        if not self._attachment_node.detach():
            return self._fail("Failed to detach meter on desktop")
        self._meter_is_attached = False
        self._transition(RemovalState.DETACHED)

        # 阶段 12：张开夹爪，完成机械意义上的释放。
        if not self._command_gripper(
            float(self._gripper_config["open_position_m"]),
            "release on desktop",
        ):
            return self._fail("Failed to release meter on desktop")
        self._transition(RemovalState.RELEASED)

        # 阶段 13：沿世界 Z 方向直线上撤，避免扫到桌面电表。
        if not self._run_motion(
            "meter_desktop_place_retreat",
            lambda: self._plan_linear_pose_goal(
                self._pose("approach"), "meter_desktop_place_retreat"
            ),
        ):
            return self._fail("Desktop retreat failed")
        self._transition(RemovalState.RETREATED)

        # 阶段 14：关节空间返回 Home。
        if not self._run_motion(
            "return_home",
            lambda: self._plan_joint_goal(
                self._home_joint_positions, "return_home"
            ),
            float(self._planning["return_home_max_joint_travel_rad"]),
        ):
            return self._fail("Return-home motion failed")
        self._transition(RemovalState.HOME)
        self.get_logger().info(
            f"Meter removal process completed: removed from "
            f"'{self._target_slot}' and placed on the desktop"
        )
        return True


# ROS 2 正式入口：
#   ros2 run meter_grasp run_meter_remove
#
# 默认 execute=false，只做预检；必须显式传入 -p execute:=true 才会运动。
def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MeterRemoveStateMachine()
    success = False
    try:
        success = node.run_process()
    finally:
        node.close()
        rclpy.shutdown()
    if not success:
        raise RuntimeError("Meter removal state machine failed")
