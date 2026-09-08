#!/usr/bin/env python3
"""Execute the validated single-arm meter grasp workflow."""

from __future__ import annotations

import math
import time
from enum import Enum, auto
from typing import Callable

import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import GripperCommand
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.action import ActionClient
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState

from .object_attachment import MeterAttachment
from .pick_sequence import PickSequence
from .slot_manager import slot_state_parameter_name


# 完整工艺可能处于的离散状态。
#
# 这些枚举值本身不会让机械臂运动，它们相当于流程中的“路标”：
# 每完成一个经过验证的动作，状态机才会进入下一个状态。FAILED 是统一
# 失败出口，便于日志、恢复逻辑和上位系统判断任务停在了什么阶段。
class ProcessState(Enum):
    START = auto()
    SLOT_RESERVED = auto()
    GRIPPER_OPEN = auto()
    PREGRASP = auto()
    AT_PICK = auto()
    GRIPPED = auto()
    ATTACHED = auto()
    LIFTED = auto()
    SEMANTIC_FLIPPED = auto()
    AT_SLOT_APPROACH = auto()
    AT_SLOT = auto()
    DETACHED = auto()
    SLOT_OCCUPIED = auto()
    RELEASED = auto()
    RETREATED = auto()
    HOME = auto()
    FAILED = auto()


# 单臂电表抓取与安装状态机。
#
# 这里继承 PickSequence，而不是重新编写 MoveIt 通信、IK、轨迹规划和
# 安全检查。PickSequence 提供“怎样规划一个动作”，本类负责“按什么
# 顺序执行这些动作，以及每一步失败后怎样停止”，这就是复用通用能力、
# 只实现本项目工艺逻辑的分层方式。
class MeterGraspStateMachine(PickSequence):
    """Coordinate the validated meter grasp, transfer, and release workflow."""

    # 初始化状态机所需的配置、反馈订阅、Action 客户端和 Service 客户端。
    #
    # 注意：构造函数只建立通信和读取参数，不会命令机械臂运动。
    # stop_after_state 是开发阶段的“阶段闸门”，允许抓取后停在 LIFTED
    # 或 SEMANTIC_FLIPPED，便于先验证下一段规划再继续执行。
    def __init__(
        self,
        node_name: str = "meter_grasp_state_machine",
        workflow_label: str = "desktop-to-slot meter installation",
    ) -> None:
        # node_name 可由复用本类公共通信/执行能力的其他工艺状态机覆盖。
        # 默认值保持不变，因此原有 run_meter_grasp 入口完全兼容。
        super().__init__(node_name, announce_target_stage=False)
        self.get_logger().info(f"Selected workflow: {workflow_label}")
        self._state_machine = self._config["state_machine"]
        self._gripper_config = self._config["gripper"]
        self._state = ProcessState.START
        self._meter_is_attached = False
        self._slot_is_reserved = False
        self._slot_is_occupied = False
        self.declare_parameter("stop_after_state", "")
        self._stop_after_state = str(
            self.get_parameter("stop_after_state").value
        ).strip().upper()
        valid_stop_states = {
            "",
            ProcessState.LIFTED.name,
            ProcessState.SEMANTIC_FLIPPED.name,
        }
        if self._stop_after_state not in valid_stop_states:
            raise ValueError(
                "stop_after_state accepts '', 'LIFTED', or 'SEMANTIC_FLIPPED'"
            )
        self._latest_joint_state: JointState | None = None
        self._joint_state_subscription = self.create_subscription(
            JointState,
            str(self._state_machine["joint_state_topic"]),
            self._joint_state_callback,
            10,
        )
        self._gripper = ActionClient(
            self,
            GripperCommand,
            str(self._gripper_config["action_name"]),
        )
        self._attachment_node = MeterAttachment()
        slot_manager_name = str(
            self._state_machine["slot_manager_node"]
        ).rstrip("/")
        self._slot_get_parameters = self.create_client(
            GetParameters,
            f"{slot_manager_name}/get_parameters",
        )
        self._slot_set_parameters = self.create_client(
            SetParameters,
            f"{slot_manager_name}/set_parameters",
        )

    # /joint_states 订阅回调：保存机器人最近一次真实/模拟关节反馈。
    #
    # 后续的 Home 检查和“轨迹确实执行到位”检查都读取这里保存的数据，
    # 而不是仅仅相信 MoveIt 返回的“执行成功”。
    def _joint_state_callback(self, message: JointState) -> None:
        self._latest_joint_state = message

    # 记录一次经过确认的状态转换，并把新状态打印到终端日志。
    #
    # 所有主要动作成功后才调用该函数，因此日志里的 STATE -> XXX
    # 可以看作流程实际走到哪里的审计记录。
    def _transition(self, state: ProcessState) -> None:
        self._state = state
        self.get_logger().info(f"STATE -> {state.name}")

    # 等待第一条关节状态消息，超时时返回 None。
    #
    # 状态机必须知道机械臂当前各关节角，才能判断是否处于 Home，并在
    # 每次轨迹执行后对照目标终点。这里使用 spin_once 处理 ROS 回调，
    # 没有用固定 sleep 假设消息一定会到达。
    def _wait_for_joint_state(self) -> JointState | None:
        deadline = time.monotonic() + float(self._planning["server_timeout_s"])
        while self._latest_joint_state is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._latest_joint_state is None:
            self.get_logger().error("No joint state was received")
        return self._latest_joint_state

    # 向 meter_slot_manager 查询目标表位当前状态。
    #
    # 返回值通常是 EMPTY、RESERVED 或 OCCUPIED；通信超时、参数不存在
    # 或类型错误时返回 None。安装前必须确认表位 EMPTY，避免把新电表
    # 放进已经占用的位置。
    def _get_target_slot_state(self) -> str | None:
        timeout = float(self._planning["server_timeout_s"])
        if not self._slot_get_parameters.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(
                "Meter-slot state service was not available"
            )
            return None
        request = GetParameters.Request()
        request.names = [slot_state_parameter_name(self._target_slot)]
        future = self._slot_get_parameters.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done() or future.result() is None:
            self.get_logger().error("Meter-slot state request timed out")
            return None
        values = future.result().values
        if (
            len(values) != 1
            or values[0].type != ParameterType.PARAMETER_STRING
        ):
            self.get_logger().error(
                f"Slot manager does not define '{self._target_slot}'"
            )
            return None
        return str(values[0].string_value).upper()

    # 修改目标表位状态，并立即回读确认，形成一次状态握手。
    #
    # 例如先把 EMPTY 改成 RESERVED，防止其他任务同时使用该表位；安装
    # 成功后再改成 OCCUPIED。函数只有在“设置成功且回读值一致”时返回
    # True，不能仅凭 Service 请求已发出就认为状态更新成功。
    def _set_target_slot_state(self, state: str) -> bool:
        timeout = float(self._planning["server_timeout_s"])
        if not self._slot_set_parameters.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(
                "Meter-slot update service was not available"
            )
            return False
        request = SetParameters.Request()
        request.parameters = [
            Parameter(
                slot_state_parameter_name(self._target_slot),
                value=state,
            ).to_parameter_msg()
        ]
        future = self._slot_set_parameters.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done() or future.result() is None:
            self.get_logger().error("Meter-slot update request timed out")
            return False
        results = future.result().results
        if len(results) != 1 or not results[0].successful:
            reason = results[0].reason if results else "missing result"
            self.get_logger().error(
                f"Failed to set '{self._target_slot}' to {state}: {reason}"
            )
            return False
        confirmed = self._get_target_slot_state()
        if confirmed != state:
            self.get_logger().error(
                f"Slot-state handshake failed: requested={state}, "
                f"confirmed={confirmed}"
            )
            return False
        self.get_logger().info(
            f"Slot-state handshake confirmed: "
            f"{self._target_slot}={state}"
        )
        return True

    # 正式运动前的统一安全预检。
    #
    # 依次确认：
    # 1. MoveIt 场景中存在电表和必需碰撞物；
    # 2. 电表当前没有错误地附着在夹爪上；
    # 3. 目标表位为空；
    # 4. 桌面电表实际位姿与配置的抓取位姿足够接近；
    # 5. joint_states 齐全，并且机械臂处于配置的 Home 姿态。
    #
    # 任意条件不满足都返回 False，而且不会发出运动命令。
    def _preflight(self) -> bool:
        if not self._scene_contains_meter():
            return False
        object_id = str(self._attachment["object_id"])
        if object_id in self._attached_object_ids:
            self.get_logger().error(
                f"Preflight failed: '{object_id}' is already attached"
            )
            return False

        slot_state = self._get_target_slot_state()
        if slot_state != "EMPTY":
            self.get_logger().error(
                f"Preflight failed: target slot '{self._target_slot}' "
                f"is {slot_state or 'UNAVAILABLE'}, expected EMPTY"
            )
            return False

        world_pose = self._world_object_poses.get(object_id)
        if world_pose is None:
            self.get_logger().error(
                f"Preflight failed: world pose for '{object_id}' is unavailable"
            )
            return False
        pickup = self._candidate["pose"]["position_m"]
        position_error = math.sqrt(
            (world_pose.position.x - float(pickup["x"])) ** 2
            + (world_pose.position.y - float(pickup["y"])) ** 2
            + (world_pose.position.z - float(pickup["z"])) ** 2
        )
        pickup_tolerance = float(
            self._state_machine["pickup_position_tolerance_m"]
        )
        if position_error > pickup_tolerance:
            self.get_logger().error(
                "Preflight failed: meter is not at the pickup pose; "
                f"position error={position_error:.4f} m"
            )
            return False

        joint_state = self._wait_for_joint_state()
        if joint_state is None:
            return False
        current = dict(zip(joint_state.name, joint_state.position))
        home_tolerance = float(self._state_machine["home_tolerance_rad"])
        errors = {
            name: abs(float(current[name]) - target)
            for name, target in self._home_joint_positions.items()
            if name in current and abs(float(current[name]) - target) > home_tolerance
        }
        missing = [name for name in self._home_joint_positions if name not in current]
        if missing or errors:
            self.get_logger().error(
                f"Preflight failed: robot is not at home; missing={missing}, "
                f"errors={errors}"
            )
            return False
        self.get_logger().info("Preflight passed: home, pickup pose, and scene are valid")
        return True

    # 通过标准 GripperCommand Action 控制夹爪，并等待闭环结果。
    #
    # position 来自 poses.yaml，label 只用于产生可读日志。函数不仅检查
    # Action 是否被接受，还检查最终状态、reached_goal 和位置误差。
    # 当前连接的是模拟夹爪控制器，真实夹爪到货后需要替换参数和接口。
    def _command_gripper(self, position: float, label: str) -> bool:
        timeout = float(self._planning["server_timeout_s"])
        if not self._gripper.wait_for_server(timeout_sec=timeout):
            self.get_logger().error("Gripper action server was not available")
            return False

        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = float(
            self._gripper_config["simulation_max_effort"]
        )
        self.get_logger().info(f"Commanding gripper: {label} ({position:.4f} m)")
        send_future = self._gripper.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=timeout)
        if not send_future.done() or send_future.result() is None:
            self.get_logger().error(f"Gripper '{label}' goal request timed out")
            return False
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f"Gripper '{label}' goal was rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout)
        if not result_future.done() or result_future.result() is None:
            self.get_logger().error(f"Gripper '{label}' result timed out")
            return False
        wrapped_result = result_future.result()
        result = wrapped_result.result
        tolerance = float(
            self._state_machine["gripper_position_tolerance_m"]
        )
        if (
            wrapped_result.status != GoalStatus.STATUS_SUCCEEDED
            or not result.reached_goal
            or abs(float(result.position) - position) > tolerance
        ):
            self.get_logger().error(
                f"Gripper '{label}' failed: status={wrapped_result.status}, "
                f"reached_goal={result.reached_goal}, position={result.position}"
            )
            return False
        return True

    # 执行一个机械臂动作的统一模板：规划 -> 安全检查 -> 执行 -> 到位验证。
    #
    # planner 是一个“稍后调用的规划函数”，例如关节空间规划或 Pilz LIN
    # 直线规划。maximum_travel 可为特定阶段收紧单关节最大行程。把四步
    # 封装在这里，可防止状态机某个阶段漏掉安全检查或执行后验证。
    def _run_motion(
        self,
        name: str,
        planner: Callable,
        maximum_travel: float | None = None,
    ) -> bool:
        trajectory = planner()
        if trajectory is None:
            return False
        if not self._trajectory_is_safe(trajectory, maximum_travel):
            return False
        if not self._execute_planned_trajectory(trajectory, name):
            return False
        return self._verify_trajectory_end(trajectory, name)

    # 使用最新 /joint_states 验证机械臂是否真正到达轨迹最后一个点。
    #
    # MoveIt/控制器返回成功只说明通信与控制流程完成；这里进一步把每个
    # 关节的反馈角与轨迹末点比较。缺少关节或误差超过配置阈值都视为失败，
    # 因而下一工艺阶段不会继续。
    def _verify_trajectory_end(self, trajectory, name: str) -> bool:
        rclpy.spin_once(self, timeout_sec=0.1)
        joint_state = self._latest_joint_state
        if joint_state is None:
            self.get_logger().error(
                f"'{name}' actual-state verification failed: no joint state"
            )
            return False

        current = dict(zip(joint_state.name, joint_state.position))
        joint_trajectory = trajectory.joint_trajectory
        target = joint_trajectory.points[-1].positions
        tolerance = float(
            self._state_machine["execution_joint_tolerance_rad"]
        )
        missing = [
            joint_name
            for joint_name in joint_trajectory.joint_names
            if joint_name not in current
        ]
        errors = {
            joint_name: abs(float(current[joint_name]) - float(target[index]))
            for index, joint_name in enumerate(joint_trajectory.joint_names)
            if joint_name in current
            and abs(float(current[joint_name]) - float(target[index])) > tolerance
        }
        if missing or errors:
            self.get_logger().error(
                f"'{name}' actual-state verification failed: "
                f"missing={missing}, errors={errors}"
            )
            return False
        self.get_logger().info(
            f"'{name}' actual joint-state verification passed"
        )
        return True

    # 统一失败处理：停止后续动作，并根据失败发生阶段进行保守恢复。
    #
    # - 如果还未释放电表，尝试解除目标表位的 RESERVED 状态；
    # - 如果电表仍附着在夹爪上，保持夹紧，不自动张开，防止掉落；
    # - 如果已经解除附着，则不再擅自移动机械臂；
    # - 所有情况都转入 FAILED，并要求操作员根据日志检查。
    #
    # 这里的原则是“失败后保持已知安全状态”，而不是为了自动恢复而继续
    # 执行一串未经重新确认的动作。
    def _fail(self, reason: str) -> bool:
        previous_state = self._state
        self._transition(ProcessState.FAILED)
        if (
            self._slot_is_reserved
            and previous_state.value < ProcessState.DETACHED.value
        ):
            if self._set_target_slot_state("EMPTY"):
                self._slot_is_reserved = False
                self.get_logger().info(
                    "Released target-slot reservation after failure"
                )
            else:
                self.get_logger().error(
                    "Target-slot reservation could not be released; "
                    "operator reset is required"
                )
        if self._meter_is_attached:
            self.get_logger().error(
                f"{reason}. Safe hold active: meter remains attached and the "
                "gripper will not be opened automatically."
            )
        elif previous_state.value >= ProcessState.DETACHED.value:
            self.get_logger().error(
                f"{reason}. Meter is detached on the table; no further arm "
                "motion will be attempted."
            )
        else:
            self.get_logger().error(
                f"{reason}. Motion stopped before a verified attachment; "
                "operator inspection is required."
            )
        return False

    # 完整电表抓取与安装工艺的主流程。
    #
    # 调用顺序：
    # PRECHECK
    #   -> 预占目标表位
    #   -> 张开夹爪
    #   -> 关节空间移动到预抓取位
    #   -> 直线下降到抓取位
    #   -> 夹紧并把电表附着到夹爪碰撞模型
    #   -> 直线抬升
    #   -> joint5 语义翻转，使电表由桌面姿态变为安装姿态
    #   -> 规划到目标表位前方
    #   -> 直线插入
    #   -> 解除附着、标记表位 OCCUPIED、张开夹爪
    #   -> 直线撤离
    #   -> 返回 Home
    #
    # 每个阶段都采用“失败立即进入 _fail()，成功才更新状态”的写法。
    # execute=false 时只进行预检，不执行上述运动，这是默认安全行为。
    def run_process(self) -> bool:
        # 阶段 0：检查场景、表位、电表和机器人初始状态。
        if not self._preflight():
            return self._fail("Preflight failed")
        if not self._execute:
            self.get_logger().info(
                "Preflight-only check complete (execute is disabled)"
            )
            return True

        # 阶段 1：预占目标表位，防止并发任务选择同一个安装位置。
        if not self._set_target_slot_state("RESERVED"):
            return self._fail("Failed to reserve target slot")
        self._slot_is_reserved = True
        self._transition(ProcessState.SLOT_RESERVED)

        # 阶段 2：先张开夹爪，为从上方包住桌面电表留出空间。
        if not self._command_gripper(
            float(self._gripper_config["open_position_m"]), "open"
        ):
            return self._fail("Failed to open gripper")
        self._transition(ProcessState.GRIPPER_OPEN)

        # 阶段 3：求解预抓取位 IK，并用关节空间规划移动到电表正上方。
        pregrasp_joints = self._solve_ik(self._pose("approach"))
        if pregrasp_joints is None or not self._run_motion(
            "meter_pregrasp",
            lambda: self._plan_joint_goal(pregrasp_joints, "meter_pregrasp"),
        ):
            return self._fail("Pregrasp motion failed")
        self._transition(ProcessState.PREGRASP)

        # 阶段 4：使用 Pilz LIN 从预抓取位沿直线下降到精确抓取位。
        if not self._run_motion(
            "meter_centered_grasp_candidate",
            lambda: self._plan_linear_pose_goal(
                self._pose(), "meter_centered_grasp_candidate"
            ),
        ):
            return self._fail("Pick descent failed")
        self._transition(ProcessState.AT_PICK)

        # 阶段 5：闭合夹爪；只有夹爪控制器确认到位后才继续。
        if not self._command_gripper(
            float(self._gripper_config["meter_grasp_position_m"]), "meter grasp"
        ):
            return self._fail("Failed to close gripper")
        self._transition(ProcessState.GRIPPED)

        # 阶段 6：在 MoveIt PlanningScene 中把电表附着到夹爪。
        #
        # 这一步不会产生真实吸力/摩擦，而是告诉碰撞规划器：后续电表要
        # 随夹爪一起运动，并且需要作为机器人携带物参与碰撞检测。
        if not self._attachment_node.attach():
            return self._fail("Failed to attach meter")
        self._meter_is_attached = True
        self._transition(ProcessState.ATTACHED)

        # 阶段 7：携带电表沿世界坐标 Z 方向直线抬升。
        if not self._run_motion(
            "meter_vertical_lift",
            lambda: self._plan_linear_pose_goal(
                self._pose("lift"), "meter_vertical_lift"
            ),
        ):
            return self._fail("Lift failed")
        self._transition(ProcessState.LIFTED)
        if self._stop_after_state == ProcessState.LIFTED.name:
            self.get_logger().info(
                "Stage gate reached at LIFTED; meter remains attached for "
                "the next plan-only slot-transfer validation"
            )
            return True

        # 阶段 8：保留当前其他关节，只改变配置指定的 joint5，把电表从
        # 桌面朝向翻转为“顶部向上、背面朝安装面”的语义安装姿态。
        current_joint_positions = (
            dict(
                zip(
                    self._latest_joint_state.name,
                    self._latest_joint_state.position,
                )
            )
            if self._latest_joint_state is not None
            else {}
        )
        semantic_flip_goal = self._semantic_flip_goal(current_joint_positions)
        if semantic_flip_goal is None or not self._run_motion(
            "meter_semantic_flip",
            lambda: self._plan_joint_goal(
                semantic_flip_goal, "meter_semantic_flip"
            ),
            float(self._planning["semantic_flip_max_joint_travel_rad"]),
        ):
            return self._fail("Semantic meter flip failed")
        self._transition(ProcessState.SEMANTIC_FLIPPED)
        if self._stop_after_state == ProcessState.SEMANTIC_FLIPPED.name:
            self.get_logger().info(
                "Stage gate reached at SEMANTIC_FLIPPED; meter remains "
                "attached for slot-transfer validation"
            )
            return True

        # 阶段 9：从当前姿态搜索多个无碰 IK 分支并评分，选择较舒展、
        # 单关节变化较小的解，再移动到目标表位前方的 approach 位。
        slot_approach = self._slot_pose("approach")
        current_joint_positions = (
            dict(
                zip(
                    self._latest_joint_state.name,
                    self._latest_joint_state.position,
                )
            )
            if self._latest_joint_state is not None
            else {}
        )
        slot_approach_joints = (
            self._solve_best_ik(slot_approach, current_joint_positions)
            if slot_approach is not None
            else None
        )
        if slot_approach_joints is None or not self._run_motion(
            "meter_slot_transfer",
            lambda: self._plan_joint_goal(
                slot_approach_joints, "meter_slot_transfer"
            ),
            float(self._planning["slot_transfer_max_joint_travel_rad"]),
        ):
            return self._fail("Transfer failed")
        self._transition(ProcessState.AT_SLOT_APPROACH)

        # 阶段 10：保持安装朝向，从 approach 位沿表位坐标轴直线插入。
        slot_install = self._slot_pose()
        if slot_install is None or not self._run_motion(
            "meter_slot_insert",
            lambda: self._plan_linear_pose_goal(
                slot_install, "meter_slot_insert"
            ),
        ):
            return self._fail("Slot insertion failed")
        self._transition(ProcessState.AT_SLOT)

        # 阶段 11：先从夹爪解除电表碰撞附着，再通过握手把表位标成
        # OCCUPIED。此后电表属于世界场景中的已安装物体。
        if not self._attachment_node.detach():
            return self._fail("Failed to detach meter")
        self._meter_is_attached = False
        self._transition(ProcessState.DETACHED)
        if not self._set_target_slot_state("OCCUPIED"):
            return self._fail("Failed to mark target slot occupied")
        self._slot_is_reserved = False
        self._slot_is_occupied = True
        self._transition(ProcessState.SLOT_OCCUPIED)

        # 阶段 12：张开夹爪，完成机械意义上的释放。
        if not self._command_gripper(
            float(self._gripper_config["open_position_m"]), "release"
        ):
            return self._fail("Failed to release meter")
        self._transition(ProcessState.RELEASED)

        # 阶段 13：沿插入路径反向直线撤离，避免夹爪扫到电表或箱体。
        if not self._run_motion(
            "meter_slot_retreat",
            lambda: self._plan_linear_pose_goal(
                slot_approach, "meter_slot_retreat"
            ),
        ):
            return self._fail("Retreat failed")
        self._transition(ProcessState.RETREATED)

        # 阶段 14：使用关节空间规划返回配置的 Home 姿态。
        if not self._run_motion(
            "return_home",
            lambda: self._plan_joint_goal(
                self._home_joint_positions, "return_home"
            ),
            float(self._planning["return_home_max_joint_travel_rad"]),
        ):
            return self._fail("Return-home motion failed")
        self._transition(ProcessState.HOME)
        self.get_logger().info(
            f"Meter grasp process completed: installed in '{self._target_slot}'"
        )
        return True

    # 释放本类额外创建的 ROS 节点资源。
    #
    # MeterAttachment 是独立 Node，需要先销毁它，再销毁状态机节点。
    # 该函数不执行机械臂恢复动作；机械臂动作只能在 run_process 中发生。
    def close(self) -> None:
        self._attachment_node.destroy_node()
        self.destroy_node()


# ROS 2 console script 的程序入口，对应：
#   ros2 run meter_grasp run_meter_grasp
#
# 入口负责初始化 rclpy、创建状态机、运行主流程，并保证无论成功还是抛出
# 异常都释放节点和关闭 rclpy。流程失败时抛出 RuntimeError，使终端返回
# 非零退出码，方便脚本或未来上位系统识别任务失败。
def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MeterGraspStateMachine()
    success = False
    try:
        success = node.run_process()
    finally:
        node.close()
        rclpy.shutdown()
    if not success:
        raise RuntimeError("Meter grasp state machine failed")
