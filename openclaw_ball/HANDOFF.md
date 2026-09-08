# openclaw_ball 子项目交接

更新时间：2026-09-04

## 1. 新对话的任务边界

`/home/haoran/Arm/openclaw_ball` 是一个独立的 OpenClaw 演示子项目。

- 新对话只负责 `openclaw_ball` 的设计、实现和演示。
- 原项目 `/home/haoran/Arm/meter_replacement` 继续在原对话开发。
- 可以复用原项目已经验证的视觉、坐标转换、ROS 2、MoveIt 和 CR5 接口。
- 不要复制原项目的 `build/`、`install/`、`log/`、ROS 日志或训练数据集。
- 未经用户明确要求，不要修改 `meter_replacement` 源码。
- OpenClaw 当前是否安装、采用哪个版本、演示目标是什么，尚未检查；新对话应先确认本机现状和演示需求，不要直接假设。

当前 `openclaw_ball` 除本交接文档外没有代码，尚未确定最终目录结构。

## 2. 可复用项目概况

原项目根目录：

```text
/home/haoran/Arm/meter_replacement
```

现有系统已经验证的主链路是：

```text
D435 RGB-D
  → YOLO识别电表
  → 稳定目标相机坐标
  → 相机坐标转换至CR5 User0/base
  → MoveIt两段路径规划
  → 当前真机位置
  → 安全过渡位
  → 第一个电表的视觉渐进位
```

这条链路已在真实 CR5 上走通，但只到达电表前方渐进位：

- 尚未自动下降接触电表；
- 尚未驱动实体夹爪；
- 尚未自动抓取；
- 当前属于功能闭环验证，不是完整生产工艺。

## 3. 推荐的复用方式

不要把原项目代码复制进 `openclaw_ball`。优先使用以下边界：

1. 演示只需要视觉结果：读取稳定目标 CSV。
2. 演示需要路径规划：调用原项目统一入口的 `--plan-only`。
3. 演示需要显示机械臂状态：订阅 `/real_joint_states`。
4. 只有用户明确授权真机运动时，才允许调用默认执行入口。

建议让 OpenClaw 负责“理解演示指令和编排任务”，原项目继续负责视觉、规划和受保护的硬件执行。不要在新项目中重新实现 CR5 TCP 协议或复制 MoveIt 状态机。

## 4. 现有统一入口

文件：

```text
/home/haoran/Arm/meter_replacement/tools/start_vision_guided_grasp.py
```

只检查环境和重复进程，不启动设备：

```bash
cd /home/haoran/Arm/meter_replacement
/usr/bin/python3 tools/start_vision_guided_grasp.py --check
```

只做视觉与两段规划，不发送真机运动：

```bash
cd /home/haoran/Arm/meter_replacement
/usr/bin/python3 tools/start_vision_guided_grasp.py --plan-only
```

以下命令默认规划并执行真实 CR5 两段运动，不能作为无确认的普通演示命令：

```bash
cd /home/haoran/Arm/meter_replacement
/usr/bin/python3 tools/start_vision_guided_grasp.py
```

它不会自动使能、清错、下降或夹取，但会向已经使能且处于 TCP/IP 二次开发模式的 CR5 发送 ServoJ 轨迹。

内部协调器是：

```text
/home/haoran/Arm/meter_replacement/tools/plan_first_stable_meter.py
```

它不是第二个面向操作员的入口。它由统一入口和 ROS launch 自动启动。

## 5. D435 与 YOLO

视觉子项目：

```text
/home/haoran/Arm/meter_replacement/vision/D435_Project_new
```

实时检测程序：

```text
realtime_multi_meter_d435.py
```

当前模型：

```text
runs/detect/train-4/weights/best.pt
```

该程序需要 Conda `arm` 环境。不要先 `conda deactivate` 后直接用系统 Python 运行 YOLO。独立启动方式是：

```bash
/home/haoran/miniconda3/bin/conda run --no-capture-output -n arm \
  python /home/haoran/Arm/meter_replacement/vision/D435_Project_new/realtime_multi_meter_d435.py
```

统一入口已经自动隔离两套环境：

- ROS 2/MoveIt 使用 `/usr/bin/python3` 和 ROS Humble；
- YOLO/D435 子进程使用 Conda `arm`；
- 不需要操作员手工切换 Conda 环境。

视觉程序默认使用最近 10 个稳定帧，稳定阈值为 `0.03 m`。关键输出：

```text
/home/haoran/Arm/meter_replacement/vision/D435_Project_new/logs/robot_ready_targets.csv
```

其中 `status=ready` 的行才可作为稳定目标。统一协调器默认选择 `target_id=1`，并冻结本次目标快照，不会持续跟随每一帧抖动。

## 6. 坐标转换链路

转换顺序：

```text
D435相机坐标（m）
  → camera_to_cr5_transform.json
CR5 User0/base坐标（mm）
  → user2_frame_config.json
CR5 User2坐标（mm）
```

相关文件：

```text
vision/D435_Project_new/convert_ready_targets_to_cr5.py
vision/D435_Project_new/convert_cr5_targets_user0_to_user2.py
vision/D435_Project_new/calibration/result/camera_to_cr5_transform.json
vision/D435_Project_new/user2_frame_config.json
```

当前 User2 参数：

```text
X=-431.496 mm
Y=-155.383 mm
Z=702.852 mm
Rx=87.310 deg
Ry=4.072 deg
Rz=135.530 deg
rotation_convention=rzyx
```

当前相机到 CR5 标定报告：

```text
有效点：44
平均误差：2.705 mm
中位误差：2.475 mm
最大误差：6.038 mm
```

重要区别：User2 CSV 用于作业坐标表达和人工核对；当前 MoveIt 动态渐进位规划实际读取的是转换后的 CR5 User0/base XYZ，因为 MoveIt 场景位于 base/world 链路中。不要把 User2 数值直接当成 MoveIt world 坐标。

若 D435、CR5 基座或 User2 被重新标定或移动，必须更新对应外参，不能继续使用旧转换文件。

## 7. ROS 2、MoveIt 与真机关节状态

系统版本：ROS 2 Humble、MoveIt 2、RViz。

真机只读反馈发布器：

```text
/home/haoran/Arm/meter_replacement/robot/real/cr5_tcp/publish_cr5_joint_states.py
```

真机规划只允许使用：

```text
/real_joint_states
```

不要使用 `/joint_states` 作为真机规划起点，因为它可能带有 mock 或旧的持久化数据。

CR5 网络参数：

```text
IP：192.168.5.1
Dashboard：29999
Motion：30003
Feedback：30004
```

真机运动前提：

- DobotStudio Pro 已连接；
- CR5 已使能且无报警；
- 已切换到 TCP/IP 二次开发模式；
- 现场人员确认工作区无人、急停可用；
- 规划起点来自实时 `/real_joint_states`；
- 路径通过 MoveIt 碰撞检查。

## 8. 两段规划逻辑

第一段：任意实时当前位置到固定安全过渡位。

第二段：安全过渡位到本次 D435 检测出的第一个电表渐进位。

安全过渡位位于：

```text
/home/haoran/Arm/meter_replacement/config/poses.yaml
home_profiles.vision_safe_transition
```

当前关节角（rad）：

```text
joint1=-1.658635833347920
joint2= 0.322039252398809
joint3=-2.119355230558892
joint4= 1.496029598138698
joint5= 1.176119141281840
joint6= 1.744114799540915
```

该位置由用户现场摆放，并已对当前电表箱、D435 立柱、横梁和相机碰撞模型做过验证。若工作站几何发生变化，必须重新验证。

每次运行都会重新规划两段路径：当前位置会变化，D435 检测出的电表位置也可能变化，因此不能把旧轨迹当成固定模板直接复用。

## 9. 速度控制现状

统一入口中的手动参数：

```python
MOTION_SPEED_PERCENT = 100
```

位置：

```text
/home/haoran/Arm/meter_replacement/tools/start_vision_guided_grasp.py
```

当前实现同时：

- 调用越疆 `SpeedFactor(N)`；
- 使用 `100/N` 缩放 ServoJ 主机发送时间。

因此：

```text
25% → time_scale=4
50% → time_scale=2
100% → time_scale=1
```

执行回执会记录：

```text
dobot_global_speed_factor_percent
servoj_time_scale
```

## 10. RViz 和进程约束

- 同一时间只允许一套 RViz/MoveIt 场景。
- 关闭 RViz 窗口不代表后台 launch 已退出。
- 启动前用 `pgrep -a rviz2` 确认没有旧 RViz。
- 只终止确认属于旧场景的完整 launch，不要误停 RealSense 或其他节点。
- `publish_cr5_joint_states.py` 同时只允许一个实例。
- `/real_joint_states` 应恰好只有一个发布者。

统一协调器完成后会关闭整套 launch。RViz、MoveIt 或 Conda 子进程在退出阶段偶尔显示 `-11`、`-15` 或 `exit code 1`；判断业务是否成功应优先查看：

```text
VISION TWO-STAGE WORKFLOW PASSED
```

以及以下回执的最终 `status`：

```text
/home/haoran/Arm/meter_replacement/logs/commissioning/cr5_vision_safe_transition.json
/home/haoran/Arm/meter_replacement/logs/commissioning/cr5_vision_meter_approach.json
```

若协调器自身退出码为 1，则不是正常清理，应查看最新目录中的 `launch.log`：

```text
/home/haoran/.ros/log/<最新时间目录>/launch.log
```

## 11. 新项目第一阶段建议

新对话先完成以下最小闭环：

1. 检查本机是否已有 OpenClaw、版本和启动方式。
2. 与用户确认演示输入、预期画面和是否需要真实硬件。
3. 默认先做无运动演示：OpenClaw 接收指令并读取一份稳定目标 CSV。
4. 再接 `--plan-only`，展示 D435 目标到 MoveIt 候选轨迹。
5. 真机运动单独作为显式授权步骤，不作为默认行为。

不要一开始复制整个电表项目，也不要先设计复杂插件体系。先证明 OpenClaw 能可靠调用一个只读接口，再逐步接入规划和硬件执行。

## 12. 可复制到新对话的开场说明

```text
请先阅读 /home/haoran/Arm/openclaw_ball/HANDOFF.md。
这是独立的 OpenClaw 演示子项目，只修改 /home/haoran/Arm/openclaw_ball。
可以复用 /home/haoran/Arm/meter_replacement 的稳定接口，但不要复制生成目录，
也不要修改原项目或发送 CR5 真机运动，除非我明确授权。
请先检查 openclaw_ball、本机 OpenClaw 安装状态和现有环境，再和我确认演示目标。
```

