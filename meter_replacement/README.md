# 越疆 CR5 单臂电表安装与拆取项目

更新时间：2026-08-07

本目录是“越疆双 CR5 机械臂自动更换电表项目”的正式开发目录。当前先聚焦
实验室右侧单臂，在 ROS 2 Humble、MoveIt 2、RViz、mock ros2_control 和
MuJoCo 中完成可验证的软件与仿真闭环，再逐步迁移到真实 CR5、真实夹爪、
电批和视觉系统。

当前已经完成两个端到端模拟工艺：

```text
安装：桌面电表 → 抓取 → 抬升 → 姿态翻转 → 插入指定表位 → 释放 → Home
拆取：指定表位 → 插入抓取 → 抽出 → 姿态还原 → 放回桌面 → 释放 → Home
```

两个正式业务入口分别是：

```bash
ros2 run meter_grasp run_meter_grasp
ros2 run meter_grasp run_meter_remove
```

真机只读同步、MoveIt/RViz 和 D435 检测可用一个安全入口启动：

```bash
cd ~/Arm/meter_replacement
/usr/bin/python3 tools/start_vision_guided_grasp.py --check
/usr/bin/python3 tools/start_vision_guided_grasp.py --plan-only
```

不需要手工执行 `conda activate` 或 `conda deactivate`。公开入口会让 ROS 使用
系统 Python，并只让 YOLO 子进程通过 `conda run -n arm` 使用 `arm` 环境。
`tools/plan_first_stable_meter.py` 是该入口自动启动的内部协调器，不是第二个操作入口。

该入口启动后会冻结 D435 的第一个稳定电表目标，依次生成“任意实时当前位置
→ 已验证安全过渡位 → 视觉渐进位”两段连续 MoveIt 候选轨迹；第二段明确以
第一段终点作为规划起点。`--plan-only` 只生成候选轨迹。一键规划并执行两段轨迹：

```bash
/usr/bin/python3 tools/start_vision_guided_grasp.py
```

每次运行都会依据实时真机关节状态和本次 D435 坐标重新规划，两段执行之间
必须通过实际终点反馈检查。入口不会使能 CR5，不会下降或夹取电表。
按 `Ctrl+C` 会统一停止它启动的进程。
若暂时不使用 D435，可增加 `--scene-only`。

> 安全边界：统一入口默认调用 `robot/real/cr5_tcp/` 下带哈希、起点、模式和
> 终点反馈校验的执行器；它不会自动使能、清错、下降或夹取电表。

## 1. 当前阶段结论

### 已验证

- CR5 与占位夹爪可在 RViz/MoveIt 中正确显示和规划；
- 工作台、电表箱、断路器区域和四个电表位已按实测尺寸建模；
- 电表 CAD 标称尺寸为 `0.160 × 0.110 × 0.070 m`，并定义了正面、背面、顶部；
- OMPL 用于自由空间运动，Pilz LIN 用于下降、插入、抽出和撤离；
- 多种子碰撞感知 IK 可筛选较自然的关节分支，避免明显绕圈；
- 安装状态机已端到端执行成功，表位握手为
  `EMPTY → RESERVED → OCCUPIED`；
- 拆取状态机已端到端执行成功，表位握手为
  `OCCUPIED → RESERVED → EMPTY`；
- 每段轨迹执行后都会读取 `/joint_states` 检查实际终点；
- 电表抓住后作为附着碰撞物参与 MoveIt 碰撞检查；
- MuJoCo 中已完成依靠夹持力和摩擦、而非附着约束的重力夹持实验；
- Linux 虚拟机已能访问真实 CR5 的 `192.168.5.1`，端口
  `29999`、`30003`、`30004` 均可建立 TCP 连接；
- 已保存越疆 TCP/IP Python V3 示例库，但尚未向真机发送运动命令。

### 尚未完成

- 统一的 mock/真机机器人执行接口；
- 真实 CR5 状态读取、低速点动和轨迹执行；
- 真实夹爪、电批、线束工具模型与控制接口；
- Intel RealSense D435 与海康眼在手上相机的标定和感知；
- 开盖、拆螺丝、拆线、复核等真实工艺；
- 双臂任务调度、区域锁和安全 PLC 联调；
- 结构化业务日志和自动化单元测试。

## 2. 系统架构

这些工具不是同一级软件，也不承担相同工作：

```text
URDF/Xacro/STL                 SRDF + MoveIt 配置
机器人几何、关节、碰撞体          规划组、末端、Home、限位
          \                         /
           +------ MoveIt 2 -------+
                  IK、碰撞、路径
                         |
                         v
             FollowJointTrajectory
                         |
                         v
                  ros2_control
              当前 mock / 未来真机适配器
                         |
                         v
               CR5 关节状态与执行结果

Python 工艺状态机
  ├── 调用 MoveIt 规划和执行
  ├── 调用夹爪 Action
  ├── 管理电表附着
  ├── 管理表位状态
  └── 负责阶段握手、失败停止和恢复边界

RViz：显示模型、TF、碰撞环境和轨迹
MuJoCo：独立验证重力、接触、摩擦和夹持动力学
```

核心分工：

| 组件 | 作用 |
|---|---|
| URDF/Xacro | 描述机器人连杆、关节、碰撞体、惯量、夹爪和 TCP |
| SRDF | 描述 MoveIt 规划组、末端、命名姿态和允许碰撞矩阵 |
| MoveIt | 逆运动学、碰撞检测、路径规划和轨迹执行管理 |
| OMPL | 规划自由空间无碰路径 |
| Pilz LIN | 规划工艺要求明确的直线末端运动 |
| ros2_control | 接收轨迹并交给模拟硬件或未来真机硬件接口 |
| TF | 管理 `world`、机器人基座、电表箱、表位和相机坐标 |
| RViz | 可视化，不负责接触动力学 |
| MuJoCo | 接触、摩擦、重力和动力学实验，不替代 MoveIt |
| Python 状态机 | 编排本项目的安装和拆取工艺 |

## 3. 项目目录

```text
meter_replacement/
├── README.md
├── app/                              # 未来任务级应用入口，当前为占位
├── config/
│   ├── poses.yaml                    # 动作、Home、规划器和安全参数
│   └── scene.json                    # 工作区、碰撞物、表位和动力学参数
├── logs/                             # 未来结构化业务日志
├── robot/
│   ├── src/
│   │   ├── cr5_description/          # CR5、网格、占位夹爪和 TCP
│   │   └── cr5_moveit_config/        # SRDF、MoveIt、控制器和 RViz 配置
│   ├── build/                        # colcon 中间产物，不手工修改
│   ├── install/                      # 本工作区的安装视图
│   └── log/                          # colcon 构建日志
├── simulation/
│   ├── mujoco/                       # 接触动力学与夹持实验
│   └── ros2/meter_grasp/             # 当前业务 ROS 2 Python 包
├── tests/                            # 自动化测试目录，仍待完善
├── third_party/
│   └── TCP-IP-Python-V3/             # 越疆 TCP/IP Python 示例库
├── tools/                            # 末端工具占位模块
└── vision/                           # 相机、标定和图像存储占位模块
```

源码主要位于：

- `robot/src/`
- `simulation/ros2/`
- `simulation/mujoco/`
- `config/`

以下目录是生成物，不应手工修改：

- `robot/build/`
- `robot/install/`
- `robot/log/`
- `__pycache__/`

## 4. 三个主要 ROS 2 包

### 4.1 `cr5_description`

路径：`robot/src/cr5_description/`

它是机器人“身体说明书”：

| 文件 | 作用 |
|---|---|
| `urdf/cr5_robot.urdf` | CR5 六轴连杆、关节、轴方向、限位、惯量和网格引用 |
| `meshes/*.STL` | CR5 外观网格 |
| `urdf/epg40_100_gripper.xacro` | CHA EPG40-100-C7 本体、活动滑块、简化碰撞体和兼容 TCP |
| `meshes/epg40_100/*.stl` | 从供应商 STEP 转换得到的夹爪运行时外观网格 |
| `urdf/cr5_with_gripper.urdf.xacro` | 把 CR5 与夹爪组合 |
| `launch/display.launch.py` | 仅查看模型和关节的简单入口 |
| `rviz/cr5.rviz` | 简单模型查看布局 |

需要真正理解的是 `joint1` 至 `joint6` 的父子关系、`origin`、`axis`、
`limit`，以及 `base_link`、`Link6`、`gripper_base_link` 和
`gripper_tcp` 的含义。夹爪供应商 CAD 不包含接触电表的定制指爪，目前保留
240 mm 兼容 TCP；定制指爪到位后必须重做 TCP、碰撞包络和力学参数。
STL 顶点和自动生成的惯量 XML 不需要背诵。

当前仓库缺少基础 URDF/STL 的厂商版本号和来源校验，因此不能把它们直接
宣称为越疆官方模型。真实部署前必须确认来源、版本和坐标约定。

### 4.2 `cr5_moveit_config`

路径：`robot/src/cr5_moveit_config/`

它是 MoveIt 的“规划说明书”，由 MoveIt Setup Assistant 生成后再经项目
调整。

常用配置：

| 文件 | 作用 |
|---|---|
| `cr5_with_gripper.srdf` | 规划组、Home、末端执行器和允许碰撞矩阵 |
| `cr5_with_gripper.ros2_control.xacro` | 关节控制接口；当前使用 mock 硬件 |
| `initial_positions.yaml` | 模拟启动关节初值 |
| `joint_limits.yaml` | 规划速度和加速度限制 |
| `kinematics.yaml` | KDL 逆运动学插件配置 |
| `moveit_controllers.yaml` | MoveIt 应把轨迹发到哪个控制器 |
| `ros2_controllers.yaml` | ros2_control 实际加载哪些控制器和关节 |
| `pilz_cartesian_limits.yaml` | Pilz 直线规划速度和加速度限制 |
| `moveit.rviz` | RViz 的 MoveIt、轨迹和表位 Marker 布局 |

常用 launch：

| 文件 | 用途 |
|---|---|
| `demo.launch.py` | 启动完整 MoveIt mock 演示基础设施 |
| `move_group.launch.py` | 只启动 MoveIt 规划服务器 |
| `moveit_rviz.launch.py` | 只打开 MoveIt RViz |
| `rsp.launch.py` | 只启动 robot_state_publisher |
| `spawn_controllers.launch.py` | 加载 ros2_control 控制器 |
| `setup_assistant.launch.py` | 修改 SRDF、规划组和碰撞矩阵 |

日常不需要逐个启动这些文件。业务场景入口会复用 `demo.launch.py`。

### 4.3 `meter_grasp`

路径：`simulation/ros2/meter_grasp/`

这是本项目自己的业务 ROS 2 包：

| 文件 | 作用 |
|---|---|
| `launch/single_arm_scene.launch.py` | 启动 MoveIt、mock 控制器、RViz、碰撞场景和表位管理器 |
| `planning_scene.py` | 从 `scene.json` 创建桌面、电表箱、D435 支架和电表碰撞物 |
| `workcell_camera_visualizer.py` | 显示 D435 官方 CAD，并发布临时 `world → workcell_d435_link` 固定坐标 |
| `slot_manager.py` | 发布四个表位 TF、Marker 和槽位状态 |
| `meter_visualizer.py` | 根据 MoveIt 中的电表状态显示并跟随 SolidWorks CAD 外观 |
| `meshes/electric_meter_*.stl` | 电表主体和透明盖的精细视觉网格 |
| `pick_sequence.py` | 单段动作规划、IK、OMPL/Pilz、轨迹检查和执行 |
| `object_attachment.py` | 电表附着与解除附着 |
| `meter_grasp_state_machine.py` | 完整安装状态机 |
| `meter_remove_state_machine.py` | 完整拆取状态机 |
| `setup.py` | ROS 2 命令入口和安装规则 |

`pick_sequence.py` 是通用动作能力层，两个状态机是业务编排层。拆取状态机
继承并复用了既有 MoveIt、夹爪、关节反馈、附着和槽位服务，没有重新实现
规划器。

电表采用“精细视觉模型 + 简化碰撞模型”双层表达：两个 STL 只用于 RViz
和 MuJoCo 显示，`0.160 × 0.110 × 0.070 m` 盒体继续承担 MoveIt 碰撞检查
和 MuJoCo 接触计算。这样既保留 CAD 外观，也避免复杂网格降低规划和接触
求解的稳定性。

## 5. 场景和配置

### 5.1 坐标与尺寸

`world` 原点位于右侧 CR5 底座中心，统一使用 SI 单位：

- 长度：m
- 角度：rad
- 质量：kg
- 时间：s

实测/暂定工作区：

| 对象 | 当前尺寸或位置 |
|---|---|
| 工作台 | `2.0 × 1.0 m`，厚度暂定 `0.06 m` |
| CR5 | 底座中心距右短边 `0.60 m`，距桌前长边 `0.15 m` |
| 电表箱 | `0.70 × 0.80 × 0.20 m` |
| 左侧箱体 | 宽 `0.30 m`，断路器区域 |
| 右侧箱体 | 宽 `0.40 m`，四个电表位 |
| 电表 | CAD 标称外形 `0.160 × 0.110 × 0.070 m`，暂定质量 `1.5 kg` |

四个槽位：

```text
slot_r1_c1    slot_r1_c2
slot_r2_c1    slot_r2_c2
```

当前完整流程已在 `slot_r1_c2` 上验证。绿色、黄色和蓝色 Marker 分别表示
`EMPTY`、`RESERVED` 和 `OCCUPIED`。槽位框是任务目标和可视标记，不是
实体碰撞物；安装后的电表本身才是世界碰撞物。

### 5.2 `config/scene.json`

该文件也保存工作区 D435 的安装初值、官方外形包络以及竖杆/横梁碰撞体。
当前相机位置采用实测的“CR5 左侧 0.44 m、中心高 0.54 m”；Y 方向和光轴
姿态仍是照片推定值，支架固定后应以外参标定结果替换，不能把当前数值当作
最终视觉坐标。

定义：

- world 与工作区坐标约定；
- 桌面和电表箱碰撞几何；
- 电表尺寸、质量和语义轴；
- 四个槽位位姿和初始状态；
- MuJoCo 接触、摩擦和占位夹爪参数；
- 哪些数据来自实测，哪些仍是假设。

### 5.3 `config/poses.yaml`

定义：

- MoveIt group、Action 和 Service 名称；
- OMPL、Pilz LIN、速度、加速度和超时；
- Home 关节角和轨迹安全阈值；
- 夹爪开度；
- 桌面抓取、抬升、语义翻转和表位安装位姿；
- 多种子 IK 搜索和人体工学软偏好；
- 电表位置、Home 和执行结果容差。

位姿、速度、IP、超时和安全阈值应优先配置化，不应散落在业务代码中。

## 6. 两条完整状态机

### 6.1 安装：`run_meter_grasp`

```text
PREFLIGHT
→ SLOT_RESERVED
→ GRIPPER_OPEN
→ PREGRASP
→ AT_PICK
→ GRIPPED
→ ATTACHED
→ LIFTED
→ SEMANTIC_FLIPPED
→ AT_SLOT_APPROACH
→ AT_SLOT
→ DETACHED
→ SLOT_OCCUPIED
→ RELEASED
→ RETREATED
→ HOME
```

关键点：

- 启动时要求机械臂在 Home、电表在桌面、目标槽位为 `EMPTY`；
- OMPL 规划到预抓取位；
- Pilz LIN 直线下降与抬升；
- joint5 完成电表桌面姿态到安装姿态的语义翻转；
- 多分支 IK 为箱前动作选择自然姿态；
- 直线插入后解除附着并把槽位标记为 `OCCUPIED`；
- 任何阶段失败都停止后续动作。

### 6.2 拆取：`run_meter_remove`

```text
PREFLIGHT
→ SLOT_RESERVED
→ GRIPPER_OPEN
→ AT_SLOT_APPROACH
→ AT_SLOT
→ GRIPPED
→ ATTACHED
→ EXTRACTED
→ SLOT_EMPTY
→ AT_UNINSTALL_WAYPOINT
→ SEMANTIC_UNFLIPPED
→ AT_DESKTOP
→ DETACHED
→ RELEASED
→ RETREATED
→ HOME
```

关键点：

- 启动时要求机械臂在 Home、电表位于目标表位、槽位为 `OCCUPIED`；
- 先锁定槽位，再接近和直线插入；
- 夹紧并附着后沿原路径直线抽出；
- 抽出成功后把槽位标记为 `EMPTY`；
- 先移动到柜外安全中转位，再通过 joint5 还原桌面姿态；
- 在桌面解除附着、张开夹爪、撤离并返回 Home；
- 抓住后若发生错误，保持夹紧，不自动松爪。

这只是螺丝、透明盖、导线等前置工艺已经完成后的“机械拆取搬运”子流程，
不是完整真实拆表过程。

## 7. 从零启动和运行

关闭所有终端后，需要两个终端。精简版命令同时保存在
`/home/haoran/Arm/review`。

### 7.1 终端一：启动基础场景

```bash
cd ~/Arm/meter_replacement
conda deactivate 2>/dev/null || true
source /opt/ros/humble/setup.bash
source robot/install/setup.bash
ros2 launch meter_grasp single_arm_scene.launch.py
```

保持该终端运行。它会启动 MoveIt、RViz、mock ros2_control、控制器、TF、
碰撞场景和表位管理器，但不会自动执行工艺。

### 7.2 终端二：执行安装

```bash
cd ~/Arm/meter_replacement
conda deactivate 2>/dev/null || true
source /opt/ros/humble/setup.bash
source robot/install/setup.bash

ros2 run meter_grasp run_meter_grasp \
  --ros-args \
  -p target_slot:=slot_r1_c2 \
  -p execute:=true
```

新场景初始状态是电表在桌面、四个槽位均为空，因此重启后的第一条完整任务
应当是安装。成功标志：

```text
Meter grasp process completed: installed in 'slot_r1_c2'
```

### 7.3 终端二：随后执行拆取

安装成功且机械臂回到 Home 后，可在同一终端执行：

```bash
ros2 run meter_grasp run_meter_remove \
  --ros-args \
  -p target_slot:=slot_r1_c2 \
  -p execute:=true
```

成功标志：

```text
Meter removal process completed: removed from 'slot_r1_c2' and placed on the desktop
```

两个状态机默认均为 `execute:=false`。省略 `-p execute:=true` 时只做预检，
不会移动模拟机械臂。

### 7.4 单段调试

开发时可用：

```bash
ros2 run meter_grasp run_pick_sequence \
  --ros-args \
  -p target_stage:=slot_retreat \
  -p target_slot:=slot_r1_c2 \
  -p execute:=false
```

单段入口用于计划验证，不是日常生产入口。

## 8. 构建与 source

修改 Python 包入口、launch 或配置后，在项目根目录构建：

```bash
cd ~/Arm/meter_replacement
source /opt/ros/humble/setup.bash

colcon --log-base robot/log build \
  --base-paths robot/src simulation/ros2 \
  --build-base robot/build \
  --install-base robot/install \
  --symlink-install
```

`colcon` 是 ROS 2 多包工作区的构建编排器。它读取 `package.xml`，按依赖
顺序调用 CMake/ament 或 setuptools，并生成：

| 目录 | 含义 |
|---|---|
| `robot/build/` | 构建中间文件 |
| `robot/install/` | 本项目 ROS 包的可发现安装视图 |
| `robot/log/` | 构建日志 |

系统 ROS 2、MoveIt 和 RViz 安装在 `/opt/ros/humble`，不是安装在
`robot/install/`。后者只属于本工作区。

每个新终端都要重新 `source`，因为 `source` 只修改当前 shell 的环境变量：

```bash
source /opt/ros/humble/setup.bash
source robot/install/setup.bash
```

## 9. 规划方法与工程经验

### 9.1 OMPL 与 Pilz LIN

- Home、桌面预抓取、箱前转移和返回 Home：OMPL；
- 抓取下降、抬升、表位插入、抽出和撤离：Pilz LIN。

自由空间动作追求“找到无碰路径”，接触附近动作追求“沿明确方向直线运动”。

### 9.2 末端到达不代表轨迹合理

同一个 TCP 位姿可能有多组逆解。早期曾出现：

- 单关节转动过大或绕圈；
- 肘部突然切换分支；
- 机械臂从桌面下方绕行；
- 电表位置正确但正反、上下错误；
- 接近腕部奇异区域后出现不自然动作。

当前程序通过碰撞感知 IK、多种子搜索、关节行程阈值、人体工学软偏好和实际
关节反馈共同筛选动作。软偏好只用于排序可行解，不是把最终关节角写死。

### 9.3 状态机不能依靠 `sleep`

流程推进依据：

- MoveIt Action 结果；
- 夹爪 `GripperCommand` 结果；
- PlanningScene Service 结果；
- 槽位参数设置后的回读确认；
- `/joint_states` 实际终点误差。

固定等待时间不能证明机械臂已经完成动作，因此不能作为流程握手。

## 10. MuJoCo 动力学实验

文件：

```text
simulation/mujoco/grasp_contact_test.py
```

回归测试：

```bash
cd ~/Arm/meter_replacement
conda activate arm
python simulation/mujoco/grasp_contact_test.py --check
```

预期：

```text
nominal_hold_check: PASS
low_friction_drop_check: PASS
grasp_contact_test: PASS
```

查看器：

```bash
python simulation/mujoco/grasp_contact_test.py
```

实验启用重力，电表不使用焊接或附着约束。标称摩擦下电表应被双指保持，低
摩擦下应明显滑落。当前质量、摩擦、刚度、阻尼、夹爪力和惯量仍是工程估计，
只能用于算法与参数敏感性验证，不能视为真实夹持力预测。

MuJoCo 查看器显示 SolidWorks 导出的电表 CAD 外观，但动力学接触仍由简化
盒体计算；CAD 网格本身设置为不可碰撞、零附加质量，不会重复改变电表质量。

MoveIt 与 MuJoCo 当前是互补的独立链路：

- MoveIt 负责几何可达性、碰撞和轨迹；
- MuJoCo 负责重力、接触、摩擦和夹持稳定性；
- 尚未建立完整的 MoveIt 轨迹到 MuJoCo 执行桥。

## 11. 真机准备情况

### 已掌握的信息

```text
CR5 IP: 192.168.5.1
子网掩码: 255.255.255.0
Windows 有线网卡: 192.168.5.100
Linux 虚拟机: 192.168.12.130
```

Linux 虚拟机已验证：

- 能 `ping 192.168.5.1`；
- TCP 端口 `29999`、`30003`、`30004` 可连接。

厂商示例库：

```text
third_party/TCP-IP-Python-V3/
commit: 7fb060c9ad88332ec6fe8e768d72551c65af24f8
```

这些结果只证明网络和 TCP 服务可达，不代表机器人已经获得运动授权，也不
代表 MoveIt 轨迹可以直接发送。

### 真机迁移原则

业务状态机不应直接散落 `MovJ`/`MovL` 字符串。下一阶段建立统一执行接口：

```text
Meter workflow
      |
RobotExecutor 抽象接口
      |
      +-- MockMoveItExecutor
      |
      +-- DobotCR5Executor
```

状态机继续描述“做什么”，执行适配器负责“通过 mock 还是真实 CR5 去做”。
真机接入顺序必须是：

1. 只读机器人模式、报警和关节状态；
2. 验证急停、使能、速度限制和工作区；
3. 低速、小范围、单段动作；
4. 比较规划关节与真实反馈；
5. 才允许完整流程。

### 当前真机迁移步骤（只读阶段）

真实 CR5 网络和安全开关集中在 `config/hardware.yaml`。只读诊断入口为：

```bash
cd ~/Arm/meter_replacement
conda deactivate 2>/dev/null || true
/usr/bin/python3 robot/real/cr5_tcp/read_cr5_state.py --samples 1
```

该程序只连接反馈端口 `30004`，复用固定提交版本的越疆官方反馈数据结构，
打印机器人模式、使能/运行/报警状态、厂商原始 `J1..J6` 和 TCP 位姿。它不
连接运动端口，不发送任何命令，也不会自动使能或清错。

在真实夹爪尚未到货期间，MoveIt 继续保留占位夹爪的长度、TCP 和碰撞包络；
真机工艺中的夹爪开合暂定为 `virtual_handshake` 软件握手。不能向控制器设置
虚假的工具质量，也不能把“虚拟夹紧成功”误认为真实工件已经被夹住。

在厂商关节反馈与 ROS 模型的关节顺序、角度单位、零位和正方向完成逐轴校验
前，禁止发送既有仿真 Home 或整段工艺轨迹。

只读 RViz 姿态对照分两个终端启动：`display.launch.py` 使用
`external_joint_states:=true` 后不再启动本地关节滑块发布器；
`publish_cr5_joint_states.py` 只把 30004 反馈转换为隔离的 ROS `/real_joint_states`，避免残留 mock `/joint_states` 污染真机规划起点。
当前映射依据 CR5 ROS 参考驱动采用同序、同方向、零偏置和度转弧度。2026-08-07
已将真实反馈姿态与 RViz 模型完成外观对照，配置状态更新为
`visual_validation_passed`；该结论只批准状态显示，尚未批准运动。

下一道只读安全门是 `robot/real/cr5_tcp/preflight_cr5.py`。它通过 29999 仅查询
`RobotMode/GetAngle/GetPose/GetErrorID`，再与 30004 反馈交叉核对；源码中不
包含使能、清错、调速或运动调用。

首次实机动作使用 `commission_j6_step.py`：默认只计算 J6 `+2°` 目标且不连接
30003。真正执行还需把配置从 `read_only` 显式切换到 `commissioning`、人工
使能、准确确认字符串以及模式/报警/反馈一致性全部通过；程序只执行一段，
不会自动返回或衔接既有工艺路径。

2026-08-07 已完成首次运动前的现场人工安全确认，`access_mode` 仅为
`first_j6_positive_2deg_only` 切换到 `commissioning`。完整 Home、安装和拆取
流程仍明确标记为未授权。

首次尝试中 `JointMovJ` 返回接收成功，但因进入 TCP 模式后未再次调用
`EnableRobot()`，TCP 运动队列没有开启，机械臂未运动且 `Sync()` 返回 `-1`。
按照越疆协议，在再次使能前先用 `clear_pending_queue.py` 调用 `ResetRobot()`
停止并清空遗留队列。后续调试必须明确区分“命令被队列接收”和“运动执行并由
反馈验证完成”。

### 已登记、暂缓实现：可翻转透明盖

电表 CAD 已将主体和透明盖拆成独立视觉网格。后续将把透明盖建模为带铰链轴、
开启角度限制和盖扣接触参数的可动部件，并分别建立 MoveIt 工艺碰撞近似和
MuJoCo 接触模型。开始该任务前仍需补测铰链轴位置、最大开启角和卡扣结构；
本阶段先保留，不影响 CR5 只读接入与无负载轨迹迁移。

## 12. 代码质量与后续重构

当前代码属于“行为已经验证的研究原型”：

- 通用规划能力和业务状态机已经分层；
- 配置集中在 YAML/JSON；
- 安装与拆取是两个独立状态机；
- 使用标准 MoveIt、Pilz、TF、Action 和 Service，没有重写规划算法；
- 失败时显式停止并保留安全状态。

仍需改进：

- `pick_sequence.py` 职责较多，应继续拆出 MoveIt 客户端、IK 选择和轨迹
  安全模块；
- 自动化测试不足；
- `logs/` 尚未记录任务 ID、阶段耗时、失败原因和真机报警；
- mock 与真机配置尚未明确分离；
- 工艺位姿仍是固定候选值，未来要由视觉和标定输入。

不要因为文件多就简单合并。机器人项目天然包含模型、配置、启动文件、
控制器、规划器和业务节点。判断结构是否健康，应看职责是否清楚、参数是否
集中、生成物是否与源码分离。

## 13. 下一阶段

下一项明确任务是“统一机器人执行接口”：

1. 定义与业务状态机无关的机器人状态和执行接口；
2. 用现有 MoveIt/mock 行为实现第一个适配器，保证回归行为不变；
3. 为 Dobot TCP/IP V3 建立只读适配器；
4. 把 IP、端口、速度、超时和执行模式放入配置；
5. 加入真机必须满足的模式、报警、使能和授权检查；
6. 完成只读验证后，再讨论低速运动命令。

视觉、真实夹爪、拆螺丝和双臂协同将在执行接口稳定后逐步接入，不在状态机
中直接耦合厂商通信细节。
