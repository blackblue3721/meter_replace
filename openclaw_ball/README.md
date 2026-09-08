# OpenClaw Ball

四色小球的确定性视觉后端。当前里程碑只开放离线图片检测；MoveIt 和真机执行保持锁定。

## 运行

```bash
cd /home/haoran/Arm/openclaw_ball
/usr/bin/python3 cli.py red --image /path/to/image.jpg
/usr/bin/python3 server.py
```

HTTP 示例：

```bash
curl -s http://127.0.0.1:8765/health
curl -s -X POST http://127.0.0.1:8765/v1/tasks \
  -H 'Content-Type: application/json' \
  -d '{"color":"red","mode":"detect","source":"image","image_path":"/path/to/image.jpg"}'
```

D435 实时检测需要 Conda `arm` 环境（该环境已安装 `pyrealsense2`）：

```bash
/home/haoran/miniconda3/bin/conda run --no-capture-output -n arm \
  python cli.py red --camera

curl -s -X POST http://127.0.0.1:8765/v1/tasks \
  -H 'Content-Type: application/json' \
  -d '{"color":"red","mode":"detect","source":"camera"}'
```

相机启动后默认预热 60 帧（约 2 秒），等待自动白平衡和自动曝光稳定后再检测。

本项目使用 `calibration/camera_to_cr5_corrected.json`。它保留原项目 44 点手眼旋转矩阵，
并记录 2026-09-08 绿色方块人工确认夹取位得到的单点全局平移补偿；四种颜色共用该变换。
此前红色方块的经验修正仅存档于 `calibration/camera_to_cr5_red_legacy_20260907.json`，不再使用。单点补偿不能替代完整的
多点手眼重标定；相机、机械臂基座或工作台移动后必须重新标定。

HSV 阈值位于 `config.py`，应使用现场 D435 图像调参。相机到 CR5 base 的旋转沿用原项目标定；
平移使用本项目独立记录的真机试抓补偿，不修改原项目文件。

现场调参（拖动滑块，按 `q` 输出阈值）：

```bash
/home/haoran/miniconda3/bin/conda run --no-capture-output -n arm \
  python tools/tune_hsv.py red --camera
```

红色跨越 HSV 色相首尾，必要时分别调试低红和高红两段范围。

一次取帧检查四种颜色并保存坐标和标注图：

```bash
/home/haoran/miniconda3/bin/conda run --no-capture-output -n arm \
  python tools/check_colors.py
```

结果写入 `logs/color_check_*.json` 和 `logs/color_check_*.jpg`。方块的理论圆度约为
`0.785`，可以通过当前圆度门限；它们适合验证颜色和坐标，但最终仍需用实际小球验证抓取目标。

## EPG40-100 夹爪

夹爪通过 CR5 末端 RS485 控制，现场地址为 `9`。只读状态与持续通信：

```bash
/usr/bin/python3 robot/gripper.py status
/usr/bin/python3 robot/gripper.py monitor
```

实体动作必须提供确认字符串：

```bash
/usr/bin/python3 robot/gripper.py enable --confirmation MOVE_REAL_EPG40_100
/usr/bin/python3 robot/gripper.py open --confirmation MOVE_REAL_EPG40_100
/usr/bin/python3 robot/gripper.py close --confirmation MOVE_REAL_EPG40_100
```

蓝灯慢闪表示超过一秒没有485通信。运行 `monitor` 时持续轮询，蓝灯应保持常亮。

## 方块正面抓取规划

最终接触点在方块几何中心基础上沿相机视线方向继续前进 15 mm；预抓取位仍位于
几何中心前方 150 mm。夹爪到达接触点后执行低力低速完全闭合，保持夹持完成撤离和安全返回，
确认到达默认安全位后才重新完全打开。

以下命令只拍摄一次并冻结目标，然后启动 MoveIt/RViz 规划；不会发送真机运动：

```bash
/usr/bin/python3 tools/plan_front_grasp.py red
```

规划产物和冻结坐标写入 `logs/grasp/<时间>/`。预抓取点位于方块中心沿相机视线
反方向150 mm，接触段使用 Pilz LIN 水平推进。机械臂遮挡目标后不会重新读取相机。

默认安全撤离位记录在 `config/default_safe_pose.json`。2026-09-08 13:36 的当前版本
由操作员指定，并直接从真机实时反馈采集；后续任务应先直线退出目标区域，再规划回该关节位。

## OpenClaw 真机统一入口

真机工作区检查完成、CR5已使能并处于 TCP/IP 模式后，启动常驻服务：

```bash
cd /home/haoran/Arm/openclaw_ball
/home/haoran/miniconda3/bin/conda run --no-capture-output -n arm \
  python server.py --allow-motion
```

OpenClaw 的“抓取某色方块”调用 `openclaw/grasp_color.sh <color>`。每次都会重新采集相机、
冻结该次三维坐标并完整规划；相机、MoveIt 场景和 HTTP 服务保持常驻。不加
`--allow-motion` 时，仍可识别，但任何真机请求都会被拒绝。任务从 `/real_joint_states`
的当前真机状态起步；默认安全位只是完成抓取后的返回目标，不是启动前置条件。
若已使能但 TCP 运动队列暂停，执行前会复用原项目的受保护 `continue()` 恢复入口；
只有机械臂静止、无报警且 29999/30004 反馈一致时才会恢复。
每段真机轨迹完成后保留 1 秒控制器状态收敛时间，再启动下一段安全预检。
