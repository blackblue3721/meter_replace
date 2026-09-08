# CAD 资产归档

这里保存供应商或设计源文件；ROS、MoveIt 和 MuJoCo 运行时不直接解析这些
STEP、SolidWorks 或 PDF 文件。运行时网格统一放在对应 ROS 包的 `meshes/`
目录，路径通过项目配置引用。

## 夹爪

- 型号：CHA EPG40-100-C7
- 源文件：`gripper/epg40-100/source/EPG40-100.step`
- 图纸：`gripper/epg40-100/source/CHAEPG40-100-C7_drawing.pdf`
- 运行时网格：`robot/src/cr5_description/meshes/epg40_100/`
- 图纸总行程：40 mm
- 当前状态：本体与两个活动滑块已建模；接触电表的定制指爪尚无 CAD。
- 兼容策略：`gripper_tcp` 暂保留原验证路径使用的 240 mm；实体指爪安装后
  必须重新做 TCP、碰撞包络、质量惯量和夹持力标定。

## 电批

- 型号文件：`electric_screwdriver/e350-dsf0045/source/E350-DSF0045.step`
- 图纸：`electric_screwdriver/e350-dsf0045/source/E350-DSF0045_drawing.pdf`
- 当前状态：仅归档，未加入 URDF、MoveIt 或 MuJoCo。

## 电表

- STEP：`electric_meter/source/electric_meter.step`
- SolidWorks 零件/装配体：`electric_meter/source/`
- 运行时网格：`simulation/ros2/meter_grasp/meshes/electric_meter_*.stl`
- 工程外形基准：160 × 110 × 70 mm。
- MoveIt 和 MuJoCo 接触仍使用简化盒体；CAD STL 仅负责外观显示。

## 工作区 D435

- 型号：Intel RealSense D435
- 用户提供源文件：`camera/intel-realsense-d435/source/D435_Solid.SLDPRT`
- 运行时外观：复用 ROS Humble `realsense2_description` 提供的官方
  `package://realsense2_description/meshes/d435.dae`
- 官方外形碰撞包络：90 × 25.05 × 25 mm
- 当前安装初值：相对右侧 CR5 底座向物理左侧 0.44 m、相机中心离桌面
  0.54 m；Y 位置和光轴方向等待支架固定后外参标定。
- 支架：竖杆和横梁使用照片推定的保守盒体参与 MoveIt 碰撞检查，最终尺寸
  必须现场复测。

不要在此目录和 ROS 包内重复保存同一份 STL。源格式负责追溯，运行时 STL
负责加载。
