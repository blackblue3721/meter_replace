# 项目运行约束

- 修改场景后如需重启 RViz/MoveIt，先终止旧实例所属的完整 `ros2 launch` 进程。
- 关闭 RViz 窗口不代表后台节点已经退出；启动前必须用 `pgrep -a rviz2` 确认 RViz 数量为 0。
- 每次只允许启动一套 RViz/MoveIt 场景，启动后再次确认只有一个 `rviz2` 和一个对应 launch。
- 清理场景时只终止已确认的旧 launch，不误停 RealSense、真实 CR5 只读反馈或其他无关节点。
- 真机只读反馈固定发布到 `/real_joint_states`；真机 MoveIt/RViz 只订阅该隔离话题，禁止用可能残留 mock 数据的 `/joint_states` 作为真机规划起点。
- `publish_cr5_joint_states.py` 同一时间只允许一个实例；启动真机场景前确认 `/real_joint_states` 恰好只有一个发布者。
