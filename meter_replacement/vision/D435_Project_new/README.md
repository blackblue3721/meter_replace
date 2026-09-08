# D435 电表定位

最小工作流：

```text
realtime_multi_meter_d435.py
  -> logs/robot_ready_targets.csv          # D435 稳定坐标
convert_ready_targets_to_cr5.py
  -> logs/cr5_ready_targets.csv            # CR5 User0/base 坐标
convert_cr5_targets_user0_to_user2.py
  -> logs/cr5_ready_targets_user2.csv      # CR5 User2 坐标
```

运行检测：

```bash
python realtime_multi_meter_d435.py
```

转换坐标：

```bash
python convert_ready_targets_to_cr5.py
python convert_cr5_targets_user0_to_user2.py
```

关键输入：

- `runs/detect/train-4/weights/best.pt`：当前 YOLO 模型。
- `calibration/result/camera_to_cr5_transform.json`：D435 到 CR5 User0/base 的外参。
- `user2_frame_config.json`：User2 在 User0 中的位姿。

重新标定只使用：

```bash
python collect_cr5_chessboard_9points.py
python calibrate_camera_to_cr5_9points.py
```

标定数据采集时使用 CR5 `User 0` 和针尖 `Tool 2`。
