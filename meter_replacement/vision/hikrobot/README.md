# HIKROBOT MV-CE050-30UC 实时预览

相机是 USB3 Vision/GenICam 工业相机，不会出现为 `/dev/video0`。

## 安装结构

- 海康官方 MVS 5.0.2：`/opt/MVS`
- 机械臂 Python：`/home/haoran/miniconda3/envs/arm/bin/python`
- Python 官方绑定：`/opt/MVS/Samples/64/Python/MvImport`
- 项目程序：`mvs_live_view.py`

MVS 驱动和运行库安装在系统；conda 环境只提供 Python、NumPy 和 OpenCV。

## 自动实时预览

```bash
./mvs_live_view.py
```

脚本使用官方 `MvCameraControl` SDK 自动连接序列号 `00DA8822107`，画面会显示实时帧号和 FPS。按 `Q` / `Esc` 或点击窗口右上角关闭按钮即可停止取流并释放相机。

程序内部会自动使用 `arm` conda Python、`/opt/MVS` 官方 SDK 和官方 Python API，不需要 shell 启动脚本。

需要手动调整曝光、增益、ROI 或触发参数时，可从桌面打开官方 MVS 客户端。客户端和 Python 预览不能同时独占同一台相机。
