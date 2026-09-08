#!/home/haoran/miniconda3/envs/arm/bin/python
"""使用海康官方 MVS SDK 自动打开第一台 USB 相机并实时预览。"""

import os
import sys
import time
from ctypes import CDLL, POINTER, RTLD_GLOBAL, byref, c_ubyte, cast, memset, sizeof


MVS_ROOT = "/opt/MVS"
MVS_PYTHON_API = f"{MVS_ROOT}/Samples/64/Python/MvImport"
os.environ["MVCAM_SDK_PATH"] = MVS_ROOT
os.environ["MVCAM_COMMON_RUNENV"] = f"{MVS_ROOT}/lib"
os.environ["MVCAM_GENICAM_CLPROTOCOL"] = f"{MVS_ROOT}/lib/CLProtocol"
sys.path.insert(0, MVS_PYTHON_API)
CDLL(f"{MVS_ROOT}/lib/64/libMvCameraControl.so", mode=RTLD_GLOBAL)

import cv2
import numpy as np

from MvCameraControl_class import (
    MV_ACCESS_Exclusive,
    MV_CC_DEVICE_INFO,
    MV_CC_DEVICE_INFO_LIST,
    MV_CC_PIXEL_CONVERT_PARAM_EX,
    MV_FRAME_OUT,
    MV_TRIGGER_MODE_OFF,
    MV_USB_DEVICE,
    MvCamera,
    PixelType_Gvsp_BGR8_Packed,
)


WINDOW_NAME = "HIKROBOT MV-CE050-30UC - MVS Live"


def decode_text(value) -> str:
    return bytes(value).split(b"\0", 1)[0].decode("utf-8", errors="replace")


def check(ret: int, action: str) -> None:
    if ret != 0:
        raise RuntimeError(f"{action} failed: 0x{ret & 0xFFFFFFFF:08x}")


def main() -> int:
    camera = None
    initialized = False
    opened = False
    grabbing = False

    try:
        check(MvCamera.MV_CC_Initialize(), "initialize SDK")
        initialized = True

        devices = MV_CC_DEVICE_INFO_LIST()
        check(MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE, devices), "enumerate devices")
        if devices.nDeviceNum == 0:
            raise RuntimeError("未找到海康 USB 工业相机")

        selected = None
        for index in range(devices.nDeviceNum):
            info = cast(devices.pDeviceInfo[index], POINTER(MV_CC_DEVICE_INFO)).contents
            usb = info.SpecialInfo.stUsb3VInfo
            model = decode_text(usb.chModelName)
            serial = decode_text(usb.chSerialNumber)
            print(f"[{index}] {model}, serial={serial}")
            if selected is None or model == "MV-CE050-30UC":
                selected = info
            if model == "MV-CE050-30UC":
                break

        camera = MvCamera()
        check(camera.MV_CC_CreateHandle(selected), "create camera handle")
        check(camera.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0), "open camera")
        opened = True
        check(camera.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF), "disable trigger")
        check(camera.MV_CC_StartGrabbing(), "start grabbing")
        grabbing = True

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW_NAME, 1280, 960)
        print("实时取流已开始；按 Q 或 Esc 退出。")

        frame = MV_FRAME_OUT()
        output_buffer = None
        output_size = 0
        fps = 0.0
        frame_count = 0
        fps_started = time.monotonic()

        while True:
            memset(byref(frame), 0, sizeof(frame))
            ret = camera.MV_CC_GetImageBuffer(frame, 1000)
            if ret != 0 or not frame.pBufAddr:
                print(f"\r等待图像: 0x{ret & 0xFFFFFFFF:08x}", end="", flush=True)
                key = cv2.waitKey(1) & 0xFF
                window_closed = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
                if key in (ord("q"), ord("Q"), 27) or window_closed:
                    break
                continue

            try:
                width = frame.stFrameInfo.nWidth
                height = frame.stFrameInfo.nHeight
                required_size = width * height * 3
                if output_buffer is None or output_size != required_size:
                    output_buffer = (c_ubyte * required_size)()
                    output_size = required_size

                convert = MV_CC_PIXEL_CONVERT_PARAM_EX()
                memset(byref(convert), 0, sizeof(convert))
                convert.nWidth = width
                convert.nHeight = height
                convert.pSrcData = frame.pBufAddr
                convert.nSrcDataLen = frame.stFrameInfo.nFrameLen
                convert.enSrcPixelType = frame.stFrameInfo.enPixelType
                convert.enDstPixelType = PixelType_Gvsp_BGR8_Packed
                convert.pDstBuffer = output_buffer
                convert.nDstBufferSize = output_size
                check(camera.MV_CC_ConvertPixelTypeEx(convert), "convert pixel format")

                image = np.frombuffer(output_buffer, dtype=np.uint8, count=output_size)
                image = image.reshape(height, width, 3)

                frame_count += 1
                elapsed = time.monotonic() - fps_started
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    fps_started = time.monotonic()
                    print(
                        f"\rFrame {frame.stFrameInfo.nFrameNum}, {fps:.1f} FPS",
                        end="",
                        flush=True,
                    )

                cv2.putText(
                    image,
                    f"Frame {frame.stFrameInfo.nFrameNum}  {fps:.1f} FPS",
                    (24, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(WINDOW_NAME, image)
            finally:
                camera.MV_CC_FreeImageBuffer(frame)

            key = cv2.waitKey(1) & 0xFF
            window_closed = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
            if key in (ord("q"), ord("Q"), 27) or window_closed:
                break

        return 0
    except KeyboardInterrupt:
        print("\n已停止实时预览。")
        return 0
    except Exception as exc:
        print(f"\n相机错误: {exc}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()
        if camera is not None and grabbing:
            camera.MV_CC_StopGrabbing()
        if camera is not None and opened:
            camera.MV_CC_CloseDevice()
        if camera is not None:
            camera.MV_CC_DestroyHandle()
        if initialized:
            MvCamera.MV_CC_Finalize()


if __name__ == "__main__":
    raise SystemExit(main())
