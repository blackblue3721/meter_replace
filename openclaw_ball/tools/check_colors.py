#!/usr/bin/env python3
"""Capture one aligned D435 frame and verify all four configured colors."""
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from executor import BallTaskExecutor
from task_types import COLORS, TaskRequest
from vision.camera import RealSenseCamera


DRAW_COLORS = {
    "red": (0, 0, 255), "yellow": (0, 255, 255),
    "green": (0, 255, 0), "blue": (255, 0, 0),
}


def main() -> int:
    executor = BallTaskExecutor()
    with RealSenseCamera() as camera:
        image, depth, intrinsics = camera.capture()
    raw_image = image.copy()

    results = {}
    for color in sorted(COLORS):
        found = executor.detect(TaskRequest(color), image, depth, intrinsics)
        results[color] = [item.to_dict() for item in found]
        for item in found:
            center = (item.pixel_x, item.pixel_y)
            cv2.circle(image, center, round(item.radius_px), DRAW_COLORS[color], 2)
            xyz = item.robot_xyz_mm
            label = color if xyz is None else f"{color} ({xyz[0]:.1f},{xyz[1]:.1f},{xyz[2]:.1f})mm"
            cv2.putText(image, label, (center[0] + 8, center[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, DRAW_COLORS[color], 2)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "logs"
    output_dir.mkdir(exist_ok=True)
    image_path = output_dir / f"color_check_{stamp}.jpg"
    raw_path = output_dir / f"color_check_{stamp}_raw.jpg"
    json_path = output_dir / f"color_check_{stamp}.json"
    cv2.imwrite(str(raw_path), raw_image)
    cv2.imwrite(str(image_path), image)
    payload = {
        "status": "ok", "image": str(image_path), "raw_image": str(raw_path),
        "detections": results,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    missing = [color for color, found in results.items() if not found]
    if missing:
        print("未检测到：" + ", ".join(missing), file=sys.stderr)
        return 1
    print(f"四色检查通过；标注图：{image_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
