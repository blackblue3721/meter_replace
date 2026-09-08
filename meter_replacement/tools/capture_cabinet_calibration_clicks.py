#!/usr/bin/env python3
"""用 RViz 的 Publish Point 采集电表箱与四个表位的配准基准点。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node


CLICK_TARGETS = (
    ("cabinet_top_left", "箱体主体外框左上角"),
    ("cabinet_top_right", "箱体主体外框右上角"),
    ("cabinet_bottom_left", "箱体主体外框左下角"),
    ("slot_r1_c1_face", "上排左侧电表正面中心"),
    ("slot_r1_c2_face", "上排右侧电表正面中心"),
    ("slot_r2_c1_face", "下排左侧电表正面中心"),
    ("slot_r2_c2_face", "下排右侧电表正面中心"),
)


class ClickCollector(Node):
    """一次只接收一个 RViz 点击，避免误把连续消息当作多个基准点。"""

    def __init__(self) -> None:
        super().__init__("cabinet_calibration_click_collector")
        self.latest: PointStamped | None = None
        self.create_subscription(PointStamped, "/clicked_point", self._on_click, 10)

    def _on_click(self, message: PointStamped) -> None:
        self.latest = message

    def wait_for_click(self) -> PointStamped:
        self.latest = None
        while rclpy.ok() and self.latest is None:
            rclpy.spin_once(self, timeout_sec=0.2)
        if self.latest is None:
            raise KeyboardInterrupt
        return self.latest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/calibration/cabinet_rviz_clicks.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = ClickCollector()
    records: dict[str, dict[str, float | str]] = {}
    try:
        print("\n请在 RViz 顶部选择 Publish Point，然后严格按提示点击红色 D435 点云。")
        print("注意：点击箱体主体外框，不要点击左右两扇打开的门。\n")
        for key, description in CLICK_TARGETS:
            print(f"等待点击：{description}", flush=True)
            message = node.wait_for_click()
            if message.header.frame_id != "world":
                raise RuntimeError(
                    f"RViz Fixed Frame 必须是 world，当前点击来自 {message.header.frame_id!r}"
                )
            point = message.point
            records[key] = {
                "frame_id": message.header.frame_id,
                "x_m": point.x,
                "y_m": point.y,
                "z_m": point.z,
            }
            print(f"  已记录 ({point.x:.6f}, {point.y:.6f}, {point.z:.6f}) m\n")
    finally:
        node.destroy_node()
        rclpy.shutdown()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "format": "rviz_cabinet_calibration_clicks_v1",
                "topic": "/clicked_point",
                "points": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"采集完成：{args.output.resolve()}")
    print("该文件只是测量记录，不会自动修改 MoveIt 场景。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
