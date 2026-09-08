#!/usr/bin/env python3
import argparse
import json

import cv2

from executor import BallTaskExecutor
from task_types import COLORS, TaskRequest


def main():
    parser = argparse.ArgumentParser(description="Detect a colored ball in an image")
    parser.add_argument("color", choices=sorted(COLORS))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image")
    source.add_argument("--camera", action="store_true")
    args = parser.parse_args()
    executor = BallTaskExecutor()
    if args.camera:
        result = executor.run_camera(TaskRequest(args.color))
    else:
        image = cv2.imread(args.image)
        if image is None:
            parser.error("image is not readable")
        result = executor.run(TaskRequest(args.color), image)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
