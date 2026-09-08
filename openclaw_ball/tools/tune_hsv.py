#!/usr/bin/env python3
"""Interactive HSV threshold tuner for a saved image or live D435 feed."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import HSV_RANGES
from task_types import COLORS
from vision.camera import RealSenseCamera


def nothing(_):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("color", choices=sorted(COLORS))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image")
    source.add_argument("--camera", action="store_true")
    args = parser.parse_args()

    camera = RealSenseCamera() if args.camera else None
    fixed_image = None if camera else cv2.imread(args.image)
    if not camera and fixed_image is None:
        parser.error("image is not readable")

    lower, upper = HSV_RANGES[args.color][0]
    cv2.namedWindow("HSV tuner")
    for name, value, maximum in zip(
        ("H min", "S min", "V min", "H max", "S max", "V max"),
        (*lower, *upper), (179, 255, 255, 179, 255, 255),
    ):
        cv2.createTrackbar(name, "HSV tuner", value, maximum, nothing)

    try:
        while True:
            image = camera.capture()[0] if camera else fixed_image
            values = [cv2.getTrackbarPos(name, "HSV tuner") for name in
                      ("H min", "S min", "V min", "H max", "S max", "V max")]
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array(values[:3]), np.array(values[3:]))
            preview = cv2.bitwise_and(image, image, mask=mask)
            cv2.imshow("HSV tuner", np.hstack((image, preview)))
            if cv2.waitKey(30) & 0xFF in (27, ord("q")):
                print(f"{args.color}: (({tuple(values[:3])}, {tuple(values[3:])}),)")
                break
    finally:
        if camera:
            camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
