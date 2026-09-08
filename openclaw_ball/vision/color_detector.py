import cv2
import numpy as np

from config import HSV_RANGES, MIN_AREA_PX, MIN_CIRCULARITY
from task_types import BallDetection


class ColorBallDetector:
    def detect(self, bgr_image, color: str) -> list[BallDetection]:
        if color not in HSV_RANGES:
            raise ValueError(f"unsupported color: {color}")
        if bgr_image is None or bgr_image.ndim != 3:
            raise ValueError("expected a BGR image")

        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in HSV_RANGES[color]:
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter else 0
            if area < MIN_AREA_PX or circularity < MIN_CIRCULARITY:
                continue
            (x, y), radius = cv2.minEnclosingCircle(contour)
            detections.append(BallDetection(color, round(x), round(y), radius, area, circularity))
        return sorted(detections, key=lambda item: item.area_px, reverse=True)

