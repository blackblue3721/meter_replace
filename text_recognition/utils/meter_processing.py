from pathlib import Path

import cv2
import numpy as np


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from a path that may contain non-ASCII characters."""
    image_data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    return image


def write_image(image_path: Path, image: np.ndarray) -> None:
    """Write an image to a path that may contain non-ASCII characters."""
    image_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(image_path.suffix, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {image_path}")
    encoded.tofile(str(image_path))


def preprocess_edges(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Create a Canny edge image on a resized copy for stable contour search."""
    max_side = max(image.shape[:2])
    scale = min(1.0, 1600.0 / max_side)
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    return resized, closed, scale


def find_meter_corners(edge_image: np.ndarray, scale: float, original_shape: tuple[int, int, int]) -> np.ndarray:
    """Find the largest rectangular contour and return its corners in original-image coordinates."""
    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contours found. Check 02_edges.jpg and adjust preprocessing parameters.")

    image_area = edge_image.shape[0] * edge_image.shape[1]
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.02:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            candidates.append((area, approx.reshape(4, 2).astype(np.float32)))

    if candidates:
        corners = max(candidates, key=lambda item: item[0])[1]
    else:
        # Fallback: use the minimum-area rectangle of the largest contour.
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < image_area * 0.02:
            raise RuntimeError(
                "Could not find a large meter-like contour. "
                "Make sure the meter outer border is visible in the image."
            )
        corners = cv2.boxPoints(cv2.minAreaRect(largest)).astype(np.float32)

    corners = corners / scale
    h, w = original_shape[:2]
    corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)
    return order_points(corners)


def order_points(points: np.ndarray) -> np.ndarray:
    """Sort four points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    point_sum = points.sum(axis=1)
    point_diff = np.diff(points, axis=1).reshape(-1)

    rect[0] = points[np.argmin(point_sum)]
    rect[2] = points[np.argmax(point_sum)]
    rect[1] = points[np.argmin(point_diff)]
    rect[3] = points[np.argmax(point_diff)]
    return rect


def draw_meter_contour(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Draw the detected meter contour on a copy of the original image."""
    output = image.copy()
    contour = corners.reshape(4, 1, 2).astype(np.int32)
    cv2.polylines(output, [contour], isClosed=True, color=(0, 0, 255), thickness=8)
    for index, point in enumerate(corners.astype(int), start=1):
        cv2.circle(output, tuple(point), 18, (0, 255, 0), -1)
        cv2.putText(output, str(index), tuple(point + 25), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
    return output


def warp_meter(image: np.ndarray, corners: np.ndarray, width: int, height: int) -> np.ndarray:
    """Perspective-correct the meter into a fixed front-facing image."""
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), destination)
    return cv2.warpPerspective(image, matrix, (width, height))


def crop_roi(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a fixed coordinate ROI from an image."""
    x1, y1, x2, y2 = roi
    h, w = image.shape[:2]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid ROI after clipping: {(x1, y1, x2, y2)}")
    return image[y1:y2, x1:x2]


def preprocess_display_roi(display_roi: np.ndarray) -> np.ndarray:
    """Prepare the LCD/seven-segment ROI for OCR/debugging."""
    gray = cv2.cvtColor(display_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        7,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)


def preprocess_barcode_roi(barcode_roi: np.ndarray) -> np.ndarray:
    """Prepare a barcode ROI for debugging and fallback OCR."""
    gray = cv2.cvtColor(barcode_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def recognize_display(display_roi: np.ndarray, processed_display: np.ndarray, decimal_places: int | None = None) -> str:
    """Recognize display digits with EasyOCR; a rule-based seven-segment hook can be added here."""
    seven_segment_text = recognize_display_with_seven_segment(display_roi)

    try:
        import easyocr
    except ImportError:
        if seven_segment_text:
            return _normalize_display_value(seven_segment_text, decimal_places)
        return "<easyocr not installed>"

    reader = easyocr.Reader(["en"], gpu=False)
    candidates = []
    allowlist = "0123456789."

    raw_results = reader.readtext(
        display_roi,
        detail=1,
        paragraph=False,
        allowlist=allowlist,
        decoder="greedy",
        mag_ratio=2.0,
        contrast_ths=0.05,
        adjust_contrast=0.7,
    )
    candidates.append(_ocr_candidate(raw_results))

    processed_bgr = cv2.cvtColor(processed_display, cv2.COLOR_GRAY2BGR)
    processed_results = reader.readtext(
        processed_bgr,
        detail=1,
        paragraph=False,
        allowlist=allowlist,
        decoder="greedy",
        mag_ratio=1.0,
        contrast_ths=0.05,
        adjust_contrast=0.7,
    )
    candidates.append(_ocr_candidate(processed_results))

    gray = cv2.cvtColor(display_roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    whole_results = reader.recognize(
        gray,
        horizontal_list=[[0, w, 0, h]],
        free_list=[],
        detail=1,
        paragraph=False,
        allowlist=allowlist,
        decoder="greedy",
        contrast_ths=0.05,
        adjust_contrast=0.7,
    )
    candidates.append(_ocr_candidate(whole_results))
    if seven_segment_text:
        candidates.append({"text": seven_segment_text, "confidence": 0.95, "source": "seven_segment"})

    candidates = [candidate for candidate in candidates if candidate["text"]]
    if not candidates:
        return "<not recognized>"

    best = max(candidates, key=_display_candidate_score)
    return _normalize_display_value(best["text"], decimal_places)


def recognize_display_with_seven_segment(display_roi: np.ndarray) -> str:
    """Rule-based seven-segment reader for dark LCD digits."""
    binary = _threshold_display_digits(display_roi)
    components = _find_digit_components(binary)
    if not components:
        return ""

    digits = []
    decimal_after = set()
    for index, (x, y, w, h) in enumerate(components):
        digit_image = binary[y : y + h, x : x + w]
        digit = _decode_single_seven_segment_digit(digit_image)
        if digit is None:
            continue
        digits.append(digit)

        gap_start = x + w
        gap_end = components[index + 1][0] if index + 1 < len(components) else min(binary.shape[1], x + w + int(w * 0.6))
        if _has_decimal_point(binary, gap_start, gap_end, y, h):
            decimal_after.add(len(digits) - 1)

    if not digits:
        return ""

    output = []
    for index, digit in enumerate(digits):
        output.append(digit)
        if index in decimal_after:
            output.append(".")
    return "".join(output).rstrip(".")


def render_display_digit_boxes(display_roi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Render seven-segment threshold and detected digit boxes for debugging."""
    binary = _threshold_display_digits(display_roi)
    boxes = _find_digit_components(binary)
    boxed = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in boxes:
        digit = _decode_single_seven_segment_digit(binary[y : y + h, x : x + w])
        cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            boxed,
            str(digit),
            (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
    return binary, boxed


def _threshold_display_digits(display_roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(display_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # LCD active segments are much darker than the green-gray background.
    # Cap the adaptive percentile to avoid turning the whole LCD texture white.
    threshold = min(float(np.percentile(gray, 30)), 65.0)
    binary = np.where(gray <= threshold, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)


def _find_digit_components(binary: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = binary.shape[:2]
    digit_band_y1 = int(height * 0.35)
    digit_band_y2 = int(height * 0.92)
    digit_band = binary[digit_band_y1:digit_band_y2, :]

    column_projection = np.count_nonzero(digit_band, axis=0)
    active_columns = np.where(column_projection > max(3, digit_band.shape[0] * 0.08))[0]
    intervals = _split_wide_digit_intervals(
        _projection_intervals(active_columns, max_gap=8),
        column_projection,
        width,
    )

    boxes = []
    for x1, x2 in intervals:
        if x2 < width * 0.10 or x1 > width * 0.92:
            continue

        sub_band = digit_band[:, x1 : x2 + 1]
        active_rows = np.where(np.count_nonzero(sub_band, axis=1) > 0)[0]
        if active_rows.size == 0:
            continue

        local_y1 = int(active_rows[0])
        local_y2 = int(active_rows[-1])
        h = local_y2 - local_y1 + 1
        w = x2 - x1 + 1
        if h < digit_band.shape[0] * 0.32:
            continue
        if h > digit_band.shape[0] * 0.90:
            continue

        if w < width * 0.018 or w > width * 0.17:
            continue

        padded_x1 = max(0, int(x1 - width * 0.01))
        padded_x2 = min(width, int(x2 + width * 0.01))
        padded_y1 = max(0, digit_band_y1 + local_y1 - 8)
        padded_y2 = min(height, digit_band_y1 + local_y2 + 8)
        boxes.append((padded_x1, padded_y1, padded_x2 - padded_x1, padded_y2 - padded_y1))

    return boxes


def _split_wide_digit_intervals(
    intervals: list[tuple[int, int]],
    column_projection: np.ndarray,
    image_width: int,
) -> list[tuple[int, int]]:
    split_intervals = []
    max_digit_width = image_width * 0.14
    for x1, x2 in intervals:
        if x2 - x1 + 1 <= max_digit_width:
            split_intervals.append((x1, x2))
            continue

        local_active = np.where(column_projection[x1 : x2 + 1] > 6)[0] + x1
        split_intervals.extend(_projection_intervals(local_active, max_gap=3))
    return split_intervals


def _projection_intervals(active_indexes: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    if active_indexes.size == 0:
        return []

    intervals = []
    start = int(active_indexes[0])
    previous = int(active_indexes[0])
    for index in active_indexes[1:]:
        index = int(index)
        if index - previous > max_gap:
            intervals.append((start, previous))
            start = index
        previous = index
    intervals.append((start, previous))
    return intervals


def _decode_single_seven_segment_digit(digit_image: np.ndarray) -> str | None:
    original_h, original_w = digit_image.shape[:2]
    if original_w < original_h * 0.32:
        return "1"

    digit_image = cv2.resize(digit_image, (70, 120), interpolation=cv2.INTER_NEAREST)
    h, w = digit_image.shape[:2]
    segment_boxes = {
        "a": (int(w * 0.25), 0, int(w * 0.78), int(h * 0.18)),
        "b": (int(w * 0.62), int(h * 0.12), w, int(h * 0.48)),
        "c": (int(w * 0.62), int(h * 0.52), w, int(h * 0.88)),
        "d": (int(w * 0.25), int(h * 0.82), int(w * 0.78), h),
        "e": (0, int(h * 0.52), int(w * 0.38), int(h * 0.88)),
        "f": (0, int(h * 0.12), int(w * 0.38), int(h * 0.48)),
        "g": (int(w * 0.25), int(h * 0.40), int(w * 0.78), int(h * 0.60)),
    }

    bits = []
    ratios = {}
    segment_thresholds = {
        "a": 0.12,
        "b": 0.25,
        "c": 0.25,
        "d": 0.12,
        "e": 0.25,
        "f": 0.25,
        "g": 0.12,
    }
    for name in ("a", "b", "c", "d", "e", "f", "g"):
        x1, y1, x2, y2 = segment_boxes[name]
        segment = digit_image[y1:y2, x1:x2]
        ratio = float(np.mean(segment > 0))
        ratios[name] = ratio
        bits.append("1" if ratio > segment_thresholds[name] else "0")

    patterns = {
        "1111110": "0",
        "0110000": "1",
        "1101101": "2",
        "1111001": "3",
        "0110011": "4",
        "1011011": "5",
        "1011111": "6",
        "1110000": "7",
        "1111111": "8",
        "1111011": "9",
    }
    pattern = "".join(bits)
    if pattern == "1111111" and ratios["g"] < 0.45:
        return "0"

    # LCD 7s often show faint middle/left ghosts; treat this family as 7 when
    # top and right-side segments dominate and bottom/left-bottom are absent.
    if (
        ratios["a"] > 0.25
        and ratios["b"] > 0.45
        and ratios["c"] > 0.35
        and ratios["d"] < 0.25
        and ratios["e"] < 0.08
    ):
        return "7"

    if pattern in patterns:
        return patterns[pattern]

    return _nearest_seven_segment_digit(pattern, patterns)


def _nearest_seven_segment_digit(pattern: str, patterns: dict[str, str]) -> str | None:
    distances = [(sum(a != b for a, b in zip(pattern, key)), value) for key, value in patterns.items()]
    distance, digit = min(distances, key=lambda item: item[0])
    return digit if distance <= 1 else None


def _has_decimal_point(binary: np.ndarray, x1: int, x2: int, digit_y: int, digit_h: int) -> bool:
    h, w = binary.shape[:2]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, digit_y + int(digit_h * 0.75)))
    y2 = max(0, min(h, digit_y + digit_h + int(digit_h * 0.15)))
    if x2 <= x1 or y2 <= y1:
        return False
    dot_region = binary[y1:y2, x1:x2]
    return int(np.count_nonzero(dot_region)) >= max(8, dot_region.size * 0.02)


def decode_barcode(barcode_roi: np.ndarray) -> str:
    """Decode a barcode ROI with native barcode decoders before OCR fallback."""
    zxing_value = _decode_barcode_with_zxingcpp(barcode_roi)
    if zxing_value:
        return zxing_value

    pyzbar_value = _decode_barcode_with_pyzbar(barcode_roi)
    if pyzbar_value:
        return pyzbar_value

    opencv_value = _decode_barcode_with_opencv(barcode_roi)
    if opencv_value:
        return opencv_value

    fallback_value = _read_barcode_text_with_easyocr(barcode_roi)
    if fallback_value:
        return fallback_value

    return "<not decoded>"


def _decode_barcode_with_zxingcpp(barcode_roi: np.ndarray) -> str:
    try:
        import zxingcpp
    except ImportError:
        return ""

    for image in _barcode_decode_variants(barcode_roi):
        results = zxingcpp.read_barcodes(
            image,
            formats=zxingcpp.BarcodeFormat.Code128,
            try_rotate=True,
            try_downscale=True,
            try_invert=True,
        )
        values = [result.text for result in results if result.text]
        if values:
            return values[0]
    return ""


def _barcode_decode_variants(barcode_roi: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(barcode_roi, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    variants = [barcode_roi, gray, binary]
    for scale in (2, 3, 4):
        variants.append(cv2.resize(barcode_roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))
        variants.append(cv2.resize(binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST))
    return variants


def _decode_barcode_with_pyzbar(barcode_roi: np.ndarray) -> str:
    try:
        from pyzbar.pyzbar import decode
    except Exception as error:
        print(f"pyzbar is unavailable, falling back to OpenCV barcode detector: {error}")
        return ""

    decoded = decode(barcode_roi)
    if not decoded:
        return ""
    return decoded[0].data.decode("utf-8", errors="replace")


def _decode_barcode_with_opencv(barcode_roi: np.ndarray) -> str:
    if not hasattr(cv2, "barcode_BarcodeDetector"):
        return ""

    detector = cv2.barcode_BarcodeDetector()
    ok, decoded_info, _decoded_type, _points = detector.detectAndDecodeWithType(barcode_roi)
    if ok:
        values = [value for value in decoded_info if value]
        if values:
            return values[0]
    return ""


def _read_barcode_text_with_easyocr(barcode_roi: np.ndarray) -> str:
    """Fallback: OCR the human-readable barcode number when decoders fail."""
    try:
        import easyocr
    except ImportError:
        return ""

    reader = easyocr.Reader(["en"], gpu=False)
    results = reader.readtext(
        barcode_roi,
        detail=1,
        paragraph=False,
        allowlist="0123456789",
        decoder="greedy",
        mag_ratio=2.0,
        contrast_ths=0.05,
        adjust_contrast=0.7,
    )
    candidates = [str(item[1]) for item in results if str(item[1]).isdigit()]
    if not candidates:
        return ""
    return max(candidates, key=len)


def _ocr_candidate(results) -> dict[str, float | str]:
    text = " ".join(str(item[1]) for item in results).strip()
    confidence = 0.0
    if results:
        confidence = sum(float(item[2]) for item in results) / len(results)
    return {"text": text, "confidence": confidence}


def _display_candidate_score(candidate: dict[str, float | str]) -> float:
    text = str(candidate["text"])
    digit_count = sum(char.isdigit() for char in text)
    has_decimal = "." in text
    source_bonus = 0.35 if candidate.get("source") == "seven_segment" else 0.0
    return float(candidate["confidence"]) + min(digit_count, 8) * 0.08 + (0.08 if has_decimal else 0.0) + source_bonus


def _normalize_display_value(text: str, decimal_places: int | None) -> str:
    text = text.strip()
    digits = "".join(char for char in text if char.isdigit())
    if not digits:
        return text or "<not recognized>"

    if decimal_places is None:
        cleaned = []
        decimal_used = False
        for char in text:
            if char.isdigit():
                cleaned.append(char)
            elif char == "." and not decimal_used and cleaned:
                cleaned.append(char)
                decimal_used = True
        normalized = "".join(cleaned).strip(".")
        return normalized or digits

    if decimal_places is None or decimal_places <= 0 or len(digits) <= decimal_places:
        return digits

    integer_part = digits[:-decimal_places].lstrip("0") or "0"
    fractional_part = digits[-decimal_places:]
    return f"{integer_part}.{fractional_part}"
