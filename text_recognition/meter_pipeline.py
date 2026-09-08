import argparse
from pathlib import Path

import cv2

from config import (
    BARCODE_ROI,
    DISPLAY_DECIMAL_PLACES,
    DISPLAY_ROI,
    METER_DEBUG_DIR,
    POWER_METER_IMAGE_PATH,
    WARPED_METER_HEIGHT,
    WARPED_METER_WIDTH,
)
from utils.meter_processing import (
    crop_roi,
    decode_barcode,
    draw_meter_contour,
    find_meter_corners,
    preprocess_barcode_roi,
    preprocess_display_roi,
    preprocess_edges,
    read_image,
    recognize_display,
    render_display_digit_boxes,
    warp_meter,
    write_image,
)


def run_meter_pipeline(image_path: Path = POWER_METER_IMAGE_PATH) -> tuple[str, str]:
    """Run meter correction, fixed ROI cropping, display OCR, and barcode decoding."""
    METER_DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    original = read_image(image_path)
    write_image(METER_DEBUG_DIR / "01_original.jpg", original)

    _resized_for_edges, edges, scale = preprocess_edges(original)
    write_image(METER_DEBUG_DIR / "02_edges.jpg", edges)

    corners = find_meter_corners(edges, scale, original.shape)
    contour_image = draw_meter_contour(original, corners)
    write_image(METER_DEBUG_DIR / "03_meter_contour.jpg", contour_image)

    warped = warp_meter(original, corners, WARPED_METER_WIDTH, WARPED_METER_HEIGHT)
    write_image(METER_DEBUG_DIR / "04_warped_meter.jpg", warped)

    display_roi = crop_roi(warped, DISPLAY_ROI)
    barcode_roi = crop_roi(warped, BARCODE_ROI)
    write_image(METER_DEBUG_DIR / "05_display_roi.jpg", display_roi)
    write_image(METER_DEBUG_DIR / "06_barcode_roi.jpg", barcode_roi)

    processed_display = preprocess_display_roi(display_roi)
    write_image(METER_DEBUG_DIR / "07_display_preprocessed.jpg", processed_display)
    processed_barcode = preprocess_barcode_roi(barcode_roi)
    write_image(METER_DEBUG_DIR / "08_barcode_preprocessed.jpg", processed_barcode)
    display_binary, display_boxes = render_display_digit_boxes(display_roi)
    write_image(METER_DEBUG_DIR / "09_display_binary.jpg", display_binary)
    write_image(METER_DEBUG_DIR / "10_display_digit_boxes.jpg", display_boxes)

    display_value = recognize_display(display_roi, processed_display, decimal_places=DISPLAY_DECIMAL_PLACES)
    barcode_value = decode_barcode(barcode_roi)

    print(f"display_value: {display_value}")
    print(f"barcode_value: {barcode_value}")
    print(f"debug_dir: {METER_DEBUG_DIR}")
    return display_value, barcode_value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=str(POWER_METER_IMAGE_PATH), help="path to the meter image")
    args = parser.parse_args()

    try:
        run_meter_pipeline(Path(args.image))
    except RuntimeError as error:
        print(f"Meter pipeline failed: {error}")
        print(f"Check debug images in: {METER_DEBUG_DIR}")
        raise SystemExit(1)
    except cv2.error as error:
        print(f"OpenCV failed while processing the meter image: {error}")
        print(f"Check debug images in: {METER_DEBUG_DIR}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
