import argparse

from config import (
    OCR_GPU,
    OCR_LANGUAGES,
    PREDICTION_DETAIL_PATH,
    PREDICTION_RESULT_PATH,
    ROI_DIR,
)


def predict(languages=None):
    """Run OCR on ROI images and save text results."""
    roi_images = sorted(ROI_DIR.glob("roi_*.jpg"))

    if not roi_images:
        print(f"No ROI images found in: {ROI_DIR}")
        print("Run this first: python main.py")
        return

    return predict_with_easyocr(roi_images, languages=languages)


def predict_with_easyocr(roi_images, languages=None):
    """Run pretrained EasyOCR on ROI images."""
    try:
        import easyocr
    except ImportError:
        print("EasyOCR is not installed in the current Python environment.")
        print("Install it with: pip install easyocr")
        return

    PREDICTION_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    languages = languages or OCR_LANGUAGES
    general_reader = easyocr.Reader(languages, gpu=OCR_GPU)
    digit_reader = general_reader if languages == ["en"] else easyocr.Reader(["en"], gpu=OCR_GPU)
    output_lines = []
    detail_lines = []

    for image_path in roi_images:
        image = _read_color_image(image_path)
        candidates = _easyocr_candidates(image, general_reader, digit_reader)
        best = _select_best_candidate(candidates)
        text = best["text"] or "<not recognized>"
        confidence = best["confidence"]

        line = (
            f"{image_path.name}\t{text}\tconfidence={confidence:.4f}"
            f"\tengine=easyocr\tmode={best['mode']}\tlanguages={','.join(languages)}"
        )
        output_lines.append(line)
        print(line)

        for candidate in candidates:
            detail_lines.append(
                f"{image_path.name}\t{candidate['mode']}\t{candidate['text'] or '<empty>'}"
                f"\tconfidence={candidate['confidence']:.4f}\tscore={_candidate_score(candidate):.4f}"
            )

    PREDICTION_RESULT_PATH.write_text("\n".join(output_lines), encoding="utf-8")
    PREDICTION_DETAIL_PATH.write_text("\n".join(detail_lines), encoding="utf-8")
    print(f"Saved OCR results to: {PREDICTION_RESULT_PATH}")
    print(f"Saved OCR candidate details to: {PREDICTION_DETAIL_PATH}")


def _read_color_image(image_path):
    """Read an image from a Windows path that may contain Chinese characters."""
    import cv2
    import numpy as np

    image_data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    return image


def _easyocr_candidates(image, general_reader, digit_reader):
    import cv2

    candidates = []
    candidates.append(
        _candidate_from_readtext(
            "mixed_detect_zh_en",
            general_reader.readtext(
                image,
                detail=1,
                paragraph=False,
            ),
        )
    )
    candidates.append(
        _candidate_from_readtext(
            "mixed_detect_en",
            digit_reader.readtext(
                image,
                detail=1,
                paragraph=False,
            ),
        )
    )
    candidates.append(
        _candidate_from_readtext(
            "general_detect",
            general_reader.readtext(
                image,
                detail=1,
                paragraph=False,
                decoder="greedy",
                contrast_ths=0.05,
                adjust_contrast=0.7,
                mag_ratio=2.0,
                add_margin=0.05,
            ),
        )
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    candidates.append(_candidate_from_recognize("general_whole_roi", general_reader, gray))
    candidates.append(
        _candidate_from_readtext(
            "digit_detect_en",
            digit_reader.readtext(
                image,
                detail=1,
                paragraph=False,
                allowlist="0123456789.-:/",
                decoder="greedy",
                contrast_ths=0.05,
                adjust_contrast=0.7,
                mag_ratio=2.0,
                add_margin=0.05,
            ),
        )
    )

    scaled_gray = _scale_gray_for_recognition(gray)
    candidates.append(
        _candidate_from_recognize(
            "digit_whole_roi_en",
            digit_reader,
            scaled_gray,
            allowlist="0123456789.-:/",
        )
    )
    return candidates


def _candidate_from_recognize(mode, reader, gray_image, allowlist=None):
    height, width = gray_image.shape[:2]
    results = reader.recognize(
        gray_image,
        horizontal_list=[[0, width, 0, height]],
        free_list=[],
        detail=1,
        paragraph=False,
        allowlist=allowlist,
        decoder="greedy",
        contrast_ths=0.05,
        adjust_contrast=0.7,
    )
    return _candidate_from_readtext(mode, results)


def _candidate_from_readtext(mode, results):
    text = " ".join(str(item[1]) for item in results).strip()
    return {
        "mode": mode,
        "text": text,
        "confidence": _mean_confidence(results),
    }


def _select_best_candidate(candidates):
    non_empty = [candidate for candidate in candidates if candidate["text"]]
    if not non_empty:
        return {"mode": "none", "text": "", "confidence": 0.0}
    return max(non_empty, key=_candidate_score)


def _candidate_score(candidate):
    score = candidate["confidence"]
    if _contains_chinese(candidate["text"]):
        score += 0.10
    if candidate["mode"] == "mixed_detect_zh_en":
        score += 0.03
    return score


def _contains_chinese(text):
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _scale_gray_for_recognition(gray_image):
    import cv2

    height, width = gray_image.shape[:2]
    scale = 4 if height < 48 else 2
    return cv2.resize(gray_image, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)


def _mean_confidence(results):
    if not results:
        return 0.0
    return sum(float(item[2]) for item in results) / len(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="EasyOCR languages, for example: --languages ch_sim en",
    )
    args = parser.parse_args()
    predict(languages=args.languages)
