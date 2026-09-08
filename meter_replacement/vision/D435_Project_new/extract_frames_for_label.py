from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEO_PATH = PROJECT_DIR / "data/raw/dianbiao_video.mp4"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "video_dataset/manual_label_images"


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def frame_quality_ok(frame: np.ndarray, min_blur: float, min_std: float) -> tuple[bool, float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    std = float(gray.std())
    return blur >= min_blur and std >= min_std, blur, std


def extract_frames(
    video_path: Path,
    output_dir: Path,
    frame_interval: int,
    start_second: float,
    end_second: float | None,
    max_images: int | None,
    min_blur: float,
    min_std: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame = max(0, int(round(start_second * fps)))
    end_frame = total_frames - 1 if end_second is None else int(round(end_second * fps))
    if total_frames > 0:
        end_frame = min(end_frame, total_frames - 1)
    if end_frame < start_frame:
        raise ValueError("--end-second must be greater than or equal to --start-second")

    report_rows = []
    saved = 0
    scanned = 0

    frame_index = start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    while frame_index <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        if (frame_index - start_frame) % frame_interval == 0:
            scanned += 1
            quality_ok, blur, std = frame_quality_ok(frame, min_blur=min_blur, min_std=min_std)
            image_name = f"full_frame_{frame_index:06d}.jpg"
            image_path = output_dir / image_name

            if quality_ok and (max_images is None or saved < max_images):
                write_image(image_path, frame)
                saved += 1
                status = "saved"
            elif not quality_ok:
                status = "skipped_low_quality"
            else:
                status = "skipped_max_images_reached"

            report_rows.append(
                {
                    "frame_index": frame_index,
                    "timestamp_second": f"{frame_index / fps:.3f}",
                    "image_name": image_name if status == "saved" else "",
                    "status": status,
                    "blur": f"{blur:.4f}",
                    "std": f"{std:.4f}",
                }
            )

            if max_images is not None and saved >= max_images:
                break

        frame_index += 1

    cap.release()

    report_path = output_dir / "extract_report.csv"
    with report_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["frame_index", "timestamp_second", "image_name", "status", "blur", "std"],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"Video: {video_path}")
    print(f"FPS: {fps:.3f}")
    print(f"Total frames: {total_frames}")
    print(f"Scanned samples: {scanned}")
    print(f"Saved images: {saved}")
    print(f"Output: {output_dir}")
    print(f"Report: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract full video frames for manual YOLO labeling.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO_PATH, help="input video path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="output image folder")
    parser.add_argument("--frame-interval", type=int, default=30, help="save one image every N frames")
    parser.add_argument("--start-second", type=float, default=0.0, help="start time in seconds")
    parser.add_argument("--end-second", type=float, default=None, help="end time in seconds")
    parser.add_argument("--max-images", type=int, default=250, help="maximum saved images; use 0 for unlimited")
    parser.add_argument("--min-blur", type=float, default=20.0, help="minimum Laplacian blur score")
    parser.add_argument("--min-std", type=float, default=12.0, help="minimum grayscale standard deviation")
    args = parser.parse_args()

    if args.frame_interval <= 0:
        raise ValueError("--frame-interval must be greater than 0")
    if args.start_second < 0:
        raise ValueError("--start-second cannot be negative")
    if args.end_second is not None and args.end_second < 0:
        raise ValueError("--end-second cannot be negative")
    if args.max_images is not None and args.max_images < 0:
        raise ValueError("--max-images cannot be negative")
    if args.max_images == 0:
        args.max_images = None
    return args


def main() -> None:
    args = parse_args()
    extract_frames(
        video_path=args.video,
        output_dir=args.output,
        frame_interval=args.frame_interval,
        start_second=args.start_second,
        end_second=args.end_second,
        max_images=args.max_images,
        min_blur=args.min_blur,
        min_std=args.min_std,
    )


if __name__ == "__main__":
    main()
