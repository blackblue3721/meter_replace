import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import cv2


def build_video_roi_dataset(
    video_path: Path,
    output_dir: Path,
    num_rois: int,
    frame_interval: int,
    start_second: float,
    end_second: Optional[float],
    reuse_roi_config: Optional[Path],
    save_full_frames: bool,
) -> None:
    video_path = video_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if fps <= 0:
        fps = 30.0

    start_frame = max(0, int(round(start_second * fps)))
    end_frame = total_frames - 1 if end_second is None else int(round(end_second * fps))

    if total_frames > 0:
        end_frame = min(end_frame, total_frames - 1)

    if end_frame < start_frame:
        raise ValueError("end_second must be greater than or equal to start_second.")

    rois = load_or_select_rois(
        cap=cap,
        video_path=video_path,
        output_dir=output_dir,
        reuse_roi_config=reuse_roi_config,
        num_rois=num_rois,
        start_frame=start_frame,
        video_info={
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
        },
    )

    full_frame_dir = output_dir / "full_frames"
    if save_full_frames:
        full_frame_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    saved_count = 0
    frame_index = start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while frame_index <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        if (frame_index - start_frame) % frame_interval == 0:
            timestamp_second = frame_index / fps
            full_frame_name = ""

            if save_full_frames:
                full_frame_name = f"frame_{frame_index:06d}.jpg"
                write_image(full_frame_dir / full_frame_name, frame)

            for roi in rois:
                crop = crop_roi(frame, roi)
                roi_dir = output_dir / roi["name"]
                roi_dir.mkdir(parents=True, exist_ok=True)

                image_name = f"{roi['name']}_frame_{frame_index:06d}.jpg"
                image_path = roi_dir / image_name
                write_image(image_path, crop)

                rows.append(
                    {
                        "image_path": str(image_path.relative_to(output_dir)).replace("\\", "/"),
                        "roi_name": roi["name"],
                        "frame_index": frame_index,
                        "timestamp_second": f"{timestamp_second:.3f}",
                        "x": roi["x"],
                        "y": roi["y"],
                        "width": roi["width"],
                        "height": roi["height"],
                        "label": "",
                        "full_frame": full_frame_name,
                    }
                )
                saved_count += 1

        frame_index += 1

    cap.release()

    write_csv(output_dir / "labels.csv", rows)
    write_summary(
        output_dir=output_dir,
        video_path=video_path,
        fps=fps,
        total_frames=total_frames,
        start_frame=start_frame,
        end_frame=end_frame,
        frame_interval=frame_interval,
        rois=rois,
        saved_count=saved_count,
    )

    print("Done.")
    print(f"Dataset folder: {output_dir}")
    print(f"Saved ROI images: {saved_count}")
    print(f"Labels CSV: {output_dir / 'labels.csv'}")
    print(f"ROI config: {output_dir / 'roi_config.json'}")
    print(f"Summary: {output_dir / 'summary.json'}")


def load_or_select_rois(
    cap,
    video_path: Path,
    output_dir: Path,
    reuse_roi_config: Optional[Path],
    num_rois: int,
    start_frame: int,
    video_info: dict,
) -> list[dict]:
    if reuse_roi_config:
        config_path = reuse_roi_config.resolve()
        data = json.loads(config_path.read_text(encoding="utf-8"))
        rois = data["rois"]
        print(f"Loaded ROI config: {config_path}")
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {start_frame} from video: {video_path}")

        rois = select_rois(frame, num_rois)

    validate_rois(rois, video_info["width"], video_info["height"])

    roi_config = {
        "video_path": str(video_path),
        "video_info": video_info,
        "rois": rois,
    }

    (output_dir / "roi_config.json").write_text(
        json.dumps(roi_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return rois


def select_rois(frame, num_rois: int) -> list[dict]:
    rois = []

    print()
    print("ROI selection window will open.")
    print("Drag a rectangle with your mouse.")
    print("Press Enter or Space to confirm each ROI.")
    print()

    for index in range(num_rois):
        window_name = f"Select ROI {index + 1}/{num_rois}, press Enter or Space"
        x, y, width, height = cv2.selectROI(window_name, frame, showCrosshair=True)
        cv2.destroyWindow(window_name)

        if width <= 0 or height <= 0:
            raise ValueError(f"ROI {index + 1} was not selected.")

        roi_name = f"roi_{index + 1:02d}"
        roi = {
            "name": roi_name,
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
        }

        rois.append(roi)
        print(f"{roi_name}: x={x}, y={y}, width={width}, height={height}")

    cv2.destroyAllWindows()
    return rois


def validate_rois(rois: list[dict], video_width: int, video_height: int) -> None:
    for roi in rois:
        x = int(roi["x"])
        y = int(roi["y"])
        width = int(roi["width"])
        height = int(roi["height"])

        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid ROI size: {roi}")

        if x < 0 or y < 0 or x + width > video_width or y + height > video_height:
            raise ValueError(f"ROI is outside the video frame: {roi}")


def crop_roi(frame, roi: dict):
    x = int(roi["x"])
    y = int(roi["y"])
    width = int(roi["width"])
    height = int(roi["height"])
    return frame[y:y + height, x:x + width]


def write_image(image_path: Path, image) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded_image = cv2.imencode(image_path.suffix, image)

    if not ok:
        raise RuntimeError(f"Failed to encode image: {image_path}")

    encoded_image.tofile(str(image_path))


def write_csv(csv_path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "image_path",
        "roi_name",
        "frame_index",
        "timestamp_second",
        "x",
        "y",
        "width",
        "height",
        "label",
        "full_frame",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    output_dir: Path,
    video_path: Path,
    fps: float,
    total_frames: int,
    start_frame: int,
    end_frame: int,
    frame_interval: int,
    rois: list[dict],
    saved_count: int,
) -> None:
    summary = {
        "video_path": str(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frame_interval": frame_interval,
        "roi_count": len(rois),
        "saved_roi_images": saved_count,
        "rois": rois,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a cropped ROI image dataset from a video."
    )

    parser.add_argument("--video", type=Path, required=True, help="input video path")
    parser.add_argument("--output", type=Path, default=Path("video_dataset"), help="dataset output folder")
    parser.add_argument("--num-rois", type=int, default=1, help="number of ROI areas to select")
    parser.add_argument("--frame-interval", type=int, default=30, help="save one sample every N frames")
    parser.add_argument("--start-second", type=float, default=0.0, help="start time in seconds")
    parser.add_argument("--end-second", type=float, default=None, help="end time in seconds")
    parser.add_argument("--reuse-roi-config", type=Path, default=None, help="reuse an existing roi_config.json")
    parser.add_argument("--save-full-frames", action="store_true", help="also save original full sampled frames")

    args = parser.parse_args()

    if args.num_rois <= 0:
        raise ValueError("--num-rois must be greater than 0")
    if args.frame_interval <= 0:
        raise ValueError("--frame-interval must be greater than 0")
    if args.start_second < 0:
        raise ValueError("--start-second cannot be negative")
    if args.end_second is not None and args.end_second < 0:
        raise ValueError("--end-second cannot be negative")

    return args


def main() -> None:
    args = parse_args()

    build_video_roi_dataset(
        video_path=args.video,
        output_dir=args.output,
        num_rois=args.num_rois,
        frame_interval=args.frame_interval,
        start_second=args.start_second,
        end_second=args.end_second,
        reuse_roi_config=args.reuse_roi_config,
        save_full_frames=args.save_full_frames,
    )


if __name__ == "__main__":
    main()