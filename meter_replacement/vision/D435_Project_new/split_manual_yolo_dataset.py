from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGES_DIR = PROJECT_DIR / "video_dataset/manual_label_images"
DEFAULT_LABELS_DIR = PROJECT_DIR / "video_dataset/manual_label_labels"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "video_dataset/full_frame_yolo_dataset_manual"
DEFAULT_CLASSES_PATH = PROJECT_DIR / "video_dataset/classes.txt"


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def read_classes(classes_path: Path) -> list[str]:
    classes = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not classes:
        raise ValueError(f"No classes found in: {classes_path}")
    return classes


def validate_label_file(label_path: Path, class_count: int) -> list[str]:
    errors = []
    if not label_path.exists():
        return [f"missing label file: {label_path.name}"]

    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return [f"empty label file: {label_path.name}"]

    for line_number, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label_path.name}:{line_number} expected 5 values, got {len(parts)}")
            continue
        try:
            class_id = int(float(parts[0]))
            values = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"{label_path.name}:{line_number} contains non-numeric values")
            continue
        if class_id < 0 or class_id >= class_count:
            errors.append(f"{label_path.name}:{line_number} invalid class id {class_id}")
        if any(value < 0.0 or value > 1.0 for value in values):
            errors.append(f"{label_path.name}:{line_number} box values must be in [0, 1]")
        if values[2] <= 0.0 or values[3] <= 0.0:
            errors.append(f"{label_path.name}:{line_number} width/height must be greater than 0")
    return errors


def collect_pairs(images_dir: Path, labels_dir: Path, class_count: int) -> tuple[list[tuple[Path, Path]], list[dict]]:
    pairs = []
    report_rows = []
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        errors = validate_label_file(label_path, class_count=class_count)
        if errors:
            report_rows.append(
                {
                    "image": image_path.name,
                    "label": label_path.name,
                    "status": "invalid",
                    "message": " | ".join(errors),
                }
            )
        else:
            pairs.append((image_path, label_path))
            report_rows.append(
                {
                    "image": image_path.name,
                    "label": label_path.name,
                    "status": "valid",
                    "message": "",
                }
            )
    return pairs, report_rows


def split_pairs(
    pairs: list[tuple[Path, Path]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    chronological: bool,
) -> dict[str, list[tuple[Path, Path]]]:
    items = list(pairs)
    if not chronological:
        random.Random(seed).shuffle(items)

    total = len(items)
    train_count = int(round(total * train_ratio))
    val_count = int(round(total * val_ratio))
    if total >= 3:
        train_count = max(1, min(train_count, total - 2))
        val_count = max(1, min(val_count, total - train_count - 1))

    return {
        "train": items[:train_count],
        "val": items[train_count : train_count + val_count],
        "test": items[train_count + val_count :],
    }


def copy_split(split_items: dict[str, list[tuple[Path, Path]]], output_dir: Path) -> list[dict]:
    if output_dir.exists():
        shutil.rmtree(output_dir)

    rows = []
    for split in ["train", "val", "test"]:
        image_out = output_dir / "images" / split
        label_out = output_dir / "labels" / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        for image_path, label_path in split_items[split]:
            dst_image = image_out / image_path.name
            dst_label = label_out / label_path.name
            shutil.copy2(image_path, dst_image)
            shutil.copy2(label_path, dst_label)
            rows.append(
                {
                    "split": split,
                    "image": str(dst_image.relative_to(output_dir)).replace("\\", "/"),
                    "label": str(dst_label.relative_to(output_dir)).replace("\\", "/"),
                }
            )
    return rows


def write_data_yaml(output_dir: Path, classes: list[str]) -> None:
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(classes))
    data_yaml = (
        f"path: {str(output_dir).replace(chr(92), '/')}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        f"{names}\n"
    )
    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_dataset(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    classes_path: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    chronological: bool,
) -> None:
    classes = read_classes(classes_path)
    pairs, validation_rows = collect_pairs(images_dir, labels_dir, class_count=len(classes))
    write_csv(
        output_dir / "validation_report.csv",
        validation_rows,
        fieldnames=["image", "label", "status", "message"],
    )

    invalid_count = sum(1 for row in validation_rows if row["status"] == "invalid")
    if invalid_count:
        raise RuntimeError(
            f"Found {invalid_count} invalid image/label pair(s). "
            f"Fix them first. Report: {output_dir / 'validation_report.csv'}"
        )
    if len(pairs) < 10:
        raise RuntimeError(f"Only {len(pairs)} valid labeled images found. Label more images before splitting.")

    split_items = split_pairs(
        pairs=pairs,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        chronological=chronological,
    )
    split_rows = copy_split(split_items, output_dir)
    write_data_yaml(output_dir, classes)
    write_csv(output_dir / "split_report.csv", split_rows, fieldnames=["split", "image", "label"])

    print(f"Valid labeled images: {len(pairs)}")
    print(f"Output dataset: {output_dir}")
    for split in ["train", "val", "test"]:
        print(f"{split}: {len(split_items[split])}")
    print(f"data.yaml: {output_dir / 'data.yaml'}")
    print(f"split_report.csv: {output_dir / 'split_report.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and split manually labeled YOLO dataset.")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES_DIR, help="manual image folder")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_DIR, help="manual YOLO label folder")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="YOLO dataset output folder")
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES_PATH, help="classes.txt path")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.20, help="val split ratio")
    parser.add_argument("--seed", type=int, default=42, help="random split seed")
    parser.add_argument("--chronological", action="store_true", help="do not shuffle before splitting")
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio <= 0:
        raise ValueError("--train-ratio and --val-ratio must be greater than 0")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("--train-ratio + --val-ratio must be less than 1")
    return args


def main() -> None:
    args = parse_args()
    build_dataset(
        images_dir=args.images,
        labels_dir=args.labels,
        output_dir=args.output,
        classes_path=args.classes,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        chronological=args.chronological,
    )


if __name__ == "__main__":
    main()
