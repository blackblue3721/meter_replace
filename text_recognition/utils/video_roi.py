from pathlib import Path

from config import NUM_ROIS, RAW_VIDEO_PATH, ROI_DIR


def extract_first_frame_rois(
    video_path: Path = RAW_VIDEO_PATH,
    output_dir: Path = ROI_DIR,
    num_rois: int = NUM_ROIS,
) -> list[Path]:
    """Read the first video frame, select ROIs, and save one crop per ROI."""
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    ok, first_frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Cannot read first frame from video: {video_path}")

    saved_paths = []

    for roi_index in range(num_rois):
        x, y, w, h = _select_roi(first_frame, roi_index, num_rois)

        if w == 0 or h == 0:
            raise ValueError(f"ROI {roi_index + 1} was not selected.")

        roi = first_frame[y : y + h, x : x + w]
        save_path = output_dir / f"roi_{roi_index + 1:02d}.jpg"
        _write_image(save_path, roi)
        saved_paths.append(save_path)

        print(f"ROI {roi_index + 1}: x={x}, y={y}, width={w}, height={h}")
        print(f"Saved: {save_path}")

    print(f"Saved {len(saved_paths)} ROI image(s) to {output_dir}")
    return saved_paths


def extract_roi_images(
    video_path: Path = RAW_VIDEO_PATH,
    output_dir: Path = ROI_DIR,
    num_rois: int = NUM_ROIS,
) -> list[Path]:
    """Backward-compatible wrapper for the old command name."""
    return extract_first_frame_rois(video_path=video_path, output_dir=output_dir, num_rois=num_rois)


def _select_roi(frame, roi_index: int, num_rois: int) -> tuple[int, int, int, int]:
    import cv2

    window_name = f"Select ROI {roi_index + 1}/{num_rois}, then press Enter"
    try:
        roi = cv2.selectROI(window_name, frame, showCrosshair=True)
        cv2.destroyWindow(window_name)
        return tuple(int(value) for value in roi)
    except cv2.error as error:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

        print("OpenCV ROI window is unavailable in this environment.")
        print(f"Falling back to Matplotlib ROI selector for ROI {roi_index + 1}/{num_rois}.")
        print(f"Original OpenCV error: {error}")
        return _select_roi_with_matplotlib(frame, roi_index, num_rois)


def _select_roi_with_matplotlib(frame, roi_index: int, num_rois: int) -> tuple[int, int, int, int]:
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    selected_roi = {"value": None}

    fig, ax = plt.subplots()
    ax.imshow(rgb_frame)
    ax.set_title(f"Drag ROI {roi_index + 1}/{num_rois}, then close this window")
    ax.axis("off")

    def on_select(click_event, release_event):
        if click_event.xdata is None or release_event.xdata is None:
            return

        x1, y1 = int(round(click_event.xdata)), int(round(click_event.ydata))
        x2, y2 = int(round(release_event.xdata)), int(round(release_event.ydata))
        x = max(0, min(x1, x2))
        y = max(0, min(y1, y2))
        w = min(frame.shape[1], max(x1, x2)) - x
        h = min(frame.shape[0], max(y1, y2)) - y
        selected_roi["value"] = (x, y, w, h)

    def on_key_press(event):
        if event.key in {"enter", " "} and selected_roi["value"] is not None:
            plt.close(fig)

    selector = RectangleSelector(
        ax,
        on_select,
        useblit=True,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=True,
    )
    fig.canvas.mpl_connect("key_press_event", on_key_press)
    plt.show()

    if selected_roi["value"] is None:
        return 0, 0, 0, 0

    _ = selector
    return selected_roi["value"]


def _write_image(save_path: Path, image) -> None:
    """Write an image on Windows paths that may contain Chinese characters."""
    import cv2

    ok, encoded_image = cv2.imencode(save_path.suffix, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {save_path}")

    encoded_image.tofile(str(save_path))
    if not save_path.exists() or save_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to save image: {save_path}")
