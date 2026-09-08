from config import NUM_ROIS, RAW_VIDEO_PATH, ROI_DIR
from utils.video_roi import extract_first_frame_rois


def show_info():
    """Print project information."""
    print("ROI EasyOCR project")
    print(f"Raw video: {RAW_VIDEO_PATH}")
    print(f"ROI output: {ROI_DIR}")
    print("Useful commands:")
    print("python main.py --num-rois 2")
    print("python main.py --num-rois 2 --skip-predict")
    print("python predict.py")
    print("python meter_pipeline.py")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--info", action="store_true", help="show project information")
    parser.add_argument("--num-rois", type=int, default=NUM_ROIS, help="number of ROI areas to select")
    parser.add_argument("--skip-predict", action="store_true", help="only save ROI images, do not run OCR")
    args = parser.parse_args()

    if args.info:
        show_info()
    else:
        extract_first_frame_rois(num_rois=args.num_rois)
        if not args.skip_predict:
            from predict import predict

            predict()
