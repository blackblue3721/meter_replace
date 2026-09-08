from pathlib import Path


# Project root
ROOT_DIR = Path(__file__).resolve().parent

# Data paths
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ROI_DIR = DATA_DIR / "roi"

# Input video
RAW_VIDEO_PATH = RAW_DATA_DIR / "RawVideo.mp4"
POWER_METER_IMAGE_PATH = RAW_DATA_DIR / "Power_Meter.jpg"

# Model/output paths
OUTPUT_DIR = ROOT_DIR / "outputs"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
PREDICTION_RESULT_PATH = PREDICTION_DIR / "result.txt"
PREDICTION_DETAIL_PATH = PREDICTION_DIR / "result_details.txt"
METER_DEBUG_DIR = OUTPUT_DIR / "meter_debug"

# ROI selection
NUM_ROIS = 2

# OCR settings. Use ["ch_sim", "en"] for simplified Chinese + English.
OCR_LANGUAGES = ["ch_sim", "en"]
OCR_GPU = False

# Meter perspective correction settings
WARPED_METER_WIDTH = 800
WARPED_METER_HEIGHT = 1200

# Fixed ROI coordinates on the warped 800x1200 meter image.
# Format: (x1, y1, x2, y2). Adjust these after checking outputs/meter_debug.
DISPLAY_ROI = (185, 145, 625, 345)
BARCODE_ROI = (105, 740, 625, 930)

# If set to an integer, force a decimal point before this many trailing digits.
# Keep it as None when the meter display may or may not show a decimal point.
DISPLAY_DECIMAL_PLACES = None
