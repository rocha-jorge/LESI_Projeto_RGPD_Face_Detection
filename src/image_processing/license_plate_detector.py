import argparse
import os
import logging
import sys
from pathlib import Path
from typing import Iterable

from ultralytics import YOLO

from utils.logging_setup import init_logging
from utils.paths import LOG_DIR, LOG_FILE
from image_processing.list_images import list_images
from image_processing.face_blur import blur_faces
from utils.setup_model import setup_model


def _read_threshold(var_name: str, default: float) -> float:
    try:
        val = float(os.environ.get(var_name, str(default)))
        if 0.0 <= val <= 1.0:
            return val
        logging.warning(f"Invalid {var_name} value '{val}', using default {default}")
        return default
    except Exception:
        logging.warning(f"Could not parse {var_name}, using default {default}")
        return default


def detect_license_plates_on_image(img_path: Path, model: YOLO) -> list[tuple[int, int, int, int]]:
    """Run license plate detection on a single image and return list of boxes (x,y,w,h)."""
    plates: list[tuple[int, int, int, int]] = []
    try:
        conf_thr = _read_threshold("PLATE_CONF_THRESHOLD", 0.25)
        logging.debug(f"Confidence threshold (plate): {conf_thr}")
        results = model(str(img_path), conf=conf_thr, verbose=False)
    except Exception as exc:
        logging.error(f"Model inference failed for {img_path.name}: {exc}", exc_info=True)
        return plates

    for result in results:
        # Log Ultralytics per-stage speed in our format
        spd = getattr(result, "speed", None)
        if isinstance(spd, dict):
            pp = spd.get("preprocess")
            inf = spd.get("inference")
            post = spd.get("postprocess")
            if pp is not None and inf is not None and post is not None:
                logging.debug(
                    f"Ultralytics inference stats: preprocess={pp:.1f}ms inference={inf:.1f}ms postprocess={post:.1f}ms"
                )
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            plates.append((x1, y1, x2 - x1, y2 - y1))
    return plates


def detect_license_plates(model: YOLO, image_paths: Iterable[Path]) -> None:
    for img_path in image_paths:
        logging.info("=" * 60)
        logging.info(f"Running license plate detection on {img_path.name}")
        plates = detect_license_plates_on_image(img_path, model)
        for (x, y, w, h) in plates:
            logging.info(f"Plate: x={x}, y={y}, w={w}, h={h}")

        if not plates:
            logging.info("No license plates detected.")
        logging.info(f"Completed inference for {img_path.name}")

        if plates:
            logging.info("Applying blur over detected plates for verification...")
            success = blur_faces(img_path, plates)
            if not success:
                logging.warning("License plate blur failed; check logs for details.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run YOLOv8 license plate detection on a folder of images.")
    parser.add_argument("--input", "-i", type=Path, default=None, help="Directory containing input images.")
    parser.add_argument("--model-cache", dest="model_key", default="license_plate", help="Model key (face or license_plate)")
    args = parser.parse_args()

    init_logging(LOG_DIR, LOG_FILE, logging.INFO)

    input_dir = args.input or Path().cwd()
    if not input_dir.exists():
        logging.error("Input directory does not exist: %s", input_dir)
        return 1

    images = list_images(input_dir)
    if not images:
        logging.warning("No supported images found in %s", input_dir)
        return 0

    model = setup_model(model_key=args.model_key, input_dir=input_dir)
    detect_license_plates(model, images)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
