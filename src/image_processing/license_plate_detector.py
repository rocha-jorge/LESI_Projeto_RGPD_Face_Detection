import argparse
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


def detect_license_plates_on_image(img_path: Path, model: YOLO) -> list[tuple[int, int, int, int]]:
    """Run license plate detection on a single image and return list of boxes (x,y,w,h)."""
    plates: list[tuple[int, int, int, int]] = []
    try:
        results = model(str(img_path))
    except Exception as exc:
        logging.error(f"Model inference failed for {img_path.name}: {exc}", exc_info=True)
        return plates

    for result in results:
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
