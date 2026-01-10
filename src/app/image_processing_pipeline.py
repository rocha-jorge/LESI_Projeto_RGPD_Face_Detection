import logging
import time
from pathlib import Path

from utils.paths import IMAGE_INPUT, IMAGE_OUTPUT, IMAGE_ERROR
from utils.system_metrics import get_process_usage, get_system_usage
from input_output.rename_with_timestamp import rename_with_timestamp
from input_output.copy_original import copy_original_to_output
from image_processing.convert_image import convert
from image_processing.strip_metadata import strip_all_metadata
from image_processing.detector_face import detect_faces
from image_processing.face_blur import blur_faces
from image_processing.license_plate_detector import detect_license_plates_on_image
from utils.setup_model import setup_model
from input_output.finalize_output import move_anon_image_to_output


def process_image(
    img: Path,
    model,
) -> bool:
    """Run the full per-image pipeline. Returns True on success, else False.

    Each helper logs and moves the file to error on failure; this function
    simply orchestrates and returns a boolean.
    """
    start_ts = time.perf_counter()

    logging.info("" + "═" * 60)
    logging.info(f"▶ Processing image: {img.name}")
    logging.info("" + "═" * 60)

    # 1) Timestamp rename
    renamed = rename_with_timestamp(img)
    if renamed is None:
        return False
    img = renamed
    # Metrics suppressed per user request; single batch snapshot logged in watcher

    # 2) Copy original to output
    if not copy_original_to_output(img):
        return False
    logging.info("Original stored in encrypted_originals")
    # Metrics suppressed

    # 3) Convert if needed (BMP/GIF -> JPEG)
    ok, converted = convert(img)
    if not ok or converted is None:
        return False
    img = converted
    # Metrics suppressed

    # 4) Delete original metadata to ensure privacy
    if not strip_all_metadata(img):
        return False
    logging.info("Original metadata stripped")

    # 5) Detect faces
    face_start = time.perf_counter()
    ok, faces = detect_faces(img, model)
    if not ok:
        return False
    face_elapsed = time.perf_counter() - face_start
    # Performance log for face model (unified format)
    device_str = str(getattr(model, "device", "cpu"))
    logging.info(
        f"Face detection | Time: {int(face_elapsed*1000)} ms | Device: {device_str}"
    )
    # Metrics suppressed

    # 6) Detect license plates (returns list of boxes)
    try:
        lp_model = setup_model("license_plate", input_dir=IMAGE_INPUT)
        plate_start = time.perf_counter()
        plates = detect_license_plates_on_image(img, lp_model)
        plate_elapsed = time.perf_counter() - plate_start
        lp_device_str = str(getattr(lp_model, "device", "cpu"))
        logging.info(
            f"Plate detection | Time: {int(plate_elapsed*1000)} ms | Device: {lp_device_str}"
        )
    except Exception:
        logging.error(f"Could not evaluate license plate detection for {img.name}", exc_info=True)
        return False

    # 7) Blur regions if any (faces + plates)
    combined_regions = []
    if faces:
        combined_regions.extend(faces)
    if plates:
        combined_regions.extend(plates)

    personal_data_detected = bool(combined_regions)
    logging.info(f"Detected regions: faces={len(faces) if faces else 0}, plates={len(plates) if plates else 0}")

    if combined_regions:
        logging.info("Personal data detected: yes")
        if not blur_faces(img, combined_regions):
            return False
    else:
        # Explicit clean-image logs for RGPD auditing
        logging.info("Personal data detected: no")
        logging.info("Image passed without anonymization")

    # 8) Move anonymized file to output
    if not move_anon_image_to_output(img):
        return False
    logging.info("No personal data stored beyond processing window")
    # Metrics suppressed

    elapsed = time.perf_counter() - start_ts
    hw = device_str if personal_data_detected else str(getattr(model, "device", "cpu"))
    logging.info(f"Processing time: {int(elapsed*1000)} ms ({hw})")
    logging.info("✔ Image processed successfully")
    return True
