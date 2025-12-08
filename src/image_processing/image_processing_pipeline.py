import logging
import time
from pathlib import Path

from utils.paths import IMAGE_INPUT, IMAGE_OUTPUT, IMAGE_ERROR
from io.rename_with_timestamp_id import rename_with_timestamp
from io.copy_original import copy_original_to_output
from image_processing.convert_image import convert
from image_processing.detector import detect_faces
from image_processing.face_blur import blur_faces
from io.finalize_output import move_anon_image_to_output


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

    # 2) Copy original to output
    if not copy_original_to_output(img):
        return False

    # 3) Convert if needed (BMP/GIF -> JPEG)
    ok, converted = convert(img)
    if not ok or converted is None:
        return False
    img = converted

    # 4) Detect faces
    ok, faces = detect_faces(img, model)
    if not ok:
        return False

    # 5) Blur faces if any
    if faces and len(faces) > 0:
        success = blur_faces(img, faces)
        if not success:
            return False
    else:
        logging.info(f"No faces reported by detector for {img.name}. Skipping blur")

    # 6) Move anonymized file to output
    if not move_anon_image_to_output(img):
        return False

    elapsed = time.perf_counter() - start_ts
    logging.info(f"Image processing time for {img.name}: {elapsed:.2f} seconds")
    return True
