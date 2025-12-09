import logging
from pathlib import Path

from utils.paths import IMAGE_OUTPUT, IMAGE_ERROR


def move_anon_image_to_output(img: Path) -> bool:
    """Move anonymized image to centralized output.

    Uses IMAGE_OUTPUT/IMAGE_ERROR from utils.paths; no args required.
    Returns True on success, False otherwise.
    """
    try:
        destination = IMAGE_OUTPUT / img.name
        logging.info(f"Moving anonymized image: {img.name} -> {destination}")
        img.replace(destination)
        logging.info(f"✓ Successfully finalized: {img.name}")
        return True
    except Exception:
        logging.error(
            "Unable to move the anonymized image to the output folder", exc_info=True
        )
        try:
            err_dest = IMAGE_ERROR / img.name
            logging.info(f"Redirecting anonymized image to error: {img.name} -> {err_dest}")
            img.replace(err_dest)
        except Exception:
            logging.error("Failed moving anonymized image to error", exc_info=True)
        return False
