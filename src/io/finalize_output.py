import logging
import shutil
from pathlib import Path

from error_handling.move_to_error import move_to_error


def move_anon_photo_to_output(img: Path, photo_output: Path, photo_error: Path) -> bool:
    """Move anonymized image to output as anonymized_<name>.

    On failure, logs the error and moves the file to the error folder.
    Returns True on success, False otherwise.
    """
    try:
        destination = photo_output / f"anonymized_{img.name}"
        logging.info(f"Moving image from {img.name} to {str(destination)}")
        shutil.move(str(img), str(destination))
        logging.info(
            f"✓ Successfully processed and moved: {img.name} to {destination.name}"
        )
        return True
    except Exception:
        logging.error(
            "Unable to move the photo to the output folder", exc_info=True
        )
        move_to_error(img, photo_error)
        return False

# Alias removed after updating all imports to move_anon_photo_to_output
