import logging
import shutil
from pathlib import Path
from utils.paths import IMAGE_OUTPUT, IMAGE_ERROR
from error_handling.move_to_error import move_to_error

def copy_original_to_output(img: Path) -> bool:
    """Copy the original file to image_output/original_<name>. On failure, move to error and return False."""
    try:
        copy_original_to_output = IMAGE_OUTPUT / f"original_{img.name}"
        logging.info(f"Copying original to output: {img.name} -> {copy_original_to_output.name}")
        shutil.copy2(str(img), str(copy_original_to_output))
        return True
    except Exception:
        logging.error(f"Failed to copy original {img.name} to the output folder", exc_info=True)
        move_to_error(img, IMAGE_ERROR)
        return False
