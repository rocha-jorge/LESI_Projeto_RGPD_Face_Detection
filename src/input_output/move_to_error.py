from pathlib import Path
import shutil
import logging
from utils.paths import IMAGE_ERROR

def move_to_error(img: Path) -> Path:
    """Move a file to the error folder under the name original_<filename>.

    Returns the destination path. Logs failures but does not raise.
    """
    logging.info(f"✗ Processing {img.name} was unsuccessful. Moving to error folder.")

    try:
        dest = IMAGE_ERROR / f"original_{img.name}"
        shutil.move(str(img), str(dest))
        logging.info(f"Moved {img.name} to error folder: {dest}")
        return dest
    except Exception as e:
        logging.error(f"Failed to move {img.name} to error folder: {e}", exc_info=True)
        try:
            img.unlink(missing_ok=True)
            logging.info(f"Deleted problematic file from input: {img}")
        except Exception as del_err:
            logging.error(f"Failed to delete problematic file {img}: {del_err}", exc_info=True)
        return img