import logging
import shutil
from pathlib import Path
from utils.paths import IMAGE_OUTPUT, IMAGE_ERROR
from input_output.rename_with_timestamp import rename_with_timestamp
from input_output.move_to_error import move_to_error


def handle_unsupported_file(bad: Path) -> None:
    """Handle a single unsupported file: timestamp rename, copy original, then move to error."""
    try:
        renamed = rename_with_timestamp(bad)
        bad = renamed if renamed is not None else bad
        copy_original_to_output = IMAGE_OUTPUT / f"original_{bad.name}"
        logging.info(
            f"Timestamp rename successful, copying unsupported original to output: {bad.name} -> {copy_original_to_output.name}"
        )
        shutil.copy2(str(bad), str(copy_original_to_output))
    except Exception:
        logging.error(
            f"Failed preparing unsupported file {bad.name}", exc_info=True
        )
    finally:
            move_to_error(bad)
