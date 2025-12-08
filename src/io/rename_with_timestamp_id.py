from datetime import datetime
from pathlib import Path
import logging

def _generate_timestamp_name(path: Path) -> str:
    """Generate a timestamped filename based on the original name.

    Format: YYYYMMDD_HHMMSS_mmm_originalname.ext
    """
    now = datetime.now()
    base = path.name
    ts = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # millisecond precision
    return f"{ts}_{base}"

def _rename_image(path: Path, new_name: str) -> Path:
    """Rename the image in-place (same directory) to new_name and return the new Path."""
    new_path = path.parent / new_name
    path.rename(new_path)
    return new_path

def rename_with_timestamp(img: Path) -> Path | None:
    """Rename an input file with a timestamp in-place. On failure, return None."""
    try:
        new_name = _generate_timestamp_name(img)
        renamed = _rename_image(img, new_name)
        logging.info(f"Renamed: {img} to {new_name}")
        return renamed
    except Exception:
        logging.error(f"Failed generating/including timestamp ID for {img.name}", exc_info=True)
        return None
