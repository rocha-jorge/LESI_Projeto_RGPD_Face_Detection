from pathlib import Path
import time

def generate_timestamp_name(img_path: Path) -> str:
    """Return a timestamp-prefixed filename (YYYYMMDD_HHMMSS_<originalname>)."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{img_path.name}"

def rename_photo(img_path: Path, input_dir: Path, new_name: str) -> Path:
    """
    Rename the given image in-place within input_dir to new_name.
    Returns the new Path. Does not handle collisions; caller ensures uniqueness.
    """
    new_path = input_dir / new_name
    img_path.rename(new_path)
    return new_path
