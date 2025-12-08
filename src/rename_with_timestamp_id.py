from pathlib import Path
from datetime import datetime

def generate_timestamp_name(img_path: Path) -> str:
    """Return a timestamp-prefixed filename with milliseconds: YYYYMMDD_HHMMSS_mmm_<originalname>."""
    now = datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S")
    ms = f"{now.microsecond // 1000:03d}"  # milliseconds
    return f"{ts}_{ms}_{img_path.name}"

def rename_photo(img_path: Path, input_dir: Path, new_name: str) -> Path:
    """
    Rename the given image in-place within input_dir to new_name.
    Returns the new Path. Does not handle collisions; caller ensures uniqueness.
    """
    new_path = input_dir / new_name
    img_path.rename(new_path)
    return new_path
