from typing import List
from pathlib import Path

# file extensions to consider
EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif", ".heic"]

def list_images(directory: Path) -> list[Path]:
    """Return all image files in a directory matching supported extensions.

    If the directory doesn't exist, return an empty list to avoid errors.
    """
    if not directory.exists():
        return []
    return [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS]