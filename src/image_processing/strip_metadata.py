import logging
from pathlib import Path

from PIL import Image
import piexif

from input_output.move_to_error import move_to_error

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _strip_jpeg_tiff_exif(path: Path) -> None:
    """Remove EXIF from JPEG/TIFF using piexif.remove."""
    piexif.remove(str(path))


def _rewrite_without_metadata(path: Path) -> None:
    """Rewrite image dropping metadata by re-saving without EXIF/XMP/PNG chunks."""
    img = Image.open(str(path))
    fmt = img.format  # e.g., 'PNG', 'JPEG', 'WEBP'
    # Re-save the image to the same path without passing metadata/exif/pnginfo
    img.save(str(path), format=fmt)


def strip_all_metadata(img_file: Path) -> bool:
    """Delete all metadata from the image in-place.

    Returns True on success, False on failure (moves file to error on exception).
    """
    try:
        if img_file.suffix.lower() not in SUPPORTED_EXTS:
            logging.info(f"Metadata strip skipped for unsupported format: {img_file.name}")
            return True

        logging.info(f"Stripping metadata from {img_file.name}")
        ext = img_file.suffix.lower()
        if ext in {".jpg", ".jpeg", ".tif", ".tiff"}:
            _strip_jpeg_tiff_exif(img_file)
        else:
            _rewrite_without_metadata(img_file)
        logging.info(f"✓ Metadata removed for {img_file.name}")
        return True
    except Exception:
        logging.error(f"Failed to strip metadata for {img_file.name}", exc_info=True)
        try:
            move_to_error(img_file)
        except Exception:
            logging.warning("Moving to error folder failed.", exc_info=True)
        return False
