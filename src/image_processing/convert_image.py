from pathlib import Path
import logging
from PIL import Image
from utils.paths import IMAGE_ERROR
from input_output.move_to_error import move_to_error


def ensure_processable_image(img_path: Path) -> Path:
    """
    If the image is BMP or GIF, convert to JPEG in the input directory and
    return the converted path. Otherwise return the original path.

    - Drops alpha channel if present.
    - Uses quality=90 for JPEG.
    - On conversion failure, returns the original path.
    """

    try:
        converted_path = img_path.with_suffix(".jpg")
        if not converted_path.exists():
            with Image.open(img_path) as im:
                if im.mode in ("RGBA", "P"):
                    im = im.convert("RGB")
                elif im.mode != "RGB":
                    im = im.convert("RGB")
                im.save(converted_path, format="JPEG", quality=90)
        print(f"Converted {img_path.name} -> {converted_path.name}")
        return converted_path
    except Exception as e:
        print(f"Conversion failed for {img_path.name}: {e}")
        return img_path

def convert(img: Path) -> tuple[bool, Path | None]:
    """Convert BMP/GIF to JPEG if needed. On failure, move to error. Returns (ok, new_path_or_none)."""
    try:
        ext = img.suffix.lower()
        if ext in {".bmp", ".gif"}:
            logging.info(f"Image {img.name} is {ext}. Converting to JPEG for metadata insertion")
            new_img = ensure_processable_image(img)
            return True, new_img
        else:
            logging.info(f"Image {img.name} is in {ext} format. Conversion not required")
            return True, img
    except Exception:
        logging.error(f"Could not determine/convert image type for {img.name}", exc_info=True)
        move_to_error(img)
        return False, None
