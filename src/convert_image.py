from pathlib import Path
from typing import Optional
from PIL import Image


def ensure_processable_image(img_path: Path, input_dir: Path) -> Path:
    """
    If the image is BMP or GIF, convert to JPEG in the input directory and
    return the converted path. Otherwise return the original path.

    - Drops alpha channel if present.
    - Uses quality=90 for JPEG.
    - On conversion failure, returns the original path.
    """

    try:
        converted_path = input_dir / f"{img_path.stem}.jpg"
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
