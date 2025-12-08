import os
import shutil
import time
from pathlib import Path
import sys
import cv2
from PIL import Image
import piexif
import logging
from utils.paths import IMAGE_OUTPUT, IMAGE_ERROR
from error_handling.move_to_error import move_to_error

# --- CONFIG ---
BLUR_STRENGTH = 100  # Higher value = stronger blur


# --- HELPER TO EXTRACT FACE COORDINATES FROM EXIF ---
def _get_faces_from_exif(image_path):
    try:
        img = Image.open(image_path)
        exif_data = img.info.get("exif", b"")
        
        if not exif_data:
            return []
        
        exif_dict = piexif.load(exif_data)
        description = exif_dict["0th"].get(piexif.ImageIFD.ImageDescription, b"").decode("utf-8")
        
        if not description:
            return []
        
        faces = []
        for face_str in description.split("; "):
            if face_str:
                x, y, w, h = map(int, face_str.split(","))
                faces.append((x, y, w, h))
        
        return faces
    except Exception as e:
        print(f"Error reading EXIF data: {e}")
        return []

# --- HELPER TO BLUR FACES ---
def _apply_blur(image_path, output_path, faces):
    img = cv2.imread(str(image_path))
    if img is None:
        raise Exception("Could not read image")
    
    for (x, y, w, h) in faces:
        x2 = x + w
        y2 = y + h
        # Blur the face region
        roi = img[y:y2, x:x2]
        blurred = cv2.blur(roi, (BLUR_STRENGTH, BLUR_STRENGTH))
        img[y:y2, x:x2] = blurred
        print(f"Blurred face at: x={x}, y={y}, w={w}, h={h}")
    
    cv2.imwrite(str(output_path), img)

def face_blur(img_file: Path, faces: list[tuple[int, int, int, int]] | None = None) -> bool:
    """Anonymize a single detected image.
    If `faces` is provided, uses them directly; otherwise, reads from EXIF.
    Returns True if anonymization succeeded, else False.
    """
    if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
        return False

    start_time = time.time()
    faces = faces if faces is not None else _get_faces_from_exif(img_file)
    if not faces:
        logging.info(f"No faces found in EXIF for {img_file.name}")
        return False

    # Overwrite the processing copy in image_output
    output_path = img_file
    logging.info(f"Processing image {img_file.name} for blur")
    try:
        _apply_blur(img_file, output_path, faces)
        logging.info(f"Anonymized image saved to {output_path.name}")
        elapsed_time = time.time() - start_time
        logging.info(f"✓ Blur completed in {elapsed_time:.2f} seconds")
        return True
    except Exception:
        logging.error(f"Error processing {img_file.name} during blur", exc_info=True)
        return False

def blur_faces(img: Path, faces: list[tuple[int, int, int, int]] | None) -> bool:
    """Wrapper around face_blur returning success boolean; on failure, move to error."""
    try:
        return face_blur(img, faces)
    except Exception:
        logging.error(f"Could not evaluate or apply face blur for {img.name}", exc_info=True)
        move_to_error(img, IMAGE_ERROR)
        return False


# --- SCRIPT ENTRY: optional single-file CLI ---
if __name__ == "__main__":
    single_file = None
    if len(sys.argv) > 1:
        single_file = Path(sys.argv[1])
        if not single_file.is_absolute():
            single_file = (Path(__file__).parent.parent / single_file).resolve()

    files_iter = [single_file] if single_file else IMAGE_OUTPUT.glob("*.*")
    for img in files_iter:
        face_blur(img)
