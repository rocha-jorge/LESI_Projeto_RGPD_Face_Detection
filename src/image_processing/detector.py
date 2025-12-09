# src/detector.py
import time
from pathlib import Path
from ultralytics import YOLO
from input_output.move_to_error import move_to_error
from utils.paths import IMAGE_INPUT, IMAGE_OUTPUT, IMAGE_ERROR
import cv2
from PIL import Image
import logging
from input_output.move_to_error import move_to_error
from utils.paths import IMAGE_INPUT, IMAGE_OUTPUT, IMAGE_ERROR
import piexif

# --- CONFIG ---
# Centralized paths imported from utils.paths
SRC_DIR = IMAGE_INPUT
OUTPUT_DIR = IMAGE_OUTPUT
ERROR_DIR = IMAGE_ERROR
SAVE_EXIF = True  # set False if you don't want to save metadata

# Ensure output directories exist
# Directory creation is handled centrally (utils.paths.ensure_dirs)

# --- LOAD MODEL ---
MODEL_PATH = Path(__file__).parent.parent / "models" / "yolov8n-face.pt"  # path object
WEIGHTS_CACHE = Path(__file__).parent.parent / "weights" / MODEL_PATH.name
"""
Model initialization is handled by the caller (e.g., watcher.py).
This module no longer downloads/loads YOLO at import time to avoid
duplicated work and slow startup. Use the `detector(img, model)`
function with a pre-initialized model.
"""

# --- HELPER TO SAVE FACE COORDINATES TO EXIF ---
def save_faces_exif(image_path, faces):
    img = Image.open(image_path)
    exif_data = img.info.get("exif", b"")
    
    # Load or create EXIF data
    if exif_data:
        exif_dict = piexif.load(exif_data)
    else:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}
    
    faces_str = "; ".join([f"{x},{y},{w},{h}" for (x, y, w, h) in faces])
    exif_dict["0th"][piexif.ImageIFD.ImageDescription] = faces_str.encode("utf-8")
    exif_bytes = piexif.dump(exif_dict)
    img.save(image_path, exif=exif_bytes)


def detector(img_file: Path, model: YOLO) -> list:
    """
    Detect faces on a single image, save EXIF coordinates, and write the
    processed image to image_output with same filename.

    Returns list of (x, y, w, h) face boxes.
    """
    if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
        logging.error(f"Unsupported file format for detection: {img_file.name}")
        return []

    start_time = time.time()
    # Operate directly on the provided path (processing copy in image_output)
    output_path = img_file
    logging.info(f"Processing image {output_path.name} for detection")
    img = cv2.imread(str(output_path))
    if img is None:
        logging.error(f"Could not read {output_path.name} for detection")
        return []

    try:
        results = model(img)
    except Exception as e:
        logging.error(f"Error during detection for {img_file.name}: {e}", exc_info=True)
        return []

    faces_coords = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # x1,y1,x2,y2
        for box in boxes:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            faces_coords.append((int(x1), int(y1), int(w), int(h)))
            logging.debug(f"Face: x={int(x1)}, y={int(y1)}, w={int(w)}, h={int(h)}")

    if SAVE_EXIF and faces_coords:
        save_faces_exif(output_path, faces_coords)
        logging.info("Saved face coordinates to EXIF.")

    # Do not delete or move any files here; watcher manages moves

    elapsed_time = time.time() - start_time
    logging.info(f"✓ Image detection completed for {output_path.name} in {elapsed_time:.2f} seconds, found {len(faces_coords)} face(s).")
    return faces_coords

def detect_faces(img: Path, model: YOLO) -> tuple[bool, list | None]:
    """Wrapper for detection: returns (ok, faces_or_none). On failure, move to error."""
    try:
        faces = detector(img, model)
        return True, faces
    except Exception:
        logging.error(f"Could not apply face detection for {img.name}", exc_info=True)
        move_to_error(img)
        return False, None
