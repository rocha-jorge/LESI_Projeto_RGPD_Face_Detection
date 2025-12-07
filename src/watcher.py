"""Continuous watcher that processes images as they appear using a single YOLO model.

Initializes the YOLO model once, then continuously:
- Generates a per-image ID (timestamp) via generate_photo_id.py
- Copies the original to photo_output/original_<filename> for traceability
- Detects faces on the image and saves EXIF coordinates
- Blurs faces (face_blur) only when detection finds faces, writing to photo_output

Notes:
- ID generation is always performed before copying/detection.
- The model is kept in memory for efficiency (single initialization).
"""
import os
import sys
import time
import signal
import shutil
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
from convert_image import ensure_processable_image


ROOT = Path(__file__).parent.parent
PHOTO_OUTPUT = ROOT / "photo_output"
PHOTO_ERROR = ROOT / "photo_error"

from detector import detector
from face_blur import face_blur
from rename_with_timestamp_id import generate_timestamp_name, rename_photo
from list_images import list_images
from move_to_error import move_to_error

# Configure via environment variables (use absolute paths when mounting volumes)
INPUT_DIR = Path(os.environ.get("PHOTO_INPUT", ROOT / "photo_input"))
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "5"))

# Flag used by the main loop to know when to exit.
# When a shutdown signal is received (CTRL+C or service stop),
# we set this to True and let the loop break cleanly.
stop_requested = False

# Signal handler: flips the flag when termination/interruption is requested.
def handle_sigterm(signum, frame):
    global stop_requested
    stop_requested = True

# Bind both SIGTERM (service stop) and SIGINT (CTRL+C) to the handler
# so the watcher can shut down gracefully.
signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)




def main():
    # Startup: ensure required folders exist and log the monitored path
    print(f"Watcher starting. Monitoring: {INPUT_DIR}")
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    PHOTO_OUTPUT.mkdir(parents=True, exist_ok=True)
    PHOTO_ERROR.mkdir(parents=True, exist_ok=True)

    # Initialize YOLO once (kept in memory for the life of the process)
    model_path = ROOT / "models" / "yolov8n-face.pt"
    weights_cache = ROOT / "weights" / model_path.name
    if not model_path.exists() and weights_cache.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        weights_cache.replace(model_path)
        print(f"Moved cached weights from {weights_cache} to {model_path}")
    if not model_path.exists():
        print("Downloading YOLOv8-Face model...")
        model = YOLO("https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
    else:
        model = YOLO(str(model_path))
    print("Model initialized and ready.")

    while not stop_requested:
        try:
            # Poll the input directory for new images
            images = list_images(INPUT_DIR)     # image paths in input dir
            if images:
                print(f"Found {len(images)} image(s). Processing...")
                for img in images:
                   
                    print(f"\nProcessing image: {img.name}")

                    # 1) Generate and include timestamp ID in filename
                    try:
                        new_name = generate_timestamp_name(img)
                        img = rename_photo(img, INPUT_DIR, new_name)
                        print(f"Renamed: {img} to {new_name}")
                    except Exception as e:
                        print(f"Failed generating and/or include the timestamp ID for {img.name} name: {e}")
                        move_to_error(img, PHOTO_ERROR)
                        continue  # Skip all the steps for this image, jump to move decision

                    # 2) Copy original to output folder or to error folder if copy fails
                    try:
                        copy_original_to_output = PHOTO_OUTPUT / f"original_{img.name}"
                        print(f"Copying original to output: {img.name} -> {copy_original_to_output.name}")
                        shutil.copy2(str(img), str(copy_original_to_output))
                    except Exception as e:
                        print(f"Failed to copy original {img.name} to the output folder: {e}")
                        move_to_error(img, PHOTO_ERROR)
                        continue  # Skip all the steps for this image, jump to move decision

                    # 3) If BMP/GIF, convert to JPEG to enable stable metadata handling
                    try:
                        ext = img.suffix.lower()
                        if ext in {".bmp", ".gif"}:
                            print(f"Image {img.name} is {ext}. Converting to JPEG in order to enable metadata insertion.")
                            img = ensure_processable_image(img, INPUT_DIR)
                        else:
                            print(f"Image {img.name} is in {ext} format. Conversion to JPEG not required for metadata insertion.")
                    except Exception as e:
                        print(f"Warning: Could not determine the image type for {img.name}: {e}.")
                        move_to_error(img, PHOTO_ERROR)
                        continue  # Skip all the steps for this image, jump to move decision

                    # 4) Detect faces on the processing copy
                    try:
                        faces = detector(img, model)
                        if not faces:
                            print(f"No faces detected in: {img.name}")
                        else :
                            print(f"Detected {len(faces)} face(s) in: {img.name}")
                    except Exception as e:
                        print(f"Warning: Could not apply face detection for {img.name}: {e}.")
                        move_to_error(img, PHOTO_ERROR)
                        continue  # Skip all the steps for this image, jump to move decision

                    # 5) Blur faces if any detected
                    try:
                        if len(faces) > 0:
                            print(f"Blurring faces in: {img.name}")
                            success = face_blur(img)
                            if success:
                                print(f"Successfully blurred faces in: {img.name}")
                            else:
                                print(f"Failed to blur faces in: {img.name}")
                                move_to_error(img, PHOTO_ERROR)
                                continue  # VER BEM O QUE ESTE CONTINUE ESTA A FAZER, PODE SER MELHOR USAR A MOVE_TO_ERROR
                        else:
                            print(f"No faces reported by detector for {img.name}. Skipping blur")
                    except Exception as e:
                        print(f"Warning: Could not evaluate or apply face blur for {img.name}: {e}.")
                        move_to_error(img, PHOTO_ERROR)
                        continue  # Skip all the steps for this image, jump to move decision

                    # 6) Move the processed photo to the output folder
                    try:
                        destination_file_path = PHOTO_OUTPUT / os.path.basename(img)
                        shutil.move(str(img), str(destination_file_path))
                        print(f"Moving photo from {img} to {str(destination_file_path)}")
                        print(f"✓ Successfully processed and moved: {img.name}")
                    except Exception as e:
                        print(f"Warning: unable to move the photo to the output folder: {e}")
                        move_to_error(img, PHOTO_ERROR)

            # Sleep between polls to avoid busy-waiting
            time.sleep(POLL_INTERVAL)

        except Exception as e:
            print(f"Watcher error: {e}")
            # Even on unexpected errors, keep polling after a short delay
            time.sleep(POLL_INTERVAL)

    print("Stop requested — watcher exiting")

if __name__ == "__main__":
    main()
