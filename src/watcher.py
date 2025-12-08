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



# ADICIONAR 

# Configurable thresholds: Add env/config for detection confidence, IoU, min face size, and blur strength. Makes results tunable without code changes.
# Structured logging: Replace prints with Python logging (levels, file handlers with rotation). Optionally add JSON logs for easy ingestion.
# evitar processar ficheiros que ainda não acabaram de ser escritos na pasta de input // Ready file detection: Avoid processing partially copied files by verifying file size is stable (e.g., unchanged across N checks) before processing.

# Non-JPEG metadata: EXIF on PNG/WebP isn’t standard. Keep EXIF for JPEG/TIFF, but add XMP embedding for PNG/WebP to preserve face boxes cross-format. Sidecar JSON as a fallback if XMP lib isn’t available.
# HEIC/HEIF support: You list “.heic” as acceptable — add pillow-heif to read and convert to JPEG/PNG for processing and metadata.

# GPU/CPU selection: Env flag to force CPU if CUDA is missing; log the selected device for clarity.

# Unit tests: For generate_timestamp_name, rename_photo, ensure_processable_image, EXIF round-trip (JPEG), and XMP embedding (PNG).
# E2E harness: A small scripted test that drops 3–5 images (no faces + faces of different sizes) and verifies expected outputs in photo_output with timings.

# EXIF hygiene: After you write face-location metadata, consider stripping other PII EXIF tags (GPS, camera serial) unless explicitly needed.

# Filesystem events: Replace polling with watchdog for instant reaction and less CPU (still keep a fallback polling mode).

import os
import sys
import time
import signal
import shutil
from pathlib import Path
from convert_image import ensure_processable_image


ROOT = Path(__file__).parent.parent
PHOTO_OUTPUT = ROOT / "photo_output"
PHOTO_ERROR = ROOT / "photo_error"

from detector import detector
from face_blur import face_blur
from rename_with_timestamp_id import generate_timestamp_name, rename_photo
from list_images import list_images
from move_to_error import move_to_error
from setup_environment import setup_environment_and_model

# Configure via environment variables (use absolute paths when mounting volumes)
INPUT_DIR = Path(os.environ.get("PHOTO_INPUT", ROOT / "photo_input"))
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "5"))

# Flag used by the main loop to know when to exit.
# When a shutdown signal is received (CTRL+C or service stop),
# we set this to True and let the loop break cleanly.
stop_requested = False

# Signal handler: flips the flag when termination/interruption is requested.
def handle_sigterm(_signum, _frame):
    global stop_requested
    stop_requested = True

# Bind both SIGTERM (service stop) and SIGINT (CTRL+C) to the handler
# so the watcher can shut down gracefully.
signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)


# setup_environment_and_model now lives in src/setup_environment.py

def main():
    # Setup environment and initialize YOLO once (kept in memory for the life of the process)
    model = setup_environment_and_model(INPUT_DIR)

    while not stop_requested:
        try:
            # Poll the input directory for new images
            images = list_images(INPUT_DIR)     # image paths in input dir
            if images:
                print(f"Found {len(images)} image(s). Processing...")
                for img in images:

                    start_ts = time.perf_counter()

                    print("\n" + "═" * 60)
                    print(f"▶ Processing: {img.name}")
                    print("═" * 60)

                    # 1) Generate and include timestamp ID in filename
                    try:
                        new_name = generate_timestamp_name(img)
                        img = rename_photo(img, INPUT_DIR, new_name)
                        print(f"Renamed: {img} to {new_name}")
                    except Exception as e:
                        print(f"Failed generating and/or including the timestamp ID for {img.name} name: {e}")
                        move_to_error(img, PHOTO_ERROR)
                        continue  # Skip all the steps for this image, jump to move decision

                    # 2) Copy original to output folder
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
                            success = face_blur(img, faces)
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

                    # 6) Rename and move the processed photo to the output folder
                    try:
                        destination = PHOTO_OUTPUT / f"anonymized_{img.name}"
                        print(f"Moving photo from {img.name} to {str(destination)}")
                        shutil.move(str(img), str(destination))
                        print(f"✓ Successfully processed and moved: {img.name} to {destination.name}")
                    except Exception as e:
                        print(f"Warning: unable to move the photo to the output folder: {e}")
                        move_to_error(img, PHOTO_ERROR)

                    elapsed = time.perf_counter() - start_ts
                    print(f"Processing time for {img.name}: {elapsed:.2f} seconds")
                    
            # Sleep between polls to avoid busy-waiting
            time.sleep(POLL_INTERVAL)

        except Exception as e:
            print(f"Watcher error: {e}")
            # Even on unexpected errors, keep polling after a short delay
            time.sleep(POLL_INTERVAL)

    print("Stop requested — watcher exiting")

if __name__ == "__main__":
    main()
