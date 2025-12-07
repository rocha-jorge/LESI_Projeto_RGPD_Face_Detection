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
import subprocess
from ultralytics import YOLO
from pathlib import Path


ROOT = Path(__file__).parent.parent
DETECT_OUTPUT = ROOT / "photo_detection_output"
PHOTO_OUTPUT = ROOT / "photo_output"
ID_SCRIPT = ROOT / "src" / "generate_photo_id.py"

from detector import detector
from face_blur import face_blur
from generate_photo_id import generate_photo_id
from list_images import list_images

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
    DETECT_OUTPUT.mkdir(parents=True, exist_ok=True)
    PHOTO_OUTPUT.mkdir(parents=True, exist_ok=True)

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

    copied_originals: set[str] = set()
    while not stop_requested:
        try:
            # Poll the input directory for new images
            images = list_images(INPUT_DIR)
            if images:
                print(f"Found {len(images)} image(s). Processing...")
                for img in images:
                    try:
                        # 1) Generate ID (timestamps/unique naming) in-process.
                        try:
                            _ = generate_photo_id(img)
                        except Exception as e:
                            print(f"Failed to generate ID for {img.name}: {e}")

                        # 2) Copy original to output (traceability/backups)
                        try:
                            import shutil
                            original_copy = PHOTO_OUTPUT / f"original_{img.name}"
                            if img.name not in copied_originals and not original_copy.exists():
                                print(f"Copying original to output: {img.name} -> {original_copy.name}")
                                shutil.copy2(str(img), str(original_copy))
                                copied_originals.add(img.name)
                            else:
                                print(f"Original already copied: {img.name}")
                        except Exception as e:
                            print(f"Failed to copy original for {img.name}: {e}")

                        # 3) Detect faces; only blur when detector reports faces
                        faces = detector(img, model)

                        if isinstance(faces, (list, tuple)) and len(faces) > 0:
                            detected_path = DETECT_OUTPUT / img.name
                            if detected_path.exists():
                                print(f"Blurring faces in: {img.name}")
                                face_blur(detected_path)
                            else:
                                print(f"Detected image not found at {detected_path}")
                        else:
                            print(f"No faces reported by detector for {img.name}; skipping blur")

                    except Exception as e:
                        print(f"Error processing {img.name}: {e}")

            # Sleep between polls to avoid busy-waiting
            time.sleep(POLL_INTERVAL)

        except Exception as e:
            print(f"Watcher error: {e}")
            # Even on unexpected errors, keep polling after a short delay
            time.sleep(POLL_INTERVAL)

    print("Stop requested — watcher exiting")

if __name__ == "__main__":
    main()
