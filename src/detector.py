# src/detector.py
import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import cv2
from PIL import Image
import piexif

# --- CONFIG ---
SRC_DIR = Path(__file__).parent.parent / "photo_input"
OUTPUT_DIR = Path(__file__).parent.parent / "photo_detection_output"
ERROR_DIR = Path(__file__).parent.parent / "photo_detection_error"
SAVE_EXIF = True  # set False if you don't want to save metadata

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

# --- LOAD MODEL ---
MODEL_PATH = Path(__file__).parent.parent / "models" / "yolov8n-face.pt" # path object

if not MODEL_PATH.exists():  # check if file exists
    print("Downloading YOLOv8-face model...")
    model = YOLO("yolov8n-face")  # this fetches the model from Ultralytics
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)  # make sure models/ folder exists
    model.save(MODEL_PATH)        # saves it to models/
else:
    model = YOLO(str(MODEL_PATH))  # load local copy


# --- HELPER TO SAVE FACE COORDINATES TO EXIF ---
def save_faces_exif(image_path, faces):
    img = Image.open(image_path)
    exif_dict = piexif.load(img.info.get("exif", b""))
    faces_str = "; ".join([f"{x},{y},{w},{h}" for (x, y, w, h) in faces])
    exif_dict["0th"][piexif.ImageIFD.ImageDescription] = faces_str.encode("utf-8")
    exif_bytes = piexif.dump(exif_dict)
    img.save(image_path, exif=exif_bytes)

# --- HELPER TO MOVE IMAGE TO ERROR FOLDER ---
def move_to_error(src_path, output_path, error_dir, reason=""):
    error_path = error_dir / src_path.name
    if output_path.exists():
        shutil.move(str(output_path), str(error_path))
    if src_path.exists():
        src_path.unlink()
    print(f"Moved {src_path.name} to error folder. {reason}")

# --- PROCESS IMAGES ---
for img_file in SRC_DIR.glob("*.*"):
    if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
        print(f"\nUnsupported file format: {img_file.name}. Moving to error folder.")
        move_to_error(img_file, None, ERROR_DIR, "Unsupported file extension")
        continue

    # Move image to output directory
    output_path = OUTPUT_DIR / img_file.name
    print(f"\nMoving {img_file.name} to output folder...")
    shutil.copy2(str(img_file), str(output_path))
    
    print(f"Processing {img_file.name}...")
    img = cv2.imread(str(output_path))
    if img is None:
        print(f"Error: could not read {output_path.name}. Moving to error folder.")
        move_to_error(img_file, output_path, ERROR_DIR, "Could not read image")
        continue
    
    try:
        results = model(img)
    except Exception as e:
        print(f"Error processing {img_file.name}: {e}. Moving to error folder.")
        move_to_error(img_file, output_path, ERROR_DIR, f"Processing error: {e}")
        continue

    faces_coords = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # x1,y1,x2,y2
        for box in boxes:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            faces_coords.append((int(x1), int(y1), int(w), int(h)))
            print(f"Face: x={int(x1)}, y={int(y1)}, w={int(w)}, h={int(h)}")

    if SAVE_EXIF and faces_coords:
        save_faces_exif(output_path, faces_coords)
        print("Saved face coordinates to EXIF.")
    
    # Remove original file from input folder
    img_file.unlink()
    print(f"Removed {img_file.name} from input folder.")
