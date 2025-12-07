import os
import shutil
import time
from pathlib import Path
import cv2
from PIL import Image
import piexif

# --- CONFIG ---
DETECTION_OUTPUT_DIR = Path(__file__).parent.parent / "photo_detection_output"
ANONYMIZATION_OUTPUT_DIR = Path(__file__).parent.parent / "photo_anonymization_output"
ERROR_DIR = Path(__file__).parent.parent / "photo_detection_error"
BLUR_STRENGTH = 100  # Higher value = stronger blur

# Ensure output directory exists
ANONYMIZATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- HELPER TO MOVE IMAGE TO ERROR FOLDER ---
def move_to_error(src_path, output_path, error_dir, reason=""):
    error_path = error_dir / src_path.name
    if output_path.exists():
        shutil.move(str(output_path), str(error_path))
    if src_path.exists():
        src_path.unlink()
    print(f"Moved {src_path.name} to error folder. {reason}")

# --- HELPER TO EXTRACT FACE COORDINATES FROM EXIF ---
def get_faces_from_exif(image_path):
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
def blur_faces(image_path, output_path, faces):
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

# --- PROCESS IMAGES ---
for img_file in DETECTION_OUTPUT_DIR.glob("*.*"):
    if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
        continue

    start_time = time.time()
    
    # Get face coordinates from EXIF
    faces = get_faces_from_exif(img_file)
    
    if not faces:
        print(f"\nNo faces found in EXIF for {img_file.name}")
        continue
    
    # Create output path
    output_path = ANONYMIZATION_OUTPUT_DIR / img_file.name
    print(f"\nProcessing {img_file.name}...")
    
    try:
        blur_faces(img_file, output_path, faces)
        print(f"Anonymized image saved to {output_path.name}")
        
        # Move original to anonymization folder as backup
        shutil.move(str(img_file), str(ANONYMIZATION_OUTPUT_DIR / f"original_{img_file.name}"))
        print(f"Moved original to anonymization folder as backup.")
        
        elapsed_time = time.time() - start_time
        print(f"✓ Completed in {elapsed_time:.2f} seconds")
    
    except Exception as e:
        print(f"Error processing {img_file.name}: {e}. Moving to error folder.")
        move_to_error(img_file, output_path, ERROR_DIR, f"Anonymization error: {e}")
