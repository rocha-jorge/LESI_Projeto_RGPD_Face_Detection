#!/usr/bin/env python
"""
FACE DETECTION AND ANONYMIZATION PIPELINE

WHAT IT DOES:
- Step 1: Creates backup copies of all photos with unique timestamp IDs
- Step 2: Detects faces in each photo and saves face locations to metadata (EXIF)
- Step 3: Copies processed photos to the output folder

WHY WE DO THIS:
- We want to find and track faces in photos for privacy compliance
- We keep original backups in case something goes wrong
- We separate concerns: detection, anonymization, and ID generation are in different scripts

HOW IT WORKS:
1. Calls generate_photo_id.py to create backups with timestamps
2. For each photo:
   - Read the photo from photo_input/
   - Use AI (YOLO) to find where faces are located
   - Save face locations as metadata inside the photo (EXIF)
   - Copy the photo to photo_output/ with "_anonymized" suffix
   - Delete original from photo_input/
3. If any errors occur, move the problematic photo to photo_detection_error/
"""

# IMPORTS: These are libraries (tools) we use to do our work
from pathlib import Path          # Working with file paths
import shutil                     # Copying and moving files
import subprocess                 # Running other Python scripts
import sys                        # System utilities
import time                       # Measuring how long things take
import cv2                        # Reading images (OpenCV library)
from PIL import Image             # Another image library
import piexif                     # Reading/writing metadata (EXIF) in photos
from ultralytics import YOLO      # The AI model that detects faces

# --- CONFIG ---
# These are settings we can easily change

ROOT = Path(__file__).parent.parent  # Get the main project folder
INPUT_DIR = ROOT / "photo_input"     # Folder where we PUT photos to process
OUTPUT_DIR = ROOT / "photo_output"   # Folder where we GET processed photos
ERROR_DIR = ROOT / "photo_detection_error"  # Folder for photos that have problems
BLUR_STRENGTH = 100  # How much blur (not used yet, but available for later)
SAVE_EXIF = True     # Should we save face locations to the photo metadata?

# Create the output folders if they don't exist yet
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

# --- LOAD MODEL ---
# We use an AI model called YOLOv8 to detect faces
# This is the AI "brain" that finds faces in photos

MODEL_PATH = ROOT / "models" / "yolov8n-face.pt"  # Where we store the AI model file
WEIGHTS_CACHE = ROOT / "weights" / MODEL_PATH.name  # Sometimes it gets saved elsewhere

# If the model was downloaded to a temporary cache folder, move it to our models folder
if not MODEL_PATH.exists() and WEIGHTS_CACHE.exists():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS_CACHE.replace(MODEL_PATH)
    print(f"Moved cached weights from {WEIGHTS_CACHE} to {MODEL_PATH}")

# If we don't have the model yet, download it
# If we already have it, just load it
if not MODEL_PATH.exists():
    print("Downloading YOLOv8-Face model...")
    model = YOLO("https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
else:
    model = YOLO(str(MODEL_PATH))  # Load existing model

# --- HELPER FUNCTIONS ---
# These are small tools we use repeatedly

def move_to_error(src_path, error_dir, reason=""):
    """
    If something goes wrong with a photo, move it to the error folder
    WHY: We don't want bad photos mixed with good ones
    HOW: Move the file and print a message
    """
    error_path = error_dir / src_path.name
    if src_path.exists():
        shutil.move(str(src_path), str(error_path))
    print(f"    Moved to error folder. {reason}")

def save_faces_exif(image_path, faces):
    """
    Save the face locations into the photo's metadata (EXIF)
    WHY: So we (or the anonymizer script) know where the faces are
    HOW: 
    1. Open the photo
    2. Get its existing metadata
    3. Create a text version of face coordinates (x, y, width, height)
    4. Save this text into the photo's metadata
    5. Write the modified photo back to disk
    
    EXAMPLE: If faces are at [100,150,80,100] and [200,200,90,90],
    we save the text: "100,150,80,100; 200,200,90,90"
    """
    img = Image.open(image_path)
    exif_data = img.info.get("exif", b"")
    
    # If photo has no existing metadata, create empty metadata structure
    if exif_data:
        exif_dict = piexif.load(exif_data)
    else:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}
    
    # Convert face coordinates to text format
    faces_str = "; ".join([f"{x},{y},{w},{h}" for (x, y, w, h) in faces])
    
    # Store the face coordinates in the metadata
    exif_dict["0th"][piexif.ImageIFD.ImageDescription] = faces_str.encode("utf-8")
    exif_bytes = piexif.dump(exif_dict)
    
    # Save photo with updated metadata
    img.save(image_path, exif=exif_bytes)

# --- PROCESS IMAGES ONE BY ONE ---
def process_images():
    """
    Main detection logic: For each photo, detect faces and save coordinates
    WHY: This is the core of what we do
    HOW:
    1. Loop through all photos in photo_input/
    2. For each valid photo:
       a. Read it with the AI model
       b. AI tells us where faces are
       c. Save face locations to the photo's metadata
       d. Copy photo to output folder
       e. Delete original from input folder
    """
    print(f"{'='*60}")
    print("STEP 2: Processing images (detect + anonymize)")
    print(f"{'='*60}\n")
    
    processed_count = 0  # Count how many photos we successfully processed
    
    # Loop through all files in the input folder
    for img_file in INPUT_DIR.glob("*.*"):
        # Only process common photo formats
        if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            print(f"Unsupported file format: {img_file.name}")
            move_to_error(img_file, ERROR_DIR, "Unsupported file extension")
            continue
        
        # Start measuring processing time for this photo
        image_start_time = time.time()
        print(f"{'-'*60}")
        print(f"Image: {img_file.name}")
        print(f"{'-'*60}")
        
        # Create the output filename with "_anonymized" suffix
        # Example: "photo.jpg" becomes "photo_anonymized.jpg"
        name_without_ext = img_file.stem
        output_filename = f"{name_without_ext}_anonymized{img_file.suffix}"
        output_path = OUTPUT_DIR / output_filename
        
        try:
            # --- DETECT FACES ---
            print(f"  [1/2] Detecting faces...")
            
            # Read the photo as an image
            img = cv2.imread(str(img_file))
            if img is None:
                raise Exception("Could not read image")
            
            # Time how long detection takes
            detect_start = time.time()
            results = model(img)  # Run AI model on the image
            detect_time = time.time() - detect_start
            
            # Extract face locations from AI results
            faces_coords = []
            for result in results:
                # Each result has a "boxes" field with face locations
                boxes = result.boxes.xyxy.cpu().numpy()  # x1,y1,x2,y2 format
                
                # Convert corner coordinates to x,y,width,height format
                for box in boxes:
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    faces_coords.append((int(x1), int(y1), int(w), int(h)))
                    print(f"    Face: x={int(x1)}, y={int(y1)}, w={int(w)}, h={int(h)}")
            
            # Save face locations to the photo's metadata
            if SAVE_EXIF and faces_coords:
                save_faces_exif(img_file, faces_coords)
                print(f"    Saved {len(faces_coords)} face coordinate(s) to EXIF")
            
            print(f"    Detection completed in {detect_time:.2f} seconds")
            
            # --- COPY TO OUTPUT ---
            # Copy the photo (with face coordinates now in metadata) to output folder
            print(f"  [2/2] Copying to output...")
            shutil.copy2(str(img_file), str(output_path))
            if faces_coords:
                print(f"    Image with {len(faces_coords)} face(s) copied (ready for anonymization)")
            else:
                print(f"    Image copied (no faces detected)")
            
            # Delete the original from input folder (we have a backup in output folder)
            img_file.unlink()
            
            # Calculate total time for this photo
            total_time = time.time() - image_start_time
            print(f"  ✓ Completed in {total_time:.2f} seconds\n")
            processed_count += 1
        
        except Exception as e:
            # If something goes wrong, move the photo to error folder
            print(f"  ✗ Error: {e}")
            move_to_error(img_file, ERROR_DIR, str(e))
    
    # Print summary
    print(f"{'='*60}")
    print(f"Processing complete: {processed_count} image(s) processed")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Error folder: {ERROR_DIR}")
    print(f"{'='*60}\n")

# --- MAIN PIPELINE ---
def main():
    """
    This is where everything starts
    It orchestrates (coordinates) the whole process
    """
    root = Path(__file__).parent.parent
    generate_photo_id_script = root / "src" / "generate_photo_id.py"
    
    print(f"\n{'#'*60}")
    print("# FACE DETECTION AND ANONYMIZATION PIPELINE")
    print(f"{'#'*60}")
    
    # Step 1: Call the generate_photo_id.py script to create backups with timestamps
    print(f"\n{'='*60}")
    print("STEP 1: Generating photo IDs and backups")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, str(generate_photo_id_script)], cwd=str(root))
    if result.returncode != 0:
        print("Error running generate_photo_id.py")
        return
    
    # Step 2: Process all images (detect faces and save coordinates)
    process_images()
    
    print("✓ Pipeline execution complete!")


if __name__ == "__main__":
    # This line means: "Only run main() if this script is run directly"
    # (not if it's imported by another script)
    main()


