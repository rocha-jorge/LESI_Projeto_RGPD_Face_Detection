#!/usr/bin/env python
"""Generate photo IDs and create backups.

For each photo in photo_input/:
1. Generate a unique ID with current timestamp (milliseconds)
2. Copy the original photo to photo_output/ with suffix "_original"
3. Leave the original in photo_input/ for processing by detector.py
"""
from pathlib import Path
import shutil
import sys
from datetime import datetime

# --- CONFIG ---
INPUT_DIR = Path(__file__).parent.parent / "photo_input"
OUTPUT_DIR = Path(__file__).parent.parent / "photo_output"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- API: single-file ID generation ---
def generate_photo_id(img_file: Path) -> str:
    """Generate a timestamp-based ID for one image.

        Returns the new filename with ID appended (does not move/copy the original).
        Watcher or caller is responsible for copying backups to photo_output.
        """
    if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
        raise ValueError(f"Unsupported format: {img_file.name}")

    timestamp_ms = int(datetime.now().timestamp() * 1000)
    name_without_ext = img_file.stem
    new_name = f"{name_without_ext}_{timestamp_ms}{img_file.suffix}"

    return new_name


# --- CLI: process images in a folder or a single file ---
def main():
    processed_count = 0

    # Optional: process a single file passed as argument
    single_file = None
    if len(sys.argv) > 1:
        single_file = Path(sys.argv[1])
        if not single_file.is_absolute():
            single_file = (Path(__file__).parent.parent / single_file).resolve()

    files_iter = [single_file] if single_file else INPUT_DIR.glob("*.*")

    for img_file in files_iter:
        if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            print(f"Skipping unsupported format: {img_file.name}")
            continue
        
        # Generate ID only (no copying performed here)
        timestamp_ms = int(datetime.now().timestamp() * 1000)
        name_without_ext = img_file.stem
        new_name = f"{name_without_ext}_{timestamp_ms}{img_file.suffix}"
        
        print(f"\nProcessing: {img_file.name}")
        print(f"  Generated ID: {new_name}")
        print(f"  (Caller will handle copying backup)")
        
        try:
            # No file operations here; just report the ID
            processed_count += 1
        except Exception as e:
            print(f"  ✗ Error creating backup: {e}")
    
    print(f"\n{'='*60}")
    print(f"Photo ID generation complete: {processed_count} image(s) processed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
