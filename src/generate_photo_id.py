#!/usr/bin/env python
"""Generate photo IDs and create backups.

For each photo in photo_input/:
1. Generate a unique ID with current timestamp (milliseconds)
2. Copy the original photo to photo_output/ with suffix "_original"
3. Leave the original in photo_input/ for processing by detector.py
"""
from pathlib import Path
import shutil
from datetime import datetime

# --- CONFIG ---
INPUT_DIR = Path(__file__).parent.parent / "photo_input"
OUTPUT_DIR = Path(__file__).parent.parent / "photo_output"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- PROCESS IMAGES ---
def main():
    processed_count = 0
    
    for img_file in INPUT_DIR.glob("*.*"):
        if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            print(f"Skipping unsupported format: {img_file.name}")
            continue
        
        # Generate timestamp in milliseconds
        timestamp_ms = int(datetime.now().timestamp() * 1000)
        
        # Create new filename: original_name_<timestamp>.extension
        name_without_ext = img_file.stem
        new_name = f"{name_without_ext}_{timestamp_ms}{img_file.suffix}"
        
        # Create backup in photo_output with "_original" suffix
        backup_name = f"{new_name.replace(img_file.suffix, '')}_original{img_file.suffix}"
        backup_path = OUTPUT_DIR / backup_name
        
        print(f"\nProcessing: {img_file.name}")
        print(f"  Generated ID: {new_name}")
        print(f"  Creating backup: {backup_name}")
        
        try:
            # Copy original to output folder as backup
            shutil.copy2(str(img_file), str(backup_path))
            print(f"  ✓ Backup created in photo_output/")
            processed_count += 1
        except Exception as e:
            print(f"  ✗ Error creating backup: {e}")
    
    print(f"\n{'='*60}")
    print(f"Photo ID generation complete: {processed_count} image(s) processed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
