from pathlib import Path
import shutil


def move_to_error(img: Path, photo_error: Path) -> Path:
    """Move a file to the error folder under the name original_<filename>.

    Returns the destination path. Logs failures but does not raise.
    """
    print(f"✗ Processing {img.name} was unsuccessful. Moving to error folder.")

    try:
        dest = photo_error / f"original_{img.name}"
        shutil.move(str(img), str(dest))
        print(f"Moved {img.name} to error folder: {dest}")
        return dest
    except Exception as e:
        print(f"Failed to move {img.name} to error folder: {e}")
        try:
            img.unlink(missing_ok=True)
            print(f"Deleted problematic file from input: {img}")
        except Exception as del_err:
            print(f"Failed to delete problematic file {img}: {del_err}")
        return img