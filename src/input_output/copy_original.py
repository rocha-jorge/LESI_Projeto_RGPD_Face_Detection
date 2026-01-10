import logging
import shutil
from pathlib import Path
from utils.paths import IMAGE_OUTPUT, IMAGE_ERROR
from input_output.move_to_error import move_to_error
from input_output.encrypt_original import encrypt_original, get_hardcoded_password

def copy_original_to_output(img: Path) -> bool:
    """Encrypt or copy the original file into encrypted_originals.

    If env var ORIGINALS_ENCRYPTION_PASSWORD is set, encrypt to .enc;
    otherwise, perform a direct copy to original_<name>.
    On failure, move to error and return False.
    """
    try:
        password = get_hardcoded_password()
        if password:
            ok = encrypt_original(img, password)
            if not ok:
                raise Exception("Encryption failed")
            else:
                logging.info("Original image encrypted successfully")
        else:
            dest = IMAGE_OUTPUT / "encrypted_originals" / f"original_{img.name}"
            logging.info(f"Copying original to output: {img.name} -> {dest}")
            shutil.copy2(str(img), str(dest))
        return True
    except Exception:
        logging.error(f"Failed to copy original {img.name} to the output folder", exc_info=True)
        move_to_error(img)
        return False
