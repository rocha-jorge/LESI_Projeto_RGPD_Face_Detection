import os
import logging
from pathlib import Path
from typing import Optional

from utils.paths import IMAGE_OUTPUT

def encrypt_to_aesgcm(in_file: Path, out_file: Path, password: str) -> bool:
    """Encrypt a file using AES-GCM with a password-derived key.

    File format: [magic 4B 'ENC1'][salt 16B][nonce 12B][ciphertext...][tag 16B]
    """
    try:
        import importlib, secrets
        pbkdf2_mod = importlib.import_module('cryptography.hazmat.primitives.kdf.pbkdf2')
        hashes_mod = importlib.import_module('cryptography.hazmat.primitives.hashes')
        aead_mod = importlib.import_module('cryptography.hazmat.primitives.ciphers.aead')
        PBKDF2HMAC = getattr(pbkdf2_mod, 'PBKDF2HMAC')
        AESGCM = getattr(aead_mod, 'AESGCM')
        SHA256 = getattr(hashes_mod, 'SHA256')
        salt = secrets.token_bytes(16)
        kdf = PBKDF2HMAC(
            algorithm=SHA256(),
            length=32,
            salt=salt,
            iterations=200_000,
        )
        key = kdf.derive(password.encode("utf-8"))
        aesgcm = AESGCM(key)
        nonce = secrets.token_bytes(12)
        data = in_file.read_bytes()
        ct = aesgcm.encrypt(nonce, data, associated_data=None)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open("wb") as f:
            f.write(b"ENC1")
            f.write(salt)
            f.write(nonce)
            f.write(ct)
        return True
    except ImportError:
        logging.error("Encryption library missing. Install 'cryptography' to enable AES-GCM.")
        return False
    except Exception:
        logging.error("Failed to encrypt original image", exc_info=True)
        return False

def encrypt_original(img: Path, password: Optional[str]) -> bool:
    """Encrypt the original image into image_output/encrypted_originals as .enc.

    If password is None or empty, returns False (no encryption performed).
    """
    if not password:
        return False
    dest = IMAGE_OUTPUT / "encrypted_originals" / f"original_{img.name}.enc"
    logging.info(f"Encrypting original to: {dest}")
    return encrypt_to_aesgcm(img, dest, password)

# Hardcoded password (set this to enable encryption). Leave empty to disable.
ENCRYPTION_PASSWORD: str = "password"

def get_hardcoded_password() -> Optional[str]:
    return ENCRYPTION_PASSWORD.strip() or None
