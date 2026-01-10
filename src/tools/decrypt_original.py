import sys
import logging
from pathlib import Path

from utils.paths import PROJECT_ROOT, IMAGE_OUTPUT

try:
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except Exception:
    AESGCM = None
    PBKDF2HMAC = None
    hashes = None

def _derive_key(password: str, salt: bytes, iterations: int = 200_000) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=iterations,
    )
    return kdf.derive(password.encode("utf-8"))

def decrypt_file(enc_file: Path, out_file: Path, password: str) -> bool:
    if AESGCM is None:
        logging.error("Cryptography not installed. Please install 'cryptography'.")
        return False
    try:
        blob = enc_file.read_bytes()
        if blob[:4] != b"ENC1":
            logging.error("Unsupported file format.")
            return False
        salt = blob[4:20]
        nonce = blob[20:32]
        ct = blob[32:]
        key = _derive_key(password, salt)
        aesgcm = AESGCM(key)
        data = aesgcm.decrypt(nonce, ct, associated_data=None)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_bytes(data)
        logging.info(f"Decrypted to: {out_file}")
        return True
    except Exception:
        logging.error("Failed to decrypt file", exc_info=True)
        return False

def main():
    if len(sys.argv) < 3:
        print("Usage: python -m src.tools.decrypt_original <path> <password> [output]")
        print("- <path>: .enc file OR a directory containing .enc files")
        print("- [output]: for file, output path; for directory, output directory (defaults to '<path>/decrypted')")
        sys.exit(1)
    in_path = Path(sys.argv[1])
    password = sys.argv[2]
    # Directory mode: decrypt all .enc files
    if in_path.is_dir():
        if len(sys.argv) >= 4:
            out_dir = Path(sys.argv[3])
        else:
            out_dir = in_path / "decrypted"
        out_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        failed = 0
        for enc_file in in_path.glob("*.enc"):
            stem = enc_file.stem
            if stem.startswith("original_"):
                stem = stem[len("original_"):]
            out_file = out_dir / stem
            ok = decrypt_file(enc_file, out_file, password)
            if ok:
                count += 1
            else:
                failed += 1
        print(f"Decryption completed: {count} succeeded, {failed} failed. Output: {out_dir}")
        sys.exit(0 if failed == 0 else 3)
    # Single-file mode
    else:
        enc_path = in_path
        if len(sys.argv) >= 4:
            out_path = Path(sys.argv[3])
        else:
            stem = enc_path.stem
            if stem.startswith("original_"):
                stem = stem[len("original_"):]
            out_path = enc_path.with_name(stem)
        ok = decrypt_file(enc_path, out_path, password)
        sys.exit(0 if ok else 2)

if __name__ == "__main__":
    main()
