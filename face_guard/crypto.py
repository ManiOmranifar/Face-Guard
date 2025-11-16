import base64
import io
from pathlib import Path
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet
import os
import json
from .utils import atomic_write_bytes, DATA_FOLDER

KDF_ITERATIONS = 390000
MASTER_SALT_SIZE = 16

def gen_salt(n=MASTER_SALT_SIZE) -> bytes:
    return os.urandom(n)

def derive_key_from_password(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=KDF_ITERATIONS,
        backend=default_backend()
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key

def encrypt_bytes(key: bytes, data: bytes) -> bytes:
    f = Fernet(key)
    return f.encrypt(data)

def decrypt_bytes(key: bytes, token: bytes) -> bytes:
    f = Fernet(key)
    return f.decrypt(token)

def save_encrypted_file(path: Path, key: bytes, data: bytes):
    enc = encrypt_bytes(key, data)
    atomic_write_bytes(path, enc)

def load_encrypted_file(path: Path, key: bytes) -> bytes:
    from pathlib import Path
    p = Path(path)
    with p.open('rb') as f:
        enc = f.read()
    return decrypt_bytes(key, enc)
