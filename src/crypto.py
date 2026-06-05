import base64
import os

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

SALT_LENGTH = 16
PBKDF2_ITERATIONS = 600_000


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
    )
    return base64.urlsafe_b64encode(kdf.derive(passphrase.encode("utf-8")))


def encrypt(plaintext: str, passphrase: str) -> str:
    if not plaintext or not passphrase:
        raise ValueError("Plaintext and passphrase are required")
    salt = os.urandom(SALT_LENGTH)
    key = _derive_key(passphrase, salt)
    token = Fernet(key).encrypt(plaintext.encode("utf-8"))
    return base64.b64encode(salt + token).decode("ascii")


def decrypt(encrypted: str, passphrase: str) -> str:
    if not encrypted or not passphrase:
        raise ValueError("Encrypted data and passphrase are required")
    try:
        raw = base64.b64decode(encrypted)
        salt = raw[:SALT_LENGTH]
        token = raw[SALT_LENGTH:]
        key = _derive_key(passphrase, salt)
        return Fernet(key).decrypt(token).decode("utf-8")
    except Exception:
        raise ValueError("Falha ao descriptografar. Senha incorreta ou dados corrompidos.")
