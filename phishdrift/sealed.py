"""Encrypted storage for files that contain feed URLs. See DATA_HANDLING.md.

OpenPhish's terms allow research use but forbid making the feed's contents
available to third parties, so files holding OpenPhish (or URLhaus) URLs are
committed encrypted: `<name>.csv.gz.enc`, a Fernet token wrapping the
gzipped CSV. Committing the ciphertext before scoring keeps the prospective
record's guarantee (each day is fixed in git before it is used) without
publishing the URLs. `SEALED_SHA256.txt` lists the SHA-256 of every file's
decrypted bytes, so anyone given the key can check nothing was swapped.

The key comes from the FEED_KEY environment variable (a GitHub Actions
secret in CI) or from `.feed_key` at the repository root (gitignored).
"""

from __future__ import annotations

import gzip
import io
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
KEY_FILE = ROOT / ".feed_key"
MANIFEST = ROOT / "SEALED_SHA256.txt"
SUFFIX = ".enc"


def _fernet():
    from cryptography.fernet import Fernet
    key = os.environ.get("FEED_KEY") or (KEY_FILE.read_text().strip() if KEY_FILE.exists() else "")
    if not key:
        raise RuntimeError("FEED_KEY is not set and .feed_key does not exist; feed files are "
                           "encrypted (see DATA_HANDLING.md)")
    return Fernet(key.encode())


def seal(data: bytes) -> bytes:
    return _fernet().encrypt(data)


def unseal(token: bytes) -> bytes:
    return _fernet().decrypt(token)


def read_csv(path: Path) -> pd.DataFrame:
    """A `.csv.gz.enc` file is decrypted in memory; a plain `.csv.gz` is read as is."""
    path = Path(path)
    if path.name.endswith(SUFFIX):
        return pd.read_csv(io.BytesIO(gzip.decompress(unseal(path.read_bytes()))))
    return pd.read_csv(path)


def write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Write `frame` as an encrypted gzipped CSV; `path` gets `.enc` appended if missing."""
    path = Path(path)
    if not path.name.endswith(SUFFIX):
        path = path.with_name(path.name + SUFFIX)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as fh:
        fh.write(frame.to_csv(index=False).encode("utf-8"))
    path.write_bytes(seal(buf.getvalue()))
    _record(path, gzip.decompress(buf.getvalue()))
    return path


def _record(path: Path, csv: bytes) -> None:
    """Append the file's decrypted-CSV hash to SEALED_SHA256.txt (repo files only)."""
    import hashlib
    try:
        rel = path.resolve().relative_to(ROOT).as_posix()
    except ValueError:                                   # outside the repo, e.g. a test's temp dir
        return
    if MANIFEST.exists():
        with open(MANIFEST, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(f"{hashlib.sha256(csv).hexdigest()}  {rel}\n")


def stem(path: Path) -> str:
    """'2026-09-21.csv.gz.enc' or '2026-09-21.csv.gz' -> '2026-09-21'."""
    return Path(path).name.removesuffix(SUFFIX).removesuffix(".csv.gz")


def data_files(folder: Path) -> list[Path]:
    """Every snapshot in `folder`, encrypted or not, sorted by name."""
    folder = Path(folder)
    return sorted(list(folder.glob("*.csv.gz" + SUFFIX)) + list(folder.glob("*.csv.gz")),
                  key=lambda p: p.name)
