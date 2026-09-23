"""Fetch the versioned official Gesture Recognizer model, without importing ROS."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)
DEFAULT_MODEL_PATH = Path.home() / ".local/share/ghost_game/gesture_recognizer.task"
# Integrity pin of the version-1 bundle previously retrieved from MODEL_URL.
# This is a project checksum, not a claim of an upstream-signed SHA-256 manifest.
MODEL_SHA256 = "97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482"
MODEL_BYTES = 8_373_440


def verify_model(path: Path) -> str:
    """Reject truncation, HTTP error pages and unexpected version changes."""
    if path.stat().st_size != MODEL_BYTES:
        raise ValueError(f"Unexpected model size: {path.stat().st_size}, expected {MODEL_BYTES}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != MODEL_SHA256:
        raise ValueError(f"Model SHA-256 mismatch: {actual}")
    return actual


def download_model(output: Path = DEFAULT_MODEL_PATH, *, force: bool = False) -> Path:
    """Download to a sibling temporary file and replace only after verification."""
    output = Path(output).expanduser().absolute()
    if output.exists() and not force:
        try:
            verify_model(output)
        except (OSError, ValueError) as exc:
            raise ValueError(f"{exc}. Existing file kept; pass --force to redownload.") from exc
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=output.name + ".", suffix=".part", dir=output.parent, delete=False
        ) as sink:
            temporary = Path(sink.name)
            total = 0
            with urlopen(MODEL_URL, timeout=30) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    total += len(chunk)
                    if total > MODEL_BYTES:
                        raise ValueError("Downloaded model exceeds pinned size")
                    sink.write(chunk)
            sink.flush()
            os.fsync(sink.fileno())
        verify_model(temporary)
        os.replace(temporary, output)
        temporary = None
        return output
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--force", action="store_true", help="Replace an existing file after verification")
    args = parser.parse_args(argv)
    try:
        path = download_model(args.output, force=args.force)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Model download failed: {exc}\n")
    print(f"Model ready: {path}\nSHA-256: {MODEL_SHA256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
