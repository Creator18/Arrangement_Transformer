"""
Download + extract MAESTRO v3.0.0 (MIDI-only) into <repo-root>/data/.

The full MAESTRO dataset is ~120 GB (audio + MIDI).  We only train on the
MIDI, so the MIDI-only archive is enough — ~80 MB compressed, ~250 MB
extracted.

Usage:
    cd backend
    python -m scripts.download_maestro
    # or override the destination:
    python -m scripts.download_maestro --root /elsewhere/maestro-v3.0.0

After extraction:
    data/maestro-v3.0.0/
        maestro-v3.0.0.csv
        2004/  ... 2018/

That layout matches app.ml.config.PATHS.maestro_root, so the training CLI
finds it automatically.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.ml.config import PATHS


_MAESTRO_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"


def _download_with_progress(url: str, dest: Path) -> None:
    """Streaming download with a MiB-resolution progress indicator."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[maestro] downloading {url}")
    print(f"[maestro] -> {dest}")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk = 1 << 20  # 1 MiB
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            out.write(buf)
            downloaded += len(buf)
            if total:
                print(
                    f"\r[maestro] {downloaded // (1<<20)} / {total // (1<<20)} MiB",
                    end="", flush=True,
                )
        print()
    tmp.replace(dest)


def main() -> int:
    p = argparse.ArgumentParser(description="Download MAESTRO v3.0.0 MIDI-only")
    p.add_argument(
        "--root", type=Path, default=PATHS.maestro_root,
        help=f"Destination directory (default: {PATHS.maestro_root})",
    )
    p.add_argument(
        "--keep-zip", action="store_true",
        help="Keep the downloaded .zip after extraction",
    )
    args = p.parse_args()

    target_root: Path = args.root
    parent: Path = target_root.parent
    parent.mkdir(parents=True, exist_ok=True)

    if (target_root / "maestro-v3.0.0.csv").exists():
        print(f"[maestro] already extracted at {target_root} — nothing to do.")
        return 0

    zip_path = parent / "maestro-v3.0.0-midi.zip"
    if not zip_path.exists():
        _download_with_progress(_MAESTRO_URL, zip_path)
    else:
        print(f"[maestro] reusing existing zip at {zip_path}")

    print(f"[maestro] extracting into {parent}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(parent)

    manifest = target_root / "maestro-v3.0.0.csv"
    if not manifest.exists():
        print(f"[maestro] ERROR: expected manifest not found at {manifest}")
        return 1

    if not args.keep_zip:
        zip_path.unlink()
        print(f"[maestro] removed zip archive (use --keep-zip to retain)")

    print(f"[maestro] done.  Manifest at {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
