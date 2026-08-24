"""Download and extract the CREMA-D AudioWAV dataset without using the paginated GitHub API."""

from __future__ import annotations

import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

REPO_URL = "https://github.com/CheyneyComputerScience/CREMA-D/archive/refs/heads/master.zip"
ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT_DIR / "data" / "raw" / "crema-d"
TARGET_DIR = DATASET_ROOT / "AudioWAV"
ZIP_PATH = DATASET_ROOT / "crema-d-master.zip"


def download_zip(url: str, destination: Path) -> None:
    """Download the repository archive to a local zip path."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    context = ssl._create_unverified_context()
    with urllib.request.urlopen(request, context=context, timeout=120) as response, open(destination, "wb") as handle:
        shutil.copyfileobj(response, handle)


def extract_wavs(zip_path: Path, output_dir: Path) -> tuple[int, int]:
    """Extract only the WAV files from the repository zip into the target directory.

    Returns a tuple of (extracted_count, skipped_count).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    skipped = 0

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            normalized = member.filename.replace("\\", "/")
            lower_name = normalized.lower()
            if not lower_name.endswith(".wav"):
                continue
            if "/audiowav/" not in lower_name and "/audio.wav/" not in lower_name:
                continue

            filename = PurePosixPath(normalized).name
            if not filename:
                continue

            destination = output_dir / filename
            if destination.exists():
                skipped += 1
                continue

            with archive.open(member) as src, open(destination, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1

    return extracted, skipped


def main() -> None:
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Target directory: {TARGET_DIR}")

    if TARGET_DIR.exists():
        existing = sorted(p.name for p in TARGET_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".wav")
        print(f"Existing WAV files before download: {len(existing)}")
    else:
        print("Existing WAV files before download: 0")

    if not ZIP_PATH.exists():
        print(f"Downloading archive from {REPO_URL}")
        download_zip(REPO_URL, ZIP_PATH)
    else:
        print(f"Archive already present: {ZIP_PATH}")

    extracted, skipped = extract_wavs(ZIP_PATH, TARGET_DIR)
    final_count = len([p for p in TARGET_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".wav"])

    print(f"Extracted new WAV files: {extracted}")
    print(f"Skipped existing WAV files: {skipped}")
    print(f"Final WAV file count: {final_count}")
    print("Dataset acquisition complete.")


if __name__ == "__main__":
    main()
