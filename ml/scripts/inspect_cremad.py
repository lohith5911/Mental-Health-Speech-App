"""Inspect the local CREMA-D dataset directory and summarize audio files by emotion label."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

DATASET_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "crema-d"

EXPECTED_EMOTIONS = {
    "ANG",
    "DIS",
    "FEA",
    "HAP",
    "NEU",
    "SAD",
}


def infer_emotion_from_filename(filename: str) -> str | None:
    stem = Path(filename).stem
    if len(stem) < 3:
        return None

    code = stem.split("_")[-1].upper()
    if code in EXPECTED_EMOTIONS:
        mapping = {
            "ANG": "Anger",
            "DIS": "Disgust",
            "FEA": "Fear",
            "HAP": "Happy",
            "NEU": "Neutral",
            "SAD": "Sad",
        }
        return mapping.get(code)

    return None


def main() -> None:
    if not DATASET_DIR.exists():
        print("CREMA-D dataset not found.")
        print(f"Please place the downloaded dataset in: {DATASET_DIR}")
        print("Expected local structure: ml/data/raw/crema-d/")
        return

    audio_files = sorted(DATASET_DIR.rglob("*"))
    valid_files = [path for path in audio_files if path.is_file() and path.suffix.lower() in {".wav", ".mp3", ".ogg", ".m4a", ".webm"}]

    rows: list[dict[str, str]] = []
    invalid_files: list[str] = []

    for file_path in valid_files:
        emotion = infer_emotion_from_filename(file_path.name)
        if emotion is None:
            invalid_files.append(str(file_path.relative_to(DATASET_DIR)))
            continue
        rows.append({
            "file_path": str(file_path.resolve()),
            "emotion": emotion,
        })

    df = pd.DataFrame(rows, columns=["file_path", "emotion"])

    print(f"Total audio files detected: {len(valid_files)}")
    if not df.empty:
        counts = df["emotion"].value_counts().sort_index()
        print("Files per emotion:")
        print(counts.to_string())
        print("Detected emotion labels:")
        print(sorted(df["emotion"].unique().tolist()))
    else:
        print("Files per emotion:")
        print("No valid CREMA-D files found.")
        print("Detected emotion labels:")
        print([])

    if invalid_files:
        print("\nMissing or invalid filenames detected:")
        for item in invalid_files:
            print(f" - {item}")

    if not df.empty:
        print(f"\nDataFrame preview rows: {len(df)}")
        print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
