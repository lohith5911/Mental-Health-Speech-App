"""Run emotion inference on a single WAV file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from ml.src.models.emotion_model import MODEL_PATH, predict_emotion_from_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict emotion from a WAV file.")
    parser.add_argument("wav_path", help="Path to a .wav file")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Saved model artifact path")
    args = parser.parse_args()

    result = predict_emotion_from_file(args.wav_path, model_path=args.model)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
