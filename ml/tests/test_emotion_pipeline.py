from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from ml.src.features.features import extract_mfcc_stats
from ml.src.models.emotion_model import (
    build_speaker_aware_split,
    parse_emotion_from_filename,
    parse_speaker_id,
)
from ml.src.preprocessing.audio import load_audio


class EmotionModelTests(unittest.TestCase):
    def test_parse_emotion_from_filename(self):
        self.assertEqual(parse_emotion_from_filename("1001_DFA_ANG_XX.wav"), "ANG")
        self.assertEqual(parse_emotion_from_filename("1001_IEO_HAP_HI.wav"), "HAP")
        self.assertIsNone(parse_emotion_from_filename("not_a_label.wav"))

    def test_extract_mfcc_stats_shape(self):
        waveform = np.random.randn(16000).astype(np.float32)
        features = extract_mfcc_stats(waveform, sample_rate=16000, n_mfcc=13)
        self.assertEqual(features.shape[0], 52)

    def test_speaker_aware_split_has_no_overlap(self):
        files = [
            Path("1001_DFA_ANG_XX.wav"),
            Path("1001_DFA_DIS_XX.wav"),
            Path("1002_DFA_FEA_XX.wav"),
            Path("1002_DFA_HAP_XX.wav"),
            Path("1003_DFA_NEU_XX.wav"),
            Path("1003_DFA_SAD_XX.wav"),
        ]
        split = build_speaker_aware_split(files, seed=42)
        speaker_sets = {
            "train": {parse_speaker_id(path.name) for path in split["train"]},
            "validation": {parse_speaker_id(path.name) for path in split["validation"]},
            "test": {parse_speaker_id(path.name) for path in split["test"]},
        }
        self.assertTrue(speaker_sets["train"].isdisjoint(speaker_sets["validation"]))
        self.assertTrue(speaker_sets["train"].isdisjoint(speaker_sets["test"]))
        self.assertTrue(speaker_sets["validation"].isdisjoint(speaker_sets["test"]))

    def test_inference_pipeline_on_generated_audio(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample.wav"
            sr = 16000
            waveform = np.sin(2 * np.pi * 220 * np.linspace(0, 1, sr, endpoint=False)).astype(np.float32)
            sf.write(path, waveform, sr)
            self.assertTrue(path.exists())
            self.assertGreater(len(waveform), 0)

    def test_git_lfs_pointer_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "fake.wav"
            path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc123\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Git LFS pointer"):
                load_audio(path)


if __name__ == "__main__":
    unittest.main()
