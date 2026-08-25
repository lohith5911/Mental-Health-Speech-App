from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from ml.src.features.features import (
    extract_acoustic_features_stats,
    extract_mfcc_delta_stats,
    extract_mfcc_stats,
)
from ml.src.models.cnn_emotion_model import (
    SMALL_CNN_CONFIG,
    CNNEmotionModel,
    compute_log_mel_spectrogram,
    predict_emotion_from_file_cnn,
)
from ml.src.models.v4_emotion_model import (
    Wav2Vec2EmotionClassifier,
    compute_wav2vec_embeddings,
    predict_emotion_from_file_v4,
)
from ml.src.models.emotion_model import (
    build_speaker_aware_split,
    EMOTION_MAP,
    MODEL_PATH,
    VERSION_2_MODEL_PATH,
    VERSION_3_MODEL_PATH,
    load_cremad_dataset,
    load_feature_matrix,
    list_audio_files,
    parse_emotion_from_filename,
    parse_speaker_id,
    predict_emotion_from_file,
)
from ml.src.preprocessing.audio import load_audio


class EmotionModelTests(unittest.TestCase):
    def test_load_cremad_wav(self):
        dataset_path = Path(__file__).parents[1] / "data/raw/crema-d/AudioWAV/1001_DFA_ANG_XX.wav"
        waveform, sample_rate = load_audio(dataset_path)
        self.assertEqual(sample_rate, 16000)
        self.assertGreater(waveform.size, 0)
        self.assertTrue(np.isfinite(waveform).all())

    def test_parse_emotion_from_filename(self):
        self.assertEqual(parse_emotion_from_filename("1001_DFA_ANG_XX.wav"), "ANG")
        self.assertEqual(parse_emotion_from_filename("1001_IEO_HAP_HI.wav"), "HAP")
        self.assertIsNone(parse_emotion_from_filename("not_a_label.wav"))

    def test_cremad_dataset_loads(self):
        files = list_audio_files(Path(__file__).parents[1] / "data/raw/crema-d/AudioWAV")
        self.assertEqual(len(files), 7442)
        features, labels = load_cremad_dataset()
        self.assertEqual(features.shape, (7442, 52))
        self.assertEqual(labels.shape, (7442,))
        self.assertTrue(set(labels).issubset(EMOTION_MAP))

    def test_extract_mfcc_stats_shape(self):
        waveform = np.random.randn(16000).astype(np.float32)
        features = extract_mfcc_stats(waveform, sample_rate=16000, n_mfcc=13)
        self.assertEqual(features.shape[0], 52)

    def test_extract_mfcc_delta_stats_shape(self):
        waveform = np.random.randn(16000).astype(np.float32)
        features = extract_mfcc_delta_stats(waveform, sample_rate=16000, n_mfcc=13)
        self.assertEqual(features.shape[0], 78)
        self.assertTrue(np.isfinite(features).all())

    def test_extract_v3_acoustic_stats_shape(self):
        waveform = np.random.randn(16000).astype(np.float32)
        features = extract_acoustic_features_stats(waveform, sample_rate=16000, n_mfcc=13)
        self.assertGreaterEqual(features.shape[0], 78)
        self.assertEqual(features.ndim, 1)
        self.assertTrue(np.isfinite(features).all())

    def test_extract_mfcc_stats_has_no_nan_values(self):
        waveform = np.zeros(16000, dtype=np.float32)
        waveform[0] = np.nan
        features = extract_mfcc_stats(waveform, sample_rate=16000, n_mfcc=13)
        self.assertTrue(np.isfinite(features).all())

    def test_v3_feature_extractor_has_fixed_dimension(self):
        waveform_a = np.random.RandomState(0).randn(16000).astype(np.float32)
        waveform_b = np.random.RandomState(1).randn(16000).astype(np.float32)
        left = extract_acoustic_features_stats(waveform_a, sample_rate=16000, n_mfcc=13)
        right = extract_acoustic_features_stats(waveform_b, sample_rate=16000, n_mfcc=13)
        self.assertEqual(left.shape, right.shape)
        self.assertEqual(left.ndim, 1)
        self.assertTrue(np.isfinite(left).all())
        self.assertTrue(np.isfinite(right).all())

    def test_v3_dataset_compatibility(self):
        files = list_audio_files(Path(__file__).parents[1] / "data/raw/crema-d/AudioWAV")[:20]
        feature_matrix, labels, corrupted = load_feature_matrix(files, feature_extractor=extract_acoustic_features_stats)
        self.assertEqual(feature_matrix.shape[0], len(labels))
        self.assertEqual(feature_matrix.ndim, 2)
        self.assertTrue(np.isfinite(feature_matrix).all())
        self.assertFalse(corrupted)
        self.assertTrue(set(labels).issubset(EMOTION_MAP))

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

    def test_cremad_speaker_split_has_no_overlap(self):
        files = list_audio_files(Path(__file__).parents[1] / "data/raw/crema-d/AudioWAV")
        split = build_speaker_aware_split(files, seed=42)
        speaker_sets = [
            {parse_speaker_id(path.name) for path in split[name]}
            for name in ("train", "validation", "test")
        ]
        self.assertEqual(len(set.union(*speaker_sets)), 91)
        for index, speakers in enumerate(speaker_sets):
            for other in speaker_sets[index + 1:]:
                self.assertTrue(speakers.isdisjoint(other))

    def test_inference_pipeline_on_generated_audio(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample.wav"
            sr = 16000
            waveform = np.sin(2 * np.pi * 220 * np.linspace(0, 1, sr, endpoint=False)).astype(np.float32)
            sf.write(path, waveform, sr)
            self.assertTrue(path.exists())
            self.assertGreater(len(waveform), 0)

    def test_inference_from_saved_cremad_model(self):
        audio_path = Path(__file__).parents[1] / "data/raw/crema-d/AudioWAV/1001_DFA_ANG_XX.wav"
        result = predict_emotion_from_file(audio_path, model_path=MODEL_PATH)
        self.assertIn(result["predicted_emotion"], EMOTION_MAP)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_inference_from_saved_v2_cremad_model(self):
        audio_path = Path(__file__).parents[1] / "data/raw/crema-d/AudioWAV/1001_DFA_ANG_XX.wav"
        result = predict_emotion_from_file(audio_path, model_path=VERSION_2_MODEL_PATH)
        self.assertIn(result["predicted_emotion"], EMOTION_MAP)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertTrue(np.isfinite(result["confidence"]))

    def test_inference_from_saved_v3_cremad_model(self):
        audio_path = Path(__file__).parents[1] / "data/raw/crema-d/AudioWAV/1001_DFA_ANG_XX.wav"
        result = predict_emotion_from_file(audio_path, model_path=VERSION_3_MODEL_PATH)
        self.assertIn(result["predicted_emotion"], EMOTION_MAP)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertTrue(np.isfinite(result["confidence"]))

    def test_mel_spectrogram_shape(self):
        waveform, sample_rate = load_audio(Path(__file__).parents[1] / "data/raw/crema-d/AudioWAV/1001_DFA_ANG_XX.wav")
        mel = compute_log_mel_spectrogram(waveform, sample_rate=sample_rate, n_mels=SMALL_CNN_CONFIG["n_mels"], n_frames=SMALL_CNN_CONFIG["n_frames"])
        self.assertEqual(mel.shape, (SMALL_CNN_CONFIG["n_mels"], SMALL_CNN_CONFIG["n_frames"]))
        self.assertTrue(np.isfinite(mel).all())

    def test_cnn_model_forward_pass(self):
        model = CNNEmotionModel(
            n_mels=SMALL_CNN_CONFIG["n_mels"],
            n_frames=SMALL_CNN_CONFIG["n_frames"],
            n_classes=len(EMOTION_MAP),
        )
        x = torch.randn(2, 1, SMALL_CNN_CONFIG["n_mels"], SMALL_CNN_CONFIG["n_frames"])
        logits = model(x)
        self.assertEqual(logits.shape, (2, len(EMOTION_MAP)))

    def test_cnn_label_validity(self):
        self.assertEqual(set(EMOTION_MAP), {"ANG", "DIS", "FEA", "HAP", "NEU", "SAD"})

    def test_cnn_inference_output(self):
        audio_path = Path(__file__).parents[1] / "data/raw/crema-d/AudioWAV/1001_DFA_ANG_XX.wav"
        result = predict_emotion_from_file_cnn(audio_path)
        self.assertIn(result["predicted_emotion"], EMOTION_MAP)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertTrue(np.isfinite(result["confidence"]))

    def test_v4_encoder_output_shape(self):
        audio_path = Path(__file__).parents[1] / "data/raw/crema-d/AudioWAV/1001_DFA_ANG_XX.wav"
        embeddings = compute_wav2vec_embeddings(audio_path)
        self.assertEqual(embeddings.ndim, 1)
        self.assertGreater(embeddings.shape[0], 0)
        self.assertTrue(np.isfinite(embeddings).all())

    def test_v4_classifier_forward_pass(self):
        model = Wav2Vec2EmotionClassifier(embedding_dim=768, n_classes=len(EMOTION_MAP))
        x = torch.randn(2, 768)
        logits = model(x)
        self.assertEqual(logits.shape, (2, len(EMOTION_MAP)))

    def test_v4_inference_output(self):
        audio_path = Path(__file__).parents[1] / "data/raw/crema-d/AudioWAV/1001_DFA_ANG_XX.wav"
        result = predict_emotion_from_file_v4(audio_path)
        self.assertIn(result["predicted_emotion"], EMOTION_MAP)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertTrue(np.isfinite(result["confidence"]))

    def test_git_lfs_pointer_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "fake.wav"
            path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc123\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Git LFS pointer"):
                load_audio(path)


if __name__ == "__main__":
    unittest.main()
