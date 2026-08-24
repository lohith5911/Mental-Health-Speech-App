# Speech Emotion Detection (Milestone 4)

## Project title
AI-Powered Mental Health Screening Through Daily Speech

## Current scope
This milestone focuses on speech emotion detection for emotional self-monitoring. The work here is limited to the CREMA-D speech dataset and a baseline speaker-aware ML pipeline.

Emotion detection is used for emotional self-monitoring and does not constitute a clinical diagnosis.

## Dataset
The dataset is CREMA-D, a speech corpus with recordings labeled with six emotions:
- Anger
- Disgust
- Fear
- Happy
- Neutral
- Sad

The local dataset lives at:

ml/data/raw/crema-d/AudioWAV/

Do not commit the dataset to Git. Keep ml/data/ ignored.

## ML environment setup
The isolated environment is:

ml/.venv

Use the packages listed in ml/requirements.txt.

## Preprocessing
The project loads WAV files with a safe audio loader that:
- reads the waveform
- converts to mono if needed
- resamples to 16 kHz
- normalizes amplitude
- handles unreadable or empty files gracefully without altering the original dataset

## Feature extraction
The baseline feature vector uses MFCC statistics:
- mean MFCC coefficients
- standard deviation MFCC coefficients
- delta statistics
- delta-delta statistics

This produces a fixed-length representation suitable for classical ML models.

## Label parsing
The implementation reads emotion codes from CREMA-D filenames:
- ANG = angry
- DIS = disgust
- FEA = fear
- HAP = happy
- NEU = neutral
- SAD = sad

## Speaker-aware split
The data is split by speaker ID so that no speaker appears in more than one split:
- ~70% train
- ~15% validation
- ~15% test

The split is reproducible with a fixed random seed.

## Model used
The initial baseline is a classical SVM pipeline:
- StandardScaler
- SVC with RBF kernel
- class_weight="balanced"

## Evaluation
The pipeline reports:
- accuracy
- precision
- recall
- F1-score
- macro F1
- weighted F1
- confusion matrix
- per-emotion metrics

## Training and inference
Use the training script:

ml/.venv/bin/python ml/scripts/train_emotion_model.py

Use the inference script:

ml/.venv/bin/python ml/scripts/infer_emotion.py path/to/audio.wav

## Artifacts
The model artifact is saved under:

ml/artifacts/

This directory is kept small and does not include the dataset itself.

## Validation
The project includes unit tests for:
- filename parsing
- feature extraction shape
- speaker-aware splitting
- inference pipeline readiness

## Future FastAPI integration
This project is not yet connected to FastAPI inference. The current backend upload flow remains unchanged and independent from the machine learning pipeline.

## Future enhancement: depression screening
Depression screening and other clinical mental-health detection are explicitly out of scope for this milestone and reserved for a future enhancement.
