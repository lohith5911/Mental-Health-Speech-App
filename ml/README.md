# Speech Emotion Detection (Milestone 3)

## Project title
AI-Powered Mental Health Screening Through Daily Speech

## Current scope
This project is currently focused on speech emotion detection for emotional self-monitoring. The work here is limited to processing and preparing the CREMA-D dataset for future emotion-recognition experiments.

Emotion detection is used for emotional self-monitoring and does not constitute a clinical diagnosis.

## Dataset
The initial dataset for this project is CREMA-D, a speech dataset containing recordings labeled with six emotions:
- Anger
- Disgust
- Fear
- Happy
- Neutral
- Sad

The developer should manually download and place the CREMA-D dataset at:

ml/data/raw/crema-d/

Do not commit the dataset to Git. Do not copy it into backend/uploads.

## ML environment setup
Create the isolated environment at:

ml/.venv

Use the packages listed in ml/requirements.txt.

## Dataset inspection
The script at ml/scripts/inspect_cremad.py scans the local dataset, extracts the emotion label from CREMA-D filenames, and prints a summary of the audio count and emotion distribution.

If the dataset is not present yet, the script explains where to place it and exits cleanly.

## Future preprocessing and features
The project includes placeholder modules for future use:
- audio preprocessing
- MFCC feature extraction
- mel spectrogram extraction
- spectral centroid extraction
- zero-crossing rate extraction
- RMS energy extraction

These are not final training features yet and are prepared for later model development.

## Future model training
The eventual workflow is:

CREMA-D
↓
Audio preprocessing
↓
Feature extraction
↓
Speaker-aware split
↓
Baseline classifier
↓
Evaluation
↓
Best model
↓
Saved model

The current milestone does not train the final model.

## Future FastAPI integration
This project is not yet connected to FastAPI inference. The current backend upload flow remains unchanged and independent from the machine learning pipeline.

## Future dashboard emotion tracking
A future milestone may add emotion summaries to the dashboard for daily emotional self-monitoring.

## Future enhancement: depression screening
Depression screening and other clinical mental-health detection are explicitly out of scope for this milestone and are reserved for a future enhancement.
