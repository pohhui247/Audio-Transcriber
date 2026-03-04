import librosa
import numpy as np

def detect_pitch(audio, sr):
    """
    Returns an array of pitch values (Hz) over time
    """
    pitches = librosa.yin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr
    )

    return pitches