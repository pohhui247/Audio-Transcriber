import librosa

def hz_to_note(freq):
    """
    Convert frequency (Hz) to note name (e.g. A4)
    """
    if freq <= 0:
        return None
    return librosa.hz_to_note(freq)