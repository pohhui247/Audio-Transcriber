import sounddevice as sd
import numpy as np
import librosa

# ---------------- CONFIG ----------------
SAMPLE_RATE = 44100
BUFFER_SIZE = 2048
MIN_FREQ = librosa.note_to_hz("C2")   # ~65 Hz
MAX_FREQ = librosa.note_to_hz("C7")   # ~2093 Hz
VOLUME_THRESHOLD = 0.005               # ignore quiet input (lowered for sensitivity)
DEBUG = True                            # print per-frame debug info
MIC_INDEX = 2                           # force input device index (set to 2)
# ----------------------------------------

last_note = None
stable_count = 0
STABLE_FRAMES = 3  # require note to persist for 3 frames before printing

def freq_to_note(freq):
    """Convert frequency (Hz) to closest note name."""
    note_num = 12 * np.log2(freq / 440.0) + 69
    rounded = int(round(note_num))
    note_index = rounded % 12
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (rounded // 12) - 1
    return f"{note_names[note_index]}{octave}"

def audio_callback(indata, frames, time, status):
    global last_note, stable_count

    if status:
        print(status)
    
    # Flatten input (in case stereo)
    audio_data = indata[:, 0] if indata.ndim > 1 else indata

    # Compute RMS volume for debugging and thresholding
    rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
    # Debug RMS logging temporarily disabled

    # Ignore quiet frames (don't clobber last_note, just reset stability)
    if rms < VOLUME_THRESHOLD:
        stable_count = 0
        return

    # Apply a Hann window to reduce spectral leakage
    windowed = audio_data * np.hanning(len(audio_data))

    # Zero-pad FFT to improve frequency resolution
    n_fft = len(windowed) * 4
    fft = np.fft.rfft(windowed, n=n_fft)
    fft_magnitude = np.abs(fft)
    freqs = np.fft.rfftfreq(n_fft, 1 / SAMPLE_RATE)

    # Focus only on frequencies in desired range
    idx = np.where((freqs >= MIN_FREQ) & (freqs <= MAX_FREQ))[0]
    if len(idx) == 0:
        return

    # Pick peak within the band
    band_mags = fft_magnitude[idx]
    peak_rel = np.argmax(band_mags)
    peak_idx = idx[peak_rel]

    # Parabolic interpolation to refine peak frequency
    if 1 <= peak_idx < len(fft_magnitude) - 1:
        alpha = fft_magnitude[peak_idx - 1]
        beta = fft_magnitude[peak_idx]
        gamma = fft_magnitude[peak_idx + 1]
        denom = (alpha - 2 * beta + gamma)
        if denom != 0:
            delta = 0.5 * (alpha - gamma) / denom
        else:
            delta = 0.0
    else:
        delta = 0.0

    refined_bin = peak_idx + delta
    detected_freq = refined_bin * SAMPLE_RATE / n_fft

    note = freq_to_note(detected_freq)

    # Only print note if it stays stable for several frames
    if note == last_note:
        stable_count += 1
    else:
        stable_count = 1
        last_note = note

    if stable_count >= STABLE_FRAMES:
        print(f"{note} ({detected_freq:.1f} Hz)")
        stable_count = 0  # reset so it prints again on next stable detection

# ---------------- START STREAM ----------------
print("Default input device:", sd.default.device)
try:
    print("Available input devices:\n", sd.query_devices(kind='input'))
except Exception:
    print("Could not query input devices")

print(f"Forcing input device index: {MIC_INDEX}")

with sd.InputStream(device=MIC_INDEX, channels=1, callback=audio_callback,
                    samplerate=SAMPLE_RATE, blocksize=BUFFER_SIZE):
    print("Listening... Play some notes!")
    while True:
        sd.sleep(1000)