import sounddevice as sd
import numpy as np
import librosa
import time

# -----------------------------
# Configuration
# -----------------------------
SAMPLE_RATE = 44100
BUFFER_SIZE = 2048
MIN_FREQ = librosa.note_to_hz("C2")
MAX_FREQ = librosa.note_to_hz("C7")
MIC_INDEX = 2           # <--- your working microphone index
AMPLIFY = 5             # optional boost for quiet mic
# -----------------------------

last_note = None

def audio_callback(indata, frames, time_info, status):
    global last_note

    if status:
        print(status)

    # Convert stereo to mono and amplify
    audio = np.mean(indata, axis=1) * AMPLIFY

    # Detect pitch using YIN
    pitch = librosa.yin(
        audio,
        fmin=MIN_FREQ,
        fmax=MAX_FREQ,
        sr=SAMPLE_RATE
    )

    # Take median pitch of the block
    freq = np.median(pitch)

    if freq > 0:
        note = librosa.hz_to_note(freq)
        # Only print when the note changes
        if note != last_note:
            print(note)
            last_note = note

def main():
    print("🎤 Listening... Play a single-note instrument (Ctrl+C to stop)")
    try:
        with sd.InputStream(
            device=MIC_INDEX,
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=BUFFER_SIZE,
            callback=audio_callback
        ):
            while True:
                time.sleep(0.1)  # keeps stream alive
    except KeyboardInterrupt:
        print("\n🛑 Stopped cleanly")

if __name__ == "__main__":
    main()