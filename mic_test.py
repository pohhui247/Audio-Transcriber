import sounddevice as sd
import numpy as np
import time

SAMPLE_RATE = 44100
MIC_INDEX = 2  # try 1 first, then 2 if needed

def callback(indata, frames, time_info, status):
    volume = np.linalg.norm(indata) * 10
    print(f"Volume: {volume:.2f}")

print("🎤 Speak or clap. Ctrl+C to stop.")

try:
    with sd.InputStream(
        device=MIC_INDEX,
        channels=1,
        samplerate=SAMPLE_RATE,
        callback=callback
    ):
        while True:
            time.sleep(0.1)
except KeyboardInterrupt:
    print("\n🛑 Done")