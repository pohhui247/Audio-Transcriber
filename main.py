import librosa
from pitch import detect_pitch
from utils import hz_to_note

AUDIO_FILE = "example.wav"  # put a monophonic recording here

def main():
    # Load audio
    audio, sr = librosa.load(AUDIO_FILE, sr=None)

    # Detect pitch
    pitches = detect_pitch(audio, sr)

    # Convert to notes (remove silence)
    notes = []
    for freq in pitches:
        note = hz_to_note(freq)
        if note:
            notes.append(note)

    # Print results
    print("Detected notes:")
    print(notes[:30])  # first 30 notes for sanity

if __name__ == "__main__":
    main()