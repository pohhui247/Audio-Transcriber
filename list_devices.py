import sounddevice as sd

print("All audio devices:\n")
print(sd.query_devices())

print("\nDefault input device:")
print(sd.query_devices(kind='input'))