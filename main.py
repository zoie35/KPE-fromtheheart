
import faster_whisper
model = faster_whisper.WhisperModel('ivrit-ai/whisper-large-v3-turbo-ct2')

segs, _ = model.transcribe('C:/Users/USER/Desktop/לימודים/רפואה/KPE/Traumatic_024.wav', language='he')

for segment in segs:
    print(f"[{segment.start:.2f} --> {segment.end:.2f}] {segment.text}")


