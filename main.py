
import faster_whisper
import openai
import certifi
import httpx

openAiKey = ""
model = faster_whisper.WhisperModel('ivrit-ai/whisper-large-v3-turbo-ct2')

segs, _ = model.transcribe('C:/Users/USER/Desktop/לימודים/רפואה/KPE/Traumatic_024.wav', language='he')
lines = []
for segment in segs:
    lines.append((f"[{segment.start:.2f} --> {segment.end:.2f}] {segment.text}\t"))
client = openai.OpenAI(api_key=openAiKey,
    http_client=httpx.Client(verify=False))
# Make the request
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {"role": "system", "content": "אתה מקבל טקסט שבנוי מסגמנט של התחלה וסגמנט של סוף ובכל אחד מהם יש מילים - אני רוצה שתמצא לי את ה-30 שניות שהכי עשויות להפעיל פוסט טראומה מבחינת עוררות ועוצמה רגשית - תחזיר לי בתשובה רק התחלה וסוף בלי עוד מלים בפורמט start:end"},
        {"role": "user", "content": lines}
    ]
)
# Print the response
print(response.choices[0].message.content)
