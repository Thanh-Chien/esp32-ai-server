from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from openai import OpenAI
import uuid
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()
sessions = {}   # session_id → message history

# =====================
# DATA MODEL
# =====================
class TextReq(BaseModel):
    text: str
    session_id: str | None = None

# =====================
@app.get("/")
def root():
    return {"status": "ESP32 AI Server OK"}

# =====================
# TEXT → GPT → TEXT
# =====================
@app.post("/chat")
def chat(req: TextReq):
    sid = req.session_id or str(uuid.uuid4())

    if sid not in sessions:
        sessions[sid] = []

    sessions[sid].append({"role": "user", "content": req.text})

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=sessions[sid],
        temperature=0.6
    )

    reply = resp.choices[0].message.content
    sessions[sid].append({"role": "assistant", "content": reply})

    return {
        "reply": reply,
        "session_id": sid
    }

# =====================
# TEXT → SPEECH (TTS)
# =====================
@app.post("/tts")
def tts(req: TextReq):
    audio = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=req.text
    )

    return {
        "audio_base64": audio.data
    }
