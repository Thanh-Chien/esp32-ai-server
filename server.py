from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import uuid
import os
import base64

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

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=sessions[sid],
        temperature=0.6
    )

    reply = resp.output_text

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

    audio_base64 = base64.b64encode(audio.read()).decode("utf-8")

    return {
        "audio_base64": audio_base64
    }
