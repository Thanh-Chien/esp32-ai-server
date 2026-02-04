from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
import os
import uuid
import time

# =============================
# OPENAI CLIENT
# =============================
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=20.0,          # CỰC KỲ QUAN TRỌNG
)

# =============================
# FASTAPI
# =============================
app = FastAPI(title="ESP32 AI Server FINAL")

# =============================
# DATA MODEL
# =============================
class ChatReq(BaseModel):
    text: str
    session_id: str | None = None

# =============================
@app.get("/")
def root():
    return {
        "status": "OK",
        "server": "ESP32 AI FINAL",
        "time": int(time.time())
    }

# =============================
# CHAT ENDPOINT (FINAL)
# =============================
@app.post("/chat")
def chat(req: ChatReq):
    sid = req.session_id or str(uuid.uuid4())

    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",   # NHẸ – NHANH – ỔN
            input=req.text,
            max_output_tokens=150
        )

        reply = resp.output_text

        if not reply:
            reply = "Em chua nghi ra cau tra loi phu hop."

        return JSONResponse({
            "ok": True,
            "reply": reply,
            "session_id": sid
        })

    except Exception as e:
        # ⚠️ KHÔNG ĐỂ ESP32 TREO
        return JSONResponse({
            "ok": False,
            "reply": "AI dang ban, Sep thu lai sau nhe.",
            "session_id": sid
        })
