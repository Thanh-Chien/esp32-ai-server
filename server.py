from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import uuid
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()
sessions = {}

class TextReq(BaseModel):
    text: str
    session_id: str | None = None

@app.post("/chat")
def chat(req: TextReq):
    try:
        sid = req.session_id or str(uuid.uuid4())

        if sid not in sessions:
            sessions[sid] = []

        sessions[sid].append({"role": "user", "content": req.text})

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=sessions[sid],
            temperature=0.6,
            timeout=10   # ⭐ CỰC KỲ QUAN TRỌNG
        )

        reply = resp.choices[0].message.content
        sessions[sid].append({"role": "assistant", "content": reply})

        return {
            "reply": reply,
            "session_id": sid
        }

    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail="AI error")
