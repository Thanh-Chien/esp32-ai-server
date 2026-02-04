from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

class Req(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "ESP32 AI server is running"}

@app.post("/chat")
def chat(req: Req):
    r = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": req.text}]
    )
    return {"reply": r.choices[0].message.content}
