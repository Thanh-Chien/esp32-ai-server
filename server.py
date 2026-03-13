import os
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from google.genai import Client
from dotenv import load_dotenv

# =========================
# Load ENV
# =========================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")

# =========================
# App + Client
# =========================
app = FastAPI(title="Gemini AI Server")

client = Client(api_key=GEMINI_API_KEY)

MODEL_NAME = "models/gemini-2.5-flash-lite"

# =========================
# Request schema
# =========================
class ChatRequest(BaseModel):
    prompt: str
    history: list = []

# =========================
# Health check
# =========================
@app.get("/")
async def health():
    return {"status": "online"}

# =========================
# HTTP API (test bằng Postman dễ)
# =========================
@app.post("/chat")
async def chat_api(data: ChatRequest):

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=data.prompt,
        config={
            "max_output_tokens": 80,
            "temperature": 0.4
        }
    )

    return {
        "response": response.text,
        "model": MODEL_NAME
    }

# =========================
# WebSocket API
# =========================
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    try:
        while True:
            raw = await ws.receive_text()
            payload = json.loads(raw)

            prompt = payload.get("prompt", "")

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config={
                    "max_output_tokens": 80,
                    "temperature": 0.4
                }
            )

            result = {
                "response": response.text,
                "model": MODEL_NAME
            }

            await ws.send_text(json.dumps(result))

    except WebSocketDisconnect:
        print("Client disconnected")

    except Exception as e:
        print("WebSocket error:", e)
        await ws.close()
