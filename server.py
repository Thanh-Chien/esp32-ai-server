import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google.genai import Client
from dotenv import load_dotenv

# =========================
# Load environment variables
# =========================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in environment variables")

# =========================
# Create FastAPI app
# =========================
app = FastAPI(title="Gemini WebSocket Server")

# =========================
# Create Gemini client
# =========================
client = Client(api_key=GEMINI_API_KEY)

# =========================
# Health check endpoint
# =========================
@app.get("/")
async def health():
    return {"status": "online"}

# =========================
# WebSocket endpoint
# =========================
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    # Create chat session
    chat = client.chats.create(
        model="gemini-1.5-flash"
    )

    try:
        while True:
            # Receive message from client
            message = await ws.receive_text()

            # Send to Gemini
            response = chat.send_message(message)

            # Send response back
            await ws.send_text(response.text)

    except WebSocketDisconnect:
        print("Client disconnected")

    except Exception as e:
        print("WebSocket error:", e)
        await ws.close()
