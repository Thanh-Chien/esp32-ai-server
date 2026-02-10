import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google.genai import Client
from dotenv import load_dotenv
import uvicorn

load_dotenv()

app = FastAPI()

# ===== Gemini client =====
client = Client(api_key=os.getenv("GEMINI_API_KEY"))

# ===== Health check =====
@app.get("/")
async def health():
    return {"status": "online"}

# ===== WebSocket endpoint =====
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    chat = client.chats.create(
        model="models/gemini-flash-lite-latest"
    )

    try:
        while True:
            message = await ws.receive_text()

            response = chat.send_message(message)

            await ws.send_text(response.text)

    except WebSocketDisconnect:
        print("Client disconnected")

    except Exception as e:
        print("WS error:", e)
        await ws.close()


# ===== Run server =====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
