import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.genai import Client
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("Lỗi: Thiếu GEMINI_API_KEY!")

# Cấu hình Client ép sử dụng phiên bản API v1 để tránh lỗi 404 Not Found
client = Client(
    api_key=API_KEY,
    http_options={'api_version': 'v1'}
)

app = FastAPI(title="Gemini AI Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    prompt: str
    history: list = []

@app.get("/")
def health_check():
    return {"status": "online"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Sử dụng model ID chuẩn
        chat_session = client.chats.create(
            model="gemini-1.5-flash",
            history=request.history
        )
        
        response = chat_session.send_message(request.prompt)
        
        return {
            "reply": response.text,
            "status": "success"
        }

    except Exception as e:
        error_msg = str(e)
        print(f"Lỗi: {error_msg}")
        # Nếu vẫn 404, hãy thử đổi model sang 'gemini-1.5-flash-001'
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
