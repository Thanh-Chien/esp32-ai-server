import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.genai import Client
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Khởi tạo Client ép về API v1 để ổn định hạn mức
client = Client(
    api_key=API_KEY,
    http_options={'api_version': 'v1'}
)

app = FastAPI(title="Gemini ESP32 Server")

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
async def root():
    return {"status": "online", "model": "gemini-1.5-flash"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Sử dụng gemini-1.5-flash vì có Quota ổn định nhất cho tài khoản Free
        chat_session = client.chats.create(
            model="models/gemini-1.5-flash", 
            history=request.history
        )
        
        response = chat_session.send_message(request.prompt)
        return {"reply": response.text}

    except Exception as e:
        error_msg = str(e)
        print(f"DEBUG LOG: {error_msg}")
        
        # Xử lý lỗi 429 (Hết hạn mức)
        if "429" in error_msg:
            raise HTTPException(
                status_code=429, 
                detail="Hết hạn mức yêu cầu (Quota). Vui lòng đợi 10-60 giây rồi thử lại."
            )
        
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/list-models")
async def list_models():
    try:
        return {"available_models": [m.name for m in client.models.list()]}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
