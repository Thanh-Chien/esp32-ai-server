import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.genai import Client
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Khởi tạo Client mặc định (để thư viện tự chọn phiên bản tốt nhất)
client = Client(api_key=API_KEY)

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str
    history: list = []

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # GIẢI PHÁP: Thử dùng model ID đầy đủ hoặc model thay thế
        # Thử 1: gemini-1.5-flash
        # Thử 2 (nếu lỗi): gemini-1.5-flash-001
        
        chat_session = client.chats.create(
            model="gemini-1.5-flash", 
            history=request.history
        )
        
        response = chat_session.send_message(request.prompt)
        return {"reply": response.text}

    except Exception as e:
        print(f"DEBUG LOG: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="Model không phản hồi. Hãy kiểm tra API Key hoặc khu vực hỗ trợ."
        )

# API này để bạn kiểm tra xem Key của bạn dùng được model nào
@app.get("/list-models")
async def list_models():
    try:
        models = []
        for m in client.models.list():
            models.append({"name": m.name, "methods": m.supported_generation_methods})
        return {"available_models": models}
    except Exception as e:
        return {"error": str(e)}
