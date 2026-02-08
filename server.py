import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.genai import Client
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("Thiếu GEMINI_API_KEY!")

# Khởi tạo Client với API v1 để tránh lỗi 404 (model not found on v1beta)
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
    return {"message": "Server đã online. Dùng /chat hoặc /list-models"}

# API liệt kê model (Đã sửa lỗi Attribute)
@app.get("/list-models")
async def list_models():
    try:
        models = []
        # Thư viện mới dùng client.models.list()
        for m in client.models.list():
            models.append({
                "name": m.name,
                "display_name": m.display_name
            })
        return {"available_models": models}
    except Exception as e:
        return {"error": str(e)}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # SỬA TÊN MODEL TẠI ĐÂY (Lấy chính xác từ danh sách bạn đã quét được)
        chat_session = client.chats.create(
            model="models/gemini-2.0-flash", 
            history=request.history
        )
        
        response = chat_session.send_message(request.prompt)
        return {"reply": response.text}

    except Exception as e:
        print(f"DEBUG LOG: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
