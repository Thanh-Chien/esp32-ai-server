import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.genai import Client
from dotenv import load_dotenv

load_dotenv()
# Khởi tạo Client mặc định (để SDK tự chọn bản v1 hoặc v1beta phù hợp với Key Free)
client = Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str
    history: list = []

@app.get("/")
async def root():
    return {"status": "success", "mode": "Free Tier Optimized"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Dùng bản Lite để tránh lỗi 429 (Resource Exhausted)
        chat_session = client.chats.create(
            model="gemini-2.0-flash-lite-preview-02-05", 
            history=request.history
        )
        
        response = chat_session.send_message(request.prompt)
        return {"reply": response.text}

    except Exception as e:
        error_msg = str(e)
        # Nếu bị hết hạn mức (429), trả về thông báo thay vì sập server
        if "429" in error_msg:
            return {"reply": "Hệ thống đang bận do hạn mức Free đã hết. Thử lại sau 1 phút nhé!"}
        
        # Nếu model không tồn tại (404), thử dùng bản Lite cơ bản
        if "404" in error_msg:
             return {"reply": "Lỗi cấu hình Model. Hãy kiểm tra lại tên model trong list-models."}
             
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/list-models")
async def list_models():
    # Giúp bạn luôn theo dõi được Key Free của bạn đang được phép dùng model nào
    return {"available_models": [m.name for m in client.models.list()]}
