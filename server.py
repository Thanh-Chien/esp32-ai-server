import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.genai import Client
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Nạp biến môi trường
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("Lỗi: Thiếu GEMINI_API_KEY. Hãy kiểm tra file .env hoặc cấu hình Render!")

# Khởi tạo
client = Client(api_key=API_KEY)
app = FastAPI(title="Gemini AI Server Professional")

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cấu trúc dữ liệu
class ChatRequest(BaseModel):
    prompt: str
    history: list = []

@app.get("/")
def health_check():
    return {"status": "online", "message": "Server Gemini đang hoạt động!"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Cách gọi chuẩn: Khởi tạo chat session với history riêng biệt
        chat_session = client.chats.create(
            model="gemini-1.5-flash",
            history=request.history
        )
        
        # Gửi tin nhắn mới
        response = chat_session.send_message(request.prompt)
        
        return {
            "reply": response.text,
            "status": "success"
        }

    except Exception as e:
        print(f"Lỗi hệ thống: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Port 10000 là mặc định của Render
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
