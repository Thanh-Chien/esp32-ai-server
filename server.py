import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.genai import Client
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# 1. Cấu hình môi trường
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("Lỗi: Thiếu GEMINI_API_KEY trong biến môi trường!")

# 2. Khởi tạo Client Gemini & FastAPI
client = Client(api_key=API_KEY)
app = FastAPI(title="Gemini AI Server")

# 3. Cấu hình CORS (Cho phép các ứng dụng khác gọi vào)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Có thể thay bằng domain cụ thể của bạn để bảo mật
    allow_methods=["*"],
    allow_headers=["*"],
)

# Định dạng dữ liệu đầu vào
class ChatRequest(BaseModel):
    prompt: str
    history: list = []

@app.get("/")
def health_check():
    return {"status": "online", "model": "gemini-1.5-flash"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Gửi tin nhắn đến Gemini (Sử dụng SDK mới nhất)
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=request.prompt,
            config={
                "history": request.history
            }
        )
        
        if not response.text:
            raise HTTPException(status_code=500, detail="AI không trả về nội dung.")

        return {"reply": response.text}

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__server__":
    import uvicorn
    # Chạy local tại port 8000
    uvicorn.run(app, host="0.0.0.0", port=10000)
