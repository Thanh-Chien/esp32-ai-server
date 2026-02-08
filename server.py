import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from dotenv import load_dotenv

# Tải biến môi trường
load_dotenv()

# Cấu hình Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

app = FastAPI()

# Cấu hình CORS để có thể gọi từ trình duyệt hoặc ứng dụng khác
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    prompt: str
    history: list = []

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Khởi tạo phiên chat với lịch sử nhận từ client
        chat_session = model.start_chat(history=request.history)
        response = chat_session.send_message(request.prompt)
        
        return {"reply": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Server Python Gemini đang chạy!"}
