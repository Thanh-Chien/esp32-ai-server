import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.genai import Client
from dotenv import load_dotenv

load_dotenv()
client = Client(api_key=os.getenv("GEMINI_API_KEY"))
app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str
    history: list = []

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Sử dụng bản Lite mới nhất từ danh sách của bạn
        # Bản này nhẹ, nhanh và ổn định nhất cho Key Free
        chat_session = client.chats.create(
            model="models/gemini-flash-lite-latest", 
            history=request.history
        )
        
        response = chat_session.send_message(request.prompt)
        return {"reply": response.text}

    except Exception as e:
        error_msg = str(e)
        print(f"Lỗi: {error_msg}")
        
        # Xử lý lỗi hết hạn mức 429
        if "429" in error_msg:
            return {"reply": "Hạn mức API Free đã hết (Quota Exceeded). Vui lòng đợi 30s-60s rồi thử lại."}
            
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/list-models")
async def list_models():
    return {"available_models": [m.name for m in client.models.list()]}

@app.get("/")
async def health():
    return {"status": "online"}
