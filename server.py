import os
import json
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

# =========================
# Load ENV
# =========================
load_dotenv()  # Đọc file .env khi chạy local, bỏ qua khi trên Render

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
SEARCH_API_KEY  = os.getenv("SEARCH_API_KEY", "")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY chưa được set!\n"
        "  - Chạy local: tạo file .env với dòng GEMINI_API_KEY=your_key\n"
        "  - Trên Render: vào Dashboard -> Environment -> thêm GEMINI_API_KEY"
    )

# =========================
# App + Gemini Client
# =========================
app = FastAPI(title="ESP32 AI Server", version="2.0")
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL = "gemini-2.0-flash"

SYSTEM_PROMPT = (
    "Bạn là ESP-Bot, trợ lý AI thân thiện chạy trên thiết bị ESP32. "
    "Luôn trả lời bằng ngôn ngữ người dùng đang dùng (Việt hoặc Anh). "
    "Ngắn gọn tối đa 4 câu. "
    "KHÔNG dùng markdown, bullet, emoji vì hiển thị trên màn hình nhỏ."
)

# Lịch sử hội thoại theo session_id
chat_histories: dict[str, list] = {}

# =========================
# Schemas
# =========================
class ChatRequest(BaseModel):
    prompt: str
    history: list = []
    session_id: str = "default"

class SearchRequest(BaseModel):
    query: str
    session_id: str = "default"

# =========================
# Helper: gọi Gemini có lịch sử
# =========================
def ask(session_id: str, prompt: str, max_tokens: int = 200) -> str:
    history = chat_histories.get(session_id, [])

    # Thêm lượt mới vào history
    history.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

    response = client.models.generate_content(
        model=MODEL,
        contents=history,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=max_tokens,
            temperature=0.7,
        )
    )
    answer = response.text.strip()

    # Lưu lại cả lượt AI trả lời
    history.append(types.Content(role="model", parts=[types.Part(text=answer)]))
    chat_histories[session_id] = history[-20:]  # Giữ 10 lượt gần nhất

    return answer

def ask_once(prompt: str, max_tokens: int = 200) -> str:
    """Gọi không lưu lịch sử (weather, news...)"""
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=0.5,
        )
    )
    return response.text.strip()

# =========================
# Health
# =========================
@app.get("/")
async def health():
    return {
        "status": "online",
        "model": MODEL,
        "sessions": len(chat_histories),
        "features": ["chat", "weather", "search", "news", "calculate", "websocket"]
    }

# =========================
# 1. CHAT - nhớ lịch sử
# =========================
@app.post("/chat")
async def chat(data: ChatRequest):
    """
    Body: {"prompt": "Xin chào", "session_id": "esp32-001"}
    """
    answer = ask(data.session_id, data.prompt)
    return {
        "response":   answer,
        "model":      MODEL,
        "session_id": data.session_id
    }

# =========================
# 2. THỜI TIẾT
# =========================
@app.get("/weather/{city}")
async def weather(city: str, lang: str = "vi"):
    if not WEATHER_API_KEY:
        prompt = (
            f"Mô tả ngắn thời tiết điển hình ở {city} vào thời điểm này trong năm. "
            f"Trả lời {'tiếng Việt' if lang == 'vi' else 'English'}, 2 câu."
        )
        return {"city": city, "response": ask_once(prompt, 100), "source": "ai"}

    try:
        async with httpx.AsyncClient() as http:
            r = await http.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"q": city, "appid": WEATHER_API_KEY,
                        "units": "metric", "lang": "vi"},
                timeout=5
            )
            d = r.json()
        info = (
            f"Nhiệt độ {d['main']['temp']}°C, "
            f"{d['weather'][0]['description']}, "
            f"độ ẩm {d['main']['humidity']}%, "
            f"gió {d['wind']['speed']} m/s"
        )
        prompt = (
            f"Thời tiết tại {city}: {info}. "
            f"Tóm tắt và gợi ý trang phục. 2 câu "
            f"{'tiếng Việt' if lang == 'vi' else 'English'}."
        )
        return {
            "city": city,
            "temp": d["main"]["temp"],
            "description": d["weather"][0]["description"],
            "response": ask_once(prompt, 120),
            "source": "openweathermap"
        }
    except Exception as e:
        return {"city": city, "response": f"Lỗi thời tiết: {e}"}

# =========================
# 3. TÌM KIẾM WEB
# =========================
@app.post("/search")
async def search(data: SearchRequest):
    if not SEARCH_API_KEY:
        answer = ask(data.session_id, data.query)
        return {"query": data.query, "response": answer, "source": "ai"}

    try:
        async with httpx.AsyncClient() as http:
            r = await http.get(
                "https://serpapi.com/search",
                params={"q": data.query, "api_key": SEARCH_API_KEY, "num": 3},
                timeout=8
            )
            results = r.json()
        snippets = [
            f"{x.get('title','')}: {x.get('snippet','')}"
            for x in results.get("organic_results", [])[:3]
            if x.get("snippet")
        ]
        context = "\n".join(snippets)
        prompt = f"Dựa trên thông tin sau, trả lời '{data.query}':\n{context}"
        answer = ask(data.session_id, prompt)
        return {"query": data.query, "response": answer, "source": "web"}
    except Exception as e:
        return {"query": data.query, "response": f"Lỗi tìm kiếm: {e}"}

# =========================
# 4. TIN TỨC
# =========================
@app.get("/news")
async def news(topic: str = "công nghệ", lang: str = "vi"):
    prompt = (
        f"Tóm tắt tin tức quan trọng về '{topic}' gần đây. "
        f"3 điểm chính, mỗi điểm 1 câu. "
        f"Trả lời {'tiếng Việt' if lang == 'vi' else 'English'}."
    )
    return {"topic": topic, "response": ask_once(prompt, 250)}

# =========================
# 5. TÍNH TOÁN / LẬP TRÌNH
# =========================
@app.post("/calculate")
async def calculate(data: ChatRequest):
    prompt = f"Giải bài toán sau, trình bày từng bước ngắn gọn:\n{data.prompt}"
    return {"response": ask_once(prompt, 300)}

# =========================
# 6. LỊCH SỬ
# =========================
@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    chat_histories.pop(session_id, None)
    return {"ok": True, "message": f"Đã xóa lịch sử '{session_id}'"}

@app.get("/history/{session_id}")
async def get_history(session_id: str):
    history = chat_histories.get(session_id, [])
    simplified = [
        {"role": m.role, "text": m.parts[0].text}
        for m in history[-10:]
    ]
    return {"session_id": session_id, "turns": len(history) // 2, "history": simplified}

# =========================
# 7. WEBSOCKET cho ESP32
# =========================
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    ESP32 gửi:  {"prompt": "...", "session_id": "esp32-001", "type": "chat"}
    Server trả: {"response": "...", "type": "chat", "session_id": "esp32-001"}

    type có thể là: chat | weather | search | news | calculate
    """
    await ws.accept()
    print("WebSocket connected")
    try:
        while True:
            raw        = await ws.receive_text()
            payload    = json.loads(raw)
            prompt     = payload.get("prompt", "")
            session_id = payload.get("session_id", "default")
            req_type   = payload.get("type", "chat")

            if req_type == "weather":
                res      = await weather(payload.get("city", prompt))
                response = res["response"]
            elif req_type == "search":
                res      = await search(SearchRequest(query=prompt, session_id=session_id))
                response = res["response"]
            elif req_type == "news":
                res      = await news(topic=payload.get("topic", prompt))
                response = res["response"]
            elif req_type == "calculate":
                res      = await calculate(ChatRequest(prompt=prompt, session_id=session_id))
                response = res["response"]
            else:
                res      = await chat(ChatRequest(prompt=prompt, session_id=session_id))
                response = res["response"]

            await ws.send_text(json.dumps({
                "response":   response,
                "type":       req_type,
                "session_id": session_id
            }, ensure_ascii=False))

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await ws.send_text(json.dumps({"error": str(e)}))
            await ws.close()
        except:
            pass
