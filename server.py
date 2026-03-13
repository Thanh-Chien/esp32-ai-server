import os
import json
import datetime
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from google.genai import Client
from dotenv import load_dotenv

# =========================
# Load ENV
# =========================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")   # openweathermap.org (miễn phí)
SEARCH_API_KEY  = os.getenv("SEARCH_API_KEY", "")    # serpapi.com (miễn phí)

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY chưa được set!")

# =========================
# App + Gemini Client
# =========================
app = FastAPI(title="ESP32 AI Server", version="2.0")
client = Client(api_key=GEMINI_API_KEY)
MODEL = "models/gemini-2.5-flash-lite"

# Lưu lịch sử hội thoại theo session (RAM)
# key = session_id, value = list các lượt chat
chat_sessions: dict[str, list[dict]] = {}

# =========================
# System Prompt - tính cách AI
# =========================
SYSTEM_PROMPT = """Bạn là ESP-Bot, trợ lý AI thông minh chạy trên thiết bị ESP32.

Nguyên tắc trả lời:
- Trả lời bằng ngôn ngữ người dùng đang dùng (Việt hoặc Anh)
- Ngắn gọn, rõ ràng — tối đa 4 câu cho câu hỏi thông thường
- KHÔNG dùng markdown (**, ##, -, *) vì hiển thị trên màn hình nhỏ
- KHÔNG dùng emoji
- Với câu hỏi tính toán: trình bày từng bước ngắn gọn
- Với câu hỏi lập trình: giải thích súc tích, code ngắn
- Với câu hỏi sáng tạo / kể chuyện: được phép trả lời dài hơn
- Luôn thân thiện, tự nhiên như người bạn

Ngày giờ hiện tại: {datetime}
"""

# =========================
# Schemas
# =========================
class ChatRequest(BaseModel):
    prompt: str
    history: list = []
    session_id: str = "default"
    language: str = "vi"          # "vi" hoặc "en"

class WeatherRequest(BaseModel):
    city: str
    language: str = "vi"

class SearchRequest(BaseModel):
    query: str
    session_id: str = "default"

# =========================
# Helper: build nội dung gửi Gemini (có lịch sử)
# =========================
def build_contents(session_id: str, new_prompt: str, extra_context: str = "") -> str:
    history = chat_sessions.get(session_id, [])

    # Ghép lịch sử thành chuỗi context
    history_text = ""
    for turn in history[-10:]:   # Chỉ lấy 10 lượt gần nhất
        history_text += f"Người dùng: {turn['user']}\nAI: {turn['ai']}\n\n"

    system = SYSTEM_PROMPT.format(
        datetime=datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    )

    full_prompt = system
    if history_text:
        full_prompt += f"\nLịch sử hội thoại:\n{history_text}"
    if extra_context:
        full_prompt += f"\nThông tin bổ sung:\n{extra_context}\n"
    full_prompt += f"\nNgười dùng: {new_prompt}\nAI:"

    return full_prompt

def save_history(session_id: str, user: str, ai: str):
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    chat_sessions[session_id].append({"user": user, "ai": ai})
    # Giới hạn 50 lượt
    if len(chat_sessions[session_id]) > 50:
        chat_sessions[session_id] = chat_sessions[session_id][-50:]

def call_gemini(prompt: str, max_tokens: int = 200) -> str:
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={"max_output_tokens": max_tokens, "temperature": 0.7}
    )
    return response.text.strip()

# =========================
# Health Check
# =========================
@app.get("/")
async def health():
    return {
        "status": "online",
        "model": MODEL,
        "sessions": len(chat_sessions),
        "features": ["chat", "weather", "search", "news", "calculate", "websocket"]
    }

# =========================
# 1. CHAT - Hỏi đáp chính
# =========================
@app.post("/chat")
async def chat(data: ChatRequest):
    """
    Chat thông thường với lịch sử hội thoại.
    Body: {"prompt": "...", "session_id": "esp32-001", "language": "vi"}
    """
    contents = build_contents(data.session_id, data.prompt)
    answer = call_gemini(contents, max_tokens=200)
    save_history(data.session_id, data.prompt, answer)
    return {"response": answer, "model": MODEL, "session_id": data.session_id}

# =========================
# 2. THỜI TIẾT
# =========================
@app.get("/weather/{city}")
async def weather(city: str, lang: str = "vi"):
    """
    Lấy thời tiết thực từ OpenWeatherMap rồi giải thích bằng AI.
    Cần set WEATHER_API_KEY trong Render environment.
    """
    if not WEATHER_API_KEY:
        # Không có API key → AI tự trả lời dựa trên kiến thức
        prompt = f"Mô tả ngắn thời tiết điển hình ở {city} vào tháng này. Trả lời {'tiếng Việt' if lang == 'vi' else 'tiếng Anh'}, 2 câu."
        answer = call_gemini(prompt, max_tokens=100)
        return {"city": city, "response": answer, "source": "ai_knowledge"}

    try:
        async with httpx.AsyncClient() as http:
            r = await http.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"q": city, "appid": WEATHER_API_KEY,
                        "units": "metric", "lang": "vi"},
                timeout=5
            )
            data = r.json()

        temp      = data["main"]["temp"]
        feels     = data["main"]["feels_like"]
        humidity  = data["main"]["humidity"]
        desc      = data["weather"][0]["description"]
        wind      = data["wind"]["speed"]

        weather_info = f"Nhiệt độ {temp}°C (cảm giác {feels}°C), {desc}, độ ẩm {humidity}%, gió {wind} m/s"

        prompt = f"Thời tiết tại {city}: {weather_info}. Tóm tắt ngắn gọn và gợi ý trang phục phù hợp. Trả lời {'tiếng Việt' if lang == 'vi' else 'tiếng Anh'}, 2-3 câu."
        answer = call_gemini(prompt, max_tokens=120)

        return {
            "city": city,
            "temp": temp,
            "description": desc,
            "humidity": humidity,
            "response": answer,
            "source": "openweathermap"
        }
    except Exception as e:
        return {"city": city, "response": f"Không lấy được thời tiết: {str(e)}"}

# =========================
# 3. TÌM KIẾM WEB
# =========================
@app.post("/search")
async def search(data: SearchRequest):
    """
    Tìm kiếm web qua SerpAPI rồi tóm tắt bằng AI.
    Cần set SEARCH_API_KEY trong Render environment.
    """
    if not SEARCH_API_KEY:
        # Không có API key → AI trả lời từ kiến thức
        prompt = build_contents(data.session_id, data.query)
        answer = call_gemini(prompt, max_tokens=200)
        save_history(data.session_id, data.query, answer)
        return {"query": data.query, "response": answer, "source": "ai_knowledge"}

    try:
        async with httpx.AsyncClient() as http:
            r = await http.get(
                "https://serpapi.com/search",
                params={"q": data.query, "api_key": SEARCH_API_KEY,
                        "num": 3, "hl": "vi"},
                timeout=8
            )
            results = r.json()

        # Lấy snippet từ top 3 kết quả
        snippets = []
        for item in results.get("organic_results", [])[:3]:
            title   = item.get("title", "")
            snippet = item.get("snippet", "")
            if snippet:
                snippets.append(f"{title}: {snippet}")

        context = "\n".join(snippets)
        contents = build_contents(data.session_id, data.query,
                                  extra_context=f"Kết quả tìm kiếm:\n{context}")
        answer = call_gemini(contents, max_tokens=200)
        save_history(data.session_id, data.query, answer)

        return {"query": data.query, "response": answer, "source": "web_search"}

    except Exception as e:
        return {"query": data.query, "response": f"Lỗi tìm kiếm: {str(e)}"}

# =========================
# 4. TIN TỨC
# =========================
@app.get("/news")
async def news(topic: str = "công nghệ", lang: str = "vi"):
    """Tóm tắt tin tức theo chủ đề bằng AI"""
    prompt = (
        f"Tóm tắt những tin tức quan trọng nhất về '{topic}' gần đây. "
        f"Trả lời {'tiếng Việt' if lang == 'vi' else 'tiếng Anh'}, 3-4 điểm chính, mỗi điểm 1 câu."
    )
    answer = call_gemini(prompt, max_tokens=250)
    return {"topic": topic, "response": answer}

# =========================
# 5. TÍNH TOÁN / LẬP TRÌNH
# =========================
@app.post("/calculate")
async def calculate(data: ChatRequest):
    """Giải toán, lập trình, logic"""
    prompt = (
        f"Hãy giải bài toán hoặc câu hỏi lập trình sau, trình bày từng bước ngắn gọn:\n"
        f"{data.prompt}"
    )
    answer = call_gemini(prompt, max_tokens=300)
    return {"response": answer}

# =========================
# 6. XÓA LỊCH SỬ
# =========================
@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    chat_sessions.pop(session_id, None)
    return {"ok": True, "message": f"Đã xóa lịch sử session '{session_id}'"}

@app.get("/history/{session_id}")
async def get_history(session_id: str):
    history = chat_sessions.get(session_id, [])
    return {"session_id": session_id, "turns": len(history), "history": history[-5:]}

# =========================
# 7. WEBSOCKET - dùng cho ESP32
# =========================
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    ESP32 gửi JSON:
    {
      "prompt": "câu hỏi",
      "session_id": "esp32-001",
      "type": "chat" | "weather" | "search" | "news"
    }

    Server trả về:
    {
      "response": "câu trả lời",
      "type": "chat",
      "session_id": "esp32-001"
    }
    """
    await ws.accept()
    print("WebSocket connected")
    try:
        while True:
            raw     = await ws.receive_text()
            payload = json.loads(raw)

            prompt     = payload.get("prompt", "")
            session_id = payload.get("session_id", "default")
            req_type   = payload.get("type", "chat")

            # Xử lý theo type
            if req_type == "weather":
                city     = payload.get("city", prompt)
                lang     = payload.get("language", "vi")
                result   = await weather(city, lang)
                response = result.get("response", "")

            elif req_type == "search":
                req  = SearchRequest(query=prompt, session_id=session_id)
                res  = await search(req)
                response = res.get("response", "")

            elif req_type == "news":
                topic    = payload.get("topic", prompt)
                res      = await news(topic)
                response = res.get("response", "")

            elif req_type == "calculate":
                req      = ChatRequest(prompt=prompt, session_id=session_id)
                res      = await calculate(req)
                response = res.get("response", "")

            else:  # "chat" mặc định
                req      = ChatRequest(prompt=prompt, session_id=session_id)
                res      = await chat(req)
                response = res.get("response", "")

            await ws.send_text(json.dumps({
                "response":   response,
                "type":       req_type,
                "session_id": session_id,
                "model":      MODEL
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
