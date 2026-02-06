from flask import Flask, request, jsonify, Response
from openai import OpenAI
from youtubesearchpython import VideosSearch
import os
import sys

app = Flask(__name__)

# Lấy API Key từ Environment Variable trên Render
# Nếu không có, server sẽ in thông báo lỗi rõ ràng ra Logs
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("ERROR: OPENAI_API_KEY chưa được thiết lập trên Render!")
    # Không dừng server để bạn vẫn truy cập được trang chủ để debug
else:
    client = OpenAI(api_key=api_key)

# =========================
# ROOT - Kiểm tra server sống
# =========================
@app.route("/")
def home():
    return jsonify({
        "status": "OK",
        "server": "ESP32 AI FINAL",
        "message": "Server đang chạy trực tuyến!"
    })

# =========================
# CHAT - Kết nối GPT-4o-mini
# =========================
@app.route("/chat", methods=["POST"])
def chat():
    try:
        # Kiểm tra JSON đầu vào
        data = request.get_json(force=True, silent=True)
        if data is None:
            return jsonify({"ok": False, "reply": "Lỗi: Dữ liệu gửi lên không phải JSON chuẩn."}), 400
            
        text = data.get("text", "")
        if not text:
            return jsonify({"ok": False, "reply": "Nội dung trống."})

        # Gọi OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Bạn là trợ lý ảo tiếng Việt."},
                {"role": "user", "content": text}
            ],
            max_tokens=150
        )

        reply = response.choices[0].message.content.strip()
        return jsonify({"ok": True, "reply": reply})

    except Exception as e:
        # In lỗi chi tiết ra Render Logs để bạn xem được
        print(f"CRITICAL CHAT ERROR: {str(e)}")
        return jsonify({
            "ok": False, 
            "reply": f"Lỗi hệ thống: {str(e)}"
        }), 500

# =========================
# SPEECH - Text-to-Speech
# =========================
@app.route("/speech", methods=["POST"])
def speech():
    try:
        data = request.get_json(force=True)
        text = data.get("text", "")
        
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        return Response(response.content, mimetype="audio/mpeg")
    except Exception as e:
        print(f"TTS ERROR: {str(e)}")
        return jsonify({"ok": False, "error": str(e)}), 500

# =========================
# MUSIC - YouTube Search
# =========================
@app.route("/music", methods=["POST"])
def music():
    try:
        data = request.get_json(force=True)
        query = data.get("query", "")
        
        search = VideosSearch(query, limit=1)
        results = search.result()["result"]
        
        if results:
            return jsonify({
                "ok": True,
                "title": results[0]["title"],
                "url": results[0]["link"]
            })
        return jsonify({"ok": False, "reply": "Không tìm thấy nhạc."})
    except Exception as e:
        print(f"MUSIC ERROR: {str(e)}")
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
