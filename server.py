from flask import Flask, request, jsonify, send_file
import openai
import yt_dlp
from gtts import gTTS
import os
import time

app = Flask(__name__)

# ===== OPENAI KEY =====
openai.api_key = "YOUR_OPENAI_KEY"

# =========================
# POST /chat  → hỏi GPT
# =========================
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    text = data.get("text","")

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
          {"role":"system","content":"Tra loi tieng Viet, ngan gon"},
          {"role":"user","content": text}
        ]
    )
    return jsonify({
        "reply": res.choices[0].message.content
    })

# =========================
# POST /speech → text -> mp3
# =========================
@app.route("/speech", methods=["POST"])
def speech():
    text = request.json["text"]

    audio = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="female",
        input=text
    )

    return Response(audio, mimetype="audio/mpeg")

# =========================
# POST /music → tìm YouTube
# =========================
@app.route("/music", methods=["POST"])
def music():
    query = request.json["query"]

    result = YoutubeSearch(query, max_results=1).to_dict()
    video_id = result[0]["id"]

    stream_url = f"https://www.youtube.com/watch?v={video_id}"
    return jsonify({"url": stream_url})

# =========================

@app.route("/")
def home():
    return jsonify({
        "status": "OK",
        "server": "ESP32 AI FINAL"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
