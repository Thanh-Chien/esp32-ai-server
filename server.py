from flask import Flask, request, jsonify, Response
from openai import OpenAI
from youtubesearchpython import VideosSearch
import requests, os

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/")
def home():
    return jsonify({"status": "OK", "server": "ESP32 AI FINAL"})

# ===== CHAT =====
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    text = data.get("text","")

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"Tra loi tieng Viet ngan gon"},
            {"role":"user","content": text}
        ]
    )

    reply = res.choices[0].message.content
    is_music = ("nhac" in text.lower()) or ("phat" in text.lower())

    return jsonify({"reply": reply, "music": is_music})

# ===== TTS =====
@app.route("/speech", methods=["POST"])
def speech():
    text = request.json["text"]

    audio = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text
    )

    return Response(audio.read(), mimetype="audio/mpeg")

# ===== MUSIC =====
@app.route("/music", methods=["POST"])
def music():
    query = request.json["query"]

    search = VideosSearch(query, limit=1)
    link = search.result()["result"][0]["link"]

    # proxy audio stream
    stream = f"https://piped.video/api/v1/streams/{link.split('v=')[1]}"
    r = requests.get(stream, stream=True)

    return Response(r.iter_content(1024), content_type="audio/mpeg")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
