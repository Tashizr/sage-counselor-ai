import os
import sys
from flask import Flask, render_template, request, jsonify, session
sys.path.insert(0, os.path.dirname(__file__))
from counselor import Counselor

# SAGE AI Counselor - v2.0 with proper introduction handling
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24).hex())
app.config["SESSION_TYPE"] = "filesystem"

sessions = {}

def get_bot(session_id):
    if session_id not in sessions:
        sessions[session_id] = Counselor()
    return sessions[session_id]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    msg = data.get("message", "").strip()
    session_id = session.get("id")
    if not session_id:
        import uuid
        session_id = str(uuid.uuid4())
        session["id"] = session_id

    bot = get_bot(session_id)

    if not msg and bot.user_name is None:
        return jsonify({"reply": bot.greet(), "name_set": False})

    if not msg:
        return jsonify({"reply": "Take your time. I'm here whenever you're ready.", "name_set": bot.user_name is not None})

    reply = bot.respond(msg)
    return jsonify({"reply": reply, "name_set": bot.user_name is not None})

@app.route("/reset", methods=["POST"])
def reset():
    session_id = session.get("id")
    if session_id and session_id in sessions:
        del sessions[session_id]
    return jsonify({"status": "ok"})

@app.route("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
