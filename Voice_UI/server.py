from flask import Flask, jsonify, request, render_template
from command_parser import parse_command
from vehicle_service import get_state, apply_command

app = Flask(__name__)

@app.get("/")
def home():
    return render_template("index.html")

@app.get("/state")
def state():
    return jsonify(get_state())

@app.post("/command")
def command():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    
    cmd = parse_command(text)
    if cmd == "not_understood":
        return jsonify({
            "ok": False,
            "message": "Sorry, I couldn't understand you",
            "command": cmd,
            "state": get_state()
        })
    
    new_state = apply_command(cmd)
    return jsonify({
        "ok": True,
        "command": cmd,
        "state": new_state
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)