import json
import os
from threading import Lock

STATE_FILE = "vehicle_state.json"
_lock = Lock()

DEFAULT_STATE = {
    "play": False,
    "audio": 5
}

def _ensure_file():
    if not os.path.exists(STATE_FILE):
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_STATE, f, indent=2)
            
def get_state():
    _ensure_file()
    with _lock:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
            
def set_state(state: dict):
    with _lock:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            
def apply_command(cmd: str):
    state = get_state()     
    
    if cmd == "play_radio":
        state["play"] = True
        
    elif cmd == "stop_radio":
        state["play"] = False

    elif cmd == "volume_up":
        state["audio"] = min(10, int(state.get("audio", 5)) + 1)
        
    elif cmd == "volume_down":
        state["audio"] = max(0, int(state.get("audio", 5)) - 1)
        
    set_state(state)
    return state