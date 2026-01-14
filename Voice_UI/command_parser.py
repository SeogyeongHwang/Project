import re

COMMANDS = {
    "play_radio": re.compile(r"\b(turn\s*on\s*radio|play\s*radio|start\s*radio)\b", re.I),
    "stop_radio": re.compile(r"\b(turn\s*off\s*radio|stop\s*radio)]b", re.I),
    "volume_up": re.compile(r"\b(increase\s*volume|volume\s*up|turn\s*up)\b", re.I),
    "volume_down": re.compile(r"\b(reduce\s*volume|decrease\s*volume|volume\s*down|turn\s*down)\b", re.I)
}

def parse_command(text: str) -> str:
    if not text:
        return "not understood"
    
    for name, pattern in COMMANDS.items():
        if pattern.search(text):
            return name
    
    return "not_understood"