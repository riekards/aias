# aias/core.py

import os
import json
import queue
import requests
import re
import yaml
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# ─── Configuration ─────────────────────────────────────────────────────────────

_CONFIG: Optional[Dict[str, Any]] = None

def load_config(path: str = "aias/config.yaml") -> Dict[str, Any]:
    global _CONFIG
    if _CONFIG is None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing config file: {path}")
        with open(path, encoding="utf-8") as f:
            _CONFIG = yaml.safe_load(f)
    return _CONFIG

# Load once
_conf = load_config()
MODEL      = _conf["model"]
OLLAMA_URL = _conf["ollama_url"]

# ─── LLM Wrappers ─────────────────────────────────────────────────────────────

def ask_llm(prompt: str) -> str:
    """
    Send a single-prompt generate request to Ollama.
    Returns the generated text, or empty string on error.
    """
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": MODEL, "prompt": prompt}
        )
        data = resp.json()
        return data.get("response", "").strip()
    except Exception:
        return ""

def ask_chat(messages: List[Dict[str,str]]) -> str:
    """
    Send a chat-completions request to Ollama.
    Expects messages=[{"role": "...", "content": "..."}].
    Returns assistant reply, or empty string on error.
    """
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat/completions",
            json={"model": MODEL, "messages": messages},
            timeout=10
        )
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""

# ─── File Indexing & Resolution ────────────────────────────────────────────────

known_files: List[str] = []

def index_files(start_path: str) -> None:
    """
    Populate `known_files` with all .py and .yaml/.json files under start_path,
    skipping venv and hidden folders.
    """
    global known_files
    known_files.clear()
    for dirpath, dirnames, filenames in os.walk(start_path):
        # skip virtual envs and hidden
        if "venv" in dirpath or dirpath.strip().startswith("."):
            continue
        for fn in filenames:
            rel = os.path.relpath(os.path.join(dirpath, fn), start_path).replace("\\","/")
            known_files.append(rel)

def resolve_path(filename: str) -> Optional[str]:
    """
    Return the first known file whose path ends with `filename` (case-insensitive),
    or None if not found.
    """
    target = filename.lower()
    for p in known_files:
        if p.lower().endswith(target):
            return p
    return None

# ─── Command Classification ────────────────────────────────────────────────────

def classify_command(text: str) -> Dict[str, Any]:
    """
    Classify user input into types: locate, patch, create, reflect, improve, feature, or chat.
    Returns a dict with at least {"type": ..., "filenames": [...], ...}.
    """
    t = text.lower()
    # find explicit filenames
    ext_matches = re.findall(r"\b[\w/\\]+?\.\w{1,10}\b", text)
    words = re.findall(r"\b[\w]+\b", text)
    files = set()
    for tok in ext_matches + words:
        cands = [tok]
        if "." not in tok:
            for ext in ("py","json","yaml","md","log","txt"):
                cands.append(f"{tok}.{ext}")
        for c in cands:
            c_norm = c.replace("\\","/")
            for kf in known_files:
                if kf.lower().endswith(c_norm.lower()):
                    files.add(kf)
    files = list(files)

    if any(k in t for k in ("rename","move")):
        return {"type":"rename","filenames":files}
    if any(k in t for k in ("patch","update","fix","refactor","modify")):
        return {"type":"patch","filenames":files,"task":text}
    if any(k in t for k in ("create file","make new file")):
        return {"type":"create","filenames":files}
    if re.search(r"\bself\s+reflect\b", t):
        return {"type":"reflect","filenames":[]}
    if re.search(r"\bself\s+improve\b", t):
        return {"type":"improve","filenames":[]}
    if any(k in t for k in ("feature","feature request")):
        return {"type":"feature","filenames":[]}
    if any(k in t for k in ("where is","locate","find")):
        return {"type":"locate","filenames":files}
    return {"type":"chat","filenames":files}

# ─── Traceback Detection ───────────────────────────────────────────────────────

def detect_traceback_issue(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Scan text for Python traceback lines and extract (filename, description).
    """
    fn = None
    desc = None
    for line in text.splitlines():
        if "File" in line and ", line" in line:
            m = re.search(r'File "(.+?)", line (\d+)', line)
            if m:
                fn = m.group(1).replace("\\","/")
                desc = f"Error at line {m.group(2)} in {fn}"
        elif ("Error" in line or "Exception" in line) and fn:
            desc = line.strip()
    return (fn, desc) if fn and desc else (None, None)

# ─── Patch Queue Helpers ──────────────────────────────────────────────────────

background_tasks = queue.Queue()
completed_tasks: List[Tuple[str,str]] = []

def enqueue_patch(path: str, desc: str) -> None:
    """
    Add a new patch request to the background queue.
    """
    background_tasks.put((path, desc))

def get_pending_patches() -> List[Tuple[str,str]]:
    """
    Return list of patches that have been completed.
    """
    return completed_tasks.copy()

# ─── Interaction Logging ──────────────────────────────────────────────────────

LOG_FILE = Path("memory/logs.jsonl")
LOG_FILE.parent.mkdir(exist_ok=True)

def log_interaction(user: str, ai: str) -> None:
    """
    Append a JSON line with {"timestamp","user","ai"} to logs.jsonl.
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "user": user,
        "ai": ai
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
