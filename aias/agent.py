# aias/agent.py

import os
import json
import threading
import queue
import re
import requests
from datetime import datetime
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM
)

from utils.config import load_config
from utils.patcher import safe_update_file
from aias.commands.search import search_google

# ── Configuration ────────────────────────────────────────────────────────────
CONFIG         = load_config()
LOG_FILE       = os.path.join("memory", "logs.jsonl")
CONVERSATIONAL = CONFIG["modes"].get("conversational", False)

# Intent & generation models
INTENT_MODEL = "distilbert-base-uncased"
GEN_MODEL    = "facebook/bart-base"

# Load DistilBERT intent classifier
intent_tok   = AutoTokenizer.from_pretrained(INTENT_MODEL)
intent_model = AutoModelForSequenceClassification.from_pretrained(INTENT_MODEL).eval()

# Load BART generator
gen_tok      = AutoTokenizer.from_pretrained(GEN_MODEL)
gen_model    = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL).eval()

# ── Ensure memory folders ─────────────────────────────────────────────────────
os.makedirs("memory", exist_ok=True)
os.makedirs("memory/patch_notes", exist_ok=True)
open(LOG_FILE, "a").close()

# ── Global state ─────────────────────────────────────────────────────────────
known_files      = []
background_tasks = queue.Queue()
completed_tasks  = []

# ── Background patch worker ──────────────────────────────────────────────────
def background_worker():
    while True:
        task = background_tasks.get()
        if task is None:
            break
        filename, description = task
        propose_patch(filename, description)
        completed_tasks.append((filename, description))
        print(f"✅ Background patch for {filename} completed.")
        background_tasks.task_done()

threading.Thread(target=background_worker, daemon=True).start()

# ── File indexing & resolution ───────────────────────────────────────────────
def index_files(start_path: str):
    global known_files
    known_files = []
    for dirpath, _, filenames in os.walk(start_path):
        rel = os.path.relpath(dirpath, start_path).replace("\\", "/")
        if rel.startswith("venv") or rel.startswith(".git") or "/." in rel:
            continue
        for f in filenames:
            path = f"{rel}/{f}" if rel != "." else f
            known_files.append(path)

def resolve_path(filename: str) -> str:
    fn = filename.lower()
    for p in known_files:
        if p.lower().endswith(fn):
            return p
    return None

# ── Prompt construction ─────────────────────────────────────────────────────
def get_folder_overview() -> str:
    roots = sorted({p.split("/")[0] for p in known_files})
    return "\n".join(f"- {d}/" for d in roots)

def get_file_overview() -> str:
    return "\n".join(f"- {p}" for p in sorted(known_files))

def build_prompt() -> str:
    ctx = ""
    cp = Path("memory/context.json")
    if cp.exists():
        try:
            ctx = json.dumps(json.loads(cp.read_text(encoding="utf-8")), indent=2)
        except:
            ctx = "(context load failed)"
    root_path = CONFIG.get("identity", {}).get("root_path", os.getcwd())
    return f"""
You are AIAS, Ricky’s AI companion living in {root_path}.
You can read, write, and modify files. Speak conversationally and ask clarifying questions if unsure.

Current context:
{ctx}

Known folders:
{get_folder_overview()}

Known files:
{get_file_overview()}

If Ricky asks you to rename, modify, or update a file, execute that action.
"""

# ── LLM call using BART ──────────────────────────────────────────────────────
def ask_llm(prompt: str) -> str:
    inputs = gen_tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(gen_model.device)
    out    = gen_model.generate(**inputs, max_new_tokens=256, do_sample=True, top_p=0.9)
    resp   = gen_tok.decode(out[0], skip_special_tokens=True).strip()
    if re.search(r"\b(i don[’']t know|not sure)\b", resp, re.IGNORECASE):
        snippets = search_google(prompt)[:5]
        enhanced = prompt + "\n\n# Web results:\n" + "\n".join(f"- {s}" for s in snippets)
        inputs2 = gen_tok(enhanced, return_tensors="pt", truncation=True, max_length=1024).to(gen_model.device)
        out2    = gen_model.generate(**inputs2, max_new_tokens=256, do_sample=True, top_p=0.9)
        resp    = gen_tok.decode(out2[0], skip_special_tokens=True).strip()
    return resp

# ── Patch generation ─────────────────────────────────────────────────────────
def propose_patch(filename: str, task_description: str):
    if not os.path.exists(filename):
        print(f"❌ Cannot propose patch: {filename} not found.")
        return
    original = Path(filename).read_text(encoding="utf-8", errors="ignore")
    prompt = f"""
You are AIAS. Modify this Python file to accomplish the task below.

Task: {task_description}
Filename: {filename}

Original Code:
{original}

Updated Code (only include Python code, no commentary):
"""
    result = ask_llm(prompt)
    m = re.search(r"```(?:python\\n)?([\\s\\S]+?)```", result)
    code = m.group(1).rstrip() if m else result
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    patch_path = f"memory/patch_notes/{Path(filename).name.replace('.', '_')}_{stamp}.patch"
    Path(patch_path).write_text(code, encoding="utf-8")
    background_tasks.put((filename, task_description))

# ── Intent classification via DistilBERT ────────────────────────────────────
def classify_command(text: str) -> dict:
    inputs = intent_tok(text, return_tensors="pt", truncation=True, max_length=128).to(intent_model.device)
    logits = intent_model(**inputs).logits
    idx    = int(logits.argmax(-1))
    label  = intent_model.config.id2label[idx].lower()

    ext_matches = re.findall(r"\b[\w/\\]+?\.\w{1,10}\b", text)
    files = []
    for tok in ext_matches:
        norm = tok.replace("\\","/")
        p = resolve_path(norm)
        if p: files.append(p)
    files = list(dict.fromkeys(files))

    if label in ("patch","update","fix","refactor") and files:
        return {"type":"patch","filenames":files,"task":text}
    if label == "reflect":
        return {"type":"reflect"}
    if label == "improve":
        return {"type":"improve"}
    if label == "locate" and files:
        return {"type":"locate","filenames":files}
    if label == "create":
        return {"type":"create","filenames":files}
    if "feature request" in text.lower():
        return {"type":"feature"}

    return {"type":"chat"}

# ── Traceback detection ─────────────────────────────────────────────────────
def detect_traceback_issue(text: str) -> tuple:
    fn = desc = None
    for line in text.splitlines():
        if "File" in line and ", line" in line:
            m = re.search(r'File \"(.+?)\", line (\d+)', line)
            if m:
                fn = m.group(1).replace("\\","/")
                desc = f"Error at line {m.group(2)} in {fn}"
        elif desc and ("Error" in line or "Exception" in line):
            desc = line.strip()
    return (fn, desc) if fn and desc else (None, None)

# ── Main input handler ──────────────────────────────────────────────────────
def handle_input(user: str) -> str:
    user = user.strip()
    index_files(os.getcwd())
    pc = classify_command(user)

    if pc["type"] == "reflect":
        from aias.commands.SelfReflectCommand import SelfReflectCommand
        return SelfReflectCommand().execute()

    if pc["type"] == "improve":
        from aias.commands.SelfImproveCommand import SelfImproveCommand
        return SelfImproveCommand().execute()

    if pc["type"] == "feature":
        return ("🔧 Feature Request noted. Use the side panel or say "
                "`implement feature <description>` when ready.")

    if pc["type"] == "locate":
        return "\n".join(
            f"AIAS: I see '{fn}' at '{(resolve_path(fn) or 'not found')}'."
            for fn in pc["filenames"]
        )

    if pc["type"] == "patch":
        fn = pc["filenames"][0]
        p  = resolve_path(fn)
        if p:
            background_tasks.put((p, pc["task"]))
            return f"AIAS: Queued patch for {p}"
        else:
            return f"AIAS: File not found: {fn}"

    if any(kw in user.lower() for kw in ("inspect model","model stats")):
        from aias.commands.InspectModelCommand import InspectModelCommand
        return f"AIAS: {InspectModelCommand().execute()}"

    fn, desc = detect_traceback_issue(user)
    if fn:
        rp = resolve_path(Path(fn).name) or fn
        propose_patch(rp, desc)
        return f"AIAS: Proposed patch for error in {rp}"

    prompt = build_prompt() + f"\n[User]: {user}\n[AIAS]:"
    reply  = ask_llm(prompt)
    log_interaction(user, reply)
    return f"AIAS: {reply}"

def log_interaction(user: str, reply: str) -> None:
    entry = {"timestamp": datetime.now().isoformat(), "user": user, "ai": reply}
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
