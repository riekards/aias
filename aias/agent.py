import os
import json
import requests
import threading
import queue
import re
import time
from datetime import datetime
from utils.config import load_config
from utils.patcher import safe_update_file
from commands import search
from utils.nlp_engine import classify_intent, generate_response

CONFIG = load_config()
MODEL = CONFIG["model"]
OLLAMA_URL = CONFIG["ollama_url"]
LOG_FILE = os.path.join("memory", "logs.jsonl")
CONVERSATIONAL = CONFIG["modes"].get("conversational", False)
EDITOR = CONFIG.get("preferences", {}).get("editor", "unknown")
OS_NAME = CONFIG.get("identity", {}).get("os", "linux")

os.makedirs("memory", exist_ok=True)
os.makedirs("memory/patch_notes", exist_ok=True)
open(LOG_FILE, "a").close()

known_files = []
background_tasks = queue.Queue()
completed_tasks = []

CAPABILITIES = {
    "locate": True,
    "patch": True,
    "self_improve": True,
    "browse_internet": False,
}

def index_files(start_path: str) -> None:
    global known_files
    known_files.clear()
    for dirpath, _, filenames in os.walk(start_path):
        rel = os.path.relpath(dirpath, start_path).replace("\\", "/")
        if rel.startswith("venv") or "/." in rel or "__pycache__" in rel:
            continue
        for f in filenames:
            if f.endswith(".pyc"):
                continue
            path = f"{rel}/{f}" if rel != "." else f
            known_files.append(path)

def resolve_path(filename: str) -> str | None:
    fname = filename.lower()
    for p in known_files:
        if p.lower().endswith(fname):
            return p
    return None

def ask_llm(prompt: str) -> str:
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": MODEL, "prompt": prompt, "stream": False}
    ).json().get("response", "")

    if re.search(r"\b(i don[’']t know|not sure how|no idea)\b", resp, re.IGNORECASE):
        try:
            snippets = search.search_google(prompt) or []
            enhanced = prompt + "\n\n# Web results:\n" + "\n".join(f"- {s}" for s in snippets[:5])
            resp = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": MODEL, "prompt": enhanced, "stream": False}
            ).json().get("response", "")
        except Exception as e:
            print(f"⚠️ Web fallback failed: {e}")
            return resp

    return resp.strip()

def propose_patch(filename: str, description: str) -> None:
    if not os.path.exists(filename):
        print(f"❌ Cannot propose patch: {filename} not found.")
        return
    with open(filename, encoding="utf-8") as f:
        original = f.read()

    prompt = (
        f"You are AIAS. Modify this Python file to accomplish: {description}\n"
        f"Filename: {filename}\nOriginal Code:\n{original}\nUpdated Code (only Python code):"
    )
    updated = ask_llm(prompt).strip()

    if not updated or updated.lower().startswith("no results"):
        print(f"⚠️ LLM returned no usable patch for: {filename}")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    patch_file = os.path.join("memory/patch_notes", f"{os.path.basename(filename)}_{stamp}.patch")
    with open(patch_file, "w", encoding="utf-8") as f:
        f.write(updated)
    print(f"📌 Patch saved to {patch_file}")

def background_worker():
    while True:
        task = background_tasks.get()
        if task is None:
            break
        filename, description = task
        propose_patch(filename, description)
        completed_tasks.append((filename, description))
        background_tasks.task_done()

threading.Thread(target=background_worker, daemon=True).start()

def log_interaction(user: str, reply: str) -> None:
    entry = {"timestamp": datetime.now().isoformat(), "user": user, "ai": reply}
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    try:
        ctx_file = "memory/context.json"
        if os.path.exists(ctx_file):
            context = json.load(open(ctx_file, encoding="utf-8"))
        else:
            context = {}
        context["last_interaction"] = entry
        context.setdefault("session_history", []).append(entry)
        with open(ctx_file, "w", encoding="utf-8") as f:
            json.dump(context, f, indent=2)
    except Exception as e:
        print(f"⚠️ Context update failed: {e}")

def learn_from_interaction(interaction: dict):
    try:
        feedback_path = "memory/feedback.jsonl"
        archive_path = "memory/feedback_archived.jsonl"
        if not os.path.exists(feedback_path):
            return
        updated_feedback = []
        recent_feedback = []
        with open(feedback_path, encoding="utf-8") as f:
            for line in f:
                fb = json.loads(line)
                if fb.get("status") != "new":
                    updated_feedback.append(fb)
                    continue
                fb["status"] = "seen"
                recent_feedback.append(fb)
                with open(archive_path, "a", encoding="utf-8") as arch:
                    arch.write(json.dumps(fb, ensure_ascii=False) + "\n")
        with open(feedback_path, "w", encoding="utf-8") as f:
            for fb in updated_feedback:
                f.write(json.dumps(fb, ensure_ascii=False) + "\n")
        ctx_file = "memory/context.json"
        context = json.load(open(ctx_file, encoding="utf-8")) if os.path.exists(ctx_file) else {}
        context["recent_feedback"] = recent_feedback
        with open(ctx_file, "w", encoding="utf-8") as f:
            json.dump(context, f, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to learn from interaction: {e}")

def generate_stub(intent: str) -> str:
    name = intent.replace(" ", "_")
    return f"def {name}(*args, **kwargs):\n    \"\"\"TODO: implement {intent} feature.\"\"\"\n    pass\n"

def classify_command(text: str) -> dict:
    t = text.lower()
    if any(k in t for k in ("where is", "locate", "find")):
        parts = re.findall(r"\b[\w./]+\b", text)
        return {"type": "locate", "filenames": parts}
    if any(k in t for k in ("patch", "update", "fix", "refactor", "modify")):
        parts = re.findall(r"\b[\w./]+\b", text)
        return {"type": "patch", "filenames": parts, "task": text}
    if "improve" in t:
        return {"type": "self_improve", "filenames": [], "task": text}
    return {"type": "chat"}

def build_prompt() -> str:
    ctx = ""
    ctx_file = "memory/context.json"
    if os.path.exists(ctx_file):
        try:
            ctx = json.dumps(json.load(open(ctx_file, encoding="utf-8")), indent=2)
        except:
            ctx = "(could not load context)"
    overview = (
        f"You are AIAS, Ricky’s local AI companion running on a {OS_NAME} system inside {EDITOR}.\n"
        f"You load your runtime settings from config.yaml using `load_config()` — do not rename or restructure config keys.\n"
        f"You speak naturally, suggest improvements, and never assume shell commands are UNIX by default.\n\n"
        f"Current context:\n{ctx}\n\n"
        f"Known files:\n" + "\n".join(known_files)
    )
    return overview

def get_pending_patches() -> list[tuple[str, str]]:
    return list(completed_tasks)

def handle_input(user: str) -> str:
    index_files(os.getcwd())
    cmd = classify_command(user)
    intent_label = classify_intent(user)
    natural_response = generate_response(f"User input: {user}\nWhat should AIAS do?")

    if cmd["type"] != "chat" and (cmd["type"] not in CAPABILITIES or not CAPABILITIES[cmd["type"]]):
        stub = generate_stub(cmd["type"])
        return (
            f"I’m sorry, I can’t do '{cmd['type']}' yet—but I can generate "
            f"a stub function for it:\n\n```python\n{stub}```\nShall I queue this patch?"
        )

    if cmd["type"] == "locate":
        return "\n".join(
            f"Found '{fn}' at '{resolve_path(fn) or 'not found'}'"
            for fn in cmd["filenames"]
        )

    if cmd["type"] == "patch":
        for fn in cmd["filenames"]:
            path = resolve_path(fn)
            if path:
                background_tasks.put((path, cmd["task"]))
                return f"Queued patch for {path}"
        return "No files matched for patch."
    
    if cmd["type"] == "self_improve":
        from aias.commands.rltrainingcommand import RLTrainingCommand
        trainer = RLTrainingCommand()
        trainer.execute(None)
        trainer.clean_up(None)
        return "✅ Completed an RL training cycle and saved the model."

    system = (
        "You are AIAS, a friendly, proactive AI assistant.\n"
        "You suggest code when needed and ask for clarification when uncertain.\n\n"
    )
    turn = f"User: {user}\nAIAS:"
    reply = ask_llm(system + build_prompt() + "\n" + turn)

    mentioned = re.findall(r"\b([\w_/]+\.py|requirements\.txt|config\.ya?ml)\b", reply.lower())
    seen = set()
    for file in mentioned:
        if "__pycache__" in file or file.endswith(".pyc"):
            continue
        file_path = resolve_path(file)
        if not file_path or not os.path.exists(file_path):
            continue
        if file_path.endswith("config.yaml"):
            print("⚠️ Skipping patch on config.yaml — runtime dependency.")
            continue
        task = f"Improve based on LLM suggestion: {file}"
        if (file_path, task) not in completed_tasks and (file_path, task) not in seen:
            background_tasks.put((file_path, task))
            seen.add((file_path, task))

    log_interaction(user, reply)
    learn_from_interaction({"user": user, "reply": reply})
    return reply

if __name__ == "__main__":
    print(f"🧠 AIAS ready (conversational: {'ON' if CONVERSATIONAL else 'OFF'}) — Running on {OS_NAME}, Editor: {EDITOR}")
    while True:
        user = input("💬 You: ")
        if user.lower() in ("exit", "quit"):
            break
        print("🤖 AI:", handle_input(user))
    background_tasks.put(None)
