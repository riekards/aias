# aias/agent.py

import os
import sys
import threading
from datetime import datetime
from pathlib import Path

from aias.core import (
    MODEL, OLLAMA_URL,
    ask_llm, ask_chat,
    index_files, resolve_path,
    classify_command, detect_traceback_issue,
    background_tasks, completed_tasks, enqueue_patch,
    log_interaction
)
from aias.utils.patcher import safe_update_file

# Ensure memory folders
os.makedirs("memory", exist_ok=True)
os.makedirs("memory/patch_notes", exist_ok=True)
LOG_FILE = Path("memory/logs.jsonl")
LOG_FILE.touch(exist_ok=True)

# Background worker to apply patches
def _background_worker():
    while True:
        task = background_tasks.get()
        if task is None:
            break
        filename, description = task
        _propose_and_save_patch(filename, description)
        completed_tasks.append((filename, description))
        background_tasks.task_done()

threading.Thread(target=_background_worker, daemon=True).start()

def _propose_and_save_patch(filename: str, task_description: str):
    """
    Generate a code patch for `filename` based on `task_description` and
    ask the user for approval before applying.
    """
    if not os.path.exists(filename):
        print(f"❌ Cannot propose patch: {filename} not found.")
        return

    original = Path(filename).read_text(encoding="utf-8", errors="ignore")
    prompt = (
        f"You are AIAS. Modify this Python file to accomplish the task below.\n\n"
        f"Task: {task_description}\n"
        f"Filename: {filename}\n\n"
        f"Original Code:\n{original}\n\n"
        f"Updated Code (only include Python code, no commentary):"
    )
    result = ask_llm(prompt)
    # extract code block if present
    import re
    m = re.search(r"```(?:python\n)?([\s\S]+?)```", result)
    code = m.group(1).rstrip() if m else result.strip()

    # save patch to memory
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    patch_file = Path(f"memory/patch_notes/{Path(filename).stem}_{stamp}.patch")
    patch_file.write_text(code, encoding="utf-8")
    print(f"📌 Patch saved to {patch_file}")

    # ask for approval
    apply_it = input(f"❓ Apply this patch to {filename}? (y/n): ").strip().lower()
    if apply_it == "y":
        safe_update_file(filename, code)
        print(f"✅ Applied patch to {filename}.")
    else:
        print("🛑 Patch not applied.")

def handle_input(user_text: str) -> str:
    """
    Process a single user message and return AIAS's reply.
    """
    # refresh index
    index_files(os.getcwd())
    pc = classify_command(user_text)

    # Reflect
    if pc["type"] == "reflect":
        from aias.commands.SelfReflectCommand import SelfReflectCommand
        raw = SelfReflectCommand().execute()
        # also queue those insights
        lines = [l[2:].strip() for l in raw.splitlines() if l.strip().startswith("- ")]
        queued = []
        for insight in lines:
            # extract filename or default to agent.py
            import re
            m = re.search(r"`([^`]+\.py)`", insight)
            fn = m.group(1) if m else "agent.py"
            path = resolve_path(fn) or fn
            enqueue_patch(path, insight)
            queued.append(f"[{path}] {insight}")
        if queued:
            return (
                f"{raw}\n\n🛠️ I’ve queued these self-improvement tasks:\n" +
                "\n".join(f"- {q}" for q in queued)
            )
        return raw

    # Improve
    if pc["type"] == "improve":
        from aias.commands.SelfImproveCommand import SelfImproveCommand
        return SelfImproveCommand().execute()

    # Feature request
    if pc["type"] == "feature":
        return (
            "🔧 Feature Request noted. Use the side panel or say "
            "`implement feature <description>` when ready."
        )

    # Locate files
    if pc["type"] == "locate":
        replies = []
        for fn in pc["filenames"]:
            path = resolve_path(fn)
            replies.append(f"I see '{fn}' at '{path or 'not found'}'.")
        return "\n".join(replies) if replies else "No matching files found."

    # Patch on-demand
    if pc["type"] == "patch" and pc.get("filenames"):
        replies = []
        for fn in pc["filenames"]:
            path = resolve_path(fn)
            if path:
                enqueue_patch(path, pc["task"])
                replies.append(f"Queued patch for {path}")
            else:
                replies.append(f"File not found: {fn}")
        return "\n".join(replies)

    # Model inspection
    if any(k in user_text.lower() for k in ("inspect model","model stats")):
        from aias.commands.InspectModelCommand import InspectModelCommand
        stats = InspectModelCommand().execute()
        return f"🔍 Model parameter stats:\n{stats}"

    # Traceback detection
    fn, desc = detect_traceback_issue(user_text)
    if fn:
        rel = resolve_path(Path(fn).name) or fn
        enqueue_patch(rel, desc)
        return f"🔍 Detected error in {rel}, queued a proposed fix."

    # Fallback: chat via LLM
    prompt = build_prompt = (
        f"You are AIAS, Ricky’s local AI assistant.\n"
        f"You can read/write files, suggest patches, and chat fluidly.\n"
        f"Known files:\n- " + "\n- ".join(known_files) +
        f"\n\n[User]: {user_text}\n[AIAS]:"
    )
    reply = ask_llm(prompt)
    log_interaction(user_text, reply)
    return reply

# If run as script, start interactive CLI
if __name__ == "__main__":
    print("🧠 AIAS is ready.")
    index_files(os.getcwd())
    try:
        while True:
            user = input("\n💬 You: ")
            if user.lower() in ("exit","quit"):
                break
            response = handle_input(user)
            print("🤖 AIAS:", response)
    except KeyboardInterrupt:
        print("\n👋 Exiting AIAS.")
