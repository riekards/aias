import json
import os
from pathlib import Path
from typing import Any, List

class SelfReflectCommand:
    """
    Introspect Python source and return high-level improvement insights (no patches).
    """

    def __init__(self, project_root: str = None):
        self.root = Path(project_root or os.getcwd())

    def execute(self, args: Any = None) -> str:
        # Defer all agent imports to here
        from aias.agent import index_files, known_files, ask_llm

        # Re-index and collect .py files
        index_files(str(self.root))
        py_files: List[str] = [
            f for f in known_files
            if f.endswith(".py") and not f.startswith(".git/") and "/." not in f
        ]
        if not py_files:
            return "❌ No Python files found to inspect."

        # Load user context
        ctx = {}
        ctx_path = self.root / "memory" / "context.json"
        if ctx_path.exists():
            try:
                ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        # Build consolidated prompt
        prompt = (
            "You are AIAS doing self-reflection.  \n"
            f"User context: {json.dumps(ctx)}  \n\n"
            "Review ALL my Python source files listed below.  \n"
            "For each file, suggest up to TWO high-impact improvements "
            "(refactor, feature, perf/security fix).  \n\n"
            + "\n".join(f"- {f}" for f in py_files)
        )

        resp = ask_llm(prompt)
        if not resp.strip():
            return "❌ I didn’t receive any suggestions. Try again?"

        # Return insights without creating any .patch files
        return "🔍 Self-Reflection Insights:\n" + resp.strip()
