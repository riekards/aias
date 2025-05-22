import os
from pathlib import Path
from typing import Any, List
from aias.agent import index_files, known_files, propose_patch

class SelfImproveCommand:
    """
    Batch full-file patches: for each .py file, ask the LLM
    to refactor/improve it, queue one patch per file.
    Creates a backup copy before queuing.
    """

    def __init__(self, project_root: str = None):
        self.root = Path(project_root or os.getcwd())

    def execute(self, args: Any = None) -> str:
        # Re-index to get source list
        index_files(str(self.root))
        py_files: List[str] = [
            f for f in known_files
            if f.endswith(".py") and not f.startswith(".git/") and "/." not in f
        ]
        if not py_files:
            return "❌ No Python files found to improve."

        # Build a multi-file prompt header
        prompt_header = (
            "You are AIAS. Please refactor and improve the following files "
            "to make me more sentient, robust, and efficient.  \n\n"
            + "\n".join(f"- {f}" for f in py_files)
        )

        queued = []
        for rel in py_files:
            src = Path(rel)
            # Backup
            backup_dir = self.root / "memory" / "backups"
            backup_dir.mkdir(exist_ok=True, parents=True)
            bak = backup_dir / f"{src.name}.bak"
            if not bak.exists():
                bak.write_bytes((self.root/src).read_bytes())

            # Task: improve this file
            task = prompt_header + f"\n\n--- Improve {rel} ---"
            propose_patch(rel, task)
            queued.append(rel)

        return "✅ Queued full-file patches for:\n" + "\n".join(f"- {f}" for f in queued)
