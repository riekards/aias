# aias/commands/SelfImproveCommand.py

import json
import re
from pathlib import Path
from typing import List

HISTORY_PATH = Path("memory/self_improve_history.json")

class SelfImproveCommand:
    def execute(self) -> str:
        """
        1) Load previously queued insights from history.
        2) Run self-reflection to get fresh insights.
        3) Filter out duplicates.
        4) Queue new insights as patch tasks.
        5) Save updated history.
        6) Return a summary of what was queued.
        """
        # 1) Load past history
        if HISTORY_PATH.exists():
            try:
                history = set(json.loads(HISTORY_PATH.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                history = set()
        else:
            history = set()

        # 2) Get fresh insights
        from aias.commands.SelfReflectCommand import SelfReflectCommand
        raw = SelfReflectCommand().execute()
        # strip header and split into bullet lines
        body = raw.split(":", 1)[-1]
        insights = [
            line[2:].strip()
            for line in body.splitlines()
            if line.startswith("- ")
        ]

        if not insights:
            return "⚠️ No actionable insights to queue."

        # 3) Filter out already-queued ones
        new_insights = [ins for ins in insights if ins not in history]
        if not new_insights:
            return "✅ All self-improvement suggestions have already been queued."

        # 4) Queue and record them
        from aias.agent import background_tasks, resolve_path

        queued: List[str] = []
        for insight in new_insights:
            # try to extract a filename in backticks, else default to agent.py
            m = re.search(r"`([^`]+\.py)`", insight)
            fn = m.group(1) if m else "agent.py"
            path = resolve_path(fn) or fn
            background_tasks.put((path, insight))
            queued.append(f"- [{path}] {insight}")
            history.add(insight)

        # 5) Save updated history
        HISTORY_PATH.parent.mkdir(exist_ok=True)
        HISTORY_PATH.write_text(json.dumps(list(history)), encoding="utf-8")

        # 6) Build and return report
        report = ["🛠️ Queued new self-improvement tasks:"]
        report.extend(queued)
        return "\n".join(report)
