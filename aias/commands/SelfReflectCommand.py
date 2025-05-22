# aias/commands/SelfReflectCommand.py

import ast
from pathlib import Path
from radon.complexity import cc_visit
from typing import List, Set, Tuple


class SelfReflectCommand:
    """
    Analyze the AIAS codebase and produce three concise self-reflection insights:
    1) Cyclomatic complexity hotspots (CC ≥ 8)
    2) Files with TODO comments
    3) Modules missing type hints
    """

    def __init__(self):
        # Root of the project (two levels up from this file)
        self.project_root = Path(__file__).parents[2]

    def execute(self) -> str:
        """
        Run the full analysis and return a bullet-list of three insights.
        """
        py_files = list(self.project_root.rglob("*.py"))
        hotspots, todo_counts, missing_hints = self._analyze_code(py_files)
        insights = self._build_insights(hotspots, todo_counts, missing_hints)

        header = "🔍 Self-Reflection Insights:"
        return header + "\n" + "\n".join(insights)

    def _analyze_code(
        self, py_files: List[Path]
    ) -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, int]], Set[str]]:
        """
        Scan each Python file for:
          - Cyclomatic complexity (collect those ≥ 8)
          - TODO comment counts
          - Missing type hints in function definitions
        Returns three structures: hotspots, todo_counts, missing_hint_files.
        """
        hotspots: List[Tuple[str, str, int]] = []
        todo_counts: List[Tuple[str, int]] = []
        missing_hint_files: Set[str] = set()

        for f in py_files:
            # Skip virtualenv, site-packages, hidden dirs
            if (
                "venv" in f.parts
                or "site-packages" in f.parts
                or f.name.startswith(".")
            ):
                continue

            try:
                src = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            rel_path = str(f.relative_to(self.project_root))

            # 1) Cyclomatic complexity
            try:
                for comp in cc_visit(src):
                    if comp.complexity >= 8:
                        hotspots.append((rel_path, comp.name, comp.complexity))
            except Exception:
                pass

            # 2) TODO comment count
            count = src.count("TODO")
            if count:
                todo_counts.append((rel_path, count))

            # 3) Missing type hints
            try:
                tree = ast.parse(src)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # if any arg or return annotation is missing
                        args_missing = any(arg.annotation is None for arg in node.args.args)
                        returns_missing = node.returns is None
                        if args_missing or returns_missing:
                            missing_hint_files.add(rel_path)
                            break
            except Exception:
                pass

        # Sort so the “worst” offenders are first
        hotspots.sort(key=lambda x: x[2], reverse=True)
        todo_counts.sort(key=lambda x: x[1], reverse=True)

        return hotspots, todo_counts, missing_hint_files

    def _build_insights(
        self,
        hotspots: List[Tuple[str, str, int]],
        todo_counts: List[Tuple[str, int]],
        missing: Set[str],
    ) -> List[str]:
        """
        Construct exactly three bullet-point insights from the analysis data.
        """
        insights: List[str] = []

        # Insight #1: Top complexity hotspot
        if hotspots:
            fn, name, cc = hotspots[0]
            insights.append(
                f"- Function `{name}` in `{fn}` has cyclomatic complexity of {cc}, consider refactoring."
            )

        # Insight #2: File with the most TODOs
        if todo_counts:
            fn, cnt = todo_counts[0]
            insights.append(f"- File `{fn}` contains {cnt} TODO comments; address these to improve code quality.")

        # Insight #3: Modules lacking type hints
        if missing:
            insights.append(f"- {len(missing)} module(s) lack type hints; adding annotations will prevent bugs and aid IDEs.")

        # Fill up to 3 if fewer found
        while len(insights) < 3:
            insights.append("- No further high-impact code issues detected at this time.")

        return insights
