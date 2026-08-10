from __future__ import annotations

import ast

from agents.types import AgentMessage, TaskState

BANNED = ("eval(", "exec(", "os.system(", "__import__('subprocess')")


def critique(state: TaskState) -> TaskState:
    issues: list[str] = []
    code = state.code or ""

    try:
        ast.parse(code)
    except SyntaxError as exc:
        issues.append(f"syntax error: {exc.msg} (line {exc.lineno})")

    for token in BANNED:
        if token in code:
            issues.append(f"banned pattern: {token}")

    if "def " not in code:
        issues.append("expected at least one function definition")

    if "if __name__" not in code:
        issues.append("missing __main__ smoke test")

    state.critiques = issues
    verdict = "REJECT" if issues else "APPROVE"
    state.history.append(
        AgentMessage(role="critic", content=f"{verdict}\n" + ("\n".join(issues) if issues else "looks fine"))
    )
    return state


class CriticAgent:
    def review(self, code: str):
        state = TaskState(instruction="", code=code)
        critique(state)
        ok = not state.critiques
        return ("; ".join(state.critiques) if state.critiques else "ok"), ok
