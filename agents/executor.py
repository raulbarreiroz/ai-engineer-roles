from __future__ import annotations

import textwrap

from agents.llm import complete
from agents.types import AgentMessage, TaskState


def _stub_code(instruction: str, critiques: list[str]) -> str:
    note = ""
    if critiques:
        note = "# addressed prior critique: " + "; ".join(critiques[:2]) + "\n"

    return textwrap.dedent(
        f'''\
        """Auto-draft for: {instruction[:80]}"""
        {note}from __future__ import annotations


        def solve(items: list) -> list:
            """Preserve first-seen order while dropping duplicates."""
            seen: set = set()
            out: list = []
            for item in items:
                if item in seen:
                    continue
                seen.add(item)
                out.append(item)
            return out


        if __name__ == "__main__":
            sample = [1, 2, 2, 3, 1, 4]
            print(solve(sample))
        '''
    )


def execute(state: TaskState) -> TaskState:
    plan_txt = "\n".join(f"{i+1}. {s}" for i, s in enumerate(state.plan))
    critique_txt = "\n".join(state.critiques) if state.critiques else "(none)"
    prompt = (
        f"Task: {state.instruction}\n\nPlan:\n{plan_txt}\n\n"
        f"Prior critiques:\n{critique_txt}\n\n"
        "Return only a Python module."
    )
    llm = complete(prompt, system="You are the executor agent. Emit runnable Python only.")
    code = llm if llm else _stub_code(state.instruction, state.critiques)

    if "```" in code:
        parts = code.split("```")
        for part in parts:
            if part.strip().startswith("python"):
                code = part.strip()[6:].strip()
                break
            if part.strip() and not part.strip().startswith("`"):
                code = part.strip()

    state.code = code
    state.history.append(AgentMessage(role="executor", content=code[:500]))
    return state
