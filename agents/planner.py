from __future__ import annotations

from agents.llm import complete
from agents.types import AgentMessage, TaskState


def plan(state: TaskState) -> TaskState:
    llm = complete(
        f"Break this coding task into 3-5 short steps:\n{state.instruction}",
        system="You are the planner agent. Reply with a numbered list only.",
    )
    if llm:
        steps = [ln.strip(" -") for ln in llm.splitlines() if ln.strip()]
    else:
        steps = [
            "Clarify inputs/outputs and edge cases",
            "Draft a pure function with type hints",
            "Add a tiny __main__ smoke test",
            "Keep stdlib-only unless asked otherwise",
        ]

    state.plan = steps
    state.history.append(AgentMessage(role="planner", content="\n".join(f"- {s}" for s in steps)))
    return state


class PlannerAgent:
    """Thin wrapper kept for older call sites."""

    def plan(self, instruction: str):
        state = plan(TaskState(instruction=instruction))
        return state.plan
