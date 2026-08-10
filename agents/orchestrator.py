"""Crew-style orchestrator: plan -> execute -> critique -> (optional) revise."""

from __future__ import annotations

from agents.critic import critique
from agents.executor import execute
from agents.planner import plan
from agents.types import TaskState


def run_team(instruction: str, max_revisions: int = 2) -> TaskState:
    state = TaskState(instruction=instruction)
    state = plan(state)

    for _ in range(max_revisions + 1):
        state = execute(state)
        state = critique(state)
        if not state.critiques:
            break
        state.revision += 1

    return state
