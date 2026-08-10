from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentMessage:
    role: str
    content: str


@dataclass
class TaskState:
    instruction: str
    plan: list[str] = field(default_factory=list)
    code: str = ""
    critiques: list[str] = field(default_factory=list)
    revision: int = 0
    history: list[AgentMessage] = field(default_factory=list)
