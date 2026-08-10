from agents.planner import PlannerAgent
from agents.executor import ExecutorAgent
from agents.critic import CriticAgent

class MultiAgentOrchestrator:
    def __init__(self):
        self.planner = PlannerAgent()
        self.executor = ExecutorAgent()
        self.critic = CriticAgent()
        self.backend_name = 'stub'

    def run(self, instruction: str):
        plan = self.planner.plan(instruction)
        code = self.executor.generate(instruction, plan)
        critique, ok = self.critic.review(code)
        return type('R', (), {'verdict': 'approved' if ok else 'rejected', 'code': code, 'backend': self.backend_name})()
