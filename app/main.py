from fastapi import FastAPI
from pydantic import BaseModel
from orchestrator.pipeline import MultiAgentOrchestrator

app = FastAPI()
orch = MultiAgentOrchestrator()

class Req(BaseModel):
    instruction: str

@app.post('/generate')
def generate(body: Req):
    r = orch.run(body.instruction)
    return {'verdict': r.verdict, 'code': r.code, 'backend': r.backend}
