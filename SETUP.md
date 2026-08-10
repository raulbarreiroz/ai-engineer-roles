# Setup - Multi-agent Python codegen (planner / executor / critic)

## What runs locally

A CrewAI/AutoGen-style loop implemented as plain Python agents:

1. **Planner** - breaks a natural-language task into steps
2. **Executor** - drafts Python (template + stubs; can call an LLM if configured)
3. **Critic** - static checks (syntax, banned patterns) and may request a revision

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m demo.run_team "Write a function that deduplicates a list while preserving order"
```

No GPU required. Set `USE_LLM=1` and `OPENAI_API_BASE` (vLLM OpenAI-compatible server) to swap stubs for real completions.

## vLLM deployment notes

See `notes/vllm_deploy.md` for a minimal serve command, PagedAttention knobs, and how this demo talks to an OpenAI-compatible endpoint.

## Layout

```
agents/     planner, executor, critic, orchestrator
demo/       CLI entrypoint
notes/      vLLM / guardrails pointers
```
