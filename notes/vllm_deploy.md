# Serving the codegen model with vLLM

This demo defaults to stubbed generation. To wire a real model:

```bash
# example - requires GPU + vLLM installed in its own env
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --port 8001
```

Then point the agents at it:

```bash
export USE_LLM=1
export OPENAI_API_BASE=http://127.0.0.1:8001/v1
export OPENAI_API_KEY=sk-local
export LLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
python -m demo.run_team "Write a Fibonacci generator"
```

## Why vLLM here

- PagedAttention keeps KV cache memory manageable under concurrent agent calls
- Continuous batching helps when planner/executor/critic hit the endpoint back-to-back
- OpenAI-compatible API means the agent code stays thin

## Quantization

For tighter GPUs prefer AWQ/GPTQ checkpoints or `--quantization awq`. Measure latency with a short prompt before enabling the critic loop (2-3 round trips).
