# Qwen3-Coder Deployment Options

Qwen3-Coder supports several deployment shapes depending on whether you optimize for simplicity, tooling compatibility, or local throughput.

| Runtime | Best for | Notes |
|---|---|---|
| `transformers` | Single-machine experiments, notebooks, local prompt iteration | Lowest setup complexity for a first local run |
| vLLM | OpenAI-compatible serving, higher-throughput local inference, tool integrations | Good fit when IDEs or agents need a stable endpoint |
| SGLang | Structured serving with tool-calling support and long-context serving workflows | Useful when you already standardize on SGLang |
| Agent wrapper + served endpoint | Qwen Code, Claude Code, Cline, OpenClaw | Best when the model is one part of a broader coding-agent workflow |

## Which runtime to pick first

- Pick **`transformers`** if your first goal is “make one prompt work on one machine.”
- Pick **vLLM** if your first goal is “reuse one endpoint across multiple tools.”
- Pick **SGLang** if you already deploy other models with it and want operational consistency.

## Recommended rollout path

1. Validate one checkpoint locally with `transformers`.
2. Move the same checkpoint behind vLLM or SGLang if you need agent or IDE integrations.
3. Only then wire it into coding-agent tooling and larger workflows.
