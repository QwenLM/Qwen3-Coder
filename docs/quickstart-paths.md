# Qwen3-Coder Quickstart Paths

Use this guide to choose the smallest setup that matches what you want to do with Qwen3-Coder.

| Goal | Recommended path | Best when | Starting point |
|---|---|---|---|
| Run the model locally in Python | `transformers` | You want a direct local inference script or notebook | `README.md` quick start example |
| Serve the model behind an OpenAI-compatible endpoint | vLLM or SGLang | You need a reusable local / team endpoint for tools or IDEs | `README.md` and the model card links |
| Use the model through a coding agent | Qwen Code, Claude Code, OpenClaw, Cline | You want repo editing, tool use, or terminal-driven workflows | Links in the root README header |
| Evaluate a checkpoint before wider rollout | `qwencoder-eval/` | You need benchmark or tool-calling numbers | `qwencoder-eval/README.md` |
| Study older usage patterns from the previous generation | `examples/` | You want extra `transformers` examples and long-context notes | `examples/Qwen2.5-Coder*.md` |

## Suggested first steps

1. Start with the `transformers` example if you only need local prompting.
2. Move to vLLM or SGLang once you need a stable API endpoint for tools.
3. Use one of the coding-agent integrations only after the base generation path is already working.

## Coding-agent entry hints

- Choose **Qwen Code** when you want the most direct first-party agent workflow.
- Choose **Claude Code** or **Cline** when you already use those environments and only need a compatible model endpoint.
- Choose **OpenClaw** when you want to reproduce the agentic demos linked from the root README.
