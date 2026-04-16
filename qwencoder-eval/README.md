# Qwen3-Coder Evaluation Index

This folder groups the benchmark resources that are linked from the Qwen3-Coder repository.

| Path | Focus | Typical use |
|---|---|---|
| `base/` | Base-model code generation evaluation | Compare non-instruct checkpoints on coding benchmarks |
| `instruct/` | Instruct-model evaluation and quantization result references | Check instruction-following coding quality or compare quantized variants |
| `tool_calling_eval/` | Tool-calling evaluation | Measure BFCL-v3 and Tau-Bench style function/tool use |
| `reasoning/` | Reasoning-oriented evaluation assets | Extend or adapt reasoning-heavy coding tests |

## Recommended reading order

1. Start with `instruct/README.md` if you are evaluating chat or coding-agent checkpoints.
2. Use `tool_calling_eval/README.md` when your deployment depends on structured tool use.
3. Use `base/readme.md` for base-model generation comparisons.

## Practical tip

When sharing benchmark results internally, record:

- the exact checkpoint name;
- the runtime you used;
- the evaluation script or subfolder;
- any quantization or tensor-parallel setting that affects the output.
