# Prompt Pattern Examples for Qwen3-Coder

These examples are intended as lightweight starting points for common coding workflows.

## Bug fixing

```text
You are reviewing a Python service.
Find the most likely bug in the following function, explain the root cause briefly, and return a minimal patch.
```

## File-scoped feature work

```text
Add a retry wrapper to this HTTP client.
Keep the public API unchanged, avoid new dependencies, and show the final code with only the modified functions.
```

## Repository understanding

```text
Summarize this repository for a new contributor.
List the entrypoints, core modules, and the safest place to start debugging.
```

## Test generation

```text
Write focused unit tests for this function.
Cover the happy path, one edge case, and one failure case. Use the existing test style in the snippet.
```

## Fill-in-the-middle completion

```text
Complete the missing code between the prefix and suffix.
Preserve the existing naming style and do not rewrite the surrounding lines.
```

## Prompting tip

For coding tasks, include at least two of the following when possible:

- the target file or scope;
- what must stay unchanged;
- the expected output format;
- constraints on dependencies or style.
