# Table of Supported Models

Below is a comprehensive table of models supported for running leaderboard evaluations. Each model entry indicates whether it supports native Function Calling (FC) or requires a special prompt format to generate function calls. Models marked with `💻` are intended to be hosted locally (using vllm or sglang), while models without the `💻` icon are accessed via API calls. To quickly see all available models, you can also run the `bfcl models` command.

## Function Calling (FC) vs. Prompt Mode

- **Function Calling (FC) Mode:**  
  Models with native tool/function calling capabilities. For example, OpenAI GPT in FC mode uses the `tools` section as documented in the [OpenAI function calling guide](https://platform.openai.com/docs/guides/function-calling).

- **Prompt Mode:**  
  Models without native function calling capabilities rely on traditional prompt-based interactions to produce function calls in the desired format, and we supply the function definitions in the `system prompt` section as opposed to a dedicated `tools` section. Prompt mode also serve as an alternative approach for models that support FC mode but do not fully leverage its function calling ability (i.e., we only use its normal text generation capability).

## Understanding Versioned Models

For model names containing `{...}`, multiple versions are available. For example, `meta-llama/Llama-3.1-{8B,70B}-Instruct` means we support both models: `meta-llama/Llama-3.1-8B-Instruct` and `meta-llama/Llama-3.1-70B-Instruct`.

| Base Model                             | Type             | Provider       | Model ID on BFCL                                            |
| -------------------------------------- | ---------------- | -------------- | ----------------------------------------------------------- |
| Amazon-Nova-Lite-v1:0                  | Function Calling | AWS            | nova-lite-v1.0                                              |
| Amazon-Nova-Micro-v1:0                 | Function Calling | AWS            | nova-micro-v1.0                                             |
| Amazon-Nova-Pro-v1:0                   | Function Calling | AWS            | nova-pro-v1.0                                               |
| Arch-Agent-{1.5B, 3B, 7B, 32B}         | Function Calling | Self-hosted 💻 | katanemo/Arch-Agent-{1.5B, 3B, 7B, 32B}                     |
| Bielik-11B-v2.3-Instruct               | Prompt           | Self-hosted 💻 | speakleash/Bielik-11B-v2.3-Instruct                         |
| BitAgent-8B                            | Prompt           | Self-hosted 💻 | BitAgent/BitAgent-8B                                        |
| BitAgent-Bounty-8B                     | Function Calling | Self-hosted 💻 | BitAgent/BitAgent-Bounty-8B                                 |
| Claude-3.5-haiku-20241022              | Function Calling | Anthropic      | claude-3-5-haiku-20241022-FC                                |
| Claude-3.5-haiku-20241022              | Prompt           | Anthropic      | claude-3-5-haiku-20241022                                   |
| Claude-Opus-4-20250514                 | Function Calling | Anthropic      | claude-opus-4-20250514                                      |
| Claude-Opus-4-20250514                 | Prompt           | Anthropic      | claude-opus-4-20250514                                      |
| Claude-Sonnet-4-20250514               | Function Calling | Anthropic      | claude-sonnet-4-20250514                                    |
| Claude-Sonnet-4-20250514               | Prompt           | Anthropic      | claude-sonnet-4-20250514                                    |
| CoALM-{8B, 70B, 405B}                  | Function Calling | Self-hosted 💻 | uiuc-convai/CoALM-{8B,70B,405B}                             |
| Command A                              | Function Calling | Cohere         | command-a-03-2025-FC                                        |
| Command R7B                            | Function Calling | Cohere         | command-r7b-12-2024-FC                                      |
| Command-R-Plus                         | Function Calling | Cohere         | command-r-plus-FC                                           |
| DBRX-Instruct                          | Prompt           | Databricks     | databricks-dbrx-instruct                                    |
| DeepSeek-R1                            | Prompt           | Self-hosted 💻 | deepseek-ai/DeepSeek-R1                                     |
| DeepSeek-R1-0528                       | Prompt           | DeepSeek       | DeepSeek-R1-0528                                            |
| DeepSeek-R1-0528                       | Function Calling | DeepSeek       | DeepSeek-R1-0528-FC                                         |
| DeepSeek-V3-0324                       | Function Calling | DeepSeek       | DeepSeek-V3-0324-FC                                         |
| DM-Cito-8B-v2                          | Prompt           | Mininglamp     | DM-Cito-8B-v2                                               |
| Falcon3-{1B,3B,7B,10B}-Instruct        | Function Calling | Self-hosted 💻 | tiiuae/Falcon3-{1B,3B,7B,10B}-Instruct                      |
| FireFunction-v2                        | Function Calling | Fireworks AI   | firefunction-v2-FC                                          |
| Functionary-{Small,Medium}-v3.1        | Function Calling | MeetKai        | meetkai/functionary-{small,medium}-v3.1-FC                  |
| Gemini-2.5-Flash                       | Function Calling | Google         | gemini-2.5-flash-FC                                         |
| Gemini-2.5-Flash                       | Prompt           | Google         | gemini-2.5-flash                                            |
| Gemini-2.5-Flash-Lite-Preview-06-17    | Function Calling | Google         | gemini-2.5-flash-lite-preview-06-17-FC                      |
| Gemini-2.5-Flash-Lite-Preview-06-17    | Prompt           | Google         | gemini-2.5-flash-lite-preview-06-17                         |
| Gemini-2.5-Pro                         | Function Calling | Google         | gemini-2.5-pro-FC                                           |
| Gemini-2.5-Pro                         | Prompt           | Google         | gemini-2.5-pro                                              |
| Gemma-3-{1b,4b,12b,27b}-it             | Prompt           | Self-hosted 💻 | google/gemma-3-{1b,4b,12b,27b}-it                           |
| GLM-4-9b-Chat                          | Function Calling | Self-hosted 💻 | THUDM/glm-4-9b-chat                                         |
| GoGoAgent                              | Prompt           | BitAgent       | BitAgent/GoGoAgent                                          |
| Gorilla-OpenFunctions-v2               | Function Calling | Gorilla LLM    | gorilla-openfunctions-v2                                    |
| GPT-4.1-2025-04-14                     | Function Calling | OpenAI         | gpt-4.1-2025-04-14-FC                                       |
| GPT-4.1-2025-04-14                     | Prompt           | OpenAI         | gpt-4.1-2025-04-14                                          |
| GPT-4.1-mini-2025-04-14                | Function Calling | OpenAI         | gpt-4.1-mini-2025-04-14-FC                                  |
| GPT-4.1-mini-2025-04-14                | Prompt           | OpenAI         | gpt-4.1-mini-2025-04-14                                     |
| GPT-4.1-nano-2025-04-14                | Function Calling | OpenAI         | gpt-4.1-nano-2025-04-14-FC                                  |
| GPT-4.1-nano-2025-04-14                | Prompt           | OpenAI         | gpt-4.1-nano-2025-04-14                                     |
| GPT-4.5-Preview-2025-02-27             | Function Calling | OpenAI         | gpt-4.5-preview-2025-02-27-FC                               |
| GPT-4.5-Preview-2025-02-27             | Prompt           | OpenAI         | gpt-4.5-preview-2025-02-27                                  |
| GPT-4o-2024-11-20                      | Function Calling | OpenAI         | gpt-4o-2024-11-20-FC                                        |
| GPT-4o-2024-11-20                      | Prompt           | OpenAI         | gpt-4o-2024-11-20                                           |
| GPT-4o-mini-2024-07-18                 | Function Calling | OpenAI         | gpt-4o-mini-2024-07-18-FC                                   |
| GPT-4o-mini-2024-07-18                 | Prompt           | OpenAI         | gpt-4o-mini-2024-07-18                                      |
| Granite-20b-FunctionCalling            | Function Calling | Self-hosted 💻 | ibm-granite/granite-20b-functioncalling                     |
| Granite-3.1-8B-Instruct                | Function Calling | Self-hosted 💻 | ibm-granite/granite-3.1-8b-instruct                         |
| Grok-3-beta                            | Function Calling | xAI            | grok-3-beta-FC                                              |
| Grok-3-beta                            | Prompt           | xAI            | grok-3-beta                                                 |
| Grok-3-mini-beta                       | Function Calling | xAI            | grok-3-mini-beta-FC                                         |
| Grok-3-mini-beta                       | Prompt           | xAI            | grok-3-mini-beta                                            |
| Haha-7B                                | Prompt           | Self-hosted 💻 | ZJared/Haha-7B                                              |
| Hammer2.1-{7b,3b,1.5b,0.5b}            | Function Calling | Self-hosted 💻 | MadeAgents/Hammer2.1-{7b,3b,1.5b,0.5b}                      |
| Ling-lite-v1.5                         | Prompt           | Ant Group      | Ling/ling-lite-v1.5                                         |
| Llama-3.1-{8B,70B}-Instruct            | Function Calling | Self-hosted 💻 | meta-llama/Llama-3.1-{8B,70B}-Instruct-FC                   |
| Llama-3.1-{8B,70B}-Instruct            | Prompt           | Self-hosted 💻 | meta-llama/Llama-3.1-{8B,70B}-Instruct                      |
| Llama-3.1-Nemotron-Ultra-253B-v1       | Prompt           | Nvidia         | nvidia/llama-3.1-nemotron-ultra-253b-v1                     |
| Llama-3.2-{1B,3B}-Instruct             | Function Calling | Self-hosted 💻 | meta-llama/Llama-3.2-{1B,3B}-Instruct-FC                    |
| Llama-3.3-70B-Instruct                 | Function Calling | Self-hosted 💻 | meta-llama/Llama-3.3-70B-Instruct-FC                        |
| Llama-4-Maverick-17B-128E-Instruct-FP8 | Prompt           | Novita AI      | meta-llama/llama-4-maverick-17b-128e-instruct-fp8-novita    |
| Llama-4-Maverick-17B-128E-Instruct-FP8 | Function Calling | Novita AI      | meta-llama/llama-4-maverick-17b-128e-instruct-fp8-FC-novita |
| Llama-4-Maverick-17B-128E-Instruct-FP8 | Function Calling | Self-hosted 💻 | meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8-FC        |
| Llama-4-Scout-17B-16E-Instruct         | Prompt           | Novita AI      | meta-llama/llama-4-scout-17b-16e-instruct-novita            |
| Llama-4-Scout-17B-16E-Instruct         | Function Calling | Novita AI      | meta-llama/llama-4-scout-17b-16e-instruct-FC-novita         |
| Llama-4-Scout-17B-16E-Instruct         | Function Calling | Self-hosted 💻 | meta-llama/Llama-4-Scout-17B-16E-Instruct-FC                |
| MiniCPM3-4B                            | Prompt           | Self-hosted 💻 | openbmb/MiniCPM3-4B                                         |
| MiniCPM3-4B-FC                         | Function Calling | Self-hosted 💻 | openbmb/MiniCPM3-4B-FC                                      |
| MiniMax-M2.7                           | Function Calling | MiniMax        | MiniMax-M2.7-FC                                             |
| MiniMax-M2.7                           | Prompt           | MiniMax        | MiniMax-M2.7                                                |
| MiniMax-M3                             | Function Calling | MiniMax        | MiniMax-M3-FC                                               |
| MiniMax-M3                             | Prompt           | MiniMax        | MiniMax-M3                                                  |
| Ministral-8B-Instruct-2410             | Function Calling | Self-hosted 💻 | mistralai/Ministral-8B-Instruct-2410                        |
| mistral-large-2411                     | Function Calling | Mistral AI     | mistral-large-2411-FC                                       |
| mistral-large-2411                     | Prompt           | Mistral AI     | mistral-large-2411                                          |
| Mistral-Medium-2505                    | Prompt           | Mistral AI     | mistral-medium-2505                                         |
| Mistral-Medium-2505                    | Function Calling | Mistral AI     | mistral-medium-2505-FC                                      |
| Mistral-small-2503                     | Function Calling | Mistral AI     | mistral-small-2503-FC                                       |
| Mistral-Small-2503                     | Prompt           | Mistral AI     | mistral-small-2503                                          |
| Nemotron-4-340b-instruct               | Prompt           | Nvidia         | nvidia/nemotron-4-340b-instruct                             |
| Nexusflow-Raven-v2                     | Function Calling | Nexusflow      | Nexusflow-Raven-v2                                          |
| O3-2025-04-16                          | Prompt           | OpenAI         | o3-2025-04-16                                               |
| O3-2025-04-16                          | Function Calling | OpenAI         | o3-2025-04-16-FC                                            |
| O4-mini-2025-04-16                     | Prompt           | OpenAI         | o4-mini-2025-04-16                                          |
| O4-mini-2025-04-16                     | Function Calling | OpenAI         | o4-mini-2025-04-16-FC                                       |
| Open-Mistral-Nemo-2407                 | Prompt           | Mistral AI     | open-mistral-nemo-2407                                      |
| Open-Mistral-Nemo-2407                 | Function Calling | Mistral AI     | open-mistral-nemo-2407-FC                                   |
| palmyra-x-004                          | Function Calling | Writer         | palmyra-x-004                                               |
| phi-4                                  | Prompt           | Self-hosted 💻 | microsoft/phi-4                                             |
| Phi-4-mini-instruct                    | Prompt           | Self-hosted 💻 | microsoft/Phi-4-mini-instruct                               |
| Phi-4-mini-instruct                    | Function Calling | Self-hosted 💻 | microsoft/Phi-4-mini-instruct-FC                            |
| Qwen3-{0.6B,1.7B,4B,8B,14B,32B}        | Prompt           | Alibaba Cloud  | qwen3-{0.6b,1.7b,4b,8b,14b,32b}                             |
| Qwen3-{0.6B,1.7B,4B,8B,14B,32B}        | Prompt           | Self-hosted 💻 | Qwen/Qwen3-{0.6B,1.7B,4B,8B,14B,32B}                        |
| Qwen3-{0.6B,1.7B,4B,8B,14B,32B}        | Function Calling | Alibaba Cloud  | qwen3-{0.6b,1.7b,4b,8b,14b,32b}-FC                          |
| Qwen3-{0.6B,1.7B,4B,8B,14B,32B}        | Function Calling | Self-hosted 💻 | Qwen/Qwen3-{0.6B,1.7B,4B,8B,14B,32B}-FC                     |
| Qwen3-{30B-A3B,235B-A22B}              | Prompt           | Alibaba Cloud  | qwen3-{30b-a3b, 235b-a22b}                                  |
| Qwen3-{30B-A3B,235B-A22B}              | Prompt           | Self-hosted 💻 | Qwen/Qwen3-{30B-A3B,235B-A22B}                              |
| Qwen3-{30B-A3B,235B-A22B}              | Function Calling | Alibaba Cloud  | qwen3-{30b-a3b, 235b-a22b}-FC                               |
| Qwen3-{30B-A3B,235B-A22B}              | Function Calling | Self-hosted 💻 | Qwen/Qwen3-{30B-A3B,235B-A22B}-FC                           |
| QwQ-32B                                | Function Calling | Alibaba Cloud  | qwq-32b-FC                                                  |
| QwQ-32B                                | Function Calling | Novita AI      | qwen/qwq-32b-FC-novita                                      |
| QwQ-32B                                | Prompt           | Alibaba Cloud  | qwq-32b                                                     |
| QwQ-32B                                | Prompt           | Novita AI      | qwen/qwq-32b-novita                                         |
| Sky-T1-32B-Preview                     | Prompt           | Self-hosted 💻 | NovaSky-AI/Sky-T1-32B-Preview                               |
| Snowflake/snowflake-arctic-instruct    | Prompt           | Snowflake      | snowflake/arctic                                            |
| ThinkAgent-1B                          | Function Calling | Self-hosted 💻 | ThinkAgents/ThinkAgent-1B                                   |
| ToolACE-2-8B                           | Function Calling | Self-hosted 💻 | Team-ACE/ToolACE-2-8B                                       |
| watt-tool-{8B,70B}                     | Function Calling | Self-hosted 💻 | watt-ai/watt-tool-{8B,70B}                                  |
| xiaoming-14B                           | Prompt           | Mininglamp     | xiaoming-14B                                                |
| xLAM-2-1b-fc-r                         | Function Calling | Self-hosted 💻 | Salesforce/xLAM-2-1b-fc-r                                   |
| xLAM-2-32b-fc-r                        | Function Calling | Self-hosted 💻 | Salesforce/xLAM-2-32b-fc-r                                  |
| xLAM-2-3b-fc-r                         | Function Calling | Self-hosted 💻 | Salesforce/xLAM-2-3b-fc-r                                   |
| xLAM-2-70b-fc-r                        | Function Calling | Self-hosted 💻 | Salesforce/Llama-xLAM-2-70b-fc-r                            |
| xLAM-2-8b-fc-r                         | Function Calling | Self-hosted 💻 | Salesforce/Llama-xLAM-2-8b-fc-r                             |
| yi-large                               | Function Calling | 01.AI          | yi-large-fc                                                 |

---

## Additional Requirements for Certain Models

- **MiniMax Models:**
  Set `MINIMAX_API_KEY`, then choose `openai` or `anthropic` with `MINIMAX_API_FORMAT` and `global` or `cn` with `MINIMAX_REGION`. The defaults are `openai` and `global`.

  | API format  | Region | Default base URL                     |
  | ----------- | ------ | ------------------------------------ |
  | OpenAI      | global | `https://api.minimax.io/v1`          |
  | OpenAI      | cn     | `https://api.minimaxi.com/v1`        |
  | Anthropic   | global | `https://api.minimax.io/anthropic`   |
  | Anthropic   | cn     | `https://api.minimaxi.com/anthropic` |

  `MINIMAX_BASE_URL` can override the selected default. Anthropic-compatible base URLs must end with `/anthropic`; the SDK appends `/v1/messages`. `MINIMAX_THINKING` accepts `adaptive` or `disabled` for MiniMax-M3, while MiniMax-M2.7 thinking is always on. `MINIMAX_SERVICE_TIER` accepts `standard` or `priority`, and `MINIMAX_MAX_TOKENS` controls the Anthropic-compatible response limit.

- **Gemini Models:**
  For `Gemini` models, we use the Google AI Studio API for inference. Ensure you have set the `GOOGLE_API_KEY` in your `.env` file.

- **Databricks Models:**
  For `databrick-dbrx-instruct`, you must create an Azure Databricks workspace and set up a dedicated inference endpoint. Provide the endpoint URL via `DATABRICKS_AZURE_ENDPOINT_URL` in `.env`.

- **Nova Models (AWS Bedrock):**
  For `Nova` models, set your `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in your `.env` file. Make sure the necessary AWS Bedrock permissions are granted in the `us-east-1` region.

---

For more details and a summary of feature support across different models, see the [Berkeley Function Calling Leaderboard blog post](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html#prompt).
