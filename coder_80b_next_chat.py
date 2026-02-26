import time
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# --- CONFIGURATION ---
model_name = "Qwen/Qwen3-Coder-Next-FP8"
offload_path = r"c:\Users\koshe\.cache\huggingface\hub\models--Qwen--Qwen3-Coder-Next-FP8\snapshots\da6e2ed27304dd39abadd9c82ef50e8de67bdd4c" #TODO: set your path

# --- 1. DEVICE MAP SETUP ---
device_map = {}
device_map['lm_head.weight'] = "cuda"
device_map['model.embed_tokens.weight'] = "cuda"
device_map['model.norm.weight'] = "cuda"

for i in range(48):
    layer_prefix = f"model.layers.{i}"

    # Attention & Norms -> GPU
    device_map[f"{layer_prefix}.linear_attn.in_proj_qkvz.weight"] = "cuda"
    device_map[f"{layer_prefix}.self_attn.q_proj.weight"] = "cuda"
    device_map[f"{layer_prefix}.linear_attn.out_proj.weight"] = "cuda"
    device_map[f"{layer_prefix}.linear_attn.dt_bias"] = "cuda"
    device_map[f"{layer_prefix}.linear_attn.A_log"] = "cuda"
    device_map[f"{layer_prefix}.linear_attn.conv1d.weight"] = "cuda"
    device_map[f"{layer_prefix}.linear_attn.in_proj_ba.weight"] = "cuda"
    device_map[f"{layer_prefix}.input_layernorm.weight"] = "cuda"
    device_map[f"{layer_prefix}.post_attention_layernorm.weight"] = "cuda"
    device_map[f"{layer_prefix}.self_attn.k_proj.weight"] = "cuda"
    device_map[f"{layer_prefix}.self_attn.v_proj.weight"] = "cuda"
    device_map[f"{layer_prefix}.self_attn.o_proj.weight"] = "cuda"
    device_map[f"{layer_prefix}.self_attn.q_norm.weight"] = "cuda"
    device_map[f"{layer_prefix}.self_attn.k_norm.weight"] = "cuda"
    device_map[f"{layer_prefix}.linear_attn.norm.weight"] = "cuda"

    # MLP / Experts -> Disk
    device_map[f"{layer_prefix}.mlp.experts.gate_up_proj"] = "meta"
    device_map[f"{layer_prefix}.mlp.experts.down_proj"] = "meta"

    # 2. MLP / Experts -> META
    # We map them to "meta" so HuggingFace skips loading them entirely (0 VRAM/RAM).
    # They will be discarded anyway when we replace them with our custom Qwen3NextExperts.
    device_map[f"{layer_prefix}.mlp.experts.gate_proj"] = "meta"
    device_map[f"{layer_prefix}.mlp.experts.up_proj"] = "meta"
    device_map[f"{layer_prefix}.mlp.experts.down_proj"] = "meta"

    device_map[f"{layer_prefix}.mlp.gate.weight"] = "cuda"
    device_map[f"{layer_prefix}.mlp.shared_expert.gate_proj.weight"] = "cuda"
    device_map[f"{layer_prefix}.mlp.shared_expert.up_proj.weight"] = "cuda"
    device_map[f"{layer_prefix}.mlp.shared_expert.down_proj.weight"] = "cuda"
    device_map[f"{layer_prefix}.mlp.shared_expert_gate.weight"] = "cuda"

# --- 2. LOAD MODEL ---
print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map=device_map,
    max_memory={0: "6GiB", "cpu": "32GiB", "disk": "100GiB"},
    offload_folder="./offload"
)

# --- 3. INJECT CUSTOM EXPERTS ---
# Assuming this import exists in your environment
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextExperts

print("Injecting custom experts logic...")
for i in range(len(model.model.layers)):
    new_mlp = Qwen3NextExperts(model.config, layer_idx=i, offload_folder=offload_path)
    model.model.layers[i].mlp.experts = new_mlp
print("Model ready.")

# --- 4. INTERACTIVE LOOP ---

# Chat history storage
messages = []


def print_colored(text, color_code):
    print(f"\033[{color_code}m{text}\033[0m", end="")


BLUE = "34"
GREEN = "32"
YELLOW = "33"

print("\n" + "=" * 50)
print("Interactive Chat Mode. Type 'exit' or 'quit' to stop.")
print("=" * 50 + "\n")

while True:
    try:
        print_colored("\nUser: ", BLUE)
        user_input = input()

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": user_input})

        text_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        print_colored("Assistant: ", GREEN)

        generated_response = ""
        start_time = time.time()
        first_token_time = None

        for i, new_text in enumerate(streamer):
            if i == 0:
                first_token_time = time.time()
            print(new_text, end="", flush=True)
            generated_response += new_text

        end_time = time.time()

        messages.append({"role": "assistant", "content": generated_response})

        token_count = len(tokenizer.encode(generated_response, add_special_tokens=False))
        total_duration = end_time - start_time

        speed = token_count / total_duration if total_duration > 0 else 0

        print_colored(f"\n\n[Stats] ", YELLOW)
        print(f"Tokens: {token_count} | Time: {total_duration:.2f}s | Speed: {speed:.2f} t/s")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
        break
    except Exception as e:
        print(f"\nError: {e}")