import time
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# --- CONFIGURATION ---
model_name = "Qwen/Qwen3.5-122B-A10B-FP8"
offload_path = r"c:\Users\koshe\.cache\huggingface\hub\models--Qwen--Qwen3.5-122B-A10B-FP8\snapshots\fb53b9f3bdaab287c597d4e943783153ec527e06" #TODO: set your path

# --- 1. DEVICE MAP SETUP ---
device_map = {}
device_map['lm_head.weight'] = "cpu"
device_map['model.language_model.embed_tokens.weight'] = "cuda"
device_map['model.language_model.norm.weight'] = "cuda"
device_map['mtp'] = "cuda"

device_map['model.visual.patch_embed.proj.weight'] = "meta"
device_map['model.visual.patch_embed.proj.bias'] = "meta"
device_map['model.visual.pos_embed.weight'] = "meta"
for i in range(27):
    layer_prefix = f"model.visual.blocks.{i}"

    device_map[f"{layer_prefix}.norm1.weight"] = "meta"
    device_map[f"{layer_prefix}.norm1.bias"] = "meta"
    device_map[f"{layer_prefix}.norm2.weight"] = "meta"
    device_map[f"{layer_prefix}.norm2.bias"] = "meta"

    device_map[f"{layer_prefix}.attn.qkv.weight"] = "meta"
    device_map[f"{layer_prefix}.attn.qkv.bias"] = "meta"
    device_map[f"{layer_prefix}.attn.proj.weight"] = "meta"
    device_map[f"{layer_prefix}.attn.proj.bias"] = "meta"

    device_map[f"{layer_prefix}.mlp.linear_fc1.weight"] = "meta"
    device_map[f"{layer_prefix}.mlp.linear_fc1.bias"] = "meta"
    device_map[f"{layer_prefix}.mlp.linear_fc2.weight"] = "meta"
    device_map[f"{layer_prefix}.mlp.linear_fc2.bias"] = "meta"

device_map[f"model.visual.merger.norm.weight"] = "cuda"
device_map[f"model.visual.merger.norm.bias"] = "cuda"
device_map[f"model.visual.merger.linear_fc1.weight"] = "cuda"
device_map[f"model.visual.merger.linear_fc1.bias"] = "cuda"
device_map[f"model.visual.merger.linear_fc2.weight"] = "cuda"
device_map[f"model.visual.merger.linear_fc2.bias"] = "cuda"

for i in range(48):
    layer_prefix = f"model.language_model.layers.{i}"

    # Attention & Norms -> GPU
    device_map[f"{layer_prefix}.linear_attn.in_proj_qkvz.weight"] = "cpu"
    device_map[f"{layer_prefix}.linear_attn.out_proj.weight"] = "cpu"
    device_map[f"{layer_prefix}.linear_attn.dt_bias"] = "cuda"
    device_map[f"{layer_prefix}.linear_attn.A_log"] = "cuda"
    device_map[f"{layer_prefix}.linear_attn.conv1d.weight"] = "cuda"
    device_map[f"{layer_prefix}.linear_attn.in_proj_ba.weight"] = "cuda" #!!
    device_map[f"{layer_prefix}.linear_attn.in_proj_qkv.weight"] = "cpu"
    device_map[f"{layer_prefix}.linear_attn.in_proj_z.weight"] = "cpu"
    device_map[f"{layer_prefix}.linear_attn.in_proj_b.weight"] = "cuda" #!!
    device_map[f"{layer_prefix}.linear_attn.in_proj_a.weight"] = "cuda" #!!


    device_map[f"{layer_prefix}.input_layernorm.weight"] = "cuda"
    device_map[f"{layer_prefix}.post_attention_layernorm.weight"] = "cuda"

    device_map[f"{layer_prefix}.self_attn.q_proj.weight"] = "cpu"
    device_map[f"{layer_prefix}.self_attn.k_proj.weight"] = "cpu"
    device_map[f"{layer_prefix}.self_attn.v_proj.weight"] = "cpu"
    device_map[f"{layer_prefix}.self_attn.o_proj.weight"] = "cpu"
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
    device_map[f"{layer_prefix}.mlp.shared_expert.gate_proj.weight"] = "cpu"
    device_map[f"{layer_prefix}.mlp.shared_expert.up_proj.weight"] = "cpu"
    device_map[f"{layer_prefix}.mlp.shared_expert.down_proj.weight"] = "cpu"
    device_map[f"{layer_prefix}.mlp.shared_expert_gate.weight"] = "cuda" #!!

from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForConditionalGeneration
import torch

# --- 2. LOAD MODEL ---
print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("-----------111--------")
model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype="auto",
    dtype=torch.bfloat16,
    #device_map="auto"
    device_map=device_map,
    max_memory={0: "8GiB", "cpu": "32GiB", "disk": "400GiB"},
    #offload_folder="./offload"
)
print("-----------222--------")
# --- 3. INJECT CUSTOM EXPERTS ---
# Assuming this import exists in your environment
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import  Qwen3_5MoeExperts

print("Injecting custom experts logic...")
for i in range(len(model.model.language_model.layers)):
    new_mlp = Qwen3_5MoeExperts(model.model.language_model.config, layer_idx=i, offload_folder=offload_path)
    model.model.language_model.layers[i].mlp.experts = new_mlp
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
            max_new_tokens=50,
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