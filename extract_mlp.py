import os
import json
import torch
import re
import gc
from safetensors import safe_open
from safetensors.torch import save_file
from collections import defaultdict
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SOURCE_MODEL_DIR = r"C:\Users\{user}\.cache\huggingface\hub\models--Qwen--Qwen3-Coder-Next-FP8\snapshots\da6e2ed27304dd39abadd9c82ef50e8de67bdd4c\\"  # Where the downloaded files are
OFFLOAD_DIR = "./offload"  # Where to save extracted tensors
NUM_EXPERTS = 512  # Total experts per layer

os.makedirs(OFFLOAD_DIR, exist_ok=True)


def load_index(source_dir):
    index_path = os.path.join(source_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index not found: {index_path}")

    with open(index_path, "r") as f:
        index_data = json.load(f)
    return index_data["weight_map"]


def get_file_handle(handles, filename, source_dir):
    if filename not in handles:
        path = os.path.join(source_dir, filename)
        handles[filename] = safe_open(path, framework="pt", device="cpu")
    return handles[filename]


def process_conversion():
    print(f"Reading index from {SOURCE_MODEL_DIR}...")
    weight_map = load_index(SOURCE_MODEL_DIR)

    layers = defaultdict(lambda: {"experts": defaultdict(dict), "other": {}})
    global_tensors = {}

    expert_pattern = re.compile(
        r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.(weight|weight_scale_inv)$")
    layer_pattern = re.compile(r"model\.layers\.(\d+)\.(.*)")

    print("Parsing weight map...")
    for key, filename in weight_map.items():
        exp_match = expert_pattern.match(key)
        if exp_match:
            layer_idx, expert_idx, proj_type, suffix = exp_match.groups()
            layer_idx = int(layer_idx)
            expert_idx = int(expert_idx)

            layers[layer_idx]["experts"][expert_idx].setdefault(proj_type, {})
            layers[layer_idx]["experts"][expert_idx][proj_type][suffix] = (key, filename)
            continue

        lay_match = layer_pattern.match(key)
        if lay_match:
            layer_idx, remaining_key = lay_match.groups()
            layers[int(layer_idx)]["other"][key] = (key, filename)
            continue

        global_tensors[key] = (key, filename)

    sorted_layer_indices = sorted(layers.keys())
    file_handles = {}

    try:
        for layer_idx in tqdm(sorted_layer_indices, desc="Processing Layers"):
            layer_data = layers[layer_idx]
            experts_dict = layer_data["experts"]

            if len(experts_dict) > 0:
                list_gate_up_weights = []
                list_down_weights = []
                list_gate_up_scales = []
                list_down_scales = []

                for exp_id in range(NUM_EXPERTS):
                    if exp_id not in experts_dict:
                        continue

                    parts = experts_dict[exp_id]

                    required_keys = ['gate', 'up', 'down']
                    if not all(k in parts for k in required_keys):
                        print(f"SKIP Exp {exp_id}: Missing keys. Have: {list(parts.keys())}")
                        continue

                    k_g, f_g = parts['gate']['weight']
                    t_gate = get_file_handle(file_handles, f_g, SOURCE_MODEL_DIR).get_tensor(k_g)

                    k_u, f_u = parts['up']['weight']
                    t_up = get_file_handle(file_handles, f_u, SOURCE_MODEL_DIR).get_tensor(k_u)

                    k_d, f_d = parts['down']['weight']
                    t_down = get_file_handle(file_handles, f_d, SOURCE_MODEL_DIR).get_tensor(k_d)

                    s_g = s_u = s_d = None

                    if 'weight_scale_inv' in parts['gate']:
                        k, f = parts['gate']['weight_scale_inv']
                        s_g = get_file_handle(file_handles, f, SOURCE_MODEL_DIR).get_tensor(k)

                    if 'weight_scale_inv' in parts['up']:
                        k, f = parts['up']['weight_scale_inv']
                        s_u = get_file_handle(file_handles, f, SOURCE_MODEL_DIR).get_tensor(k)

                    if 'weight_scale_inv' in parts['down']:
                        k, f = parts['down']['weight_scale_inv']
                        s_d = get_file_handle(file_handles, f, SOURCE_MODEL_DIR).get_tensor(k)

                    t_gate_up = torch.cat([t_gate, t_up], dim=0)

                    list_gate_up_weights.append(t_gate_up)
                    list_down_weights.append(t_down)

                    if s_g is not None and s_u is not None:
                        if s_g.ndim == 0: s_g = s_g.view(1)
                        if s_u.ndim == 0: s_u = s_u.view(1)
                        s_gate_up = torch.cat([s_g, s_u], dim=0)
                        list_gate_up_scales.append(s_gate_up)

                    if s_d is not None:
                        list_down_scales.append(s_d)

                if list_gate_up_weights:
                    full_gate_up = torch.cat(list_gate_up_weights, dim=0)
                    full_down = torch.cat(list_down_weights, dim=0)

                    gu_path = os.path.join(OFFLOAD_DIR,
                                           f"model.layers.{layer_idx}.mlp.experts.gate_up_proj.safetensors")
                    d_path = os.path.join(OFFLOAD_DIR, f"model.layers.{layer_idx}.mlp.experts.down_proj.safetensors")

                    gu_key = f"model.layers.{layer_idx}.mlp.experts.gate_up_proj"
                    d_key = f"model.layers.{layer_idx}.mlp.experts.down_proj"

                    gu_dict = {gu_key: full_gate_up}
                    d_dict = {d_key: full_down}

                    if list_gate_up_scales:
                        full_gu_scale = torch.cat(list_gate_up_scales, dim=0)
                        gu_dict[f"{gu_key}_scale_inv"] = full_gu_scale

                    if list_down_scales:
                        full_d_scale = torch.cat(list_down_scales, dim=0)
                        d_dict[f"{d_key}_scale_inv"] = full_d_scale

                    save_file(gu_dict, gu_path)
                    save_file(d_dict, d_path)

                    del full_gate_up, full_down, list_gate_up_weights, list_down_weights
                    gc.collect()

            for key, (real_key, filename) in layer_data["other"].items():
                f = get_file_handle(file_handles, filename, SOURCE_MODEL_DIR)
                tensor = f.get_tensor(real_key)
                out_path = os.path.join(OFFLOAD_DIR, f"{real_key}.safetensors")
                save_file({real_key: tensor}, out_path)

        print("Processing global tensors...")
        for key, (real_key, filename) in global_tensors.items():
            f = get_file_handle(file_handles, filename, SOURCE_MODEL_DIR)
            tensor = f.get_tensor(real_key)
            out_path = os.path.join(OFFLOAD_DIR, f"{real_key}.safetensors")
            save_file({real_key: tensor}, out_path)

    finally:
        print("Closing file handles...")
        file_handles.clear()


if __name__ == "__main__":
    process_conversion()
    print("Conversion complete.")