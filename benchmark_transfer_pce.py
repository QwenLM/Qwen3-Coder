import os
import time
import torch
import numpy as np

# --- Configuration ---
DUMMY_DIR = "./benchmark_data"
GB = 1024 ** 3
MB = 1024 ** 2


def setup_dummy_files(size_bytes, filename):
    """Creates a dummy file of a specific size filled with random bytes."""
    os.makedirs(DUMMY_DIR, exist_ok=True)
    filepath = os.path.join(DUMMY_DIR, filename)
    if not os.path.exists(filepath) or os.path.getsize(filepath) != size_bytes:
        print(f"Creating dummy file {filename} ({size_bytes / MB:.2f} MB)...")
        with open(filepath, "wb") as f:
            f.write(os.urandom(size_bytes))
    return filepath


def cleanup():
    """Removes the dummy files."""
    for f in os.listdir(DUMMY_DIR):
        os.remove(os.path.join(DUMMY_DIR, f))
    os.rmdir(DUMMY_DIR)


# =====================================================================
# Benchmark 1: Transfer Speed vs. Tensor Size
# =====================================================================
def benchmark_size_scaling():
    print("\n" + "=" * 50)
    print("BENCHMARK 1: Transfer Speed vs. Tensor Size")
    print("=" * 50)

    # Test sizes from 1MB to 1GB
    test_sizes_mb = [1, 2, 4, 8, 16, 64]

    # Create the largest file once to read chunks from
    max_size = test_sizes_mb[-1] * MB
    filepath = setup_dummy_files(max_size, "large_dummy.bin")

    print(f"{'Size (MB)':<10} | {'Time (ms)':<15} | {'Bandwidth (GB/s)':<15}")
    print("-" * 45)

    for size_mb in test_sizes_mb:
        size_bytes = size_mb * MB

        # 1. Allocate Pinned CPU Memory and GPU Memory
        landing_buf = torch.empty(size_bytes, dtype=torch.uint8, pin_memory=True)
        vram_buf = torch.empty(size_bytes, dtype=torch.uint8, device="cuda")
        view_1d = landing_buf.numpy()

        # Warmup GPU
        vram_buf.copy_(landing_buf, non_blocking=True)
        torch.cuda.synchronize()

        with open(filepath, "rb") as f:
            # Randomize offset to try and bypass some OS caching
            f.seek(0)

            torch.cuda.synchronize()
            start_time = time.perf_counter()

            # Disk -> Pinned RAM
            f.readinto(view_1d.data)

            # Pinned RAM -> VRAM
            vram_buf.copy_(landing_buf, non_blocking=True)

            # Wait for transfer to complete
            torch.cuda.synchronize()

            end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        bandwidth = (size_bytes / GB) / elapsed_time

        print(f"{size_mb:<10} | {elapsed_time * 1000:<15.2f} | {bandwidth:<15.3f}")

        del landing_buf, vram_buf
        torch.cuda.empty_cache()


# =====================================================================
# Benchmark 2: 3 Files (Separate Reads) vs 1 File (Single Read)
# =====================================================================
def benchmark_3_vs_1():
    print("\n" + "=" * 50)
    print("BENCHMARK 2: 3 Files (Separate Reads) vs 1 File")
    print("=" * 50)

    # Assume each tensor (Gate, Up, Down) is 128MB. Total = 384MB per layer.
    g_bytes = 6 * MB
    total_bytes = 3 * g_bytes

    # Setup files
    file_gate = setup_dummy_files(g_bytes, "gate.bin")
    file_up = setup_dummy_files(g_bytes, "up.bin")
    file_down = setup_dummy_files(g_bytes, "down.bin")
    file_combined = setup_dummy_files(total_bytes, "combined.bin")

    landing_buf = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=True)
    vram_buf = torch.empty(total_bytes, dtype=torch.uint8, device="cuda")
    view_1d = landing_buf.numpy()

    iterations = 10

    # --- METHOD A: 3 Separate Files ---
    times_3_files = []
    for _ in range(iterations):
        # Open files (simulating your get_file logic)
        f_g = open(file_gate, "rb")
        f_u = open(file_up, "rb")
        f_d = open(file_down, "rb")

        torch.cuda.synchronize()
        start = time.perf_counter()

        # Read Gate
        f_g.seek(0)
        f_g.readinto(view_1d[0: g_bytes].data)
        # Read Up
        f_u.seek(0)
        f_u.readinto(view_1d[g_bytes: 2 * g_bytes].data)
        # Read Down
        f_d.seek(0)
        f_d.readinto(view_1d[2 * g_bytes:].data)

        # Async Copy
        vram_buf.copy_(landing_buf, non_blocking=True)
        torch.cuda.synchronize()

        end = time.perf_counter()
        times_3_files.append(end - start)

        f_g.close()
        f_u.close()
        f_d.close()

    # --- METHOD B: 1 Single File ---
    times_1_file = []
    for _ in range(iterations):
        f_c = open(file_combined, "rb")

        torch.cuda.synchronize()
        start = time.perf_counter()

        # Single Read
        f_c.seek(0)
        f_c.readinto(view_1d.data)

        # Async Copy
        vram_buf.copy_(landing_buf, non_blocking=True)
        torch.cuda.synchronize()

        end = time.perf_counter()
        times_1_file.append(end - start)

        f_c.close()

    # --- Print Report ---
    avg_3 = np.mean(times_3_files) * 1000
    avg_1 = np.mean(times_1_file) * 1000
    bw_3 = (total_bytes / GB) / np.mean(times_3_files)
    bw_1 = (total_bytes / GB) / np.mean(times_1_file)

    print(f"Tensor Size: 3 x {g_bytes / MB}MB (Total: {total_bytes / MB}MB)")
    print(f"{'Method':<20} | {'Avg Time (ms)':<15} | {'Bandwidth (GB/s)':<15}")
    print("-" * 55)
    print(f"{'3 Separate Reads':<20} | {avg_3:<15.2f} | {bw_3:<15.3f}")
    print(f"{'1 Combined Read':<20} | {avg_1:<15.2f} | {bw_1:<15.3f}")

    speedup = avg_3 / avg_1
    print(f"\nConclusion: 1 Combined Read is {speedup:.2f}x faster than 3 Separate Reads.")


if __name__ == "__main__":
    try:
        benchmark_size_scaling()
        benchmark_3_vs_1()
    finally:
        print("\nCleaning up dummy files...")
        cleanup()