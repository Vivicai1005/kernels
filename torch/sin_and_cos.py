import time
import torch

def get_cos_sin(
    D,
    seq_len,
    device,
    dtype,
    freq=1e4,
    F0=1.0,
    scaling_factor=1.0,
    cache=None
):
    if cache is None:
        cache = {}

    key = (D, seq_len, device, dtype)
    if key not in cache:
        inv_freq = 1.0 / (freq ** (torch.arange(0, D, 2, device=device).float() / D))
        t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
        freqs = torch.cat((freqs, freqs), dim=-1)
        cos = freqs.cos()
        sin = freqs.sin()
        cache[key] = (cos, sin)

    return cache[key]

def measure_get_cos_sin_runtime(iterations=100):
    D = 64
    seq_len = 62
    device = torch.device("cuda")  # Use "cuda" if you want to test on GPU
    dtype = torch.bfloat16
    freq = 1e4
    F0 = 1.0
    scaling_factor = 1.0

    # Shared cache
    shared_cache = {}

    # Optional warm-up call
    _ = get_cos_sin(D, seq_len, device, dtype, freq, F0, scaling_factor, shared_cache)

    # Collect times for each call
    times = []
    for _ in range(iterations):
        start_time = time.time()
        _ = get_cos_sin(D, seq_len, device, dtype, freq, F0, scaling_factor, shared_cache)
        elapsed = time.time() - start_time
        times.append(elapsed)

    avg_time = sum(times) / iterations
    max_time = max(times)
    print(f"Average time over {iterations} calls: {avg_time:.6f} seconds")
    print(f"Max time over {iterations} calls: {max_time:.6f} seconds")

if __name__ == "__main__":
    measure_get_cos_sin_runtime(100)
