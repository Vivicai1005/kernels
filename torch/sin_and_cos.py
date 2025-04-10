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
    # Hyperparameters
    D = 64
    seq_len = 1024
    device = torch.device("cuda")
    dtype = torch.bfloat16
    freq = 1e4
    F0 = 1.0
    scaling_factor = 1.0

    # Shared cache dict so subsequent calls use the same cache
    shared_cache = {}

    # Warm-up call (optional, ensures consistent device init)
    _ = get_cos_sin(
        D, seq_len, device, dtype, freq, F0, scaling_factor, cache=shared_cache
    )

    # Measure runtime
    total_time = 0.0
    for _ in range(iterations):
        start_time = time.time()
        _ = get_cos_sin(
            D, seq_len, device, dtype, freq, F0, scaling_factor, cache=shared_cache
        )
        total_time += (time.time() - start_time)

    avg_time = total_time / iterations
    print(f"Average time for running get_cos_sin {iterations} times: {avg_time:.6f} seconds")


if __name__ == "__main__":
    measure_get_cos_sin_runtime(100)