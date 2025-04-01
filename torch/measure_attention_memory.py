import torch
import torch.nn.functional as F

def measure_attention_memory(q, k, v, use_mem_efficient):
    # Clear the CUDA cache and reset peak memory stats.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # Globaly enable or disable memory-efficient scaled dot product attention
    torch.backends.cuda.enable_mem_efficient_sdp(use_mem_efficient)

    # Run the scaled dot product attention operation.
    _ = F.scaled_dot_product_attention(q, k, v)

    # Get the peak memory allocated during the operation.
    peak_memory = torch.cuda.max_memory_allocated(q.device)
    return peak_memory

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define tensor shapes.
    batch_size = 16
    seq_length = 128
    embed_dim = 64

    # Create random query, key, and value tensors
    q = torch.randn(batch_size, seq_length, embed_dim, device=device)
    k = torch.randn(batch_size, seq_length, embed_dim, device=device)
    v = torch.randn(batch_size, seq_length, embed_dim, device=device)

    # Measure peak memory with memory-efficient SDP enabled
    mem_efficient_peak= measure_attention_memory(q, k, v, use_mem_efficient=True)

    # Measure peak memory with memory-efficient SDP disabled
    non_mem_efficient_peak = measure_attention_memory(q, k, v, use_mem_efficient=False)

    # Convert memory from bytes to megabytes.
    mem_efficient_peak_mb = mem_efficient_peak / (1024 ** 2)
    non_mem_efficient_peak_mb = non_mem_efficient_peak / (1024 ** 2)

    print(f"Peak memory with mem efficient SDP enabled: {mem_efficient_peak_mb:.2f} MB")
    print(f"Peak memory with mem efficient SDP disabled: {non_mem_efficient_peak_mb:.2F} MB")

if __name__ == "__main__":
    if torch.cuda.is_available():
        main()
    else:
        print("CUDA is not available. Please run this code on a CUDA-enabled device.")

