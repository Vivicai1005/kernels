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


def measure_attention_memory_with_backward(q, k, v, use_mem_efficient):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.enable_mem_efficient_sdp(use_mem_efficient)

    # Enable gradient computation for at least one input.
    q.requires_grad_(True)

    # Forward pass.
    attn_output = F.scaled_dot_product_attention(q, k, v)

    # Dummy loss and backward pass.
    loss = attn_output.sum()
    loss.backward()

    # Get the peak memory allocated (in bytes) during the operation.
    peak_memory = torch.cuda.max_memory_allocated(q.device)
    return peak_memory

def measure_attention_memory_with_backward_default(q, k, v):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # Enable gradient computation for at least one input.
    q.requires_grad_(True)

    # Forward pass.
    attn_output = F.scaled_dot_product_attention(q, k, v)

    # Dummy loss and backward pass.
    loss = attn_output.sum()
    loss.backward()

    # Get the peak memory allocated (in bytes) during the operation.
    peak_memory = torch.cuda.max_memory_allocated(q.device)
    return peak_memory

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define tensor shapes.
    batch_size = 128
    seq_length = 1024
    embed_dim = 256

    # Create random query, key, and value tensors
    q = torch.randn(batch_size, seq_length, embed_dim, device=device)
    k = torch.randn(batch_size, seq_length, embed_dim, device=device)
    v = torch.randn(batch_size, seq_length, embed_dim, device=device)


    # Measure peak memory with memory-efficient SDP enabled
    mem_efficient_peak= measure_attention_memory_with_backward(q, k, v, use_mem_efficient=True)

    # Measure peak memory with memory-efficient SDP disabled
    non_mem_efficient_peak = measure_attention_memory_with_backward(q, k, v, use_mem_efficient=False)

    # Measure peak memory default
    default_mem_efficient_peak = measure_attention_memory_with_backward_default(q, k, v)

    # Convert memory from bytes to megabytes.
    default_mem_efficient_peak_mb = default_mem_efficient_peak / (1024 ** 2)
    mem_efficient_peak_mb = mem_efficient_peak / (1024 ** 2)
    non_mem_efficient_peak_mb = non_mem_efficient_peak / (1024 ** 2)

    print(f"Peak memory with mem efficient defaukt: {default_mem_efficient_peak_mb:.2f} MB")
    print(f"Peak memory with mem efficient SDP enabled: {mem_efficient_peak_mb:.2f} MB")
    print(f"Peak memory with mem efficient SDP disabled: {non_mem_efficient_peak_mb:.2F} MB")

if __name__ == "__main__":
    if torch.cuda.is_available():
        main()
    else:
        print("CUDA is not available. Please run this code on a CUDA-enabled device.")

