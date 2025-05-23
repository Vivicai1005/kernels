import torch
from xfuser.core.distributed.parallel_state import get_sequence_parallel_world_size, get_sequence_parallel_rank
from xfuser.core.utils.timer import func_timer_decorator, time


class RoPE1D:
    def __init__(self, freq=1e4, F0=1.0, scaling_factor=1.0):
        self.base = freq
        self.F0 = F0
        self.scaling_factor = scaling_factor
        self.cache = {}

    def get_cos_sin(self, D, seq_len, device, dtype):
        if (D, seq_len, device, dtype) not in self.cache:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)
            cos = freqs.cos()  # (Seq, Dim)
            sin = freqs.sin()
            self.cache[D, seq_len, device, dtype] = (cos, sin)
        return self.cache[D, seq_len, device, dtype]

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope1d(self, tokens, pos1d, cos, sin):
        assert pos1d.ndim == 2
        cos = torch.nn.functional.embedding(pos1d, cos)[:, :, None, :]
        sin = torch.nn.functional.embedding(pos1d, sin)[:, :, None, :]
        return (tokens * cos) + (self.rotate_half(tokens) * sin)

    def __call__(self, tokens, positions):
        """
        input:
            * tokens: batch_size x ntokens x nheads x dim
            * positions: batch_size x ntokens (t position of each token)
        output:
            * tokens after appplying RoPE2D (batch_size x ntokens x nheads x dim)
        """
        D = tokens.size(3)
        assert positions.ndim == 2  # Batch, Seq
        cos, sin = self.get_cos_sin(D, int(positions.max()) + 1, tokens.device, tokens.dtype)
        tokens = self.apply_rope1d(tokens, positions, cos, sin)
        return tokens


class RoPE3D(RoPE1D):
    def __init__(self, freq=1e4, F0=1.0, scaling_factor=1.0):
        super(RoPE3D, self).__init__(freq, F0, scaling_factor)
        self.position_cache = {}

    def get_mesh_3d(self, rope_positions, bsz):
        f, h, w = rope_positions

        if f"{f}-{h}-{w}" not in self.position_cache:
            x = torch.arange(f, device='cpu')
            y = torch.arange(h, device='cpu')
            z = torch.arange(w, device='cpu')
            self.position_cache[f"{f}-{h}-{w}"] = torch.cartesian_prod(x, y, z).view(1, f * h * w, 3).expand(bsz, -1, 3)
        return self.position_cache[f"{f}-{h}-{w}"]

    @func_timer_decorator
    def __call__(self, tokens, rope_positions, ch_split, parallel=False):
        """
        input:
            * tokens: batch_size x ntokens x nheads x dim
            * rope_positions: list of (f, h, w)
        output:
            * tokens after appplying RoPE2D (batch_size x ntokens x nheads x dim)
        """
        assert sum(ch_split) == tokens.size(-1);

        mesh_grid = self.get_mesh_3d(rope_positions, bsz=tokens.shape[0])
        out = []
        for i, (D, x) in enumerate(zip(ch_split, torch.split(tokens, ch_split, dim=-1))):
            cos, sin = self.get_cos_sin(D, int(mesh_grid.max()) + 1, tokens.device, tokens.dtype)

            if parallel:
                mesh = torch.chunk(mesh_grid[:, :, i], get_sequence_parallel_world_size(), dim=1)[
                    get_sequence_parallel_rank()].clone()
            else:
                mesh = mesh_grid[:, :, i].clone()

            x = self.apply_rope1d(x, mesh.to(tokens.device), cos, sin)
            out.append(x)

        tokens = torch.cat(out, dim=-1)
        return tokens



import torch.distributed as dist
from stepvideo.config import parse_args
from stepvideo.parallel import initialize_parall_group, get_parallel_group
from stepvideo.utils import setup_seed

if __name__ == "__main__":
    args = parse_args()
    initialize_parall_group(ring_degree=args.ring_degree, ulysses_degree=args.ulysses_degree,
                            tensor_parallel_degree=args.tensor_parallel_degree)

    local_rank = get_parallel_group().local_rank
    device = torch.device(f"cuda:{local_rank}")

    setup_seed(args.seed)

    device = torch.device("cuda")
    tokens = torch.randn(1, 75888, 48, 128, device=device)
    rope_positions = [36, 34, 62]
    rope_ch_split = [64, 32, 32]

    rope_3d = RoPE3D(freq=1e4, F0=1.0, scaling_factor=1.0)

    rope_3d(tokens, rope_positions, rope_ch_split, parallel=True)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        rope_3d(tokens, rope_positions, rope_ch_split, parallel=True)

    # Print the profiling summary, sorting by CUDA time.
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
    # Optionally, export the profile trace for deeper analysis using Chrome Trace Viewer.
    prof.export_chrome_trace("rope3d_profile_trace.json")

    dist.destroy_process_group()
