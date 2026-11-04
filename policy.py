
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return out.type_as(x) * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)

class RotaryPE(nn.Module):
    def __init__(self, d: int, base: int = 10_000):
        super().__init__()
        assert d % 2 == 0, "RoPE dimension must be even"

        self.base = base
        self.d = d
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)

    def _build_cache(self, x: torch.Tensor):
        seq_len = x.shape[-2]

        if (
            self.cos_cached is not None
            and self.cos_cached.shape[2] >= seq_len
            and self.cos_cached.device == x.device
        ):
            return

        theta = 1.0 / (
            self.base **
            (torch.arange(0, self.d, 2, device=x.device).float() / self.d)
        )

        seq_idx = torch.arange(seq_len, device=x.device).float()
        idx_theta = torch.outer(seq_idx, theta)
        idx_theta = torch.cat([idx_theta, idx_theta], dim=-1)

        # Shape: (1, 1, seq_len, d)
        self.cos_cached = idx_theta.cos()[None, None, :, :]
        self.sin_cached = idx_theta.sin()[None, None, :, :]

    def _neg_half(self, x: torch.Tensor):
        d_2 = x.shape[-1] // 2
        return torch.cat(
            [-x[..., d_2:], x[..., :d_2]], dim=-1
        )

    def forward(self, x: torch.Tensor):
        self._build_cache(x)
        seq_len = x.shape[-2]

        cos = self.cos_cached[:, :, :seq_len, :].to(dtype=x.dtype)
        sin = self.sin_cached[:, :, :seq_len, :].to(dtype=x.dtype)

        return x * cos + self._neg_half(x) * sin


class Attention(nn.Module):
    def __init__(self, dim, num_heads, use_rope=True):
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.proj = nn.Linear(dim, dim)
        self.rope = RotaryPE(self.head_dim) if use_rope else None

    def forward(self, x, context=None, context_mask=None):
        """ context_mask: B, M bool, True = attend (key padding mask for cross-attention) """
        self_attention = context is None
        context = x if context is None else context
        B, N, C = x.shape

        q = self.q(x)
        k, v = self.kv(context).chunk(2, dim=-1)

        # (B, sequence, heads, head_dim) -> (B, heads, sequence, head_dim)
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(
            B, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = v.reshape(
            B, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Normalize each head over its head_dim
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE to queries and keys, not values
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        attn_mask = None
        if self_attention and N > 1:
            # The final token can attend to itself, but not to the first N - 1 tokens.
            attn_mask = torch.ones(N, N, dtype=torch.bool, device=x.device)
            attn_mask[-1, :-1] = False
        elif context_mask is not None:
            attn_mask = context_mask[:, None, None, :].bool()  # B, 1, 1, M

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


    
def modulate(x, shift, scale):
    """ x: B, N, C; shift, scale: B, 1, C or B, N, C """
    return x * (1 + scale) + shift


class DDTBlock(nn.Module):
    """ AdaLN-Zero block: self-attn over target tokens, cross-attn to context tokens, FFN.
    Each sub-layer is modulated (shift, scale) and gated by the conditioning c """
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1, self.norm2, self.norm3, self.norm_context = RMSNorm(dim), RMSNorm(dim), RMSNorm(dim), RMSNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.cross_attn = Attention(dim, num_heads, use_rope=False)  # query / key positions are unrelated
        self.mlp = SwiGLUFFN(dim, int(2 / 3 * dim * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))

    def forward(self, x, c, context, context_mask=None):
        """ x: B, N, C target tokens; c: B, C or B, N, C conditioning; context: B, M, C """
        if c.dim() == 2:
            c = c.unsqueeze(1)  # B, 1, C

        # adaln = (1 + scale) * norm(x) + shift * c
        # zero-adaln -> at attention: x* = x * gate + attn(adaln (x, c))
        (shift_msa, scale_msa, gate_msa,
         shift_ca, scale_ca, gate_ca,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(9, dim=-1)

        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_ca * self.cross_attn(modulate(self.norm2(x), shift_ca, scale_ca), self.norm_context(context),
                                          context_mask=context_mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.linear = nn.Linear(dim, out_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, c):
        if c.dim() == 2:
            c = c.unsqueeze(1)  # B, 1, C
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))  # no gate


class GaussianFourierEmbedding(nn.Module):
    """ Timestep embedding: fixed random Fourier features -> MLP """
    def __init__(self, dim, embedding_size=256, scale=1.0):
        super().__init__()
        self.register_buffer('W', torch.randn(embedding_size) * scale)
        self.mlp = nn.Sequential(nn.Linear(embedding_size * 2, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        """ t: B -> B, dim """
        t = t[:, None] * self.W[None, :] * 2 * torch.pi
        return self.mlp(torch.cat([torch.sin(t), torch.cos(t)], dim=-1))


def sincos_pos_embed(dim, h, w):
    """ h * w, dim fixed sin-cos table: h and w get (dim // 6) * 2 channels each, the rest are the (single) time
    step's channels, as in Flow-World-Models' PositionalEncoding3D with T = 1 """
    dim_hw = (dim // 6) * 2
    dim_t = dim - 2 * dim_hw

    def sincos_1d(n, d):
        pos = torch.arange(n, dtype=torch.float32)
        omega = 1.0 / (10000 ** (2 * torch.arange(d // 2, dtype=torch.float32) / d))
        out = pos[:, None] * omega[None, :]
        return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)

    grid_h, grid_w = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    return torch.cat([
        sincos_1d(1, dim_t).expand(h * w, -1),
        sincos_1d(h, dim_hw)[grid_h.flatten()],
        sincos_1d(w, dim_hw)[grid_w.flatten()],
    ], dim=-1)

