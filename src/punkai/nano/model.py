"""A decoder-only transformer, written to be read.

Roughly 300 lines stand between a token id and a probability distribution over
the next one. That is the whole mystery. Everything a frontier model does
differently is scale, data, and post-training -- the architecture below is,
allowing for details, the same one.

Deliberately the *modern* architecture rather than the GPT-2 one everyone
teaches, because the differences are small to write and large to understand:

  RMSNorm       instead of LayerNorm  -- no mean subtraction, no bias, faster,
                                         works just as well.
  RoPE          instead of learned position embeddings -- position is rotated
                                         into the query and key vectors, so the
                                         model sees *relative* distance and can
                                         run past the context length it trained
                                         on (badly, but it can).
  SwiGLU        instead of GELU MLP    -- a gate that lets the network suppress
                                         its own activations. Three matrices
                                         instead of two, sized 8/3 rather than
                                         4x to keep the parameter count even.
  GQA           instead of MHA         -- several query heads share one key/
                                         value head, which shrinks the KV cache
                                         by that factor. This is the single
                                         biggest lever on whether long context
                                         fits in your VRAM.

Requires the `torch` extra. Everything else in punk-ai stays dependency-free.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class NanoConfig:
    """Shape of the model. Defaults are a ~15M-parameter model that trains on a
    laptop in an evening and on a consumer GPU in minutes."""

    vocab_size: int = 4096
    block_size: int = 256  # context length in tokens
    n_layer: int = 6
    n_head: int = 6
    n_kv_head: int = 2  # must divide n_head; equal to n_head means plain MHA
    n_embd: int = 384
    dropout: float = 0.0
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd {self.n_embd} must divide by n_head {self.n_head}")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(f"n_head {self.n_head} must divide by n_kv_head {self.n_kv_head}")
        if self.block_size < 1 or self.vocab_size < 1:
            raise ValueError("block_size and vocab_size must be positive")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    """x / rms(x) * gain. No mean subtraction, no bias -- it turns out the
    centering in LayerNorm was never doing much work."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # float32 for the reduction even in mixed precision: this is a
        # normalization, and doing it in bf16 costs real accuracy.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.gain.float()).to(dtype)


def precompute_rope(head_dim: int, max_seq: int, theta: float, device) -> tuple[torch.Tensor, ...]:
    """Cosines and sines for rotary embeddings, computed once and reused.

    Each pair of dimensions in a head gets its own rotation frequency. Low
    frequencies encode coarse position, high frequencies fine position -- the
    same trick as a clock's hour and second hands.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_seq, device=device).float()
    angles = torch.outer(positions, inv_freq)  # (seq, head_dim/2)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate (B, heads, T, head_dim) by the angles for each position."""
    first, second = x.chunk(2, dim=-1)
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return torch.cat([first * cos - second * sin, second * cos + first * sin], dim=-1)


class Attention(nn.Module):
    """Causal self-attention with grouped-query heads and an optional KV cache."""

    def __init__(self, config: NanoConfig) -> None:
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.repeats = config.n_head // config.n_kv_head

        # No biases anywhere: they cost parameters and buy nothing measurable.
        self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        batch, seq, _ = x.shape

        q = self.q_proj(x).view(batch, seq, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            past_k, past_v = cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)

        # Grouped-query attention: every query head reads one of the shared
        # key/value heads. This is the line that shrinks your KV cache.
        if self.repeats > 1:
            k = k.repeat_interleave(self.repeats, dim=1)
            v = v.repeat_interleave(self.repeats, dim=1)

        # is_causal only when we are processing a whole block. With a cache the
        # single new query attends to everything before it, so no mask applies.
        causal = cache is None or seq > 1
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.o_proj(out), new_cache


class SwiGLU(nn.Module):
    """gate(x) * up(x), then project down. The gate lets the layer mute itself."""

    def __init__(self, config: NanoConfig) -> None:
        super().__init__()
        # 8/3 x, rounded to a multiple of 64, so three matrices cost about what
        # two 4x matrices would.
        hidden = int(8 * config.n_embd / 3)
        hidden = 64 * ((hidden + 63) // 64)
        self.gate_proj = nn.Linear(config.n_embd, hidden, bias=False)
        self.up_proj = nn.Linear(config.n_embd, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, config.n_embd, bias=False)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class Block(nn.Module):
    """Pre-norm residual block: normalize, transform, add back.

    Pre-norm (rather than post-norm) is why deep transformers train without
    careful warmup babysitting -- the residual path stays a clean identity.
    """

    def __init__(self, config: NanoConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = Attention(config)
        self.mlp_norm = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x, cos, sin, cache=None):
        attn_out, new_cache = self.attn(self.attn_norm(x), cos, sin, cache)
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x, new_cache


class NanoLM(nn.Module):
    """The whole model."""

    def __init__(self, config: NanoConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: the matrix that turns a token into a vector is the same
        # one that turns a vector back into token scores. Saves a big chunk of
        # parameters and usually helps a small model.
        self.lm_head.weight = self.tok_emb.weight

        cos, sin = precompute_rope(
            config.head_dim, config.block_size, config.rope_theta, torch.device("cpu")
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scale the residual projections down by depth, so the residual stream
        # does not grow without bound as layers stack.
        for name, param in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # -- stats ------------------------------------------------------------

    def num_params(self, embedding: bool = True) -> int:
        total = sum(p.numel() for p in self.parameters())
        if not embedding:
            total -= self.tok_emb.weight.numel()
        return total

    def chinchilla_tokens(self) -> int:
        """Roughly how many training tokens this size of model wants.

        The Chinchilla result: for a fixed compute budget, about 20 tokens per
        parameter is where you should land. Most hobby runs are wildly
        under-trained, which is why they babble.
        """
        return 20 * self.num_params(embedding=False)

    # -- forward ----------------------------------------------------------

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        caches: list | None = None,
        pos: int = 0,
    ):
        _, seq = idx.shape
        if pos + seq > self.config.block_size:
            raise ValueError(
                f"sequence of {pos + seq} exceeds block_size {self.config.block_size}"
            )
        cos = self.rope_cos[pos : pos + seq]
        sin = self.rope_sin[pos : pos + seq]

        x = self.drop(self.tok_emb(idx))
        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = caches[i] if caches is not None else None
            x, updated = block(x, cos, sin, cache)
            new_caches.append(updated)
        x = self.norm(x)

        if targets is None:
            # Only the last position matters for generation; skip the rest of
            # the vocab projection, which is the most expensive layer.
            logits = self.lm_head(x[:, [-1], :])
            return logits, None, new_caches

        logits = self.lm_head(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
        )
        return logits, loss, new_caches

    # -- generation -------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int | None = 40,
        top_p: float | None = 0.95,
        eot_id: int | None = None,
    ) -> torch.Tensor:
        """Sample continuations, reusing the KV cache so cost stays linear."""
        self.eval()
        caches = None
        pos = 0
        feed = idx

        for _ in range(max_new_tokens):
            if pos + feed.size(1) > self.config.block_size:
                # Context is full. Drop the cache and re-read a trailing window:
                # crude, but honest about the limit rather than silently wrong.
                feed = torch.cat([idx[:, -(self.config.block_size - 1) :]], dim=1)
                caches, pos = None, 0

            logits, _, caches = self(feed, caches=caches, pos=pos)
            pos += feed.size(1)
            logits = logits[:, -1, :]

            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    threshold = torch.topk(logits, k, dim=-1).values[:, [-1]]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                if top_p is not None and 0 < top_p < 1:
                    probs = _nucleus(probs, top_p)
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_id], dim=1)
            feed = next_id
            if eot_id is not None and int(next_id.item()) == eot_id:
                break
        return idx


def _nucleus(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Keep the smallest set of tokens whose probability sums past top_p."""
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # Keep everything up to and including the token that crosses the threshold.
    remove = cumulative - sorted_probs > top_p
    sorted_probs[remove] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    return torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
