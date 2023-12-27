from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(
        -log_timescale_increment * torch.arange(channels // 2)
    )
    scaled_time = (
        torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    )
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttentionWithCache(nn.Module):
    def __init__(self, n_ctx: int, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)
        self.n_ctx = n_ctx

    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor):
        _, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k

        mask = torch.triu(torch.ones(n_ctx, n_ctx), diagonal=1).bool()
        padded_tensor = torch.zeros(qk.shape[2:], dtype=torch.bool)
        padded_tensor[:, : mask.shape[1]] = mask
        qk[:, :] = qk[:, :].masked_fill_(padded_tensor, -np.inf)

        w = F.softmax(qk.float(), dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)

    def forward(
        self,
        x: Tensor,
        k_cache: Optional[Tensor] = None,
        v_cache: Optional[Tensor] = None,
    ):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        if k_cache is not None and v_cache is not None:
            k = torch.concat([k_cache, k], dim=1)
            v = torch.concat([v_cache, v], dim=1)

        wv = self.qkv_attention(q, k, v)
        return self.out(wv), k, v


class MultiHeadAttention(nn.Module):
    layer_count = 0

    def __init__(self, n_ctx: int, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        offset: Optional[Tensor] = None,
    ):
        q = self.query(x)

        k = self.key(x if xa is None else xa)
        v = self.value(x if xa is None else xa)
        wv = self.qkv_attention(q, k, v, mask)
        return self.out(wv)

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        _, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        _, _, n_ctx, _ = qk.shape
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]

        w = F.softmax(qk.float(), dim=-1).to(q.dtype)

        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        n_ctx: int,
        n_state: int,
        n_head: int,
        cross_attention: bool = False,
    ):
        super().__init__()

        self.attn = MultiHeadAttentionWithCache(n_ctx, n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_ctx, n_state, n_head)
            if cross_attention
            else None
        )
        self.cross_attn_ln = nn.LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_mlp), nn.GELU(), nn.Linear(n_mlp, n_state)
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
    ):
        memory, k, v = self.attn(self.attn_ln(x), k_cache=k, v_cache=v)
        x = x + memory
        if self.cross_attn and self.cross_attn_ln:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa)
        x = x + self.mlp(self.mlp_ln(x))
        return x, k, v


class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(
            n_state, n_state, kernel_size=3, stride=2, padding=1
        )
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks = nn.ModuleList(
            [
                ResidualAttentionBlock(n_ctx, n_state, n_head)
                for _ in range(n_layer)
            ]
        )
        self.ln_post = nn.LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    n_ctx, n_state, n_head, cross_attention=True
                )
                for _ in range(n_layer)
            ]
        )
        self.ln = nn.LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Tensor,
        pos_emb: Tensor,
        k1: Optional[Tensor] = None,
        v1: Optional[Tensor] = None,
        k2: Optional[Tensor] = None,
        v2: Optional[Tensor] = None,
        k3: Optional[Tensor] = None,
        v3: Optional[Tensor] = None,
        k4: Optional[Tensor] = None,
        v4: Optional[Tensor] = None,
        k5: Optional[Tensor] = None,
        v5: Optional[Tensor] = None,
        k6: Optional[Tensor] = None,
        v6: Optional[Tensor] = None,
    ):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        # minus one because we pre allocate kv_cache
        x = self.token_embedding(x) + pos_emb
        x = x.to(xa.dtype)

        x, k1, v1 = self.blocks[0](x, xa, k=k1, v=v1)
        x, k2, v2 = self.blocks[1](x, xa, k=k2, v=v2)
        x, k3, v3 = self.blocks[2](x, xa, k=k3, v=v3)
        x, k4, v4 = self.blocks[3](x, xa, k=k4, v=v4)
        x, k5, v5 = self.blocks[4](x, xa, k=k5, v=v5)
        x, k6, v6 = self.blocks[5](x, xa, k=k6, v=v6)

        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits, k1, v1, k2, v2, k3, v3, k4, v4, k5, v5, k6, v6
