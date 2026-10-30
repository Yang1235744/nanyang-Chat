"""TinyGPT —— 从零实现的 decoder-only 对话大模型。

pre-LN + 因果自注意力（SDPA）+ 权重共享输出头；generate 支持温度 / top-k / EOS 停止。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..tokenizer import EOS_ID, PAD_ID, VOCAB_SIZE


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.h = heads
        self.n1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.n2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        q = q.view(B, T, self.h, C // self.h).transpose(1, 2)
        k = k.view(B, T, self.h, C // self.h).transpose(1, 2)
        v = v.view(B, T, self.h, C // self.h).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(o.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.n2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE, dim: int = 384, depth: int = 8,
                 heads: int = 6, max_len: int = 256, mlp_ratio: float = 4.0):
        super().__init__()
        self.max_len = max_len
        self.tok = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio) for _ in range(depth)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.tok.weight  # 权重共享
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids: torch.Tensor, targets: torch.Tensor = None):
        B, T = ids.shape
        assert T <= self.max_len, f"序列长度 {T} 超过 max_len={self.max_len}"
        x = self.tok(ids) + self.pos(torch.arange(T, device=ids.device))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=PAD_ID)
        return logits, loss

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, max_new: int = 200, temperature: float = 0.8,
                 top_k: int = 40, stop_id: int = EOS_ID) -> torch.Tensor:
        self.eval()
        for _ in range(max_new):
            logits, _ = self(ids[:, -self.max_len:])
            logits = logits[:, -1, :]
            if temperature <= 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k and top_k > 0:
                    thresh = torch.topk(logits, top_k).values[:, [-1]]
                    logits = logits.masked_fill(logits < thresh, float("-inf"))
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            ids = torch.cat([ids, nxt], dim=1)
            if (nxt == stop_id).all():
                break
        return ids
