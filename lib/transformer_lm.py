from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _logits_to_probs(
        logits: torch.Tensor,  # (B, V)
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
) -> torch.Tensor:
    logits = logits / max(1e-6, temperature)

    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)

        # Smallest set with cumulative prob >= top_p; keep at least one token
        cutoff = (cumprobs > top_p).float()
        cutoff[:, 0] = 0.0
        mask = cutoff.cumsum(dim=-1) > 0
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        logits = torch.zeros_like(logits).scatter(-1, sorted_idx, sorted_logits)

    return F.softmax(logits, dim=-1)


def _init_weights(module: nn.Module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    if isinstance(module, nn.Linear) and module.bias is not None:
        nn.init.zeros_(module.bias)


class TransformerLM(nn.Module):
    """
    Decoder-only LM using nn.TransformerEncoder with:
      - Causal masking
      - Optional block-diagonal masking from `reset_mask` to prevent cross-segment attention
      - Weight tying (token embedding <-> LM head)
      - Simple sampling (no KV cache)

    Args:
        vocab_size: tokenizer vocab size
        d_model: model width
        n_layers: number of transformer layers
        n_heads: attention heads (d_model must be divisible by n_heads)
        max_seq_len: max positions supported by learned positional embeddings
        dropout: dropout prob
        tie_weights: tie token embedding to output projection
        ff_mult: expansion factor for feed-forward (dim = ff_mult * d_model)
        norm_first: use Pre-LN if True
        activation: "relu" | "gelu" | "gelu(approx)" etc.

    Notes:
        - `reset_mask[b, t] == True` means position t starts a new segment;
          tokens cannot attend to tokens from *earlier* segments.
        - Set `max_seq_len` >= (max training window) and also >= (prompt_len + max_new_tokens) for generation.
    """

    def __init__(
            self,
            vocab_size: int,
            d_model: int = 768,
            n_layers: int = 12,
            n_heads: int = 12,
            max_seq_len: int = 4096,
            dropout: float = 0.1,
            tie_weights: bool = True,
            ff_mult: int = 4,
            norm_first: bool = True,
            activation: str = "gelu",
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        # Transformer encoder used with causal mask
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        self.tr = nn.TransformerEncoder(
            enc_layer,
            num_layers=n_layers,
            enable_nested_tensor=False
        )

        # LM head (tied by default)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        # Cache for causal masks
        self._mask_cache: Dict[Tuple[torch.device, torch.dtype, int], torch.Tensor] = {}

        self.apply(_init_weights)

    def _causal_mask_TT(self, T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device, dtype, T)
        if key in self._mask_cache:
            return self._mask_cache[key]
        m = torch.full((T, T), float("-inf"), device=device, dtype=dtype)
        m = torch.triu(m, diagonal=1)
        self._mask_cache[key] = m
        return m

    def _segmented_causal_mask_BTT(self, reset_mask: torch.BoolTensor, dtype: torch.dtype) -> torch.Tensor:
        assert reset_mask.dim() == 2
        B, T = reset_mask.shape
        device = reset_mask.device

        seg = torch.cumsum(reset_mask.to(torch.int32), dim=1)  # (B, T) segment ids
        same_seg = (seg[:, :, None] == seg[:, None, :])  # (B, T, T) bool

        causal_TT = self._causal_mask_TT(T, device, dtype)  # (T, T)
        causal_BTT = causal_TT.unsqueeze(0).expand(B, -1, -1)  # (B, T, T)

        zero = torch.zeros((), dtype=dtype, device=device)
        ninf = torch.full((), float("-inf"), dtype=dtype, device=device)
        block_cross = torch.where(same_seg, zero, ninf)  # (B, T, T)

        return causal_BTT + block_cross

    def forward(
            self,
            input_ids: torch.LongTensor,  # (B, T)
            reset_mask: Optional[torch.BoolTensor] = None,  # (B, T) or None
    ) -> Tuple[torch.Tensor, None]:
        B, T = input_ids.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds max_seq_len={self.max_seq_len}. "
                f"Increase max_seq_len when instantiating TransformerLM."
            )

        device = input_ids.device
        pos = torch.arange(T, device=device).unsqueeze(0)  # (1, T)

        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)

        if reset_mask is None:
            attn_mask = self._causal_mask_TT(T, device, x.dtype)  # (T, T)
        else:
            if reset_mask.shape != (B, T):
                raise ValueError(f"reset_mask must be shape {(B, T)}, got {tuple(reset_mask.shape)}")
            attn_mask = self._segmented_causal_mask_BTT(reset_mask, x.dtype)  # (B, T, T)

        try:
            x = self.tr(x, mask=attn_mask)
        except RuntimeError:
            if attn_mask.dim() == 3:
                outs = []
                for b in range(B):
                    outs.append(self.tr(x[b:b + 1], mask=attn_mask[b]))
                x = torch.cat(outs, dim=0)
            else:
                raise

        logits = self.lm_head(x)  # (B, T, V)
        return logits, None

    @torch.no_grad()
    def generate(
            self,
            input_ids: torch.Tensor,
            max_new_tokens: int = 128,
            temperature: float = 1.0,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            eos_id: Optional[int] = None,
    ) -> Tensor:
        self.eval()
        out = input_ids.clone()

        for _ in range(max_new_tokens):
            if out.size(1) > self.max_seq_len:
                raise ValueError(
                    f"Sequence length {out.size(1)} exceeds max_seq_len={self.max_seq_len} during generation. "
                    f"Re-instantiate TransformerLM with larger max_seq_len."
                )

            logits, _ = self(out, reset_mask=None)
            next_logits = logits[:, -1, :]

            if temperature <= 0:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                probs = _logits_to_probs(next_logits, temperature=temperature, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(probs, num_samples=1)

            out = torch.cat([out, next_token], dim=1)

            if eos_id is not None and torch.all(next_token.squeeze(1) == eos_id):
                break

        return out
