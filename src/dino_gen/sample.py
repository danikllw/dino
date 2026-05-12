from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import BOS, EOS, PAD


def _top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    logits = logits.clone()

    if top_k > 0:
        k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, k)
        cutoff = values[..., -1, None]
        logits[logits < cutoff] = -float("inf")

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)

        mask = cumulative > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False

        sorted_logits[mask] = -float("inf")
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    return logits


def generate_name(
    model,
    stoi,
    itos,
    max_len: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    device: str = "cpu",
) -> str:
    model.eval()
    bos = stoi[BOS]
    eos = stoi[EOS]
    pad = stoi[PAD]

    tokens = [bos]
    hidden = None

    with torch.no_grad():
        for _ in range(max_len):
            x = torch.tensor([[tokens[-1]]], dtype=torch.long, device=device)
            logits, hidden = model(x, hidden)
            step = logits[0, -1] / max(temperature, 1e-6)

            step[pad] = -float("inf")
            step[bos] = -float("inf")

            step = _top_k_top_p_filter(step, top_k=top_k, top_p=top_p)
            probs = F.softmax(step, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()

            if next_id == eos:
                break
            tokens.append(next_id)

    chars = [itos[t] for t in tokens[1:] if t in itos]
    return "".join(chars).strip()
