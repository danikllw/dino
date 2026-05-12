from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import torch

from .config import TrainConfig
from .data import load_vocab
from .model import CharRNNDecoder
from .sample import generate_name


def run_grid(
    temperatures: list[float],
    top_ks: list[int],
    top_ps: list[float],
    n_per_setup: int,
    max_len: int,
    artifact_dir: str,
):
    art = Path(artifact_dir)
    vocab_path = art / "models" / "vocab.json"
    model_path = art / "models" / "char_decoder.pt"

    stoi, itos = load_vocab(str(vocab_path))
    cfg = TrainConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CharRNNDecoder(
        vocab_size=len(stoi),
        embedding_dim=cfg.embedding_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        rnn_type=cfg.model_type,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = []
    for temp, k, p in product(temperatures, top_ks, top_ps):
        names = []
        for _ in range(n_per_setup):
            nm = generate_name(model, stoi, itos, max_len=max_len, temperature=temp, top_k=k, top_p=p, device=device)
            if nm:
                names.append(nm)
        results.append({"temperature": temp, "top_k": k, "top_p": p, "names": names})

    out = art / "samples" / "sampling_experiments.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"runs": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {out}")


def parse_list(raw: str, cast):
    return [cast(v.strip()) for v in raw.split(",") if v.strip()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--temperatures", default="0.7,1.0,2.5,4.0")
    ap.add_argument("--top-ks", default="0,10,20")
    ap.add_argument("--top-ps", default="1.0,0.95,0.9")
    ap.add_argument("--n-per-setup", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=32)
    ap.add_argument("--artifact-dir", default="artifacts")
    args = ap.parse_args()

    run_grid(
        temperatures=parse_list(args.temperatures, float),
        top_ks=parse_list(args.top_ks, int),
        top_ps=parse_list(args.top_ps, float),
        n_per_setup=args.n_per_setup,
        max_len=args.max_len,
        artifact_dir=args.artifact_dir,
    )
