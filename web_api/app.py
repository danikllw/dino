from __future__ import annotations

from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.dino_gen.config import TrainConfig
from src.dino_gen.data import load_vocab
from src.dino_gen.model import CharRNNDecoder
from src.dino_gen.sample import generate_name

ART = Path("artifacts/models")
MODEL_PATH = ART / "char_decoder.pt"
VOCAB_PATH = ART / "vocab.json"

app = FastAPI(title="Dino Name Generator API")


class GenerateRequest(BaseModel):
    n: int = Field(default=1, ge=1, le=20)
    temperature: float = Field(default=1.0, gt=0.0, le=6.0)
    top_k: int = Field(default=0, ge=0, le=100)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    max_len: int = Field(default=32, ge=4, le=80)


class GenerateResponse(BaseModel):
    names: list[str]


@app.get("/health")
def health():
    return {"ok": True, "model_ready": MODEL_PATH.exists() and VOCAB_PATH.exists()}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if not MODEL_PATH.exists() or not VOCAB_PATH.exists():
        return GenerateResponse(names=[])

    stoi, itos = load_vocab(str(VOCAB_PATH))
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
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    names = []
    for _ in range(req.n):
        nm = generate_name(
            model,
            stoi,
            itos,
            max_len=req.max_len,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            device=device,
        )
        if nm:
            names.append(nm)

    return GenerateResponse(names=names)
