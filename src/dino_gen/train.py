from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

from .config import TrainConfig
from .data import PAD, build_vocab, load_names, make_dataset, save_vocab, set_seed
from .model import CharRNNDecoder
from .sample import generate_name


def _maybe_setup_mlflow(cfg: TrainConfig):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return None
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "dino-char-rnn"))
        run = mlflow.start_run(run_name=f"{cfg.model_type}-char-decoder")
        mlflow.log_params(cfg.__dict__)
        return mlflow, run
    except Exception as e:
        print(f"warning: mlflow disabled due to error: {e}")
        return None


def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    names = load_names(cfg.data_path)
    stoi, itos = build_vocab(names)
    dataset = make_dataset(names, stoi, cfg.max_len)

    n_total = len(dataset)
    n_val = max(1, int(n_total * 0.1))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = CharRNNDecoder(
        vocab_size=len(stoi),
        embedding_dim=cfg.embedding_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        rnn_type=cfg.model_type,
    ).to(device)

    pad_id = stoi[PAD]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optim = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    mlflow_ctx = _maybe_setup_mlflow(cfg)

    for epoch in range(cfg.epochs):
        model.train()
        total_train = 0.0
        for x, y, _ in train_dl:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            total_train += loss.item()

        train_loss = total_train / max(len(train_dl), 1)

        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for x, y, _ in val_dl:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                total_val += loss.item()

        val_loss = total_val / max(len(val_dl), 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if mlflow_ctx is not None:
            mlflow, _ = mlflow_ctx
            mlflow.log_metric("train_loss", train_loss, step=epoch + 1)
            mlflow.log_metric("val_loss", val_loss, step=epoch + 1)

        if val_loss < best_val:
            best_val = val_loss
            Path(cfg.artifact_dir, "models").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), Path(cfg.artifact_dir, "models", "char_decoder.pt"))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch={epoch+1}/{cfg.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    report_dir = Path(cfg.artifact_dir, "reports")
    sample_dir = Path(cfg.artifact_dir, "samples")
    report_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    with open(report_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_loss": best_val,
                "final_train_loss": history["train_loss"][-1],
                "final_val_loss": history["val_loss"][-1],
                "history": history,
                "config": cfg.__dict__,
            },
            f,
            indent=2,
        )

    save_vocab(stoi, str(Path(cfg.artifact_dir, "models", "vocab.json")))

    model.load_state_dict(torch.load(Path(cfg.artifact_dir, "models", "char_decoder.pt"), map_location=device))
    model.eval()

    samples = []
    for _ in range(cfg.num_samples):
        name = generate_name(
            model,
            stoi,
            itos,
            max_len=cfg.max_len,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            device=device,
        )
        if name:
            samples.append(name)

    with open(sample_dir / "baseline_samples.json", "w", encoding="utf-8") as f:
        json.dump({"samples": samples}, f, ensure_ascii=False, indent=2)

    if mlflow_ctx is not None:
        mlflow, _ = mlflow_ctx
        try:
            mlflow.log_metric("best_val_loss", best_val)
            mlflow.log_artifact(str(report_dir / "metrics.json"))
            mlflow.log_artifact(str(sample_dir / "baseline_samples.json"))
            exp_file = sample_dir / "sampling_experiments.json"
            if exp_file.exists():
                mlflow.log_artifact(str(exp_file))
            mlflow.log_artifact(str(Path(cfg.artifact_dir, "models", "vocab.json")))
            mlflow.log_artifact(str(Path(cfg.artifact_dir, "models", "char_decoder.pt")))
        finally:
            mlflow.end_run()

    return best_val, samples


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="data/dinos.csv")
    p.add_argument("--artifact-dir", default="artifacts")
    p.add_argument("--model-type", default="gru")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-len", type=int, default=32)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--num-samples", type=int, default=10)
    args = p.parse_args()
    return TrainConfig(
        data_path=args.data_path,
        artifact_dir=args.artifact_dir,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_len=args.max_len,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    cfg = parse_args()
    best_val, samples = train(cfg)
    print(f"Best val loss: {best_val:.4f}")
    print("Samples:")
    for s in samples:
        print("-", s)
