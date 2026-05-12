from __future__ import annotations

import json
from pathlib import Path


def main():
    import matplotlib.pyplot as plt

    metrics_path = Path("artifacts/reports/metrics.json")
    out_path = Path("artifacts/reports/learning_curves.png")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    h = metrics["history"]

    plt.figure(figsize=(8, 4))
    plt.plot(h["train_loss"], label="train")
    plt.plot(h["val_loss"], label="val")
    plt.title("Learning Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
