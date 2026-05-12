from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"
SPECIAL = [PAD, BOS, EOS, UNK]


@dataclass
class DinoBatch:
    x: torch.Tensor
    y: torch.Tensor
    lengths: torch.Tensor


class DinoDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, lengths: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
        self.lengths = torch.tensor(lengths, dtype=torch.long)

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.lengths[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_name(name: str) -> str:
    return str(name).strip().lower()


def load_names(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path, header=None)
    names = [normalize_name(v) for v in df.iloc[:, 0].tolist()]
    return [n for n in names if n]


def build_vocab(names: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = sorted({ch for n in names for ch in n})
    vocab = SPECIAL + chars
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode_sequence(name: str, stoi: Dict[str, int], max_len: int) -> Tuple[List[int], List[int], int]:
    seq = [BOS] + list(name) + [EOS]
    ids = [stoi.get(tok, stoi[UNK]) for tok in seq]

    if len(ids) < 2:
        ids = [stoi[BOS], stoi[EOS]]

    x = ids[:-1]
    y = ids[1:]
    length = min(len(x), max_len)

    x = x[:max_len]
    y = y[:max_len]

    pad_id = stoi[PAD]
    if len(x) < max_len:
        x.extend([pad_id] * (max_len - len(x)))
        y.extend([pad_id] * (max_len - len(y)))

    return x, y, length


def make_dataset(names: List[str], stoi: Dict[str, int], max_len: int) -> DinoDataset:
    xs, ys, lens = [], [], []
    for n in names:
        x, y, l = encode_sequence(n, stoi, max_len)
        xs.append(x)
        ys.append(y)
        lens.append(l)
    return DinoDataset(np.array(xs), np.array(ys), np.array(lens))


def save_vocab(stoi: Dict[str, int], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stoi, f, ensure_ascii=False, indent=2)


def load_vocab(path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(path, "r", encoding="utf-8") as f:
        stoi = json.load(f)
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos
