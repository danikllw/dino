from dataclasses import dataclass


@dataclass
class TrainConfig:
    data_path: str = "data/dinos.csv"
    artifact_dir: str = "artifacts"
    model_type: str = "gru"  # rnn | gru | lstm
    embedding_dim: int = 64
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 64
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 42
    max_len: int = 32
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    num_samples: int = 10
