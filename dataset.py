"""
Synthetic language dataset for cerebellar LLM experiments.

Two distributions:
1. In-distribution (ID): simple structured sequences
2. Out-of-distribution (OOD): same structure with shifted patterns

Tests: does cerebellar correction improve OOD generalization?
"""
import numpy as np
import torch
from torch.utils.data import Dataset


def make_vocab(size: int = 100) -> dict:
    """Simple token vocabulary."""
    vocab = {f"t{i}": i for i in range(size)}
    vocab["<pad>"] = size - 1
    return vocab


class SequenceDataset(Dataset):
    """
    Structured sequence dataset.
    Pattern: repeating arithmetic progressions with noise.
    ID: step size 1-3, OOD: step size 4-6.
    """

    def __init__(
        self,
        n_sequences: int = 1000,
        seq_len: int = 32,
        vocab_size: int = 100,
        ood: bool = False,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        sequences = []
        for _ in range(n_sequences):
            if ood:
                step = rng.integers(4, 7)
            else:
                step = rng.integers(1, 4)
            start = rng.integers(0, vocab_size // 2)
            seq = [(start + i * step) % (vocab_size - 1) for i in range(seq_len + 1)]
            sequences.append(seq)

        self.data = torch.tensor(sequences, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.data[idx]
        return seq[:-1], seq[1:]  # input, target
