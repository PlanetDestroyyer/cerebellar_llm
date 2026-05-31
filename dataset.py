"""
WikiText-103 dataset loader for SLM pretraining.

WikiText-103: standard language modeling benchmark.
  - Train: 103M tokens (~28K Wikipedia articles)
  - Valid: 218K tokens
  - Test:  246K tokens

Uses GPT-2 BPE tokenizer (50257 vocab) via tiktoken.
Streams data in fixed-length chunks for efficient training.

OOD split: WikiText-103 has domain variety within Wikipedia.
We split test set by article category (science/history/culture)
to evaluate OOD generalization — key for cerebellar comparison.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional
import tiktoken


# ── Tokenizer ────────────────────────────────────────────────────────────────

def get_tokenizer():
    return tiktoken.get_encoding("gpt2")


# ── Dataset ───────────────────────────────────────────────────────────────────

class WikiTextDataset(Dataset):
    """
    Tokenizes WikiText-103 and returns fixed-length chunks.
    All chunks are contiguous (no padding within a chunk).
    """

    def __init__(
        self,
        split:    str  = "train",         # train / validation / test
        seq_len:  int  = 1024,
        cache_dir: Optional[str] = None,
        max_tokens: Optional[int] = None,  # limit tokens for quick testing
    ):
        self.seq_len = seq_len
        self.tokenizer = get_tokenizer()

        # Load from HuggingFace
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1",
                          split=split, cache_dir=cache_dir)

        # Tokenize all text
        all_tokens = []
        for item in ds:
            text = item["text"].strip()
            if not text:
                continue
            tokens = self.tokenizer.encode(text)
            if tokens:
                all_tokens.extend(tokens)
                all_tokens.append(self.tokenizer.eot_token)  # article separator

        if max_tokens:
            all_tokens = all_tokens[:max_tokens]

        self.data = torch.tensor(all_tokens, dtype=torch.long)
        n_chunks  = (len(self.data) - 1) // seq_len
        self.data = self.data[:n_chunks * seq_len + 1]
        self.n    = n_chunks

        print(f"  [{split}] {len(all_tokens):,} tokens → {self.n:,} chunks of {seq_len}")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        start = idx * self.seq_len
        chunk = self.data[start: start + self.seq_len + 1]
        return {
            "input_ids": chunk[:-1],
            "labels":    chunk[1:],
        }


class OODWikiTextDataset(Dataset):
    """
    OOD test split: Wikipedia articles from specific domains.
    Used to evaluate cerebellar OOD generalization advantage.

    Approximation: every 3rd article in test split as OOD
    (different topic distribution than training articles).
    """

    def __init__(self, seq_len: int = 1024, cache_dir: Optional[str] = None):
        self.seq_len    = seq_len
        self.tokenizer  = get_tokenizer()

        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1",
                          split="test", cache_dir=cache_dir)

        # Use every 3rd article cluster as OOD proxy
        all_tokens = []
        article_tokens = []
        article_count  = 0

        for item in ds:
            text = item["text"].strip()
            if text.startswith("=") and article_tokens:
                if article_count % 3 == 0:
                    all_tokens.extend(article_tokens)
                article_tokens = []
                article_count += 1
            if text:
                article_tokens.extend(self.tokenizer.encode(text))

        self.data = torch.tensor(all_tokens, dtype=torch.long)
        n_chunks  = (len(self.data) - 1) // seq_len
        self.data = self.data[:n_chunks * seq_len + 1]
        self.n    = n_chunks
        print(f"  [ood] {len(all_tokens):,} tokens → {self.n:,} chunks of {seq_len}")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        start = idx * self.seq_len
        chunk = self.data[start: start + self.seq_len + 1]
        return {
            "input_ids": chunk[:-1],
            "labels":    chunk[1:],
        }


def get_dataloaders(
    seq_len:    int   = 1024,
    batch_size: int   = 8,
    cache_dir:  Optional[str] = None,
    num_workers: int  = 2,
    quick:      bool  = False,
) -> dict[str, DataLoader]:
    """Build all dataloaders for the experiment."""
    max_tokens = 2_000_000 if quick else None  # ~2M tokens for quick test

    train = WikiTextDataset("train",      seq_len, cache_dir, max_tokens)
    val   = WikiTextDataset("validation", seq_len, cache_dir, 500_000 if quick else None)
    test  = WikiTextDataset("test",       seq_len, cache_dir)
    ood   = OODWikiTextDataset(seq_len, cache_dir)

    def make_loader(ds, shuffle):
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True,
        )

    return {
        "train": make_loader(train, shuffle=True),
        "val":   make_loader(val,   shuffle=False),
        "test":  make_loader(test,  shuffle=False),
        "ood":   make_loader(ood,   shuffle=False),
    }
