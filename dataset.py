"""
WikiText-103 dataset — Salesforce source on HuggingFace.

HF ID: Salesforce/wikitext  (config: wikitext-103-raw-v1)
  - Train: 103M tokens
  - Validation: 218K tokens
  - Test: 246K tokens

Each model uses its own tokenizer from HuggingFace.
Returns fixed-length chunks for perplexity evaluation and fine-tuning.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional


def load_wikitext(split: str = "train", cache_dir: Optional[str] = None):
    from datasets import load_dataset
    return load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split=split,
        cache_dir=cache_dir,
        trust_remote_code=False,
    )


class WikiTextDataset(Dataset):
    """
    Tokenizes WikiText-103 (Salesforce) into fixed-length chunks.
    Uses the model's own tokenizer for fair comparison.
    """

    def __init__(
        self,
        tokenizer,
        split:      str  = "train",
        seq_len:    int  = 512,
        cache_dir:  Optional[str] = None,
        max_tokens: Optional[int] = None,
    ):
        self.seq_len = seq_len
        ds = load_wikitext(split, cache_dir)

        all_ids = []
        for item in ds:
            text = item["text"].strip()
            if not text:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            if ids:
                all_ids.extend(ids)
                if tokenizer.eos_token_id is not None:
                    all_ids.append(tokenizer.eos_token_id)

        if max_tokens:
            all_ids = all_ids[:max_tokens]

        data    = torch.tensor(all_ids, dtype=torch.long)
        n_full  = (len(data) - 1) // seq_len
        self.data = data[:n_full * seq_len + 1]
        self.n    = n_full
        print(f"  [{split}] {len(all_ids):,} tokens -> {self.n:,} chunks x {seq_len}")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        s = idx * self.seq_len
        chunk = self.data[s: s + self.seq_len + 1]
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}


class OODWikiTextDataset(Dataset):
    """
    OOD subset: every 3rd article cluster from test split.
    Used to measure cerebellar OOD generalization advantage.
    """

    def __init__(self, tokenizer, seq_len: int = 512,
                 cache_dir: Optional[str] = None):
        self.seq_len = seq_len
        ds = load_wikitext("test", cache_dir)

        all_ids, article_ids, count = [], [], 0
        for item in ds:
            text = item["text"].strip()
            if text.startswith("=") and article_ids:
                if count % 3 == 0:
                    all_ids.extend(article_ids)
                article_ids = []
                count += 1
            if text:
                article_ids.extend(
                    tokenizer.encode(text, add_special_tokens=False)
                )

        data   = torch.tensor(all_ids, dtype=torch.long)
        n_full = (len(data) - 1) // seq_len
        self.data = data[:n_full * seq_len + 1]
        self.n    = n_full
        print(f"  [ood] {len(all_ids):,} tokens -> {self.n:,} chunks x {seq_len}")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        s = idx * self.seq_len
        chunk = self.data[s: s + self.seq_len + 1]
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}


def get_dataloaders(tokenizer, seq_len: int = 512, batch_size: int = 8,
                    cache_dir: Optional[str] = None, num_workers: int = 2,
                    quick: bool = False) -> dict[str, DataLoader]:
    max_tok = 1_000_000 if quick else None

    train = WikiTextDataset(tokenizer, "train",      seq_len, cache_dir, max_tok)
    val   = WikiTextDataset(tokenizer, "validation", seq_len, cache_dir,
                            300_000 if quick else None)
    test  = WikiTextDataset(tokenizer, "test",       seq_len, cache_dir)
    ood   = OODWikiTextDataset(tokenizer, seq_len, cache_dir)

    def dl(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)

    return {"train": dl(train, True), "val": dl(val, False),
            "test": dl(test, False), "ood": dl(ood, False)}
