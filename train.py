"""Training for baseline LM and cerebellar-augmented LM."""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import MiniGPT, CerebellarLLM
from cerebellum import CerebellarModule


def train_base_model(
    model: MiniGPT,
    train_loader: DataLoader,
    n_epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
) -> dict:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"loss": [], "ppl": []}

    for epoch in tqdm(range(n_epochs), desc="Training base LM"):
        epoch_loss = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, _ = model(xb)
            B, T, V = logits.shape
            loss = criterion(logits.reshape(B * T, V), yb.reshape(B * T))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss.append(loss.item())

        mean_loss = np.mean(epoch_loss)
        history["loss"].append(mean_loss)
        history["ppl"].append(float(np.exp(min(mean_loss, 20))))

    return history


def train_cerebellar(
    cerebellar_llm: CerebellarLLM,
    train_loader: DataLoader,
    n_epochs: int = 10,
    device: str = "cpu",
) -> dict:
    """
    Train the cerebellar module using the climbing-fiber error signal.
    Base model weights are FROZEN — only cerebellum adapts.
    """
    cerebellar_llm = cerebellar_llm.to(device)
    # Freeze base
    for p in cerebellar_llm.base.parameters():
        p.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    history = {"corrected_loss": [], "base_loss": [], "cerebellum_error": []}

    for epoch in tqdm(range(n_epochs), desc="Training cerebellum"):
        cl_epoch, bl_epoch, ce_epoch = [], [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            B, T = xb.shape

            with torch.no_grad():
                base_logits, hidden = cerebellar_llm.base(xb)

            # Cerebellar forward (only correction is differentiable)
            ctx = hidden.reshape(B * T, -1)
            correction, granule = cerebellar_llm.cerebellum(ctx)
            correction = correction.reshape(B, T, -1)
            corrected_logits = base_logits + cerebellar_llm.correction_scale * correction

            corrected_loss = criterion(
                corrected_logits.reshape(B * T, -1), yb.reshape(B * T)
            )
            base_loss = criterion(
                base_logits.reshape(B * T, -1), yb.reshape(B * T)
            )

            # Climbing fiber error: difference between one-hot target and base prediction
            with torch.no_grad():
                one_hot = torch.zeros_like(base_logits.reshape(B * T, -1))
                one_hot.scatter_(1, yb.reshape(B * T, 1), 1.0)
                error_signal = one_hot - torch.softmax(base_logits.reshape(B * T, -1), dim=-1)

            # Apply cerebellar learning rule
            cerebellar_llm.cerebellum.apply_climbing_fiber(granule, error_signal)

            cl_epoch.append(corrected_loss.item())
            bl_epoch.append(base_loss.item())
            ce_epoch.append(cerebellar_llm.cerebellum.mean_error)

        history["corrected_loss"].append(np.mean(cl_epoch))
        history["base_loss"].append(np.mean(bl_epoch))
        history["cerebellum_error"].append(np.mean(ce_epoch))

    return history


def evaluate_model(
    model,
    loader: DataLoader,
    device: str = "cpu",
    use_correction: bool = False,
) -> dict:
    criterion = nn.CrossEntropyLoss()
    losses = []
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            B, T = xb.shape

            if use_correction and isinstance(model, CerebellarLLM):
                logits, _, _ = model(xb, apply_correction=True)
            elif isinstance(model, CerebellarLLM):
                logits, _, _ = model(xb, apply_correction=False)
            else:
                logits, _ = model(xb)

            loss = criterion(logits.reshape(B * T, -1), yb.reshape(B * T))
            losses.append(loss.item())

            preds = logits.argmax(dim=-1)
            correct += (preds == yb).float().sum().item()
            total += B * T

    mean_loss = float(np.mean(losses))
    return {
        "loss": mean_loss,
        "perplexity": float(np.exp(min(mean_loss, 20))),
        "accuracy": float(correct / total),
    }
