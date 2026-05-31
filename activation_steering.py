"""
Activation Steering via Climbing Fiber Signal.

Graph connection: Activation-steering (isolated, 14 sources) ↔ Cerebellum/Purkinje cell mechanism.

Hypothesis: Climbing fiber error signal in the cerebellar model is mathematically
equivalent to activation steering — both inject a directional delta into
representational space to correct behavior. This module implements:

1. Activation steering vectors (computed from contrastive pairs)
2. Climbing fiber injection (error-signal weighted update to Purkinje input)
3. Comparison: which corrects OOD behavior faster?
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SteeringVector:
    direction: np.ndarray   # unit vector in activation space
    magnitude: float        # how strongly to steer
    layer: int              # which layer to intervene on


def compute_steering_vector(
    positive_activations: np.ndarray,  # (n_pos, d)
    negative_activations: np.ndarray,  # (n_neg, d)
) -> SteeringVector:
    """
    Contrastive activation steering (Zou et al. style).
    Direction = mean(positive) - mean(negative), normalized.
    """
    delta = positive_activations.mean(axis=0) - negative_activations.mean(axis=0)
    magnitude = float(np.linalg.norm(delta))
    direction = delta / (magnitude + 1e-8)
    return SteeringVector(direction=direction, magnitude=magnitude, layer=0)


def apply_steering(
    activations: np.ndarray,  # (batch, d)
    vector: SteeringVector,
    alpha: float = 1.0,       # steering strength multiplier
) -> np.ndarray:
    """Inject steering vector into activation space."""
    return activations + alpha * vector.magnitude * vector.direction[None, :]


def climbing_fiber_steer(
    granule_activations: np.ndarray,  # (batch, granule_dim)
    error_signal: np.ndarray,         # (batch, output_dim)
    purkinje_weights: np.ndarray,     # (output_dim, granule_dim)
    lr: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Climbing fiber = error-gated weight update to Purkinje layer.
    Returns: (corrected_output, updated_weights)

    This is structurally equivalent to activation steering:
    both compute a direction in representation space and
    scale by error magnitude.
    """
    # Purkinje output before correction
    output = granule_activations @ purkinje_weights.T  # (batch, output_dim)

    # Climbing fiber update (LTD: error reduces synaptic weights)
    delta_w = lr * error_signal.T @ granule_activations  # (output_dim, granule_dim)
    updated_weights = purkinje_weights - delta_w  # LTD = negative update

    corrected_output = granule_activations @ updated_weights.T
    return corrected_output, updated_weights


def equivalence_demo(
    d: int = 64,
    granule_dim: int = 256,
    batch: int = 32,
    rng_seed: int = 42,
) -> dict:
    """
    Show that climbing fiber update direction ≈ activation steering direction.

    Both methods compute: delta ∝ error * representation
    Climbing fiber: delta_W = lr * error^T @ granule
    Steering:       delta_h = alpha * (mean_pos - mean_neg)

    The equivalence: if error_signal = (target - output), then
    climbing fiber weight update projects error onto granule space —
    identical to contrastive steering if pos = target direction, neg = current.
    """
    rng = np.random.default_rng(rng_seed)

    # Simulate granule activations (sparse, like real granule cells)
    granule = rng.standard_normal((batch, granule_dim))
    granule *= (rng.random((batch, granule_dim)) < 0.05)  # sparsity

    # Target vs current output
    purkinje_w = rng.standard_normal((d, granule_dim)) * 0.01
    current_output = granule @ purkinje_w.T  # (batch, d)
    target_output = rng.standard_normal((batch, d))
    error = target_output - current_output

    # --- Climbing fiber update ---
    _, updated_w = climbing_fiber_steer(granule, error, purkinje_w, lr=0.05)
    cf_direction = (updated_w - purkinje_w).mean(axis=0)  # mean direction of weight change
    cf_direction /= np.linalg.norm(cf_direction) + 1e-8

    # --- Activation steering ---
    # Positive = target direction in granule space, negative = current
    pos_acts = target_output @ purkinje_w  # project target back to granule space
    neg_acts = current_output @ purkinje_w
    sv = compute_steering_vector(pos_acts, neg_acts)

    # Cosine similarity between the two "correction directions"
    cos_sim = float(np.dot(cf_direction[:granule_dim], sv.direction) /
                    (np.linalg.norm(cf_direction[:granule_dim]) * np.linalg.norm(sv.direction) + 1e-8))

    return {
        "climbing_fiber_weight_delta_norm": float(np.linalg.norm(updated_w - purkinje_w)),
        "steering_vector_magnitude": sv.magnitude,
        "direction_cosine_similarity": cos_sim,
        "interpretation": (
            "High cosine similarity (>0.5) confirms climbing fiber update "
            "and activation steering are parallel correction mechanisms."
        ),
    }


def ood_correction_comparison(
    n_steps: int = 50,
    d: int = 32,
    granule_dim: int = 128,
    rng_seed: int = 7,
) -> dict:
    """
    Compare steering vs climbing fiber at correcting OOD error.
    Task: map ID distribution inputs to correct outputs.
    OOD: shifted input distribution. Measure error reduction rate.
    """
    rng = np.random.default_rng(rng_seed)

    # Shared setup
    true_w = rng.standard_normal((d, granule_dim)) * 0.1

    def make_granule(n, shift=0.0):
        x = rng.standard_normal((n, granule_dim)) + shift
        x *= (rng.random((n, granule_dim)) < 0.05)
        return x

    # OOD batch
    ood_granule = make_granule(64, shift=2.0)
    target = ood_granule @ true_w.T

    # Method A: Activation steering
    w_steering = rng.standard_normal((d, granule_dim)) * 0.01
    steering_errors = []
    for _ in range(n_steps):
        out = ood_granule @ w_steering.T
        err = target - out
        # Compute steering vector from error
        sv = compute_steering_vector(target, out)
        # Apply to output space (not weights — pure activation intervention)
        corrected = apply_steering(out, sv, alpha=0.1)
        steering_errors.append(float(np.mean((corrected - target) ** 2)))

    # Method B: Climbing fiber (weight update)
    w_cf = rng.standard_normal((d, granule_dim)) * 0.01
    cf_errors = []
    for _ in range(n_steps):
        out = ood_granule @ w_cf.T
        err = target - out
        _, w_cf = climbing_fiber_steer(ood_granule, err, w_cf, lr=0.01)
        cf_errors.append(float(np.mean((ood_granule @ w_cf.T - target) ** 2)))

    return {
        "steering_final_error": steering_errors[-1],
        "climbing_fiber_final_error": cf_errors[-1],
        "steering_error_reduction": (steering_errors[0] - steering_errors[-1]) / (steering_errors[0] + 1e-8),
        "cf_error_reduction": (cf_errors[0] - cf_errors[-1]) / (cf_errors[0] + 1e-8),
        "steering_errors": steering_errors[::10],
        "cf_errors": cf_errors[::10],
    }
