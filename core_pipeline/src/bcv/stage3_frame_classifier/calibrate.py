"""Temperature scaling — fit one scalar T so p = sigmoid(logit / T) is calibrated.

``pos_weight``/focal training leave raw sigmoids miscalibrated, and Stage-4 hysteresis
assumes calibrated probabilities, so we fit T on held-out logits before windowing.
"""

from __future__ import annotations

import numpy as np
import torch


def fit_temperature(logits: np.ndarray, labels: np.ndarray, *, max_iter: int = 200) -> float:
    """Return the temperature minimizing val NLL (>=1 softens, <1 sharpens)."""
    lg = torch.as_tensor(logits, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)  # optimise log T for positivity
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lg / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    # Clamp to a sane range: a degenerate fit (tiny/uninformative val) can blow T up and
    # flatten every probability to 0.5, destroying the signal Stage 4 needs.
    return float(min(max(log_t.exp().item(), 0.25), 10.0))
