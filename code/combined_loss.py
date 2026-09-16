"""
Reconstructed combined loss for the Nishizawa et al. pCR reproduction.

The paper reports:
    Total loss = 0.5 * BCE loss + 0.5 * Focal loss

The released repository imports combined_loss but does not include its
implementation, and the publication does not report numerical alpha/gamma.
For this reproduction we use standard focal-loss settings:
    alpha = 0.25
    gamma = 2.0

These values should be documented as reproduction assumptions, not as
verified original hyperparameters.
"""

import torch
import torch.nn.functional as F


def focal_loss_with_logits(logits, targets, alpha=0.25, gamma=2.0):
    """
    Binary focal loss operating directly on logits.

    Args:
        logits:  Tensor of raw model outputs.
        targets: Tensor of binary labels (0/1), same shape as logits.
        alpha:   Weight for the positive class.
        gamma:   Focusing parameter.

    Returns:
        Scalar mean focal loss.
    """
    targets = targets.float()
    logits = logits.float()

    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )

    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)

    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)

    focal = alpha_t * ((1.0 - p_t) ** gamma) * bce
    return focal.mean()


def combined_loss(logits, targets, alpha=0.25, gamma=2.0):
    """
    Equal-weight combination reported in the paper:
        0.5 * BCE + 0.5 * Focal Loss
    """
    targets = targets.float()
    logits = logits.float()

    bce = F.binary_cross_entropy_with_logits(logits, targets)
    focal = focal_loss_with_logits(
        logits, targets, alpha=alpha, gamma=gamma
    )

    return 0.5 * bce + 0.5 * focal


if __name__ == "__main__":
    # Minimal smoke test
    logits = torch.tensor([0.2, -0.5, 1.0, -1.0])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

    loss = combined_loss(logits, targets)
    print("combined_loss:", float(loss))
    assert torch.isfinite(loss)
    print("combined_loss.py smoke test passed.")
