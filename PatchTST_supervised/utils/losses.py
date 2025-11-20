import torch
import torch.nn as nn


def huber_loss(residual, delta=1.0):
    abs_res = residual.abs()
    quadratic = 0.5 * abs_res ** 2
    linear = delta * (abs_res - 0.5 * delta)
    return torch.where(abs_res <= delta, quadratic, linear)


class NoiseAdaptiveHybridHuber(nn.Module):
    def __init__(self, delta=1.0, gamma=1.0, eps=1e-6):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.eps = eps

    def forward(self, y_pred, y_true, x_for_volatility=None):
        """
        Args:
            y_pred, y_true: tensors shaped [B, L, C]
            x_for_volatility: optional tensor used to estimate per-sample volatility
                               Accepts shapes [B, L, C], [B, L], or [B, C].
                               Defaults to ``y_true`` when not provided.
        """
        if x_for_volatility is None:
            x_for_volatility = y_true

        B = x_for_volatility.size(0)
        std_per_sample = x_for_volatility.view(B, -1).std(dim=1)

        s_ref = std_per_sample.mean().detach() + self.eps
        rel_std = std_per_sample / s_ref

        w = 1.0 / (1.0 + self.gamma * rel_std)
        w = w.view(B, 1, 1)

        residual = y_pred - y_true
        mse_term = 0.5 * residual ** 2
        huber_term = huber_loss(residual, delta=self.delta)

        loss = w * mse_term + (1.0 - w) * huber_term
        return loss.mean()
