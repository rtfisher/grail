"""Derivatives of the actual neural population adapter in physical units."""

from __future__ import annotations

import torch


def apnn_jvp(adapter, J_bar, T, n_e, tangent):
    """Differentiate the deployed adapter, including its log-input chain rule.

    This computes the neural derivative, never the reference rate-equation
    derivative. Casts retain the deployed model's precision; no weights change.
    """
    with torch.no_grad():
        _, result = torch.func.jvp(
            lambda J: adapter(J, T, n_e), (J_bar,), (tangent,)
        )
    return result.detach()


def apnn_fd_jvp(adapter, J_bar, T, n_e, tangent, h=1e-2):
    """Independent central-difference check with a relative intensity step.

    Choose s so max(abs(s*tangent/J_bar)) = h. This keeps both samples
    positive for 0 < h < 1 and removes the dependence of the stencil on the
    tangent's arbitrary magnitude. It is a check/fallback, not the default
    Krylov operator. Float32 networks require a step-convergence study.
    """
    if not 0 < h < 1:
        raise ValueError("h must be a relative step strictly between zero and one")
    if bool((J_bar <= 0).any()):
        raise ValueError("finite-difference base intensities must be positive")
    scale = (tangent / J_bar).abs().max()
    if float(scale) == 0:
        return torch.zeros_like(adapter(J_bar, T, n_e))
    step = h / scale
    with torch.no_grad():
        plus = adapter(J_bar + step * tangent, T, n_e)
        minus = adapter(J_bar - step * tangent, T, n_e)
    return (plus - minus) / (2 * step)
