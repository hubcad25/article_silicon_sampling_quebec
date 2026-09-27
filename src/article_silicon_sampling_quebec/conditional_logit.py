"""Regularized conditional logit over frozen alternative embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class ChoiceData:
    """Flattened alternatives and one selected alternative per choice set."""

    features: np.ndarray
    offsets: np.ndarray
    chosen: np.ndarray

    def __post_init__(self) -> None:
        features = np.asarray(self.features)
        offsets = np.asarray(self.offsets)
        chosen = np.asarray(self.chosen)
        if features.ndim != 2:
            raise ValueError("features must be a two-dimensional array")
        if offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0:
            raise ValueError("offsets must start at zero and contain one boundary per choice set")
        if not np.issubdtype(offsets.dtype, np.integer):
            raise ValueError("offsets must be integers")
        if offsets[-1] != len(features) or np.any(np.diff(offsets) < 2):
            raise ValueError("every choice set must contain at least two alternatives")
        if chosen.shape != (len(offsets) - 1,):
            raise ValueError("chosen must contain one local index per choice set")
        if not np.issubdtype(chosen.dtype, np.integer):
            raise ValueError("chosen indices must be integers")
        if np.any(chosen < 0) or np.any(chosen >= np.diff(offsets)):
            raise ValueError("a chosen index is outside its choice set")
        if not np.isfinite(features).all():
            raise ValueError("features contain non-finite values")

    @property
    def n_cases(self) -> int:
        return len(self.offsets) - 1

    @property
    def chosen_rows(self) -> np.ndarray:
        return self.offsets[:-1] + self.chosen


@dataclass(frozen=True)
class FitResult:
    coefficients: np.ndarray
    penalty: float
    train_nll: float
    validation_nll: float
    converged: bool
    iterations: int


def probabilities(features: np.ndarray, offsets: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Return a softmax probability for every flattened alternative."""
    utilities = np.asarray(features, dtype=np.float64) @ np.asarray(coefficients, dtype=np.float64)
    starts = np.asarray(offsets[:-1], dtype=np.int64)
    maxima = np.maximum.reduceat(utilities, starts)
    shifted = utilities - np.repeat(maxima, np.diff(offsets))
    exponentials = np.exp(shifted)
    denominators = np.add.reduceat(exponentials, starts)
    return exponentials / np.repeat(denominators, np.diff(offsets))


def negative_log_likelihood(data: ChoiceData, coefficients: np.ndarray) -> float:
    probs = probabilities(data.features, data.offsets, coefficients)
    return float(-np.log(np.clip(probs[data.chosen_rows], 1e-300, None)).mean())


def _objective(data: ChoiceData, penalty: float):
    x = np.asarray(data.features, dtype=np.float64)
    chosen_sum = x[data.chosen_rows].sum(axis=0)

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        probs = probabilities(x, data.offsets, coefficients)
        nll = -np.log(np.clip(probs[data.chosen_rows], 1e-300, None)).mean()
        loss = nll + 0.5 * penalty * float(coefficients @ coefficients)
        gradient = (x.T @ probs - chosen_sum) / data.n_cases + penalty * coefficients
        return float(loss), gradient

    return objective


def fit_conditional_logit(
    train: ChoiceData,
    validation: ChoiceData,
    penalties: Sequence[float],
    *,
    max_iter: int = 500,
) -> tuple[FitResult, list[FitResult]]:
    """Select an L2 penalty by validation NLL and return the fitted train-only model."""
    if train.features.shape[1] != validation.features.shape[1]:
        raise ValueError("training and validation feature dimensions differ")
    candidates = tuple(float(value) for value in penalties)
    if not candidates or any(not np.isfinite(value) or value <= 0 for value in candidates):
        raise ValueError("penalties must be a non-empty sequence of finite positive values")
    if len(set(candidates)) != len(candidates):
        raise ValueError("penalties must not contain duplicates")

    fits: list[FitResult] = []
    initial = np.zeros(train.features.shape[1], dtype=np.float64)
    for penalty in candidates:
        result = minimize(
            _objective(train, penalty), initial, method="L-BFGS-B", jac=True,
            options={"maxiter": max_iter, "ftol": 1e-10, "gtol": 1e-6},
        )
        coefficients = np.asarray(result.x, dtype=np.float64)
        fit = FitResult(
            coefficients=coefficients,
            penalty=penalty,
            train_nll=negative_log_likelihood(train, coefficients),
            validation_nll=negative_log_likelihood(validation, coefficients),
            converged=bool(result.success),
            iterations=int(result.nit),
        )
        fits.append(fit)
        initial = coefficients
    converged = [fit for fit in fits if fit.converged]
    if not converged:
        raise RuntimeError("no regularization candidate converged")
    best = min(converged, key=lambda fit: (fit.validation_nll, fit.penalty))
    return best, fits


__all__ = [
    "ChoiceData", "FitResult", "fit_conditional_logit", "negative_log_likelihood",
    "probabilities",
]
