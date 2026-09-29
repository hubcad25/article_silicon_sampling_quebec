"""Small residual neural utility model for variable alternative choice sets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProjectedChoiceData:
    """Projected alternatives, frozen base utilities, and observed choices."""

    features: np.ndarray
    base_utilities: np.ndarray
    offsets: np.ndarray
    chosen: np.ndarray

    def __post_init__(self) -> None:
        features = np.asarray(self.features)
        base = np.asarray(self.base_utilities)
        offsets = np.asarray(self.offsets)
        chosen = np.asarray(self.chosen)
        if features.ndim != 2:
            raise ValueError("features must be a two-dimensional array")
        if base.shape != (len(features),):
            raise ValueError("base_utilities must contain one value per alternative")
        if offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0:
            raise ValueError("offsets must start at zero")
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
        if not np.isfinite(features).all() or not np.isfinite(base).all():
            raise ValueError("features and base utilities must be finite")

    @property
    def n_cases(self) -> int:
        return len(self.offsets) - 1

    @property
    def chosen_rows(self) -> np.ndarray:
        return self.offsets[:-1] + self.chosen


@dataclass(frozen=True)
class NeuralChoiceConfig:
    """Fixed optimization and architecture settings."""

    hidden_units: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 20
    min_delta: float = 1e-5
    seed: int = 20_260_929

    def __post_init__(self) -> None:
        if self.hidden_units < 1 or self.batch_size < 1 or self.max_epochs < 1:
            raise ValueError("hidden units, batch size, and epochs must be positive")
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.min_delta < 0:
            raise ValueError("invalid learning rate, weight decay, or minimum delta")


@dataclass(frozen=True)
class NeuralChoiceModel:
    """Parameters of a one-hidden-layer residual utility model."""

    hidden_weights: np.ndarray
    hidden_bias: np.ndarray
    output_weights: np.ndarray

    def residuals(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        hidden = np.tanh(values @ self.hidden_weights + self.hidden_bias)
        return hidden @ self.output_weights

    def utilities(self, data: ProjectedChoiceData) -> np.ndarray:
        return data.base_utilities + self.residuals(data.features)

    def probabilities(self, data: ProjectedChoiceData) -> np.ndarray:
        return grouped_softmax(self.utilities(data), data.offsets)


@dataclass(frozen=True)
class NeuralFitResult:
    """Best model and deterministic validation history."""

    model: NeuralChoiceModel
    selected_epoch: int
    train_nll: float
    validation_nll: float
    baseline_validation_nll: float
    epochs_run: int
    history: tuple[tuple[int, float], ...]


def grouped_softmax(utilities: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Normalize utilities separately within each variable-length choice set."""

    values = np.asarray(utilities)
    boundaries = np.asarray(offsets, dtype=np.int64)
    if values.ndim != 1 or boundaries.ndim != 1 or boundaries[-1] != len(values):
        raise ValueError("utilities and offsets are inconsistent")
    starts = boundaries[:-1]
    maxima = np.maximum.reduceat(values, starts)
    shifted = values - np.repeat(maxima, np.diff(boundaries))
    exponentials = np.exp(shifted)
    denominators = np.add.reduceat(exponentials, starts)
    return exponentials / np.repeat(denominators, np.diff(boundaries))


def negative_log_likelihood(data: ProjectedChoiceData, model: NeuralChoiceModel) -> float:
    """Mean conditional negative log likelihood."""

    probabilities = model.probabilities(data)
    return float(-np.log(np.clip(probabilities[data.chosen_rows], 1e-30, None)).mean())


def _copy_model(parameters: dict[str, np.ndarray]) -> NeuralChoiceModel:
    return NeuralChoiceModel(
        hidden_weights=parameters["hidden_weights"].copy(),
        hidden_bias=parameters["hidden_bias"].copy(),
        output_weights=parameters["output_weights"].copy(),
    )


def _batch(data: ProjectedChoiceData, cases: np.ndarray) -> ProjectedChoiceData:
    lengths = np.diff(data.offsets)[cases]
    rows = np.concatenate([
        np.arange(data.offsets[case], data.offsets[case + 1], dtype=np.int64)
        for case in cases
    ])
    offsets = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(lengths)))
    return ProjectedChoiceData(
        data.features[rows], data.base_utilities[rows], offsets, data.chosen[cases]
    )


def fit_residual_choice_network(
    train: ProjectedChoiceData,
    validation: ProjectedChoiceData,
    config: NeuralChoiceConfig = NeuralChoiceConfig(),
) -> NeuralFitResult:
    """Fit a residual MLP with Adam and validation early stopping."""

    if train.features.shape[1] != validation.features.shape[1]:
        raise ValueError("training and validation feature dimensions differ")
    rng = np.random.default_rng(config.seed)
    input_units = train.features.shape[1]
    limit = np.sqrt(6.0 / (input_units + config.hidden_units))
    parameters = {
        "hidden_weights": rng.uniform(
            -limit, limit, size=(input_units, config.hidden_units)
        ).astype(np.float32),
        "hidden_bias": np.zeros(config.hidden_units, dtype=np.float32),
        "output_weights": np.zeros(config.hidden_units, dtype=np.float32),
    }
    first_moments = {name: np.zeros_like(value) for name, value in parameters.items()}
    second_moments = {name: np.zeros_like(value) for name, value in parameters.items()}
    step = 0

    initial = _copy_model(parameters)
    baseline_validation_nll = negative_log_likelihood(validation, initial)
    best_model = initial
    best_validation_nll = baseline_validation_nll
    selected_epoch = 0
    history: list[tuple[int, float]] = [(0, baseline_validation_nll)]
    stale_epochs = 0

    for epoch in range(1, config.max_epochs + 1):
        order = rng.permutation(train.n_cases)
        for start in range(0, train.n_cases, config.batch_size):
            batch = _batch(train, order[start:start + config.batch_size])
            hidden = np.tanh(
                batch.features @ parameters["hidden_weights"]
                + parameters["hidden_bias"]
            )
            utilities = batch.base_utilities + hidden @ parameters["output_weights"]
            delta = grouped_softmax(utilities, batch.offsets)
            delta[batch.chosen_rows] -= 1.0
            delta /= batch.n_cases

            hidden_gradient = (
                delta[:, None] * parameters["output_weights"][None, :]
            ) * (1.0 - hidden * hidden)
            gradients = {
                "hidden_weights": (
                    batch.features.T @ hidden_gradient
                    + config.weight_decay * parameters["hidden_weights"]
                ),
                "hidden_bias": hidden_gradient.sum(axis=0),
                "output_weights": (
                    hidden.T @ delta
                    + config.weight_decay * parameters["output_weights"]
                ),
            }

            step += 1
            for name in parameters:
                gradient = np.asarray(gradients[name], dtype=np.float32)
                first_moments[name] = 0.9 * first_moments[name] + 0.1 * gradient
                second_moments[name] = (
                    0.999 * second_moments[name] + 0.001 * gradient * gradient
                )
                first_hat = first_moments[name] / (1.0 - 0.9**step)
                second_hat = second_moments[name] / (1.0 - 0.999**step)
                parameters[name] -= (
                    config.learning_rate * first_hat / (np.sqrt(second_hat) + 1e-8)
                )

        candidate = _copy_model(parameters)
        validation_nll = negative_log_likelihood(validation, candidate)
        history.append((epoch, validation_nll))
        if validation_nll < best_validation_nll - config.min_delta:
            best_model = candidate
            best_validation_nll = validation_nll
            selected_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    return NeuralFitResult(
        model=best_model,
        selected_epoch=selected_epoch,
        train_nll=negative_log_likelihood(train, best_model),
        validation_nll=best_validation_nll,
        baseline_validation_nll=baseline_validation_nll,
        epochs_run=len(history) - 1,
        history=tuple(history),
    )


__all__ = [
    "NeuralChoiceConfig",
    "NeuralChoiceModel",
    "NeuralFitResult",
    "ProjectedChoiceData",
    "fit_residual_choice_network",
    "grouped_softmax",
    "negative_log_likelihood",
]
