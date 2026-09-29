from __future__ import annotations

import numpy as np
import pytest

from article_silicon_sampling_quebec.neural_choice import (
    NeuralChoiceConfig,
    NeuralChoiceModel,
    ProjectedChoiceData,
    fit_residual_choice_network,
    grouped_softmax,
)


def test_grouped_softmax_normalizes_variable_choice_sets():
    probabilities = grouped_softmax(
        np.array([2.0, 0.0, -1.0, 1.0, 0.0]), np.array([0, 3, 5])
    )

    assert probabilities[:3].sum() == pytest.approx(1.0)
    assert probabilities[3:].sum() == pytest.approx(1.0)
    assert probabilities[0] > probabilities[1] > probabilities[2]


def test_zero_output_weights_reproduce_frozen_base_utilities():
    data = ProjectedChoiceData(
        features=np.array([[1.0], [-1.0]], dtype=np.float32),
        base_utilities=np.array([0.7, -0.2]),
        offsets=np.array([0, 2]),
        chosen=np.array([0]),
    )
    model = NeuralChoiceModel(
        hidden_weights=np.ones((1, 2), dtype=np.float32),
        hidden_bias=np.zeros(2, dtype=np.float32),
        output_weights=np.zeros(2, dtype=np.float32),
    )

    assert model.probabilities(data) == pytest.approx(
        grouped_softmax(data.base_utilities, data.offsets)
    )


def test_fit_learns_residual_preference_and_is_reproducible():
    case_features = np.array([[1.0], [-1.0]], dtype=np.float32)
    features = np.tile(case_features, (40, 1))
    data = ProjectedChoiceData(
        features=features,
        base_utilities=np.zeros(len(features), dtype=np.float32),
        offsets=np.arange(0, len(features) + 1, 2),
        chosen=np.zeros(40, dtype=np.int64),
    )
    config = NeuralChoiceConfig(
        hidden_units=4,
        learning_rate=0.02,
        weight_decay=0.0,
        batch_size=10,
        max_epochs=80,
        patience=10,
        seed=7,
    )

    first = fit_residual_choice_network(data, data, config)
    second = fit_residual_choice_network(data, data, config)

    assert first.selected_epoch > 0
    assert first.validation_nll < 0.1
    assert first.validation_nll == pytest.approx(second.validation_nll)
    assert first.model.probabilities(data) == pytest.approx(
        second.model.probabilities(data)
    )


def test_projected_choice_data_rejects_non_finite_features():
    with pytest.raises(ValueError, match="finite"):
        ProjectedChoiceData(
            features=np.array([[np.nan], [0.0]]),
            base_utilities=np.zeros(2),
            offsets=np.array([0, 2]),
            chosen=np.array([0]),
        )
