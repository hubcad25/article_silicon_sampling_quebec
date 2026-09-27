from __future__ import annotations

import numpy as np
import pytest

from article_silicon_sampling_quebec.conditional_logit import (
    ChoiceData,
    fit_conditional_logit,
    negative_log_likelihood,
    probabilities,
)


def test_probabilities_sum_to_one_with_variable_choice_sets():
    features = np.array([[1.0], [0.0], [-1.0], [2.0], [0.0]])
    offsets = np.array([0, 3, 5])

    result = probabilities(features, offsets, np.array([1.0]))

    assert result[:3].sum() == pytest.approx(1.0)
    assert result[3:].sum() == pytest.approx(1.0)
    assert result[0] > result[1] > result[2]


def test_fit_learns_to_prefer_the_positive_feature():
    features = np.tile(np.array([[1.0], [-1.0]]), (20, 1))
    offsets = np.arange(0, 41, 2)
    chosen = np.zeros(20, dtype=int)
    train = ChoiceData(features, offsets, chosen)
    validation = ChoiceData(features.copy(), offsets.copy(), chosen.copy())

    best, fits = fit_conditional_logit(train, validation, (0.01, 1.0))

    assert best.coefficients[0] > 0
    assert negative_log_likelihood(validation, best.coefficients) < np.log(2)
    assert len(fits) == 2


def test_choice_data_rejects_invalid_chosen_index():
    with pytest.raises(ValueError, match="outside its choice set"):
        ChoiceData(np.ones((2, 1)), np.array([0, 2]), np.array([2]))


def test_fit_rejects_non_finite_penalty():
    data = ChoiceData(np.array([[1.0], [0.0]]), np.array([0, 2]), np.array([0]))
    with pytest.raises(ValueError, match="finite positive"):
        fit_conditional_logit(data, data, (np.nan,))
