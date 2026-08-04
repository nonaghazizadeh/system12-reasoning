import pytest

from system12.routing import entropy_statistics, reliability_scores, select_reasoning_system


def test_entropy_statistics_use_population_variance():
    mean, variance = entropy_statistics([1.0, 2.0, 3.0])
    assert mean == pytest.approx(2.0)
    assert variance == pytest.approx(2 / 3)


def test_camera_ready_weight_prefers_stable_system():
    selected, scores = select_reasoning_system(
        [0.1, 2.0, 0.1, 2.0],
        [1.0, 1.0, 1.0, 1.0],
        mean_weight=0.4,
    )
    assert selected == "system2"
    assert scores.system2_score < scores.system1_score


def test_zero_statistics_are_normalized_without_division_by_zero():
    scores = reliability_scores([0.0], [0.0])
    assert scores.system1_score == pytest.approx(0.5)
    assert scores.system2_score == pytest.approx(0.5)


def test_ties_are_deterministic():
    selected, _ = select_reasoning_system([1.0, 1.0], [1.0, 1.0])
    assert selected == "system1"


def test_invalid_weight_is_rejected():
    with pytest.raises(ValueError):
        reliability_scores([1.0], [1.0], mean_weight=1.1)
