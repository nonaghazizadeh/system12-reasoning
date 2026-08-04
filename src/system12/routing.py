"""Entropy-based arbitration between System 1 and System 2 generations.

This module implements Equations (1)--(3) from the paper.  It is deliberately
free of model dependencies so that the selection rule can be tested in CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ReliabilityScores:
    """Normalized entropy statistics and the resulting reliability scores."""

    system1_mean_entropy: float
    system2_mean_entropy: float
    system1_entropy_variance: float
    system2_entropy_variance: float
    system1_score: float
    system2_score: float


def entropy_statistics(token_entropies: Iterable[float]) -> tuple[float, float]:
    """Return population mean and variance for a generated token prefix."""

    values = np.asarray(list(token_entropies), dtype=np.float64)
    if values.size == 0:
        raise ValueError("at least one token entropy is required")
    return float(values.mean()), float(values.var(ddof=0))


def _pairwise_total_sum(first: float, second: float) -> tuple[float, float]:
    if first < 0 or second < 0:
        raise ValueError("entropy statistics must be non-negative")
    total = first + second
    if total == 0:
        return 0.5, 0.5
    return first / total, second / total


def reliability_scores(
    system1_entropies: Iterable[float],
    system2_entropies: Iterable[float],
    *,
    mean_weight: float = 0.4,
) -> ReliabilityScores:
    """Compute the paper's normalized reliability score for both systems.

    ``mean_weight`` is ``w`` in Equation (3); entropy variance receives weight
    ``1 - w``.  The camera-ready experiments use ``w=0.4``.
    """

    if not 0 <= mean_weight <= 1:
        raise ValueError("mean_weight must be between 0 and 1")

    mean1, variance1 = entropy_statistics(system1_entropies)
    mean2, variance2 = entropy_statistics(system2_entropies)
    normalized_mean1, normalized_mean2 = _pairwise_total_sum(mean1, mean2)
    normalized_variance1, normalized_variance2 = _pairwise_total_sum(
        variance1, variance2
    )

    score1 = (
        mean_weight * normalized_mean1
        + (1 - mean_weight) * normalized_variance1
    )
    score2 = (
        mean_weight * normalized_mean2
        + (1 - mean_weight) * normalized_variance2
    )
    return ReliabilityScores(
        system1_mean_entropy=mean1,
        system2_mean_entropy=mean2,
        system1_entropy_variance=variance1,
        system2_entropy_variance=variance2,
        system1_score=score1,
        system2_score=score2,
    )


def select_reasoning_system(
    system1_entropies: Iterable[float],
    system2_entropies: Iterable[float],
    *,
    mean_weight: float = 0.4,
) -> tuple[str, ReliabilityScores]:
    """Select the system with the lower reliability score.

    Exact ties deterministically select System 1.  The paper does not define a
    stochastic tie margin, so deterministic tie-breaking avoids adding an
    undocumented source of variance.
    """

    scores = reliability_scores(
        system1_entropies, system2_entropies, mean_weight=mean_weight
    )
    selected = (
        "system1"
        if scores.system1_score <= scores.system2_score
        else "system2"
    )
    return selected, scores
