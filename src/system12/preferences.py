"""Preference-pair construction for System 1/System 2 alignment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


STYLE_ALIASES = {
    "system1": "system1",
    "system 1": "system1",
    "s1": "system1",
    "0": "system1",
    "system2": "system2",
    "system 2": "system2",
    "s2": "system2",
    "1": "system2",
}


@dataclass(frozen=True)
class PreferenceSplit:
    train: pd.DataFrame
    validation: pd.DataFrame


def load_alignment_data(path: str | Path) -> pd.DataFrame:
    """Load and validate the released 2,000-question alignment dataset."""

    frame = pd.read_csv(path)
    required = {"Question", "Answer", "Strategy"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"alignment data is missing columns: {sorted(missing)}")

    frame = frame.loc[:, ["Question", "Answer", "Strategy"]].copy()
    frame["style"] = (
        frame["Strategy"].astype(str).str.strip().str.lower().map(STYLE_ALIASES)
    )
    if frame["style"].isna().any():
        unknown = sorted(frame.loc[frame["style"].isna(), "Strategy"].unique())
        raise ValueError(f"unknown reasoning-style labels: {unknown}")
    if frame[["Question", "Answer"]].isna().any().any():
        raise ValueError("Question and Answer values must not be empty")
    return frame


def build_question_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Create one row per prompt with its paired System 1/System 2 answers."""

    duplicates = frame.duplicated(subset=["Question", "style"], keep=False)
    if duplicates.any():
        prompts = frame.loc[duplicates, "Question"].drop_duplicates().tolist()
        raise ValueError(
            "each question must have exactly one answer per style; duplicate "
            f"answers found for {len(prompts)} question(s)"
        )

    paired = frame.pivot(index="Question", columns="style", values="Answer")
    expected = {"system1", "system2"}
    if not expected.issubset(paired.columns):
        raise ValueError("both System 1 and System 2 answers are required")
    incomplete = paired[list(sorted(expected))].isna().any(axis=1)
    if incomplete.any():
        raise ValueError(
            f"{int(incomplete.sum())} question(s) do not have both reasoning styles"
        )
    return (
        paired.reset_index()
        .rename(columns={"Question": "prompt"})
        .loc[:, ["prompt", "system1", "system2"]]
    )


def orient_preferences(
    pairs: pd.DataFrame,
    *,
    system1_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Choose the preferred style for a deterministic fraction of prompts."""

    if not 0 <= system1_fraction <= 1:
        raise ValueError("system1_fraction must be between 0 and 1")
    count = len(pairs)
    system1_count = int(round(count * system1_fraction))
    permutation = np.random.default_rng(seed).permutation(count)
    system1_rows = np.zeros(count, dtype=bool)
    system1_rows[permutation[:system1_count]] = True

    result = pd.DataFrame(
        {
            "prompt": pairs["prompt"],
            "chosen": np.where(
                system1_rows, pairs["system1"], pairs["system2"]
            ),
            "rejected": np.where(
                system1_rows, pairs["system2"], pairs["system1"]
            ),
            "preferred_style": np.where(
                system1_rows, "system1", "system2"
            ),
        }
    )
    if result.isna().any().any():
        raise AssertionError("preference construction unexpectedly produced missing values")
    return result


def split_preferences(
    preferences: pd.DataFrame,
    *,
    train_fraction: float = 0.8,
    seed: int = 0,
) -> PreferenceSplit:
    """Create the prompt-disjoint 80/20 train/validation split in the paper."""

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    order = np.random.default_rng(seed).permutation(len(preferences))
    train_count = int(round(len(preferences) * train_fraction))
    train = preferences.iloc[order[:train_count]].reset_index(drop=True)
    validation = preferences.iloc[order[train_count:]].reset_index(drop=True)
    overlap = set(train["prompt"]).intersection(validation["prompt"])
    if overlap:
        raise AssertionError("train and validation prompts must be disjoint")
    return PreferenceSplit(train=train, validation=validation)


def prepare_preference_splits(
    path: str | Path,
    *,
    system1_fraction: float,
    seed: int = 0,
    train_fraction: float = 0.8,
) -> PreferenceSplit:
    frame = load_alignment_data(path)
    pairs = build_question_pairs(frame)
    preferences = orient_preferences(
        pairs, system1_fraction=system1_fraction, seed=seed
    )
    return split_preferences(
        preferences, train_fraction=train_fraction, seed=seed
    )
