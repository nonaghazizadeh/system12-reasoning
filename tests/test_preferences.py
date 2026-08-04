from pathlib import Path

import pandas as pd
import pytest

from system12.preferences import (
    build_question_pairs,
    load_alignment_data,
    orient_preferences,
    prepare_preference_splits,
)


DATA_PATH = Path("data/system12/cogbias.csv")


def test_released_alignment_data_has_2000_complete_pairs():
    frame = load_alignment_data(DATA_PATH)
    pairs = build_question_pairs(frame)
    assert len(frame) == 4000
    assert len(pairs) == 2000
    assert not pairs.isna().any().any()


@pytest.mark.parametrize("fraction", [0, 0.125, 0.5, 0.875, 1])
def test_preference_fraction_is_exact_and_has_no_missing_values(fraction):
    pairs = build_question_pairs(load_alignment_data(DATA_PATH))
    preferences = orient_preferences(pairs, system1_fraction=fraction, seed=0)
    assert (preferences["preferred_style"] == "system1").sum() == round(
        len(pairs) * fraction
    )
    assert not preferences.isna().any().any()


def test_paper_split_is_prompt_disjoint_80_20():
    splits = prepare_preference_splits(DATA_PATH, system1_fraction=0.5, seed=0)
    assert len(splits.train) == 1600
    assert len(splits.validation) == 400
    assert set(splits.train.prompt).isdisjoint(splits.validation.prompt)


def test_duplicate_style_for_a_question_is_rejected():
    frame = pd.DataFrame(
        {
            "Question": ["q", "q", "q"],
            "Answer": ["a", "b", "c"],
            "style": ["system1", "system1", "system2"],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        build_question_pairs(frame)
