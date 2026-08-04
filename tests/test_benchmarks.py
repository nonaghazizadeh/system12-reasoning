from pathlib import Path

import pytest

from system12.benchmarks import BENCHMARKS, exact_match, load_benchmark, normalize_answer


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("multiarith", 600),
        ("gsm8k", 1319),
        ("addsub", 395),
        ("aqua", 254),
        ("singleeq", 508),
        ("svamp", 1000),
        ("agieval", 1000),
        ("coin_flip", 500),
        ("last_letters", 500),
        ("commonsensqa", 1221),
        ("strategyqa", 2290),
    ],
)
def test_locally_available_paper_benchmark_sizes(name, expected):
    path = Path("data/benchmark") / BENCHMARKS[name].relative_path
    if not path.exists():
        pytest.skip("third-party benchmarks are prepared outside git")
    assert len(load_benchmark(name)) == expected


def test_registry_contains_exactly_the_14_paper_benchmarks():
    assert len(BENCHMARKS) == 14
    assert {spec.category for spec in BENCHMARKS.values()} == {
        "arithmetic",
        "symbolic",
        "commonsense",
    }


@pytest.mark.parametrize(
    ("name", "output", "expected"),
    [
        ("gsm8k", "The answer is 1,200.", "1200"),
        ("gsm8k", "The answer is 10.", "10"),
        ("aqua", "After considering A and B, the answer is (D).", "D"),
        ("piqa", "Therefore, the answer is B", "B"),
        ("strategyqa", "Yes, that is correct.", "yes"),
        ("com2sense", "The statement is FALSE.", "false"),
        ("agieval", r"The final answer is $\boxed{(3,4]}$.", "(3,4]"),
    ],
)
def test_answer_normalization(name, output, expected):
    assert normalize_answer(name, output) == expected


def test_exact_match_uses_normalized_answers():
    assert exact_match("gsm8k", "Therefore: 1,200.0", "1200")
