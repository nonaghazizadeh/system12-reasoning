from pathlib import Path

import pandas as pd


def test_machine_readable_results_cover_all_reported_models_and_benchmarks():
    results = pd.read_csv(Path("results/paper_results.csv"))
    benchmark_columns = results.columns[3:]
    assert len(benchmark_columns) == 14
    assert set(results["model"]) == {
        "Llama-3-3B",
        "Llama-3-8B",
        "Llama-3-70B",
        "Mistral-7B",
        "DeepSeek-R1-Distill-Qwen-1.5B",
    }
    assert not results[benchmark_columns].isna().any().any()
