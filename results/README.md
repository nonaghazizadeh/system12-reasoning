# Released paper results

`paper_results.csv` is a machine-readable transcription of Tables 1, 8, and 9 in the
camera-ready paper. Values are exact-match accuracy percentages. The table includes all
14 benchmarks for Llama 3 (3B, 8B, and 70B), Mistral 7B, and the
DeepSeek-R1-Distill-Qwen-1.5B robustness experiment.

Fresh evaluation runs write `predictions.csv` and `metrics.json` below `experiments/`,
which is intentionally ignored because it can contain very large generated traces.
