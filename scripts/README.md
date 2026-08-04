# Script map

The camera-ready entry points are:

- `prepare_benchmarks.py`: download pinned third-party evaluation splits and verify their sizes.
- `train_paper_models.sh`: train the System 1 and System 2 endpoints.
- `train_reasoning_spectrum.sh`: train the seven 12.5%-spaced interpolation models.
- `evaluate_paper_model.sh`: run one base or aligned model on all 14 benchmarks.
- `evaluate_dynamic_model.sh`: run entropy-guided Multi-LoRA arbitration on all 14 benchmarks.

The older `train_dpo*.sh`, `train_simpo*.sh`, and `benchmark_*.sh` files are retained as
the original experiment launch records. They contain cluster-specific GPU indices,
paths, and detached `screen` commands and are not the supported reproduction path.
