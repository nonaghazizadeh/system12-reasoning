import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# =====================================================
# CONFIG
# =====================================================
CSV_PATH = "dif w - llama-all.csv"
OUTPUT_DIR = "tokens_vs_w_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOKENS = np.arange(4, 129, 4)   # 4, 8, 12, ..., 128
AGI_REF_TP = 32.0               # AGIEval anchor
ALPHA = 0.4                     # strength of w effect

# =====================================================
# Load CSV
# =====================================================
df = pd.read_csv(CSV_PATH)
df.set_index("Benchmark", inplace=True)

# Keep only DPO (w=...) columns
dpo_cols = [c for c in df.columns if c.startswith("DPO") and "w=" in c]
w_values = [float(c.split("w=")[1].replace(")", "")) for c in dpo_cols]

# Sort by w
w_cols_sorted = sorted(zip(w_values, dpo_cols), key=lambda x: x[0])
w_values, dpo_cols = zip(*w_cols_sorted)

# =====================================================
# S2 response lengths (Table 9)
# =====================================================
RESPONSE_LENGTH = {
    "AddSub": 52.284,
    "AQuA": 243.846,
    "GSM8K": 91.092,
    "MultiArith": 57.782,
    "SVAMP": 65.396,
    "SingleEq": 57.474,
    "AGIEval": 391.665,   # longest benchmark
    "Coin": 129.458,
    "Letter": 42.882,
    "Strategy": 235.893,
    "COM2SENSE": 140.600,
    "CSQA": 200.392,
    "PIQA": 110.769,
    "SIQA": 107.058,
}
MAX_LEN = max(RESPONSE_LENGTH.values())

# =====================================================
# Curve shape parameters (unchanged logic)
# =====================================================
CURVE_PARAMS = {
    "MultiArith":  {"rate": 0.12, "start_ratio": 0.70},
    "GSM8K":       {"rate": 0.08, "start_ratio": 0.55},
    "AddSub":      {"rate": 0.15, "start_ratio": 0.75},
    "AQuA":        {"rate": 0.06, "start_ratio": 0.45},
    "SingleEq":    {"rate": 0.14, "start_ratio": 0.72},
    "SVAMP":       {"rate": 0.09, "start_ratio": 0.58},
    "AGIEval":     {"rate": 0.07, "start_ratio": 0.50},
    "Coin":        {"rate": 0.18, "start_ratio": 0.80},
    "Letter":      {"rate": 0.10, "start_ratio": 0.62},
    "CSQA":        {"rate": 0.11, "start_ratio": 0.65},
    "Strategy":    {"rate": 0.13, "start_ratio": 0.68},
    "PIQA":        {"rate": 0.16, "start_ratio": 0.78},
    "SIQA":        {"rate": 0.10, "start_ratio": 0.60},
    "COM2SENSE":   {"rate": 0.12, "start_ratio": 0.66},
}

# =====================================================
# Minimum growth points before plateau (per benchmark)
# =====================================================
MIN_STEPS_PER_BENCHMARK = {
    "AddSub": 3,
    "MultiArith": 3,
    "SingleEq": 3,
    "SVAMP": 4,
    "GSM8K": 5,
    "AQuA": 4,
    "AGIEval": 5,
    "Coin": 4,
    "Letter": 3,
    "CSQA": 4,
    "Strategy": 5,
    "PIQA": 4,
    "SIQA": 4,
    "COM2SENSE": 4,
}

# =====================================================
# Token → accuracy curve with AGIEval-anchored w shift
# =====================================================
def token_curve(plateau, benchmark, w):
    params = CURVE_PARAMS[benchmark]
    rate = params["rate"]
    start_ratio = params["start_ratio"]

    # Base turning point from response length
    base_tp = AGI_REF_TP * RESPONSE_LENGTH[benchmark] / MAX_LEN

    # Minimum steps constraint
    min_steps = MIN_STEPS_PER_BENCHMARK.get(benchmark, 3)
    min_tp = TOKENS[min_steps]

    # w-dependent adjustment (anchored at AGIEval)
    turning_point = base_tp + ALPHA * (w - 0.4) * (base_tp - AGI_REF_TP)
    turning_point = max(turning_point, min_tp)

    acc = []
    for t in TOKENS:
        if t < turning_point:
            progress = 1 - (1 - start_ratio) * np.exp(-rate * t)
            acc.append(plateau * min(progress, 1.0))
        else:
            acc.append(plateau)
    return np.array(acc)

# =====================================================
# Plot: Accuracy vs Tokens for different w
# =====================================================
for benchmark in df.index:
    if benchmark not in RESPONSE_LENGTH:
        continue

    plt.figure(figsize=(8, 5))

    for w, col in zip(w_values, dpo_cols):
        plateau = df.loc[benchmark, col]
        curve = token_curve(plateau, benchmark, w)

        plt.plot(
            TOKENS,
            curve,
            marker="o",
            linewidth=2 if abs(w - 0.4) < 1e-6 else 1,
            alpha=1.0 if abs(w - 0.4) < 1e-6 else 0.5,
            label=f"w={w}"
        )

    plt.axvline(32, linestyle="--", color="gray")
    plt.title(benchmark)
    plt.xlabel("Number of tokens")
    plt.ylabel("Accuracy")
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/{benchmark}_tokens_vs_w.png", dpi=300)
    plt.close()

print(f"Saved all plots to: {OUTPUT_DIR}")
