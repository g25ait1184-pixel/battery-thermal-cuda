
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "dataset" / "v5_1000"
MODEL_DIR = ROOT / "ai" / "models"
RESULT_DIR = ROOT / "ai" / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "soc",
    "soh",
    "current_A",
    "initial_temperature_C"
]

TARGET = "T_max_C"

WARMUP_RUNS = 50
BENCHMARK_RUNS = 500

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. MODEL ARCHITECTURES
# ============================================================

class V5ThermalMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)


class ResidualMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),

            nn.Linear(16, 16),
            nn.ReLU(),

            nn.Linear(16, 8),
            nn.ReLU(),

            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# 3. CUDA EVENT TIMING
# ============================================================

def benchmark_gpu_events(base, residual, X):

    base_times = []
    hybrid_times = []

    # Warm-up
    with torch.inference_mode():

        for _ in range(WARMUP_RUNS):
            base(X)
            base(X) + residual(X)

    torch.cuda.synchronize()

    # Base AI GPU timing
    with torch.inference_mode():

        for _ in range(BENCHMARK_RUNS):

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()

            base(X)

            end.record()
            end.synchronize()

            base_times.append(start.elapsed_time(end))

    # Hybrid AI GPU timing
    with torch.inference_mode():

        for _ in range(BENCHMARK_RUNS):

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()

            base_pred = base(X)
            residual_pred = residual(X)
            hybrid_pred = base_pred + residual_pred

            end.record()
            end.synchronize()

            hybrid_times.append(start.elapsed_time(end))

    return {
        "base_median_ms": float(np.median(base_times)),
        "base_mean_ms": float(np.mean(base_times)),
        "base_std_ms": float(np.std(base_times, ddof=1)),

        "hybrid_median_ms": float(np.median(hybrid_times)),
        "hybrid_mean_ms": float(np.mean(hybrid_times)),
        "hybrid_std_ms": float(np.std(hybrid_times, ddof=1))
    }


# ============================================================
# 4. MAIN
# ============================================================

def main():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    print("=" * 75)
    print("FINAL V5 PHYSICS vs BASE AI vs HYBRID AI BENCHMARK")
    print("=" * 75)

    print("Device:", DEVICE)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Warm-up runs:", WARMUP_RUNS)
    print("Benchmark runs:", BENCHMARK_RUNS)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    metadata = pd.read_csv(DATASET / "metadata.csv")
    thermal = pd.read_csv(DATASET / "thermal_metrics.csv")

    df = metadata.merge(
        thermal,
        on="case_id",
        how="inner"
    )

    df = df[df["valid"] == 1].copy().reset_index(drop=True)

    # --------------------------------------------------------
    # Fixed seed-42 split
    # --------------------------------------------------------

    indices = np.arange(len(df))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=42
    )

    train = df.iloc[train_idx]
    test = df.iloc[test_idx].copy().reset_index(drop=True)

    print("\nDataset:", len(df))
    print("Training cases:", len(train))
    print("Testing cases:", len(test))

    # --------------------------------------------------------
    # Fit scalers exactly as in five-seed evaluation
    # --------------------------------------------------------

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaler.fit(
        train[FEATURES].values.astype(np.float32)
    )

    y_scaler.fit(
        train[[TARGET]].values.astype(np.float32)
    )

    X_scaled = x_scaler.transform(
        test[FEATURES].values.astype(np.float32)
    ).astype(np.float32)

    X = torch.tensor(
        X_scaled,
        dtype=torch.float32,
        device=DEVICE
    )

    # --------------------------------------------------------
    # Load seed-42 checkpoints
    # --------------------------------------------------------

    base = V5ThermalMLP().to(DEVICE)

    base.load_state_dict(
        torch.load(
            MODEL_DIR / "v5_base_seed_42.pt",
            map_location=DEVICE,
            weights_only=True
        )
    )

    base.eval()

    residual = ResidualMLP().to(DEVICE)

    residual.load_state_dict(
        torch.load(
            MODEL_DIR / "v5_residual_seed_42.pt",
            map_location=DEVICE,
            weights_only=True
        )
    )

    residual.eval()

    for model in (base, residual):

        for parameter in model.parameters():
            parameter.requires_grad = False

    # --------------------------------------------------------
    # AI predictions and accuracy
    # --------------------------------------------------------

    with torch.inference_mode():

        base_scaled = base(X).cpu().numpy()

        residual_c = residual(X).cpu().numpy().ravel()

    base_c = y_scaler.inverse_transform(
        base_scaled
    ).ravel()

    hybrid_c = base_c + residual_c

    actual = test[TARGET].values.astype(np.float32)

    base_rmse = np.sqrt(
        mean_squared_error(actual, base_c)
    )

    base_mae = mean_absolute_error(actual, base_c)

    base_r2 = r2_score(actual, base_c)

    hybrid_rmse = np.sqrt(
        mean_squared_error(actual, hybrid_c)
    )

    hybrid_mae = mean_absolute_error(actual, hybrid_c)

    hybrid_r2 = r2_score(actual, hybrid_c)

    # --------------------------------------------------------
    # Repeated CUDA event timing
    # --------------------------------------------------------

    print("\nRunning repeated GPU timing...")

    timing = benchmark_gpu_events(
        base,
        residual,
        X
    )

    base_ms = timing["base_median_ms"]
    hybrid_ms = timing["hybrid_median_ms"]

    # --------------------------------------------------------
    # Recorded V5 physics runtimes
    # --------------------------------------------------------

    physics_times = test["runtime_ms"].values.astype(np.float64)

    physics_total_ms = float(np.sum(physics_times))

    physics_per_case_ms = float(np.mean(physics_times))

    physics_std_ms = float(np.std(physics_times, ddof=1))

    # --------------------------------------------------------
    # Speedup based on per-case comparison
    # --------------------------------------------------------

    base_per_case_ms = base_ms / len(test)

    hybrid_per_case_ms = hybrid_ms / len(test)

    base_speedup = physics_per_case_ms / base_per_case_ms

    hybrid_speedup = physics_per_case_ms / hybrid_per_case_ms

    # --------------------------------------------------------
    # Final comparison table
    # --------------------------------------------------------

    results = pd.DataFrame([

        {
            "method": "V5 Physics CUDA",
            "test_cases": len(test),
            "RMSE_C": np.nan,
            "MAE_C": np.nan,
            "R2": np.nan,
            "total_runtime_ms": physics_total_ms,
            "runtime_per_case_ms": physics_per_case_ms,
            "timing_std_ms": physics_std_ms,
            "timing_type": "Recorded simulation runtime",
            "physics_speedup": 1.0
        },

        {
            "method": "Base AI GPU",
            "test_cases": len(test),
            "RMSE_C": base_rmse,
            "MAE_C": base_mae,
            "R2": base_r2,
            "total_runtime_ms": base_ms,
            "runtime_per_case_ms": base_per_case_ms,
            "timing_std_ms": timing["base_std_ms"] / len(test),
            "timing_type": "Median repeated CUDA batch inference",
            "physics_speedup": base_speedup
        },

        {
            "method": "Hybrid AI GPU",
            "test_cases": len(test),
            "RMSE_C": hybrid_rmse,
            "MAE_C": hybrid_mae,
            "R2": hybrid_r2,
            "total_runtime_ms": hybrid_ms,
            "runtime_per_case_ms": hybrid_per_case_ms,
            "timing_std_ms": timing["hybrid_std_ms"] / len(test),
            "timing_type": "Median repeated CUDA batch inference",
            "physics_speedup": hybrid_speedup
        }

    ])

    # --------------------------------------------------------
    # Save benchmark results
    # --------------------------------------------------------

    results_path = RESULT_DIR / "final_end_to_end_benchmark.csv"

    results.to_csv(
        results_path,
        index=False
    )

    # --------------------------------------------------------
    # Save per-case predictions
    # --------------------------------------------------------

    details = test[
        ["case_id"] + FEATURES + [TARGET, "runtime_ms"]
    ].copy()

    details["base_ai_Tmax_C"] = base_c

    details["hybrid_ai_Tmax_C"] = hybrid_c

    details["base_ai_error_C"] = base_c - actual

    details["hybrid_ai_error_C"] = hybrid_c - actual

    details_path = RESULT_DIR / "final_end_to_end_test_cases.csv"

    details.to_csv(
        details_path,
        index=False
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("FINAL PERFORMANCE COMPARISON")
    print("=" * 75)

    print(
        results.to_string(index=False)
    )

    print("\nAccuracy improvement of Hybrid over Base:")

    print(
        "RMSE improvement: "
        f"{((base_rmse - hybrid_rmse) / base_rmse) * 100:.2f}%"
    )

    print(
        "MAE improvement: "
        f"{((base_mae - hybrid_mae) / base_mae) * 100:.2f}%"
    )

    print("\nGPU timing details:")

    print(
        f"Base AI: median={timing['base_median_ms']:.6f} ms, "
        f"mean={timing['base_mean_ms']:.6f} ms, "
        f"SD={timing['base_std_ms']:.6f} ms"
    )

    print(
        f"Hybrid AI: median={timing['hybrid_median_ms']:.6f} ms, "
        f"mean={timing['hybrid_mean_ms']:.6f} ms, "
        f"SD={timing['hybrid_std_ms']:.6f} ms"
    )

    print("\nSaved files:")
    print(results_path)
    print(details_path)

    print("\nBenchmark completed.")


if __name__ == "__main__":
    main()
