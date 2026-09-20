import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "dataset/v5_1000"

BASE_MODEL_PATH = "ai/models/v5_tmax_mlp.pt"
RESIDUAL_MODEL_PATH = "ai/models/v5_residual_mlp.pt"

OUTPUT_FILE = "ai/results/v5_hybrid_benchmark.csv"

SEED = 42

BATCH_SIZES = [1, 10, 100, 1000]

WARMUP_RUNS = 20
BENCHMARK_RUNS = 100

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# MODELS
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
# LOAD DATA
# ============================================================

metadata = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "metadata.csv"
    )
)

thermal = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "thermal_metrics.csv"
    )
)

df = metadata.merge(
    thermal,
    on="case_id",
    how="inner"
)


features = [
    "soc",
    "soh",
    "current_A",
    "initial_temperature_C"
]

X = df[features].values.astype(
    np.float32
)

y = df["T_max_C"].values.astype(
    np.float32
).reshape(-1, 1)


# ============================================================
# EXACT SEED-42 SPLIT
# ============================================================

indices = np.arange(
    len(df)
)

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=SEED
)

X_train = X[train_idx]
X_test = X[test_idx]

y_test = y[test_idx]


# ============================================================
# SCALING
# ============================================================

x_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(
    X_train
)

X_test_scaled = x_scaler.transform(
    X_test
)


y_scaler = StandardScaler()

y_scaler.fit(
    y[train_idx]
)


# ============================================================
# LOAD MODELS
# ============================================================

base_model = V5ThermalMLP().to(
    DEVICE
)

base_model.load_state_dict(
    torch.load(
        BASE_MODEL_PATH,
        map_location=DEVICE
    )
)

base_model.eval()


residual_model = ResidualMLP().to(
    DEVICE
)

residual_model.load_state_dict(
    torch.load(
        RESIDUAL_MODEL_PATH,
        map_location=DEVICE
    )
)

residual_model.eval()


# ============================================================
# FREEZE MODELS
# ============================================================

for param in base_model.parameters():
    param.requires_grad = False

for param in residual_model.parameters():
    param.requires_grad = False


# ============================================================
# FULL TEST TENSOR
# ============================================================

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32,
    device=DEVICE
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def synchronize():

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def benchmark_base_gpu(x):

    with torch.inference_mode():

        for _ in range(WARMUP_RUNS):

            _ = base_model(x)

        synchronize()

        start = time.perf_counter()

        for _ in range(BENCHMARK_RUNS):

            _ = base_model(x)

        synchronize()

        end = time.perf_counter()

    return (
        (end - start)
        / BENCHMARK_RUNS
        * 1000.0
    )


def benchmark_hybrid_gpu(x):

    with torch.inference_mode():

        for _ in range(WARMUP_RUNS):

            base_prediction = base_model(x)

            residual = residual_model(x)

            _ = base_prediction + residual

        synchronize()

        start = time.perf_counter()

        for _ in range(BENCHMARK_RUNS):

            base_prediction = base_model(x)

            residual = residual_model(x)

            _ = base_prediction + residual

        synchronize()

        end = time.perf_counter()

    return (
        (end - start)
        / BENCHMARK_RUNS
        * 1000.0
    )


def benchmark_base_cpu(x):

    model = base_model.cpu()
    x = x.cpu()

    with torch.inference_mode():

        for _ in range(WARMUP_RUNS):

            _ = model(x)

        start = time.perf_counter()

        for _ in range(BENCHMARK_RUNS):

            _ = model(x)

        end = time.perf_counter()

    return (
        (end - start)
        / BENCHMARK_RUNS
        * 1000.0
    )


def benchmark_hybrid_cpu(x):

    base = base_model.cpu()
    residual = residual_model.cpu()
    x = x.cpu()

    with torch.inference_mode():

        for _ in range(WARMUP_RUNS):

            base_prediction = base(x)

            correction = residual(x)

            _ = base_prediction + correction

        start = time.perf_counter()

        for _ in range(BENCHMARK_RUNS):

            base_prediction = base(x)

            correction = residual(x)

            _ = base_prediction + correction

        end = time.perf_counter()

    return (
        (end - start)
        / BENCHMARK_RUNS
        * 1000.0
    )


# ============================================================
# VERIFY ACCURACY
# ============================================================

base_model.eval()
residual_model.eval()

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32,
    device=DEVICE
)

with torch.inference_mode():

    base_scaled = base_model(
        X_test_tensor
    ).cpu().numpy()

    residual = residual_model(
        X_test_tensor
    ).cpu().numpy()


base_prediction = y_scaler.inverse_transform(
    base_scaled
)

hybrid_prediction = (
    base_prediction +
    residual
)


# ============================================================
# ACCURACY METRICS
# ============================================================

base_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        base_prediction
    )
)

base_mae = mean_absolute_error(
    y_test,
    base_prediction
)

base_r2 = r2_score(
    y_test,
    base_prediction
)


hybrid_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        hybrid_prediction
    )
)

hybrid_mae = mean_absolute_error(
    y_test,
    hybrid_prediction
)

hybrid_r2 = r2_score(
    y_test,
    hybrid_prediction
)


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 70)
print("HYBRID AI INFERENCE BENCHMARK")
print("=" * 70)

print(
    f"Device: {DEVICE}"
)

print(
    f"Test cases: {len(X_test)}"
)

print(
    f"Warmup runs: {WARMUP_RUNS}"
)

print(
    f"Benchmark runs: {BENCHMARK_RUNS}"
)


print("\n")
print("=" * 70)
print("ACCURACY")
print("=" * 70)

print(
    f"Base AI RMSE : "
    f"{base_rmse:.6f} °C"
)

print(
    f"Base AI MAE  : "
    f"{base_mae:.6f} °C"
)

print(
    f"Base AI R²   : "
    f"{base_r2:.6f}"
)

print()

print(
    f"Hybrid RMSE  : "
    f"{hybrid_rmse:.6f} °C"
)

print(
    f"Hybrid MAE   : "
    f"{hybrid_mae:.6f} °C"
)

print(
    f"Hybrid R²    : "
    f"{hybrid_r2:.6f}"
)


# ============================================================
# BENCHMARK
# ============================================================

results = []


print("\n")
print("=" * 70)
print("INFERENCE PERFORMANCE")
print("=" * 70)

print(
    "\nBatch | Base CPU ms | Base GPU ms | "
    "Hybrid CPU ms | Hybrid GPU ms"
)

print("-" * 70)


for batch_size in BATCH_SIZES:

    # Repeat the test data to obtain the requested batch size.
    batch = np.tile(
        X_test_scaled,
        (
            int(
                np.ceil(
                    batch_size /
                    len(X_test_scaled)
                )
            ),
            1
        )
    )[:batch_size]


    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------

    batch_cpu = torch.tensor(
        batch,
        dtype=torch.float32
    )

    base_cpu_ms = benchmark_base_cpu(
        batch_cpu
    )

    hybrid_cpu_ms = benchmark_hybrid_cpu(
        batch_cpu
    )


    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    # Move models back to GPU

    base_model = base_model.to(
        DEVICE
    )

    residual_model = residual_model.to(
        DEVICE
    )

    batch_gpu = batch_cpu.to(
        DEVICE
    )

    base_gpu_ms = benchmark_base_gpu(
        batch_gpu
    )

    hybrid_gpu_ms = benchmark_hybrid_gpu(
        batch_gpu
    )


    print(
        f"{batch_size:5d} | "
        f"{base_cpu_ms:12.6f} | "
        f"{base_gpu_ms:12.6f} | "
        f"{hybrid_cpu_ms:13.6f} | "
        f"{hybrid_gpu_ms:13.6f}"
    )


    results.append({

        "batch_size":
            batch_size,

        "base_cpu_ms":
            base_cpu_ms,

        "base_gpu_ms":
            base_gpu_ms,

        "hybrid_cpu_ms":
            hybrid_cpu_ms,

        "hybrid_gpu_ms":
            hybrid_gpu_ms
    })


# ============================================================
# DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SPEEDUPS
# ============================================================

results_df["base_gpu_speedup"] = (
    results_df["base_cpu_ms"] /
    results_df["base_gpu_ms"]
)

results_df["hybrid_gpu_speedup"] = (
    results_df["hybrid_cpu_ms"] /
    results_df["hybrid_gpu_ms"]
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL TABLE
# ============================================================

print("\n")
print("=" * 70)
print("SPEEDUP")
print("=" * 70)

print(
    results_df[
        [
            "batch_size",
            "base_gpu_speedup",
            "hybrid_gpu_speedup"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print("\n")
print("=" * 70)
print("SAVED")
print("=" * 70)

print(
    f"Results: {OUTPUT_FILE}"
)

print("\nDone.")