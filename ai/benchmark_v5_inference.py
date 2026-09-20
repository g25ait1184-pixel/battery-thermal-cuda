import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. Paths
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

DATASET_DIR = os.path.join(
    PROJECT_DIR,
    "dataset",
    "v5_1000"
)

METADATA_FILE = os.path.join(
    DATASET_DIR,
    "metadata.csv"
)

THERMAL_FILE = os.path.join(
    DATASET_DIR,
    "thermal_metrics.csv"
)

MODEL_FILE = os.path.join(
    SCRIPT_DIR,
    "models",
    "v5_tmax_mlp.pt"
)


# ============================================================
# 2. Device information
# ============================================================

print("=" * 70)
print("V5 AI THERMAL PREDICTOR - CPU vs GPU INFERENCE")
print("=" * 70)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU: Not available")

print()


# ============================================================
# 3. Define the same MLP used during training
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


# ============================================================
# 4. Load dataset
# ============================================================

metadata = pd.read_csv(METADATA_FILE)
thermal = pd.read_csv(THERMAL_FILE)

df = pd.merge(
    metadata,
    thermal,
    on="case_id"
)

print("Total V5 simulations:", len(df))

if len(df) != 1000:
    print("WARNING: Expected 100 simulations.")


# ============================================================
# 5. Select input features
# ============================================================

features = [
    "soc",
    "soh",
    "current_A",
    "initial_temperature_C"
]

X = df[features].values.astype(np.float32)


# ============================================================
# 6. Normalize inputs
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X).astype(np.float32)


# ============================================================
# 7. Load trained model - CPU
# ============================================================

model_cpu = V5ThermalMLP()

model_cpu.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location="cpu"
    )
)

model_cpu.eval()


# ============================================================
# 8. Load trained model - GPU
# ============================================================

if torch.cuda.is_available():

    model_gpu = V5ThermalMLP()

    model_gpu.load_state_dict(
        torch.load(
            MODEL_FILE,
            map_location="cpu"
        )
    )

    model_gpu = model_gpu.cuda()

    model_gpu.eval()

else:

    model_gpu = None


# ============================================================
# 9. Benchmark CPU
# ============================================================

def benchmark_cpu(model, data, repeats=1000):

    x = torch.from_numpy(data)

    # Warm-up
    with torch.no_grad():
        for _ in range(20):
            model(x)

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(repeats):
            model(x)

    end = time.perf_counter()

    elapsed_ms = (end - start) * 1000

    return elapsed_ms / repeats


# ============================================================
# 10. Benchmark GPU
# ============================================================

def benchmark_gpu(model, data, repeats=1000):

    x = torch.from_numpy(data).cuda()

    # Warm-up
    with torch.no_grad():
        for _ in range(20):
            model(x)

    # Make sure warm-up GPU work is complete
    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(repeats):
            model(x)

    # IMPORTANT:
    # CUDA operations are asynchronous.
    # Synchronization is required before stopping timer.
    torch.cuda.synchronize()

    end = time.perf_counter()

    elapsed_ms = (end - start) * 1000

    return elapsed_ms / repeats


# ============================================================
# 11. Real batch sizes
# ============================================================

# IMPORTANT:
# We have only 100 real simulations.
# Therefore we benchmark only:
#
# Batch 1   -> 1 real simulation
# Batch 10  -> 10 real simulations
# Batch 100 -> all 100 real simulations
#
# No artificial repetition is performed.

batch_sizes = [
    1,
    10,
    100,
    1000
]


# ============================================================
# 12. Run benchmark
# ============================================================

print(
    f"{'Batch':>10}"
    f"{'CPU ms':>15}"
    f"{'GPU ms':>15}"
    f"{'GPU Speedup':>18}"
)

print("-" * 70)


benchmark_results = []


for batch_size in batch_sizes:

    # --------------------------------------------------------
    # Select ACTUAL samples from the dataset
    # --------------------------------------------------------

    batch = X_scaled[:batch_size]

    # --------------------------------------------------------
    # Repetition count
    # --------------------------------------------------------

    if batch_size == 1:
        repeats = 5000

    elif batch_size == 10:
        repeats = 2000

    else:
        repeats = 1000

    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------

    cpu_ms = benchmark_cpu(
        model_cpu,
        batch,
        repeats
    )

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    if model_gpu is not None:

        gpu_ms = benchmark_gpu(
            model_gpu,
            batch,
            repeats
        )

        speedup = cpu_ms / gpu_ms

    else:

        gpu_ms = np.nan
        speedup = np.nan

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"{batch_size:10d}"
        f"{cpu_ms:15.6f}"
        f"{gpu_ms:15.6f}"
        f"{speedup:18.2f}x"
    )

    benchmark_results.append({

        "batch_size": batch_size,
        "cpu_ms": cpu_ms,
        "gpu_ms": gpu_ms,
        "gpu_speedup": speedup
    })


# ============================================================
# 13. Save benchmark results
# ============================================================

results_df = pd.DataFrame(
    benchmark_results
)

RESULT_DIR = os.path.join(
    SCRIPT_DIR,
    "results"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

RESULT_FILE = os.path.join(
    RESULT_DIR,
    "v5_ai_benchmark.csv"
)

results_df.to_csv(
    RESULT_FILE,
    index=False
)


# ============================================================
# 14. Summary
# ============================================================

print()
print("=" * 70)
print("BENCHMARK SUMMARY")
print("=" * 70)

print("Real simulations used:", len(df))
print("Batch sizes tested:", batch_sizes)

print()
print("Results saved to:")
print(RESULT_FILE)

print()
print("Benchmark complete.")