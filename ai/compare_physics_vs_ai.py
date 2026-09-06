import os
import time

import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ============================================================
# 1. Paths
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    SCRIPT_DIR
)

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

RESULT_DIR = os.path.join(
    SCRIPT_DIR,
    "results"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

OUTPUT_FILE = os.path.join(
    RESULT_DIR,
    "physics_vs_ai_comparison.csv"
)


# ============================================================
# 2. Configuration
# ============================================================

SEED = 42

FEATURES = [
    "soc",
    "soh",
    "current_A",
    "initial_temperature_C"
]

TARGET = "T_max_C"


# ============================================================
# 3. Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("PHYSICS vs AI COMPARISON")
print("=" * 70)

print("\nDevice:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 4. Load dataset
# ============================================================

print("\nLoading dataset...")

metadata = pd.read_csv(
    METADATA_FILE
)

thermal = pd.read_csv(
    THERMAL_FILE
)

print(
    f"Metadata cases : {len(metadata)}"
)

print(
    f"Thermal cases  : {len(thermal)}"
)


# ============================================================
# 5. Merge dataset
# ============================================================

df = pd.merge(
    metadata,
    thermal,
    on="case_id"
)

print(
    f"Merged cases   : {len(df)}"
)


# ============================================================
# 6. Keep valid simulations
# ============================================================

if "valid" in df.columns:

    df = df[
        df["valid"] == 1
    ].copy()


df = df.dropna(
    subset=[
        "case_id",
        *FEATURES,
        TARGET
    ]
).reset_index(
    drop=True
)

print(
    f"Valid cases    : {len(df)}"
)


# ============================================================
# 7. Prepare X and y
#
# IMPORTANT:
# This exactly follows train_v5_mlp.py
# ============================================================

X = df[
    FEATURES
].values.astype(
    np.float32
)

y = df[
    TARGET
].values.astype(
    np.float32
).reshape(
    -1, 1
)


# ============================================================
# 8. EXACT SAME TRAIN / TEST SPLIT
#
# Same:
# test_size = 0.20
# random_state = 42
#
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED
)

print("\nDataset split")
print("-" * 70)

print(
    "Training cases :",
    len(X_train)
)

print(
    "Test cases     :",
    len(X_test)
)


# ============================================================
# 9. IMPORTANT:
# EXACT SAME INPUT SCALING
# ============================================================

x_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(
    X_train
)

X_test_scaled = x_scaler.transform(
    X_test
)


# ============================================================
# 10. IMPORTANT:
# EXACT SAME TARGET SCALING
# ============================================================

y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(
    y_train
)

y_test_scaled = y_scaler.transform(
    y_test
)


# ============================================================
# 11. Model definition
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
# 12. Load trained model
# ============================================================

print("\nLoading AI model...")

model = V5ThermalMLP().to(
    device
)

state_dict = torch.load(
    MODEL_FILE,
    map_location=device
)

model.load_state_dict(
    state_dict
)

model.eval()


# ============================================================
# 13. Prepare test tensor
# ============================================================

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32,
    device=device
)


# ============================================================
# 14. Warm-up
# ============================================================

with torch.no_grad():

    _ = model(
        X_test_tensor
    )


if device.type == "cuda":

    torch.cuda.synchronize()


# ============================================================
# 15. AI inference timing
# ============================================================

start = time.perf_counter()

with torch.no_grad():

    y_pred_scaled = model(
        X_test_tensor
    ).cpu().numpy()

if device.type == "cuda":

    torch.cuda.synchronize()

end = time.perf_counter()


ai_total_ms = (
    end - start
) * 1000.0

ai_ms_per_case = (
    ai_total_ms
    / len(X_test)
)


# ============================================================
# 16. Convert AI prediction back to °C
#
# THIS WAS THE MISSING STEP
# ============================================================

y_pred = y_scaler.inverse_transform(
    y_pred_scaled
)

y_actual = y_test


# ============================================================
# 17. Flatten
# ============================================================

physics_predictions = (
    y_actual.flatten()
)

ai_predictions = (
    y_pred.flatten()
)


# ============================================================
# 18. Error
# ============================================================

error = (
    ai_predictions
    - physics_predictions
)

absolute_error = np.abs(
    error
)


# ============================================================
# 19. Recover test case IDs
#
# Reproduce the same split using indices so we can
# attach case IDs to the corresponding predictions.
# ============================================================

all_indices = np.arange(
    len(df)
)

_, test_indices = train_test_split(
    all_indices,
    test_size=0.20,
    random_state=SEED
)

test_case_ids = df.iloc[
    test_indices
]["case_id"].values

test_soc = df.iloc[
    test_indices
]["soc"].values

test_soh = df.iloc[
    test_indices
]["soh"].values

test_current = df.iloc[
    test_indices
]["current_A"].values

test_initial_temperature = df.iloc[
    test_indices
]["initial_temperature_C"].values


# ============================================================
# 20. Results dataframe
# ============================================================

results = pd.DataFrame({

    "case_id":
        test_case_ids,

    "soc":
        test_soc,

    "soh":
        test_soh,

    "current_A":
        test_current,

    "initial_temperature_C":
        test_initial_temperature,

    "Tmax_physics_C":
        physics_predictions,

    "Tmax_AI_C":
        ai_predictions,

    "error_C":
        error,

    "absolute_error_C":
        absolute_error
})


# ============================================================
# 21. Metrics
# ============================================================

rmse = np.sqrt(
    mean_squared_error(
        physics_predictions,
        ai_predictions
    )
)

mae = mean_absolute_error(
    physics_predictions,
    ai_predictions
)

r2 = r2_score(
    physics_predictions,
    ai_predictions
)

max_error = np.max(
    absolute_error
)

mean_error = np.mean(
    error
)

std_error = np.std(
    error
)


# ============================================================
# 22. Save results
# ============================================================

results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 23. Print results
# ============================================================

print("\n" + "=" * 70)
print("PHYSICS vs AI RESULTS")
print("=" * 70)

print(
    f"Test cases              : "
    f"{len(results)}"
)

print(
    f"RMSE                    : "
    f"{rmse:.6f} °C"
)

print(
    f"MAE                     : "
    f"{mae:.6f} °C"
)

print(
    f"R²                      : "
    f"{r2:.6f}"
)

print(
    f"Maximum absolute error  : "
    f"{max_error:.6f} °C"
)

print(
    f"Mean error / bias       : "
    f"{mean_error:.6f} °C"
)

print(
    f"Error standard deviation: "
    f"{std_error:.6f} °C"
)

print(
    f"\nAI total inference time : "
    f"{ai_total_ms:.6f} ms"
)

print(
    f"AI time per case        : "
    f"{ai_ms_per_case:.6f} ms"
)

print(
    "\nOutput:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# 24. Worst 10 cases
# ============================================================

print(
    "\nWorst 10 prediction errors:"
)

print("-" * 70)

worst = results.sort_values(
    "absolute_error_C",
    ascending=False
).head(10)

print(
    worst.to_string(
        index=False
    )
)


# ============================================================
# 25. First 10 cases
# ============================================================

print(
    "\nFirst 10 cases:"
)

print("-" * 70)

print(
    results.head(10).to_string(
        index=False
    )
)


print(
    "\n" + "=" * 70
)

print(
    "COMPARISON COMPLETE"
)

print(
    "=" * 70
)