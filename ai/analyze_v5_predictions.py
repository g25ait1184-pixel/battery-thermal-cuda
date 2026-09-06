import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


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

RESULT_DIR = os.path.join(
    SCRIPT_DIR,
    "results"
)

PLOT_DIR = os.path.join(
    RESULT_DIR,
    "plots"
)

os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# 2. Load data
# ============================================================

metadata = pd.read_csv(METADATA_FILE)
thermal = pd.read_csv(THERMAL_FILE)

df = pd.merge(
    metadata,
    thermal,
    on="case_id"
)

features = [
    "soc",
    "soh",
    "current_A",
    "initial_temperature_C"
]

target = "T_max_C"

X = df[features].values.astype(np.float32)
y = df[target].values.astype(np.float32).reshape(-1, 1)


# ============================================================
# 3. Recreate the same test split as seed 42
# ============================================================

X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
    X,
    y,
    df,
    test_size=0.20,
    random_state=42
)


# ============================================================
# 4. Normalize
# ============================================================

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(
    X_train
).astype(np.float32)

X_test_scaled = x_scaler.transform(
    X_test
).astype(np.float32
)

y_train_scaled = y_scaler.fit_transform(
    y_train
).astype(np.float32)


# ============================================================
# 5. Define MLP
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
# 6. Load trained model
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = V5ThermalMLP().to(device)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=device
    )
)

model.eval()


# ============================================================
# 7. Predict
# ============================================================

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32,
    device=device
)

with torch.no_grad():

    prediction_scaled = model(
        X_test_tensor
    ).cpu().numpy()


prediction = y_scaler.inverse_transform(
    prediction_scaled
).flatten()

actual = y_test.flatten()

error = prediction - actual

absolute_error = np.abs(error)


# ============================================================
# 8. Create analysis dataframe
# ============================================================

analysis = df_test.copy()

analysis["Actual_Tmax_C"] = actual

analysis["Predicted_Tmax_C"] = prediction

analysis["Error_C"] = error

analysis["Absolute_Error_C"] = absolute_error


# ============================================================
# 9. Sort by largest error
# ============================================================

largest_errors = analysis.sort_values(
    "Absolute_Error_C",
    ascending=False
)

print("=" * 70)
print("V5 MLP PREDICTION ERROR ANALYSIS")
print("=" * 70)

print()
print("Total test samples:", len(analysis))

print()
print("Largest prediction errors:")

print(
    largest_errors[
        [
            "case_id",
            "soc",
            "soh",
            "current_A",
            "initial_temperature_C",
            "Actual_Tmax_C",
            "Predicted_Tmax_C",
            "Error_C",
            "Absolute_Error_C"
        ]
    ].head(10).to_string(index=False)
)


# ============================================================
# 10. Error statistics
# ============================================================

print()
print("=" * 70)
print("ERROR STATISTICS")
print("=" * 70)

print(
    f"Mean error       : {np.mean(error):.6f} °C"
)

print(
    f"Mean abs. error  : {np.mean(absolute_error):.6f} °C"
)

print(
    f"Max abs. error   : {np.max(absolute_error):.6f} °C"
)

print(
    f"Std error        : {np.std(error):.6f} °C"
)


# ============================================================
# 11. Save detailed predictions
# ============================================================

prediction_file = os.path.join(
    RESULT_DIR,
    "v5_prediction_error_analysis.csv"
)

analysis.to_csv(
    prediction_file,
    index=False
)

print()
print("Detailed results saved to:")
print(prediction_file)


# ============================================================
# 12. Plot: Actual vs Predicted
# ============================================================

plt.figure(figsize=(7, 6))

plt.scatter(
    actual,
    prediction
)

min_value = min(
    actual.min(),
    prediction.min()
)

max_value = max(
    actual.max(),
    prediction.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Tmax (°C)")
plt.ylabel("Predicted Tmax (°C)")
plt.title("V5 MLP: Actual vs Predicted Tmax")

plt.grid(True)

plt.tight_layout()

plot_file = os.path.join(
    PLOT_DIR,
    "actual_vs_predicted_tmax.png"
)

plt.savefig(
    plot_file,
    dpi=200
)

plt.close()


# ============================================================
# 13. Plot: Prediction error
# ============================================================

plt.figure(figsize=(7, 5))

plt.scatter(
    actual,
    error
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Actual Tmax (°C)")
plt.ylabel("Prediction Error (°C)")
plt.title("V5 MLP Prediction Error")

plt.grid(True)

plt.tight_layout()

plot_file = os.path.join(
    PLOT_DIR,
    "prediction_error_vs_tmax.png"
)

plt.savefig(
    plot_file,
    dpi=200
)

plt.close()


# ============================================================
# 14. Error vs SOC
# ============================================================

plt.figure(figsize=(7, 5))

plt.scatter(
    analysis["soc"],
    absolute_error
)

plt.xlabel("SOC")
plt.ylabel("Absolute Error (°C)")
plt.title("V5 MLP Error vs SOC")

plt.grid(True)

plt.tight_layout()

plot_file = os.path.join(
    PLOT_DIR,
    "error_vs_soc.png"
)

plt.savefig(
    plot_file,
    dpi=200
)

plt.close()


# ============================================================
# 15. Error vs SOH
# ============================================================

plt.figure(figsize=(7, 5))

plt.scatter(
    analysis["soh"],
    absolute_error
)

plt.xlabel("SOH")
plt.ylabel("Absolute Error (°C)")
plt.title("V5 MLP Error vs SOH")

plt.grid(True)

plt.tight_layout()

plot_file = os.path.join(
    PLOT_DIR,
    "error_vs_soh.png"
)

plt.savefig(
    plot_file,
    dpi=200
)

plt.close()


# ============================================================
# 16. Error vs Current
# ============================================================

plt.figure(figsize=(7, 5))

plt.scatter(
    analysis["current_A"],
    absolute_error
)

plt.xlabel("Current (A)")
plt.ylabel("Absolute Error (°C)")
plt.title("V5 MLP Error vs Current")

plt.grid(True)

plt.tight_layout()

plot_file = os.path.join(
    PLOT_DIR,
    "error_vs_current.png"
)

plt.savefig(
    plot_file,
    dpi=200
)

plt.close()


# ============================================================
# 17. Error vs Initial Temperature
# ============================================================

plt.figure(figsize=(7, 5))

plt.scatter(
    analysis["initial_temperature_C"],
    absolute_error
)

plt.xlabel("Initial Temperature (°C)")
plt.ylabel("Absolute Error (°C)")
plt.title("V5 MLP Error vs Initial Temperature")

plt.grid(True)

plt.tight_layout()

plot_file = os.path.join(
    PLOT_DIR,
    "error_vs_initial_temperature.png"
)

plt.savefig(
    plot_file,
    dpi=200
)

plt.close()


print()
print("=" * 70)
print("PLOTS SAVED")
print("=" * 70)

print(PLOT_DIR)

print()
print("Analysis complete.")