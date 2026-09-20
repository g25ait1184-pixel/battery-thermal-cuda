import os
import numpy as np
import pandas as pd

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

RESULT_DIR = os.path.join(
    SCRIPT_DIR,
    "results"
)

os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# 2. Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("V5 MLP - MULTI-SPLIT VALIDATION")
print("=" * 70)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print()


# ============================================================
# 3. Load dataset
# ============================================================

metadata = pd.read_csv(METADATA_FILE)
thermal = pd.read_csv(THERMAL_FILE)

df = pd.merge(
    metadata,
    thermal,
    on="case_id"
)

print("Total simulations:", len(df))


# ============================================================
# 4. Features and target
# ============================================================

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
# 5. MLP
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
# 6. Validation configuration
# ============================================================

seeds = [
    10,
    20,
    30,
    40,
    42
]

epochs = 1000
learning_rate = 0.001

results = []


# ============================================================
# 7. Run multiple train/test splits
# ============================================================

for seed in seeds:

    print()
    print("-" * 70)
    print(f"Validation seed: {seed}")
    print("-" * 70)

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(
        X_train
    ).astype(np.float32)

    X_test_scaled = x_scaler.transform(
        X_test
    ).astype(np.float32)

    y_train_scaled = y_scaler.fit_transform(
        y_train
    ).astype(np.float32)

    # --------------------------------------------------------
    # Tensors
    # --------------------------------------------------------

    X_train_tensor = torch.tensor(
        X_train_scaled,
        dtype=torch.float32,
        device=device
    )

    y_train_tensor = torch.tensor(
        y_train_scaled,
        dtype=torch.float32,
        device=device
    )

    X_test_tensor = torch.tensor(
        X_test_scaled,
        dtype=torch.float32,
        device=device
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = V5ThermalMLP().to(device)

    # --------------------------------------------------------
    # Loss + optimizer
    # --------------------------------------------------------

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(epochs):

        model.train()

        optimizer.zero_grad()

        prediction = model(
            X_train_tensor
        )

        loss = criterion(
            prediction,
            y_train_tensor
        )

        loss.backward()

        optimizer.step()

    # --------------------------------------------------------
    # Test prediction
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        y_pred_scaled = model(
            X_test_tensor
        ).cpu().numpy()

    # --------------------------------------------------------
    # Convert back to °C
    # --------------------------------------------------------

    y_pred = y_scaler.inverse_transform(
        y_pred_scaled
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    print(f"RMSE : {rmse:.6f} °C")
    print(f"MAE  : {mae:.6f} °C")
    print(f"R²   : {r2:.6f}")

    results.append({
        "seed": seed,
        "RMSE_C": rmse,
        "MAE_C": mae,
        "R2": r2
    })


# ============================================================
# 8. Results dataframe
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# 9. Statistics
# ============================================================

rmse_mean = results_df["RMSE_C"].mean()
rmse_std = results_df["RMSE_C"].std()

mae_mean = results_df["MAE_C"].mean()
mae_std = results_df["MAE_C"].std()

r2_mean = results_df["R2"].mean()
r2_std = results_df["R2"].std()


# ============================================================
# 10. Print final validation summary
# ============================================================

print()
print("=" * 70)
print("MULTI-SPLIT VALIDATION SUMMARY")
print("=" * 70)

print()
print("Individual results:")
print(results_df.to_string(index=False))

print()
print("-" * 70)

print(
    f"RMSE : {rmse_mean:.6f} ± {rmse_std:.6f} °C"
)

print(
    f"MAE  : {mae_mean:.6f} ± {mae_std:.6f} °C"
)

print(
    f"R²   : {r2_mean:.6f} ± {r2_std:.6f}"
)

print("-" * 70)


# ============================================================
# 11. Save results
# ============================================================

result_file = os.path.join(
    RESULT_DIR,
    "v5_mlp_multisplit_validation.csv"
)

results_df.to_csv(
    result_file,
    index=False
)

print()
print("Results saved to:")
print(result_file)

print()
print("Validation complete.")