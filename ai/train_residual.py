import os
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
RESULTS_PATH = "ai/results/v5_hybrid_predictions.csv"

SEED = 42
EPOCHS = 1000
LEARNING_RATE = 0.001

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


print("=" * 70)
print("PHYSICS + AI RESIDUAL LEARNING")
print("=" * 70)

print(f"Device: {DEVICE}")


# ============================================================
# LOAD DATA
# ============================================================

metadata_path = os.path.join(
    DATA_DIR,
    "metadata.csv"
)

thermal_path = os.path.join(
    DATA_DIR,
    "thermal_metrics.csv"
)

metadata = pd.read_csv(metadata_path)
thermal = pd.read_csv(thermal_path)

df = metadata.merge(
    thermal,
    on="case_id",
    how="inner"
)

print(f"\nTotal simulations: {len(df)}")


# ============================================================
# FEATURES AND PHYSICS TARGET
# ============================================================

features = [
    "soc",
    "soh",
    "current_A",
    "initial_temperature_C"
]

target = "T_max_C"

X = df[features].values.astype(
    np.float32
)

y_physics = df[target].values.astype(
    np.float32
).reshape(-1, 1)


# ============================================================
# EXACT SAME TRAIN / TEST SPLIT
# ============================================================

indices = np.arange(len(df))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=SEED
)

print(f"Training cases: {len(train_idx)}")
print(f"Test cases    : {len(test_idx)}")


X_train = X[train_idx]
X_test = X[test_idx]

y_physics_train = y_physics[train_idx]
y_physics_test = y_physics[test_idx]


# ============================================================
# INPUT SCALER
#
# EXACTLY MATCHES ORIGINAL TRAINING
# ============================================================

x_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(
    X_train
)

X_test_scaled = x_scaler.transform(
    X_test
)


# ============================================================
# TARGET SCALER
#
# IMPORTANT:
# Original MLP was trained using a StandardScaler on y.
#
# We reconstruct the same scaler using ONLY the 800
# training samples.
# ============================================================

y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(
    y_physics_train
)

y_test_scaled = y_scaler.transform(
    y_physics_test
)


# ============================================================
# BASE AI MODEL
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
# LOAD FROZEN BASE MODEL
# ============================================================

base_model = V5ThermalMLP().to(DEVICE)

base_model.load_state_dict(
    torch.load(
        BASE_MODEL_PATH,
        map_location=DEVICE
    )
)

base_model.eval()

# Freeze ALL base-model parameters
for param in base_model.parameters():
    param.requires_grad = False


print("\nBase AI model loaded:")
print(BASE_MODEL_PATH)

print("Base AI model: FROZEN")


# ============================================================
# CONVERT INPUTS TO CUDA
# ============================================================

X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32,
    device=DEVICE
)

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32,
    device=DEVICE
)


# ============================================================
# BASE AI PREDICTION
#
# Model outputs STANDARDIZED temperature.
# Convert back to °C before calculating residual.
# ============================================================

with torch.no_grad():

    y_ai_train_scaled = base_model(
        X_train_tensor
    ).cpu().numpy()

    y_ai_test_scaled = base_model(
        X_test_tensor
    ).cpu().numpy()


# Convert AI predictions from standardized units -> °C

y_ai_train = y_scaler.inverse_transform(
    y_ai_train_scaled
)

y_ai_test = y_scaler.inverse_transform(
    y_ai_test_scaled
)


# ============================================================
# VERIFY BASE AI PERFORMANCE
# ============================================================

base_rmse = np.sqrt(
    mean_squared_error(
        y_physics_test,
        y_ai_test
    )
)

base_mae = mean_absolute_error(
    y_physics_test,
    y_ai_test
)

base_r2 = r2_score(
    y_physics_test,
    y_ai_test
)


print("\n")
print("=" * 70)
print("BASE AI VALIDATION")
print("=" * 70)

print(f"RMSE : {base_rmse:.6f} °C")
print(f"MAE  : {base_mae:.6f} °C")
print(f"R²   : {base_r2:.6f}")


# ============================================================
# RESIDUAL
#
# Physics - AI
# ============================================================

residual_train = (
    y_physics_train -
    y_ai_train
)

residual_test = (
    y_physics_test -
    y_ai_test
)


print("\n")
print("=" * 70)
print("RESIDUAL STATISTICS")
print("=" * 70)

print("\nTraining residual:")
print(
    f"Mean : {np.mean(residual_train):.6f} °C"
)

print(
    f"Std  : {np.std(residual_train):.6f} °C"
)

print(
    f"MAE  : "
    f"{np.mean(np.abs(residual_train)):.6f} °C"
)

print(
    f"Max  : "
    f"{np.max(np.abs(residual_train)):.6f} °C"
)


print("\nTest residual:")
print(
    f"Mean : {np.mean(residual_test):.6f} °C"
)

print(
    f"Std  : {np.std(residual_test):.6f} °C"
)

print(
    f"MAE  : "
    f"{np.mean(np.abs(residual_test)):.6f} °C"
)

print(
    f"Max  : "
    f"{np.max(np.abs(residual_test)):.6f} °C"
)


# ============================================================
# RESIDUAL MODEL
# ============================================================

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


residual_model = ResidualMLP().to(DEVICE)


# ============================================================
# RESIDUAL TRAINING DATA
#
# IMPORTANT:
# Residual is kept directly in °C.
# No additional target scaler is used.
# ============================================================

residual_train_tensor = torch.tensor(
    residual_train,
    dtype=torch.float32,
    device=DEVICE
)


# ============================================================
# OPTIMIZER / LOSS
# ============================================================

optimizer = torch.optim.Adam(
    residual_model.parameters(),
    lr=LEARNING_RATE
)

criterion = nn.MSELoss()


# ============================================================
# TRAIN RESIDUAL MODEL
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING RESIDUAL MODEL")
print("=" * 70)

for epoch in range(EPOCHS):

    residual_model.train()

    optimizer.zero_grad()

    residual_prediction = residual_model(
        X_train_tensor
    )

    loss = criterion(
        residual_prediction,
        residual_train_tensor
    )

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 100 == 0:

        print(
            f"Epoch {epoch + 1:4d} | "
            f"Loss = {loss.item():.8f}"
        )


# ============================================================
# TEST RESIDUAL MODEL
# ============================================================

residual_model.eval()

with torch.no_grad():

    residual_pred = residual_model(
        X_test_tensor
    ).cpu().numpy()


# ============================================================
# HYBRID PREDICTION
#
# T_hybrid = T_AI + predicted residual
# ============================================================

y_hybrid = (
    y_ai_test +
    residual_pred
)


# ============================================================
# HYBRID METRICS
# ============================================================

hybrid_rmse = np.sqrt(
    mean_squared_error(
        y_physics_test,
        y_hybrid
    )
)

hybrid_mae = mean_absolute_error(
    y_physics_test,
    y_hybrid
)

hybrid_r2 = r2_score(
    y_physics_test,
    y_hybrid
)


# ============================================================
# HYBRID ERROR
# ============================================================

ai_error = (
    y_ai_test -
    y_physics_test
)

hybrid_error = (
    y_hybrid -
    y_physics_test
)


# ============================================================
# IMPROVEMENT
# ============================================================

rmse_improvement = (
    (base_rmse - hybrid_rmse)
    / base_rmse
    * 100.0
)

mae_improvement = (
    (base_mae - hybrid_mae)
    / base_mae
    * 100.0
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print("\nBASE AI")
print("-" * 70)

print(
    f"RMSE : {base_rmse:.6f} °C"
)

print(
    f"MAE  : {base_mae:.6f} °C"
)

print(
    f"R²   : {base_r2:.6f}"
)


print("\nHYBRID PHYSICS + AI")
print("-" * 70)

print(
    f"RMSE : {hybrid_rmse:.6f} °C"
)

print(
    f"MAE  : {hybrid_mae:.6f} °C"
)

print(
    f"R²   : {hybrid_r2:.6f}"
)


print("\nIMPROVEMENT")
print("-" * 70)

print(
    f"RMSE improvement : "
    f"{rmse_improvement:.2f}%"
)

print(
    f"MAE improvement  : "
    f"{mae_improvement:.2f}%"
)


# ============================================================
# WORST HYBRID ERRORS
# ============================================================

abs_hybrid_error = np.abs(
    hybrid_error.flatten()
)

worst_indices = np.argsort(
    abs_hybrid_error
)[::-1][:10]


print("\n")
print("=" * 70)
print("WORST 10 HYBRID PREDICTIONS")
print("=" * 70)

for rank, idx in enumerate(
    worst_indices,
    start=1
):

    case_id = df.iloc[
        test_idx[idx]
    ]["case_id"]

    print(
        f"{rank:2d}. "
        f"case {int(case_id):4d} | "
        f"Physics = {y_physics_test[idx,0]:8.4f} °C | "
        f"AI = {y_ai_test[idx,0]:8.4f} °C | "
        f"Hybrid = {y_hybrid[idx,0]:8.4f} °C | "
        f"Error = {hybrid_error[idx,0]:+8.4f} °C"
    )


# ============================================================
# SAVE RESIDUAL MODEL
# ============================================================

os.makedirs(
    os.path.dirname(RESIDUAL_MODEL_PATH),
    exist_ok=True
)

torch.save(
    residual_model.state_dict(),
    RESIDUAL_MODEL_PATH
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

results = df.iloc[test_idx].copy()

results["T_physics_C"] = (
    y_physics_test.flatten()
)

results["T_AI_C"] = (
    y_ai_test.flatten()
)

results["residual_actual_C"] = (
    residual_test.flatten()
)

results["residual_predicted_C"] = (
    residual_pred.flatten()
)

results["T_hybrid_C"] = (
    y_hybrid.flatten()
)

results["AI_error_C"] = (
    ai_error.flatten()
)

results["hybrid_error_C"] = (
    hybrid_error.flatten()
)

results["absolute_AI_error_C"] = (
    np.abs(ai_error.flatten())
)

results["absolute_hybrid_error_C"] = (
    np.abs(hybrid_error.flatten())
)


os.makedirs(
    os.path.dirname(RESULTS_PATH),
    exist_ok=True
)

results.to_csv(
    RESULTS_PATH,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    f"Residual model : "
    f"{RESIDUAL_MODEL_PATH}"
)

print(
    f"Predictions    : "
    f"{RESULTS_PATH}"
)

print("\nDone.")