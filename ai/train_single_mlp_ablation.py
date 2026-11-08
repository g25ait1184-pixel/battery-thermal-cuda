
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
# 1. CONFIGURATION
# ============================================================

DATA_DIR = "dataset/v5_1000"

MODEL_DIR = "ai/models"
RESULT_DIR = "ai/results"

MODEL_PATH = os.path.join(
    MODEL_DIR, "v5_single_mlp_2258.pt"
)

PREDICTION_PATH = os.path.join(
    RESULT_DIR, "single_mlp_ablation_predictions.csv"
)

METRICS_PATH = os.path.join(
    RESULT_DIR, "single_mlp_ablation_metrics.csv"
)

SEED = 42
EPOCHS = 1000
LEARNING_RATE = 0.001

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ============================================================
# 2. REPRODUCIBILITY AND DEVICE
# ============================================================

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 65)
print("PARAMETER-MATCHED SINGLE MLP ABLATION")
print("=" * 65)

print("PyTorch:", torch.__version__)
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ============================================================
# 3. LOAD DATASET
# ============================================================

metadata = pd.read_csv(
    os.path.join(DATA_DIR, "metadata.csv")
)

thermal = pd.read_csv(
    os.path.join(DATA_DIR, "thermal_metrics.csv")
)

df = metadata.merge(
    thermal,
    on="case_id",
    how="inner"
)

print("\nDataset shape:", df.shape)

# ============================================================
# 4. INPUT FEATURES AND TARGET
# ============================================================

features = [
    "soc",
    "soh",
    "current_A",
    "initial_temperature_C"
]

target = "T_max_C"

X = df[features].values.astype(np.float32)

y = df[target].values.astype(
    np.float32
).reshape(-1, 1)

# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

# ============================================================
# 6. NORMALIZATION
# ============================================================

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

y_train_scaled = y_scaler.fit_transform(y_train)

# ============================================================
# 7. PYTORCH TENSORS
# ============================================================

X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32,
    device=device
)

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32,
    device=device
)

y_train_tensor = torch.tensor(
    y_train_scaled,
    dtype=torch.float32,
    device=device
)

# ============================================================
# 8. PARAMETER-MATCHED SINGLE MLP
#
# Architecture: 4-25-37-30-1
# Total parameters: 2258
# ============================================================

class SingleMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(4, 25),
            nn.ReLU(),

            nn.Linear(25, 37),
            nn.ReLU(),

            nn.Linear(37, 30),
            nn.ReLU(),

            nn.Linear(30, 1)
        )

    def forward(self, x):
        return self.network(x)


model = SingleMLP().to(device)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\nModel architecture:")
print(model)

print("\nTrainable parameters:", parameter_count)

assert parameter_count == 2258, (
    f"Expected 2258 parameters, got {parameter_count}"
)

# ============================================================
# 9. LOSS AND OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ============================================================
# 10. TRAINING
# ============================================================

print("\nStarting training...")

for epoch in range(EPOCHS):

    model.train()

    optimizer.zero_grad()

    prediction = model(X_train_tensor)

    loss = criterion(
        prediction,
        y_train_tensor
    )

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 100 == 0:

        print(
            f"Epoch [{epoch + 1:4d}/{EPOCHS}] "
            f"Loss = {loss.item():.6f}"
        )

# ============================================================
# 11. TEST PREDICTION
# ============================================================

model.eval()

with torch.no_grad():

    y_pred_scaled = model(
        X_test_tensor
    ).cpu().numpy()

# Convert predictions back to Celsius

y_pred = y_scaler.inverse_transform(
    y_pred_scaled
)

y_actual = y_test

# ============================================================
# 12. EVALUATION METRICS
# ============================================================

rmse = np.sqrt(
    mean_squared_error(
        y_actual,
        y_pred
    )
)

mae = mean_absolute_error(
    y_actual,
    y_pred
)

r2 = r2_score(
    y_actual,
    y_pred
)

print("\n" + "=" * 65)
print("SINGLE MLP ABLATION RESULTS")
print("=" * 65)

print(f"Parameters : {parameter_count}")
print(f"RMSE       : {rmse:.6f} °C")
print(f"MAE        : {mae:.6f} °C")
print(f"R²         : {r2:.6f}")

# ============================================================
# 13. SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print("\nModel saved:", MODEL_PATH)

# ============================================================
# 14. SAVE PREDICTIONS
# ============================================================

results = pd.DataFrame({

    "Actual_Tmax_C":
        y_actual.flatten(),

    "Predicted_Tmax_C":
        y_pred.flatten(),

    "Error_C":
        (
            y_pred.flatten()
            - y_actual.flatten()
        )
})

results.to_csv(
    PREDICTION_PATH,
    index=False
)

print("Predictions saved:", PREDICTION_PATH)

# ============================================================
# 15. SAVE METRICS
# ============================================================

metrics = pd.DataFrame([{

    "Model": "Parameter-Matched Single MLP",

    "Architecture": "4-25-37-30-1",

    "Parameters": parameter_count,

    "RMSE": rmse,

    "MAE": mae,

    "R2": r2,

    "Epochs": EPOCHS,

    "Learning_Rate": LEARNING_RATE,

    "Seed": SEED
}])

metrics.to_csv(
    METRICS_PATH,
    index=False
)

print("Metrics saved:", METRICS_PATH)

# ============================================================
# 16. COMPLETION
# ============================================================

print("\n" + "=" * 65)
print("ABLATION EXPERIMENT COMPLETED")
print("=" * 65)
