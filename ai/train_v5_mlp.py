import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# 1. Configuration
# ============================================================

DATASET_DIR = "dataset/v5_1000"

METADATA_FILE = os.path.join(DATASET_DIR, "metadata.csv")
THERMAL_FILE = os.path.join(DATASET_DIR, "thermal_metrics.csv")

MODEL_DIR = "ai/models"
RESULT_DIR = "ai/results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# 2. Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("V5 AI Thermal Predictor")
print("=" * 60)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


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

print("\nDataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 4. Input features and target
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

print("\nInput features:")
for f in features:
    print("  ", f)

print("Target:", target)


# ============================================================
# 5. Train / test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 6. Normalize data
# ============================================================

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)


# ============================================================
# 7. Convert to PyTorch tensors
# ============================================================

X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32
).to(device)

y_train_tensor = torch.tensor(
    y_train_scaled,
    dtype=torch.float32
).to(device)

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32
).to(device)

y_test_tensor = torch.tensor(
    y_test_scaled,
    dtype=torch.float32
).to(device)


# ============================================================
# 8. Define MLP
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


model = V5ThermalMLP().to(device)

print("\nModel:")
print(model)


# ============================================================
# 9. Loss and optimizer
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# 10. Training
# ============================================================

epochs = 1000

print("\nStarting training...")

for epoch in range(epochs):

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
            f"Epoch [{epoch+1:4d}/{epochs}] "
            f"Loss = {loss.item():.6f}"
        )


# ============================================================
# 11. Prediction
# ============================================================

model.eval()

with torch.no_grad():

    y_pred_scaled = model(
        X_test_tensor
    ).cpu().numpy()


# Convert back to °C

y_pred = y_scaler.inverse_transform(
    y_pred_scaled
)

y_actual = y_test


# ============================================================
# 12. Accuracy metrics
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

print("\n" + "=" * 60)
print("MODEL ACCURACY")
print("=" * 60)

print(f"RMSE : {rmse:.6f} °C")
print(f"MAE  : {mae:.6f} °C")
print(f"R²   : {r2:.6f}")


# ============================================================
# 13. Save model
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "v5_tmax_mlp.pt"
)

torch.save(
    model.state_dict(),
    model_path
)

print("\nModel saved to:")
print(model_path)


# ============================================================
# 14. Save predictions
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

prediction_file = os.path.join(
    RESULT_DIR,
    "v5_tmax_predictions.csv"
)

results.to_csv(
    prediction_file,
    index=False
)

print("\nPredictions saved to:")
print(prediction_file)

print("\nFirst 10 predictions:")
print(results.head(10))

print("\nTraining complete.")