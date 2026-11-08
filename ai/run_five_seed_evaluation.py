"""
Five-seed evaluation for the V5 battery thermal AI surrogate.

Runs Base MLP and Base+Residual MLP for each seed, using:
- Fixed 80/20 train-test split (random_state=42)
- Same four input features and V5 target
- StandardScaler fitted on training data only
- 1,000 epochs and Adam(lr=0.001)
- Separate base checkpoint for each seed

Outputs:
  ai/results/five_seed_metrics.csv
  ai/results/five_seed_summary.csv
  ai/results/five_seed_predictions.csv
  ai/results/figure4_five_seed.png
  ai/models/v5_base_seed_<seed>.pt
  ai/models/v5_residual_seed_<seed>.pt

Run from project root:
  !python ai/run_five_seed_evaluation.py
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ----------------------------- Configuration -----------------------------
DATA_DIR = "dataset/v5_1000"
MODEL_DIR = "ai/models"
RESULT_DIR = "ai/results"

SEEDS = [42, 123, 2024, 7, 21]
SPLIT_SEED = 42
EPOCHS = 1000
LEARNING_RATE = 0.001
TEST_SIZE = 0.20

FEATURES = ["soc", "soh", "current_A", "initial_temperature_C"]
TARGET = "T_max_C"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class V5ThermalMLP(nn.Module):
    """Base MLP: 4-32-32-16-1 (1,761 trainable parameters)."""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.network(x)


class ResidualMLP(nn.Module):
    """Residual MLP: 4-16-16-8-1 (497 trainable parameters)."""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.network(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(model, x_tensor, y_tensor, epochs, lr, label):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        prediction = model(x_tensor)
        loss = criterion(prediction, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 250 == 0:
            print(f"    {label} | Epoch {epoch + 1:4d}/{epochs} "
                  f"| Loss={loss.item():.8f}")


def calculate_metrics(y_true, y_pred):
    return {
        "RMSE_C": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE_C": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("=" * 76)
    print("FIVE-SEED V5 BASE MLP AND RESIDUAL-LEARNING EVALUATION")
    print("=" * 76)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Seeds: {SEEDS}")
    print(f"Epochs per model: {EPOCHS}")

    metadata_path = os.path.join(DATA_DIR, "metadata.csv")
    thermal_path = os.path.join(DATA_DIR, "thermal_metrics.csv")
    metadata = pd.read_csv(metadata_path)
    thermal = pd.read_csv(thermal_path)
    df = metadata.merge(thermal, on="case_id", how="inner")

    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.float32).reshape(-1, 1)
    indices = np.arange(len(df))

    # Fixed split shared by every seed and every model.
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=SPLIT_SEED
    )
    print(f"Dataset: {len(df)} | Train: {len(train_idx)} | Test: {len(test_idx)}")

    # Scalers are fit only on the fixed training partition.
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X[train_idx]).astype(np.float32)
    X_test_scaled = x_scaler.transform(X[test_idx]).astype(np.float32)
    y_train_scaled = y_scaler.fit_transform(y[train_idx]).astype(np.float32)

    x_train_t = torch.tensor(X_train_scaled, dtype=torch.float32, device=DEVICE)
    x_test_t = torch.tensor(X_test_scaled, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32, device=DEVICE)

    y_train_c = y[train_idx]
    y_test_c = y[test_idx]
    all_metric_rows = []
    prediction_frames = []

    for seed in SEEDS:
        print("\n" + "-" * 76)
        print(f"SEED {seed}")
        print("-" * 76)
        set_seed(seed)

        # Train base MLP.
        base_model = V5ThermalMLP().to(DEVICE)
        print("  Training Base MLP...")
        train_model(base_model, x_train_t, y_train_t, EPOCHS,
                    LEARNING_RATE, "Base MLP")

        base_model.eval()
        with torch.no_grad():
            base_pred_scaled = base_model(x_test_t).cpu().numpy()
        base_pred_c = y_scaler.inverse_transform(base_pred_scaled)
        base_metrics = calculate_metrics(y_test_c, base_pred_c)

        base_path = os.path.join(MODEL_DIR, f"v5_base_seed_{seed}.pt")
        torch.save(base_model.state_dict(), base_path)

        # Freeze the seed-matched base model and train residual on its
        # training-set errors, following the uploaded train_residual.py.
        for parameter in base_model.parameters():
            parameter.requires_grad = False
        base_model.eval()

        with torch.no_grad():
            base_train_scaled = base_model(x_train_t).cpu().numpy()
        base_train_c = y_scaler.inverse_transform(base_train_scaled)
        residual_train_c = y_train_c - base_train_c

        residual_target_t = torch.tensor(
            residual_train_c, dtype=torch.float32, device=DEVICE
        )

        residual_model = ResidualMLP().to(DEVICE)
        print("  Training Residual MLP...")
        train_model(residual_model, x_train_t, residual_target_t,
                    EPOCHS, LEARNING_RATE, "Residual MLP")

        residual_model.eval()
        with torch.no_grad():
            residual_pred_c = residual_model(x_test_t).cpu().numpy()
        hybrid_pred_c = base_pred_c + residual_pred_c
        residual_metrics = calculate_metrics(y_test_c, hybrid_pred_c)

        residual_path = os.path.join(MODEL_DIR, f"v5_residual_seed_{seed}.pt")
        torch.save(residual_model.state_dict(), residual_path)

        print(f"  Base     RMSE={base_metrics['RMSE_C']:.6f} | "
              f"MAE={base_metrics['MAE_C']:.6f} | R2={base_metrics['R2']:.6f}")
        print(f"  Residual RMSE={residual_metrics['RMSE_C']:.6f} | "
              f"MAE={residual_metrics['MAE_C']:.6f} | R2={residual_metrics['R2']:.6f}")

        all_metric_rows.append({"Seed": seed, "Model": "Base MLP", **base_metrics})
        all_metric_rows.append({"Seed": seed, "Model": "Residual MLP", **residual_metrics})

        prediction_frames.append(pd.DataFrame({
            "Seed": seed,
            "case_id": df.iloc[test_idx]["case_id"].to_numpy(),
            "Actual_Tmax_C": y_test_c.flatten(),
            "Base_MLP_Prediction_C": base_pred_c.flatten(),
            "Residual_MLP_Prediction_C": hybrid_pred_c.flatten(),
            "Base_Error_C": (base_pred_c - y_test_c).flatten(),
            "Residual_Error_C": (hybrid_pred_c - y_test_c).flatten(),
        }))

        del base_model, residual_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics_df = pd.DataFrame(all_metric_rows)
    metrics_path = os.path.join(RESULT_DIR, "five_seed_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    summary_rows = []
    for model_name, group in metrics_df.groupby("Model", sort=False):
        for metric in ["RMSE_C", "MAE_C", "R2"]:
            values = group[metric].to_numpy()
            summary_rows.append({
                "Model": model_name,
                "Metric": metric,
                "Mean": float(np.mean(values)),
                "Std": float(np.std(values, ddof=1)),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(RESULT_DIR, "five_seed_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    predictions_path = os.path.join(RESULT_DIR, "five_seed_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)

    print("\n" + "=" * 76)
    print("FIVE-SEED SUMMARY (mean ± sample standard deviation)")
    print("=" * 76)
    for model_name in ["Base MLP", "Residual MLP"]:
        sub = summary_df[summary_df["Model"] == model_name].set_index("Metric")
        print(f"\n{model_name}")
        for metric in ["RMSE_C", "MAE_C", "R2"]:
            mean = sub.loc[metric, "Mean"]
            std = sub.loc[metric, "Std"]
            unit = " °C" if metric != "R2" else ""
            print(f"  {metric}: {mean:.6f} ± {std:.6f}{unit}")

    # Figure 4: mean errors with standard-deviation error bars.
    metric_labels = ["RMSE", "MAE"]
    metric_keys = ["RMSE_C", "MAE_C"]
    models = ["Base MLP", "Residual MLP"]
    x = np.arange(len(metric_labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for offset, model_name in zip([-width / 2, width / 2], models):
        means, stds = [], []
        sub = summary_df[summary_df["Model"] == model_name].set_index("Metric")
        for key in metric_keys:
            means.append(sub.loc[key, "Mean"])
            stds.append(sub.loc[key, "Std"])
        bars = ax.bar(
            x + offset, means, width,
            yerr=stds, capsize=5, label=model_name
        )
        ax.bar_label(bars, labels=[f"{v:.4f}" for v in means],
                     padding=5, fontsize=9, fontweight="bold")

    ax.set_ylabel("Error (°C)")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontweight="bold")
    ax.set_title("Five-seed thermal prediction error (mean ± SD)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()

    figure_path = os.path.join(RESULT_DIR, "figure4_five_seed.png")
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved files:")
    for path in [metrics_path, summary_path, predictions_path, figure_path]:
        print(" ", path)
    print("\nFive-seed evaluation complete.")


if __name__ == "__main__":
    main()
