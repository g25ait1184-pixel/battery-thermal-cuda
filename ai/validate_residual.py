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

OUTPUT_RESULTS = "ai/results/v5_residual_multisplit.csv"

SEEDS = [10, 20, 30, 40, 42]

EPOCHS = 1000
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL DEFINITIONS
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

y = df[target].values.astype(
    np.float32
).reshape(-1, 1)


print("=" * 70)
print("MULTI-SPLIT PHYSICS + AI RESIDUAL VALIDATION")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Total simulations: {len(df)}")
print(f"Seeds: {SEEDS}")


# ============================================================
# RESULTS STORAGE
# ============================================================

all_results = []


# ============================================================
# RUN EACH SEED
# ============================================================

for seed in SEEDS:

    print("\n")
    print("=" * 70)
    print(f"SEED {seed}")
    print("=" * 70)


    # --------------------------------------------------------
    # REPRODUCIBILITY
    # --------------------------------------------------------

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    indices = np.arange(len(df))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=seed
    )

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_physics_train = y[train_idx]
    y_physics_test = y[test_idx]


    print(
        f"Training cases: {len(train_idx)}"
    )

    print(
        f"Test cases    : {len(test_idx)}"
    )


    # --------------------------------------------------------
    # INPUT SCALER
    # --------------------------------------------------------

    x_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(
        X_train
    )

    X_test_scaled = x_scaler.transform(
        X_test
    )


    # --------------------------------------------------------
    # TARGET SCALER
    #
    # Reconstruct the scaler used by the original MLP.
    # --------------------------------------------------------

    y_scaler = StandardScaler()

    y_scaler.fit(
        y_physics_train
    )


    # --------------------------------------------------------
    # LOAD FROZEN BASE AI
    # --------------------------------------------------------

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

    for param in base_model.parameters():
        param.requires_grad = False


    # --------------------------------------------------------
    # CUDA INPUTS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # BASE AI PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        y_ai_train_scaled = base_model(
            X_train_tensor
        ).cpu().numpy()

        y_ai_test_scaled = base_model(
            X_test_tensor
        ).cpu().numpy()


    # Convert standardized AI output to °C

    y_ai_train = y_scaler.inverse_transform(
        y_ai_train_scaled
    )

    y_ai_test = y_scaler.inverse_transform(
        y_ai_test_scaled
    )


    # --------------------------------------------------------
    # BASE AI METRICS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # RESIDUAL
    #
    # Physics - AI
    # --------------------------------------------------------

    residual_train = (
        y_physics_train -
        y_ai_train
    )

    residual_test = (
        y_physics_test -
        y_ai_test
    )


    print("\nBase AI:")
    print(
        f"  RMSE = {base_rmse:.6f} °C"
    )
    print(
        f"  MAE  = {base_mae:.6f} °C"
    )
    print(
        f"  R²   = {base_r2:.6f}"
    )

    print("\nTraining residual:")
    print(
        f"  Mean = {np.mean(residual_train):.6f} °C"
    )
    print(
        f"  Std  = {np.std(residual_train):.6f} °C"
    )


    # --------------------------------------------------------
    # RESIDUAL MODEL
    # --------------------------------------------------------

    residual_model = ResidualMLP().to(
        DEVICE
    )


    residual_train_tensor = torch.tensor(
        residual_train,
        dtype=torch.float32,
        device=DEVICE
    )


    optimizer = torch.optim.Adam(
        residual_model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.MSELoss()


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # RESIDUAL TEST PREDICTION
    # --------------------------------------------------------

    residual_model.eval()

    with torch.no_grad():

        residual_pred = residual_model(
            X_test_tensor
        ).cpu().numpy()


    # --------------------------------------------------------
    # HYBRID PREDICTION
    # --------------------------------------------------------

    y_hybrid = (
        y_ai_test +
        residual_pred
    )


    # --------------------------------------------------------
    # HYBRID METRICS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # IMPROVEMENT
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print("\nHybrid Physics + AI:")

    print(
        f"  RMSE = {hybrid_rmse:.6f} °C"
    )

    print(
        f"  MAE  = {hybrid_mae:.6f} °C"
    )

    print(
        f"  R²   = {hybrid_r2:.6f}"
    )

    print(
        f"  RMSE improvement = "
        f"{rmse_improvement:.2f}%"
    )

    print(
        f"  MAE improvement  = "
        f"{mae_improvement:.2f}%"
    )


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    all_results.append({

        "seed": seed,

        "train_cases": len(train_idx),
        "test_cases": len(test_idx),

        "base_rmse_C": base_rmse,
        "base_mae_C": base_mae,
        "base_r2": base_r2,

        "hybrid_rmse_C": hybrid_rmse,
        "hybrid_mae_C": hybrid_mae,
        "hybrid_r2": hybrid_r2,

        "rmse_improvement_percent":
            rmse_improvement,

        "mae_improvement_percent":
            mae_improvement,

        "residual_train_mean_C":
            np.mean(residual_train),

        "residual_train_std_C":
            np.std(residual_train),

        "residual_test_mean_C":
            np.mean(residual_test),

        "residual_test_std_C":
            np.std(residual_test)
    })


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    all_results
)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

base_rmse_mean = results_df[
    "base_rmse_C"
].mean()

base_rmse_std = results_df[
    "base_rmse_C"
].std()

base_mae_mean = results_df[
    "base_mae_C"
].mean()

base_mae_std = results_df[
    "base_mae_C"
].std()

base_r2_mean = results_df[
    "base_r2"
].mean()

base_r2_std = results_df[
    "base_r2"
].std()


hybrid_rmse_mean = results_df[
    "hybrid_rmse_C"
].mean()

hybrid_rmse_std = results_df[
    "hybrid_rmse_C"
].std()

hybrid_mae_mean = results_df[
    "hybrid_mae_C"
].mean()

hybrid_mae_std = results_df[
    "hybrid_mae_C"
].std()

hybrid_r2_mean = results_df[
    "hybrid_r2"
].mean()

hybrid_r2_std = results_df[
    "hybrid_r2"
].std()


improvement_mean = results_df[
    "rmse_improvement_percent"
].mean()

improvement_std = results_df[
    "rmse_improvement_percent"
].std()


mae_improvement_mean = results_df[
    "mae_improvement_percent"
].mean()

mae_improvement_std = results_df[
    "mae_improvement_percent"
].std()


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("MULTI-SPLIT SUMMARY")
print("=" * 70)

print("\nBASE AI")
print("-" * 70)

print(
    f"RMSE : "
    f"{base_rmse_mean:.6f} "
    f"+/- {base_rmse_std:.6f} °C"
)

print(
    f"MAE  : "
    f"{base_mae_mean:.6f} "
    f"+/- {base_mae_std:.6f} °C"
)

print(
    f"R²   : "
    f"{base_r2_mean:.6f} "
    f"+/- {base_r2_std:.6f}"
)


print("\nHYBRID PHYSICS + AI")
print("-" * 70)

print(
    f"RMSE : "
    f"{hybrid_rmse_mean:.6f} "
    f"+/- {hybrid_rmse_std:.6f} °C"
)

print(
    f"MAE  : "
    f"{hybrid_mae_mean:.6f} "
    f"+/- {hybrid_mae_std:.6f} °C"
)

print(
    f"R²   : "
    f"{hybrid_r2_mean:.6f} "
    f"+/- {hybrid_r2_std:.6f}"
)


print("\nIMPROVEMENT")
print("-" * 70)

print(
    f"RMSE improvement : "
    f"{improvement_mean:.2f}% "
    f"+/- {improvement_std:.2f}%"
)

print(
    f"MAE improvement  : "
    f"{mae_improvement_mean:.2f}% "
    f"+/- {mae_improvement_std:.2f}%"
)


# ============================================================
# PER-SEED TABLE
# ============================================================

print("\n")
print("=" * 70)
print("PER-SEED RESULTS")
print("=" * 70)

print(
    results_df[
        [
            "seed",
            "base_rmse_C",
            "hybrid_rmse_C",
            "rmse_improvement_percent",
            "base_mae_C",
            "hybrid_mae_C",
            "base_r2",
            "hybrid_r2"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_RESULTS),
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_RESULTS,
    index=False
)


print("\n")
print("=" * 70)
print("SAVED")
print("=" * 70)

print(
    f"Results: {OUTPUT_RESULTS}"
)

print("\nDone.")