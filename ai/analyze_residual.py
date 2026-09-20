import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "ai/results/v5_hybrid_predictions.csv"

OUTPUT_DIR = "ai/results/residual_analysis"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("PHYSICS + AI RESIDUAL ERROR ANALYSIS")
print("=" * 70)

print(f"\nLoaded test cases: {len(df)}")


# ============================================================
# VERIFY REQUIRED COLUMNS
# ============================================================

required_columns = [
    "case_id",
    "soc",
    "soh",
    "current_A",
    "initial_temperature_C",
    "T_physics_C",
    "T_AI_C",
    "residual_actual_C",
    "residual_predicted_C",
    "T_hybrid_C",
    "AI_error_C",
    "hybrid_error_C",
    "absolute_AI_error_C",
    "absolute_hybrid_error_C"
]

missing = [
    c for c in required_columns
    if c not in df.columns
]

if missing:

    raise RuntimeError(
        f"Missing columns: {missing}"
    )


# ============================================================
# BASIC ERROR STATISTICS
# ============================================================

ai_error = df["AI_error_C"].values
hybrid_error = df["hybrid_error_C"].values

abs_ai = df["absolute_AI_error_C"].values
abs_hybrid = df["absolute_hybrid_error_C"].values


print("\n")
print("=" * 70)
print("ERROR STATISTICS")
print("=" * 70)


print("\nBASE AI")
print("-" * 70)

print(
    f"Mean error       : "
    f"{np.mean(ai_error):.6f} °C"
)

print(
    f"Mean abs error   : "
    f"{np.mean(abs_ai):.6f} °C"
)

print(
    f"Std error        : "
    f"{np.std(ai_error):.6f} °C"
)

print(
    f"Max abs error    : "
    f"{np.max(abs_ai):.6f} °C"
)


print("\nHYBRID PHYSICS + AI")
print("-" * 70)

print(
    f"Mean error       : "
    f"{np.mean(hybrid_error):.6f} °C"
)

print(
    f"Mean abs error   : "
    f"{np.mean(abs_hybrid):.6f} °C"
)

print(
    f"Std error        : "
    f"{np.std(hybrid_error):.6f} °C"
)

print(
    f"Max abs error    : "
    f"{np.max(abs_hybrid):.6f} °C"
)


# ============================================================
# ERROR REDUCTION PER CASE
# ============================================================

error_reduction = (
    abs_ai -
    abs_hybrid
)

improved = error_reduction > 0
worsened = error_reduction < 0
unchanged = error_reduction == 0


print("\n")
print("=" * 70)
print("CASE-BY-CASE IMPROVEMENT")
print("=" * 70)

print(
    f"Improved cases : "
    f"{np.sum(improved)}/{len(df)} "
    f"({np.mean(improved)*100:.1f}%)"
)

print(
    f"Worsened cases : "
    f"{np.sum(worsened)}/{len(df)} "
    f"({np.mean(worsened)*100:.1f}%)"
)

print(
    f"Unchanged      : "
    f"{np.sum(unchanged)}/{len(df)}"
)


# ============================================================
# RESIDUAL PREDICTION QUALITY
# ============================================================

actual_residual = df[
    "residual_actual_C"
].values

predicted_residual = df[
    "residual_predicted_C"
].values

residual_error = (
    predicted_residual -
    actual_residual
)

residual_abs_error = np.abs(
    residual_error
)


print("\n")
print("=" * 70)
print("RESIDUAL MODEL PERFORMANCE")
print("=" * 70)

print(
    f"Residual RMSE : "
    f"{np.sqrt(np.mean(residual_error**2)):.6f} °C"
)

print(
    f"Residual MAE  : "
    f"{np.mean(residual_abs_error):.6f} °C"
)

print(
    f"Residual bias : "
    f"{np.mean(residual_error):.6f} °C"
)

print(
    f"Residual max  : "
    f"{np.max(residual_abs_error):.6f} °C"
)


# ============================================================
# CORRELATION WITH ABSOLUTE ERROR
# ============================================================

print("\n")
print("=" * 70)
print("CORRELATION WITH ABSOLUTE ERROR")
print("=" * 70)

variables = {
    "SOC":
        df["soc"].values,

    "SOH":
        df["soh"].values,

    "Current":
        df["current_A"].values,

    "Initial Temperature":
        df["initial_temperature_C"].values,

    "Physics Tmax":
        df["T_physics_C"].values,

    "Base AI Error":
        ai_error,

    "Base AI Absolute Error":
        abs_ai
}

for name, values in variables.items():

    correlation = np.corrcoef(
        values,
        abs_hybrid
    )[0, 1]

    print(
        f"{name:25s}: "
        f"{correlation:+.4f}"
    )


# ============================================================
# CORRELATION: ACTUAL VS PREDICTED RESIDUAL
# ============================================================

residual_correlation = np.corrcoef(
    actual_residual,
    predicted_residual
)[0, 1]

print(
    f"\nActual vs predicted residual correlation: "
    f"{residual_correlation:.6f}"
)


# ============================================================
# ERROR THRESHOLDS
# ============================================================

print("\n")
print("=" * 70)
print("ERROR THRESHOLDS")
print("=" * 70)

thresholds = [
    0.05,
    0.10,
    0.20,
    0.30,
    0.50,
    1.00
]

for threshold in thresholds:

    ai_count = np.sum(
        abs_ai <= threshold
    )

    hybrid_count = np.sum(
        abs_hybrid <= threshold
    )

    total = len(df)

    print(
        f"|error| <= {threshold:.2f} °C : "
        f"AI {ai_count:3d}/{total} "
        f"({ai_count/total*100:5.1f}%) | "
        f"Hybrid {hybrid_count:3d}/{total} "
        f"({hybrid_count/total*100:5.1f}%)"
    )


# ============================================================
# WORST CASES
# ============================================================

print("\n")
print("=" * 70)
print("WORST 10 HYBRID ERRORS")
print("=" * 70)

worst = np.argsort(
    abs_hybrid
)[::-1][:10]

for rank, idx in enumerate(
    worst,
    start=1
):

    row = df.iloc[idx]

    print(
        f"{rank:2d}. "
        f"case {int(row['case_id']):4d} | "
        f"SOC={row['soc']:.3f} | "
        f"SOH={row['soh']:.3f} | "
        f"I={row['current_A']:.2f} A | "
        f"T0={row['initial_temperature_C']:.2f} °C | "
        f"Physics={row['T_physics_C']:.3f} °C | "
        f"AI={row['T_AI_C']:.3f} °C | "
        f"Hybrid={row['T_hybrid_C']:.3f} °C | "
        f"Hybrid error={row['hybrid_error_C']:+.3f} °C"
    )


# ============================================================
# PLOT 1
# PHYSICS vs BASE AI vs HYBRID
# ============================================================

plt.figure(figsize=(7, 6))

plt.scatter(
    df["T_physics_C"],
    df["T_AI_C"],
    alpha=0.7,
    label="Base AI"
)

plt.scatter(
    df["T_physics_C"],
    df["T_hybrid_C"],
    alpha=0.7,
    label="Hybrid"
)

minimum = min(
    df["T_physics_C"].min(),
    df["T_AI_C"].min(),
    df["T_hybrid_C"].min()
)

maximum = max(
    df["T_physics_C"].max(),
    df["T_AI_C"].max(),
    df["T_hybrid_C"].max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    label="Ideal"
)

plt.xlabel("Physics Tmax (°C)")
plt.ylabel("Predicted Tmax (°C)")

plt.title(
    "Physics vs AI and Hybrid Prediction"
)

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "01_physics_ai_hybrid.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# PLOT 2
# BASE AI ERROR vs HYBRID ERROR
# ============================================================

plt.figure(figsize=(7, 6))

plt.scatter(
    abs_ai,
    abs_hybrid,
    alpha=0.7
)

maximum = max(
    abs_ai.max(),
    abs_hybrid.max()
)

plt.plot(
    [0, maximum],
    [0, maximum],
    linestyle="--"
)

plt.xlabel(
    "Base AI Absolute Error (°C)"
)

plt.ylabel(
    "Hybrid Absolute Error (°C)"
)

plt.title(
    "Error Before vs After Residual Correction"
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "02_error_before_after.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# PLOT 3
# ACTUAL vs PREDICTED RESIDUAL
# ============================================================

plt.figure(figsize=(7, 6))

plt.scatter(
    actual_residual,
    predicted_residual,
    alpha=0.7
)

minimum = min(
    actual_residual.min(),
    predicted_residual.min()
)

maximum = max(
    actual_residual.max(),
    predicted_residual.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel(
    "Actual Residual (°C)"
)

plt.ylabel(
    "Predicted Residual (°C)"
)

plt.title(
    "Residual Model: Actual vs Predicted"
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "03_actual_vs_predicted_residual.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# PLOT 4
# HYBRID ERROR vs CURRENT
# ============================================================

plt.figure(figsize=(7, 6))

plt.scatter(
    df["current_A"],
    hybrid_error,
    alpha=0.7
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Current (A)")
plt.ylabel("Hybrid Error (°C)")

plt.title(
    "Hybrid Prediction Error vs Current"
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "04_error_vs_current.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# PLOT 5
# HYBRID ERROR vs INITIAL TEMPERATURE
# ============================================================

plt.figure(figsize=(7, 6))

plt.scatter(
    df["initial_temperature_C"],
    hybrid_error,
    alpha=0.7
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Initial Temperature (°C)"
)

plt.ylabel(
    "Hybrid Error (°C)"
)

plt.title(
    "Hybrid Prediction Error vs Initial Temperature"
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "05_error_vs_initial_temperature.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# PLOT 6
# HYBRID ERROR vs SOC
# ============================================================

plt.figure(figsize=(7, 6))

plt.scatter(
    df["soc"],
    hybrid_error,
    alpha=0.7
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("SOC")
plt.ylabel("Hybrid Error (°C)")

plt.title(
    "Hybrid Prediction Error vs SOC"
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "06_error_vs_soc.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# PLOT 7
# HYBRID ERROR vs SOH
# ============================================================

plt.figure(figsize=(7, 6))

plt.scatter(
    df["soh"],
    hybrid_error,
    alpha=0.7
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("SOH")
plt.ylabel("Hybrid Error (°C)")

plt.title(
    "Hybrid Prediction Error vs SOH"
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "07_error_vs_soh.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# PLOT 8
# ERROR DISTRIBUTION
# ============================================================

plt.figure(figsize=(7, 6))

plt.hist(
    ai_error,
    bins=30,
    alpha=0.6,
    label="Base AI"
)

plt.hist(
    hybrid_error,
    bins=30,
    alpha=0.6,
    label="Hybrid"
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel("Prediction Error (°C)")
plt.ylabel("Number of Cases")

plt.title(
    "Prediction Error Distribution"
)

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "08_error_distribution.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# PLOT 9
# ABSOLUTE ERROR REDUCTION
# ============================================================

plt.figure(figsize=(7, 6))

plt.hist(
    error_reduction,
    bins=30
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Absolute Error Reduction (°C)"
)

plt.ylabel(
    "Number of Cases"
)

plt.title(
    "Case-by-Case Absolute Error Reduction"
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "09_error_reduction_distribution.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# SAVE ANALYSIS TABLE
# ============================================================

analysis_df = df[
    [
        "case_id",
        "soc",
        "soh",
        "current_A",
        "initial_temperature_C",
        "T_physics_C",
        "T_AI_C",
        "T_hybrid_C",
        "AI_error_C",
        "hybrid_error_C",
        "absolute_AI_error_C",
        "absolute_hybrid_error_C",
        "residual_actual_C",
        "residual_predicted_C"
    ]
].copy()

analysis_df["absolute_error_reduction_C"] = (
    analysis_df["absolute_AI_error_C"] -
    analysis_df["absolute_hybrid_error_C"]
)

analysis_df["improved"] = (
    analysis_df["absolute_error_reduction_C"] > 0
)

analysis_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "residual_error_analysis.csv"
    ),
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame([

    {
        "model": "Base AI",
        "RMSE_C": np.sqrt(
            np.mean(ai_error ** 2)
        ),
        "MAE_C": np.mean(abs_ai),
        "Mean_Error_C": np.mean(ai_error),
        "Std_Error_C": np.std(ai_error),
        "Max_Abs_Error_C": np.max(abs_ai)
    },

    {
        "model": "Hybrid Physics + AI",
        "RMSE_C": np.sqrt(
            np.mean(hybrid_error ** 2)
        ),
        "MAE_C": np.mean(abs_hybrid),
        "Mean_Error_C": np.mean(hybrid_error),
        "Std_Error_C": np.std(hybrid_error),
        "Max_Abs_Error_C": np.max(abs_hybrid)
    }

])

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "residual_summary.csv"
    ),
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 70)
print("PLOTS SAVED")
print("=" * 70)

print(
    f"Output directory: {OUTPUT_DIR}"
)

for filename in sorted(
    os.listdir(OUTPUT_DIR)
):

    print(
        f"  {filename}"
    )

print("\nDone.")