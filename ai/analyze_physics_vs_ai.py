import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

RESULT_DIR = os.path.join(
    SCRIPT_DIR,
    "results"
)

INPUT_FILE = os.path.join(
    RESULT_DIR,
    "physics_vs_ai_comparison.csv"
)

PLOT_DIR = os.path.join(
    RESULT_DIR,
    "physics_vs_ai_plots"
)

os.makedirs(
    PLOT_DIR,
    exist_ok=True
)


# ============================================================
# Load results
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)

print("=" * 70)
print("PHYSICS vs AI ERROR ANALYSIS")
print("=" * 70)

print(
    f"\nLoaded test cases: {len(df)}"
)


# ============================================================
# Error statistics
# ============================================================

error = df["error_C"]

absolute_error = df[
    "absolute_error_C"
]


print("\nOverall error statistics")
print("-" * 70)

print(
    f"Mean error       : "
    f"{error.mean():.6f} °C"
)

print(
    f"Mean abs error   : "
    f"{absolute_error.mean():.6f} °C"
)

print(
    f"Std error        : "
    f"{error.std():.6f} °C"
)

print(
    f"Max abs error    : "
    f"{absolute_error.max():.6f} °C"
)


# ============================================================
# 1. Physics vs AI
# ============================================================

plt.figure(
    figsize=(7, 6)
)

plt.scatter(
    df["Tmax_physics_C"],
    df["Tmax_AI_C"],
    alpha=0.7
)

min_value = min(
    df["Tmax_physics_C"].min(),
    df["Tmax_AI_C"].min()
)

max_value = max(
    df["Tmax_physics_C"].max(),
    df["Tmax_AI_C"].max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel(
    "Physics Tmax (°C)"
)

plt.ylabel(
    "AI Tmax (°C)"
)

plt.title(
    "Physics vs AI Maximum Temperature"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "01_physics_vs_ai.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 2. Error vs Current
# ============================================================

plt.figure(
    figsize=(7, 5)
)

plt.scatter(
    df["current_A"],
    df["error_C"],
    alpha=0.7
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Current (A)"
)

plt.ylabel(
    "AI Error (°C)"
)

plt.title(
    "AI Prediction Error vs Current"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "02_error_vs_current.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 3. Error vs Initial Temperature
# ============================================================

plt.figure(
    figsize=(7, 5)
)

plt.scatter(
    df["initial_temperature_C"],
    df["error_C"],
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
    "AI Error (°C)"
)

plt.title(
    "AI Prediction Error vs Initial Temperature"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "03_error_vs_initial_temperature.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 4. Error vs SOC
# ============================================================

plt.figure(
    figsize=(7, 5)
)

plt.scatter(
    df["soc"],
    df["error_C"],
    alpha=0.7
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "SOC"
)

plt.ylabel(
    "AI Error (°C)"
)

plt.title(
    "AI Prediction Error vs SOC"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "04_error_vs_soc.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 5. Error vs SOH
# ============================================================

plt.figure(
    figsize=(7, 5)
)

plt.scatter(
    df["soh"],
    df["error_C"],
    alpha=0.7
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "SOH"
)

plt.ylabel(
    "AI Error (°C)"
)

plt.title(
    "AI Prediction Error vs SOH"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "05_error_vs_soh.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 6. Absolute error distribution
# ============================================================

plt.figure(
    figsize=(7, 5)
)

plt.hist(
    absolute_error,
    bins=25
)

plt.xlabel(
    "Absolute Error (°C)"
)

plt.ylabel(
    "Number of Cases"
)

plt.title(
    "AI Absolute Prediction Error Distribution"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "06_error_distribution.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 7. Print correlations
# ============================================================

print("\nCorrelation with absolute error")
print("-" * 70)

variables = [
    "soc",
    "soh",
    "current_A",
    "initial_temperature_C"
]

for variable in variables:

    correlation = df[
        variable
    ].corr(
        absolute_error
    )

    print(
        f"{variable:28s}: "
        f"{correlation:+.4f}"
    )


# ============================================================
# 8. Error thresholds
# ============================================================

print("\nError thresholds")
print("-" * 70)

for threshold in [
    0.1,
    0.2,
    0.3,
    0.5,
    1.0
]:

    count = np.sum(
        absolute_error <= threshold
    )

    percentage = (
        count
        / len(df)
        * 100
    )

    print(
        f"|error| <= {threshold:.1f} °C : "
        f"{count:3d}/{len(df)} "
        f"({percentage:.1f}%)"
    )


# ============================================================
# 9. Worst cases
# ============================================================

print("\nWorst 10 cases")
print("-" * 70)

print(
    df.sort_values(
        "absolute_error_C",
        ascending=False
    ).head(10).to_string(
        index=False
    )
)


print("\nPlots saved to:")
print(PLOT_DIR)

print("\n" + "=" * 70)
print("ERROR ANALYSIS COMPLETE")
print("=" * 70)