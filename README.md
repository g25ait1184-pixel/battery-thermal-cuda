# GPU-Accelerated Physics-AI Thermal Digital Twin

This repository contains the implementation and experimental artifacts associated with the paper:

**"GPU-Accelerated Physics-AI Thermal Digital Twins: A Scalable Scientific Computing Framework"**

The work presents an integrated framework combining high-fidelity electro-thermal physics modelling, Physics-AI residual learning, validity-aware model switching, and CUDA-accelerated scientific computing for battery thermal Digital Twins.

---

## 1. Overview

High-fidelity electro-thermal simulations can become computationally expensive when evaluated over large spatial grids and repeated operating conditions.

This work addresses this challenge through a multi-fidelity Physics-AI framework combining:

- High-fidelity electro-thermal physics modelling
- SOC/SOH-aware thermal modelling
- Neural-network-based thermal prediction
- Residual learning for prediction refinement
- Validity-aware Physics-AI model switching
- Physics-based fallback outside the validated AI domain
- Stateful Digital Twin execution with automatic Hybrid AI / V5 Physics switching
- CUDA-accelerated thermal stencil computation
- Shared-memory tiling
- Halo-based data reuse
- CUDA stream concurrency
- CPU-GPU numerical consistency analysis
- Roofline-based performance analysis

The physics model serves as the high-fidelity reference and fallback model, while the Physics-AI model provides a computationally efficient approximation within its validated operating region.

---

## 2. Framework Architecture

```text
                High-Fidelity
              Physics Model
                    |
                    v
        Physics-Generated Dataset
                    |
                    v
              Base AI Model
                    |
                    v
             Residual Learning
                    |
                    v
             Physics-AI Model
                    |
                    v
          Validity-Aware Switching
              /             \
             /               \
            v                 v
     Physics-AI Model    Physics Model
     (Valid Region)      (Fallback)
```

The high-fidelity thermal solver is additionally accelerated using CUDA to reduce the computational cost of repeated physics evaluations.

---

## 3. Physics Model

The reference thermal model incorporates:

- Current-dependent Joule heating
- Temperature-dependent resistance
- Entropy heat generation
- SOC-dependent behaviour
- SOH-dependent behaviour

The physics model generates reference thermal responses for AI training and evaluation and remains available as the trusted fallback outside the validated AI operating region.

---

## 4. Physics-AI Model

The base AI model predicts maximum thermal response from four operating variables:

- SOC
- SOH
- Initial Temperature
- Current

### Base AI Architecture

```text
4 → 32 → 32 → 16 → 1
```

### Residual AI Architecture

```text
4 → 16 → 16 → 8 → 1
```

The final Physics-AI prediction is formulated as:

```text
T_PAI = T_AI + g_phi(x)
```

where:

- `T_AI` is the base AI prediction.
- `g_phi(x)` is the learned residual correction.
- `T_PAI` is the final Physics-AI prediction.
- `x` represents the operating conditions.

The residual network learns the remaining prediction error of the base AI model rather than learning the complete thermal response from the beginning.

---

## 5. AI Dataset and Training Configuration

The AI dataset is generated using the high-fidelity electro-thermal model.

| Parameter | Configuration |
|---|---|
| Number of cases | 1,000 |
| SOC | 0.20–1.00 |
| SOH | 0.70–1.00 |
| Current | 10–30 A |
| Initial temperature | 15–45 °C |
| Thermal grid | 256 × 256 |
| Time steps | 100 |
| Train/Test split | 80/20 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Epochs | 1000 |
| Primary seed | 42 |

A multi-seed evaluation is also performed to assess robustness to training initialization.

---

## 6. CUDA Thermal Solver

The thermal solver is implemented using CUDA and evaluated progressively using:

1. Global-memory stencil computation
2. Shared-memory tiled execution
3. Halo-based data reuse
4. CUDA stream concurrency

The spatial resolutions evaluated are:

```text
256 × 256
512 × 512
1024 × 1024
2048 × 2048
```

The CPU and CUDA implementations use consistent physical parameters, initial conditions, boundary conditions, thermal inputs, and time-stepping configurations.

---

## 7. Compilation and Execution

### 7.1 Requirements

- NVIDIA CUDA-capable GPU
- NVIDIA CUDA Toolkit
- CUDA-compatible C++ compiler
- `nvcc` compiler
- Python 3.x
- Required Python packages used by the AI and analysis scripts

### 7.2 Navigate to the project directory

For Google Colab:

```bash
cd /content/drive/MyDrive/CUDA_Project/project
```

For a local clone:

```bash
cd battery-thermal-cuda
```

### 7.3 Compile the normal execution application

The following command enables relocatable device code to support CUDA device functions referenced across separate CUDA source files.

```bash
nvcc \
-rdc=true \
-Wno-deprecated-gpu-targets \
-DHEAT_MODE=DYNAMIC_Q_MODE \
-DEXEC_MODE=NORMAL_EXECUTION \
-Iinclude \
src/*.cu \
kernels/*.cu \
-o thermal
```

**Note:** `NORMAL_EXECUTION` and `DYNAMIC_Q_MODE` must match the definitions in the project headers and source files. The `-rdc=true` option requires the referenced device-function implementations to be present in the compiled CUDA source files.

### 7.4 Run the application

```bash
./thermal
```

The application displays the following interactive menu:

```text
=====================================
BATTERY THERMAL SIMULATION
=====================================
1. Normal V1-V5 execution
2. Digital Twin - Automatic validity-aware routing
3. Digital Twin - Physics / Base AI / Hybrid comparison
4. Digital Twin - Stateful closed-loop test
5. Run all Digital Twin tests
```

### 7.5 Menu options

| Option | Execution mode | Description |
|---:|---|---|
| 1 | Normal V1–V5 execution | Runs the standard battery thermal simulations and configured GPU benchmarks. |
| 2 | Digital Twin – Automatic validity-aware routing | Automatically selects Hybrid AI within the configured validity conditions and V5 CUDA physics otherwise. |
| 3 | Digital Twin – Physics / Base AI / Hybrid comparison | Compares predictions from V5 physics, Base AI, and Hybrid AI. |
| 4 | Digital Twin – Stateful closed-loop test | Exercises stateful Digital Twin execution and automatic switching across operating conditions. |
| 5 | Run all Digital Twin tests | Executes the configured Digital Twin test suite. |

Select the required option when prompted by the application.

### 7.6 Nsight execution build

For CUDA stream and profiling experiments, compile using:

```bash
nvcc \
-rdc=true \
-Wno-deprecated-gpu-targets \
-DHEAT_MODE=DYNAMIC_Q_MODE \
-DEXEC_MODE=NSIGHT_EXECUTION \
-Iinclude \
src/*.cu \
kernels/*.cu \
-o thermal
```

Run the executable:

```bash
./thermal
```

If Nsight Systems is installed, profile the execution using:

```bash
mkdir -p results

nsys profile \
--stats=true \
-o results/v5_nsight \
./thermal
```

The profiling configuration is intended to evaluate CUDA stream execution and workload concurrency. Compilation alone does not establish stream overlap or performance improvement.

---

## 8. Physics-AI Results

The AI models predict scalar maximum temperature (`T_max_C`) using four operating inputs. The residual network learns the Base MLP's training-set prediction error, and the Hybrid AI output is:

```text
T_hybrid = T_base + T_residual
```

### 8.1 Five-seed evaluation

A fixed 80/20 train/test split (`random_state=42`) was used for all five independent model initializations. Each initialization trained a separate Base MLP and its corresponding Residual MLP. The test set contains 200 cases.

| Metric | Base MLP (mean ± SD) | Hybrid AI (mean ± SD) |
|---|---:|---:|
| RMSE (°C) | 0.192780 ± 0.011577 | **0.168340 ± 0.013797** |
| MAE (°C) | 0.150962 ± 0.011023 | **0.127764 ± 0.012317** |
| R² | 0.999544 ± 0.000054 | **0.999651 ± 0.000058** |

The Hybrid AI reduces mean RMSE by 12.68% and mean MAE by 15.37% relative to the Base MLP. Both error metrics improved in all five initializations.

**Evaluation clarification:** This is a five-initialization experiment on one fixed train/test split; it is not five independent data splits. The reference targets are generated by the V5 physics model, so these results quantify agreement with the simulation reference rather than independent experimental battery accuracy.

The seed-42 checkpoint pair is also used for the performance comparison in Section 16.

---

## 9. CUDA Performance

The CUDA implementation provides substantial acceleration over the CPU reference.

At the largest evaluated spatial resolution:

| Parameter | Result |
|---|---:|
| Grid size | 2048 × 2048 |
| Halo runtime | 17.58 ms |
| CPU-GPU speedup | 1238.8× |

CPU-GPU numerical consistency is maintained with RMSE below:

```text
2 × 10⁻⁶ °C
```

for all evaluated grid sizes.

---

## 10. Roofline Analysis

The thermal stencil has an arithmetic intensity of approximately:

```text
0.29 FLOP/byte
```

At 2048 × 2048, the measured performance is:

| Implementation | GFLOPS | Bandwidth (GB/s) |
|---|---:|---:|
| Global Memory | 128.92 | 442.00 |
| Tiled | 147.38 | 505.31 |
| Halo | 166.69 | 571.49 |

The roofline analysis is used to characterize the computational and memory behaviour of the CUDA thermal solver.

---

## 11. CUDA Stream Concurrency

Independent thermal workloads are evaluated using multiple CUDA streams.

At 2048 × 2048:

| Streams | Batch Time (ms) | Aggregate Speedup |
|---:|---:|---:|
| 1 | 39.8352 | 1.00× |
| 2 | 62.7138 | 1.27× |
| 4 | 121.3785 | 1.31× |
| 8 | 165.6361 | 1.92× |

The results demonstrate improved aggregate throughput with increasing stream concurrency for independent thermal workloads.

---

## 12. Validity-Aware Digital Twin

The Digital Twin combines the Hybrid AI predictor with the V5 CUDA physics solver and selects the execution path according to the validated operating domain.

### 12.1 Validated AI operating region

| Parameter | Validated Range |
|---|---|
| SOC | 0.20–1.00 |
| SOH | 0.70–1.00 |
| Current | 10–30 A |
| Initial temperature | 15–45 °C |

The automatic execution path checks AI input validity and applies a physical thermal-state guard. In the current implementation, Hybrid AI is selected only when the input is within the validated domain and the retained physical `Tmax` is at or below 45 °C. Otherwise, execution falls back to V5 CUDA physics.

The 45 °C guard is an implementation-specific conservative routing threshold associated with the validated domain; it is not asserted as a universal battery safety limit.

### 12.2 Model responsibilities

- **Base MLP:** Predicts scalar maximum temperature.
- **Residual MLP:** Predicts a scalar correction in °C.
- **Hybrid AI:** Adds the residual correction to the Base MLP prediction.
- **V5 CUDA Physics:** Computes the spatial thermal field and its thermal statistics, including `Tmin`, `Tmax`, and `Tmean`.

The AI prediction does not reconstruct or advance the full spatial thermal field. The Digital Twin preserves the physical field separately when an AI scalar prediction is used.

### 12.3 Stateful and automatic-routing experiments

The automatic Digital Twin tests exercised changing operating conditions and demonstrated switching between Hybrid AI and V5 CUDA physics.

| Test | Steps | Hybrid AI steps | Physics steps | Hybrid utilization |
|---|---:|---:|---:|---:|
| Dynamic routing test | 7 | 4 | 3 | 57.14% |
| Stateful test | 10 | 5 | 5 | 50.00% |

In the 10-step stateful test, the first five steps used Hybrid AI. As the physical thermal state crossed the configured routing guard, the remaining five steps used V5 CUDA physics.

The physical temperature field continued to be maintained by the physics path; AI supplied only scalar `Tmax` estimates.

These tests verify the implemented stateful routing behaviour for the tested trajectories. They do not establish long-duration closed-loop validation or independent experimental battery performance.

---

## 13. Repository Structure

The main repository components include:

```text
battery-thermal-cuda/
│
├── dataset/
│   └── V5 physics-generated dataset and related data
│
├── include/
│   └── Thermal-model and CUDA header files
│
├── kernels/
│   └── CUDA thermal stencil implementation
│
├── src/
│   └── Thermal solver and Digital Twin source code
│
├── results/
│   └── Experimental and benchmark results
│
├── plots/
│   └── Performance and experimental plots
│
└── README.md
```

The repository also contains supporting files for dataset generation, CUDA benchmarking, thermal simulation, Digital Twin evaluation, and experimental analysis.

---

## 14. Key Experimental Artifacts

The repository contains artifacts associated with the reported experiments, including:

- Physics-generated thermal data
- V5 dataset generation components
- CUDA thermal solver implementations
- CPU reference implementation
- Global-memory CUDA implementation
- Shared-memory tiled implementation
- Halo-based implementation
- CUDA stream implementation
- Thermal maps
- AI model results
- Five-seed Base MLP and Hybrid AI metrics, summary, and per-case predictions
- Seed-specific Base and Residual model checkpoints
- Final seed-42 performance benchmark and per-case prediction CSVs
- CUDA performance results
- Roofline analysis data
- Dynamic Digital Twin routing test results
- Stateful 10-step Digital Twin test results
- Physics / Base AI / Hybrid AI comparison results

---

## 15. Reproducibility

The primary AI experiments use:

```text
Random seed: 42
```

Additional random seeds are used for multi-seed robustness evaluation.

The CUDA experiments use consistent physical and numerical configurations between the CPU and GPU implementations to enable direct numerical and performance comparison.

### Reproduction workflow

1. Clone the repository.
2. Install the required CUDA Toolkit and Python dependencies.
3. Generate or load the V5 physics dataset.
4. Train the Base MLP and Residual MLP.
5. Compile the CUDA application.
6. Run the required thermal and Digital Twin experiments.
7. Inspect the generated performance and evaluation artifacts.

The exact execution commands for individual experiments depend on the source files and scripts included in the repository.

---

## 16. Performance Summary

### 16.1 Five-seed AI accuracy

```text
Base MLP RMSE:       0.192780 ± 0.011577 °C
Hybrid AI RMSE:      0.168340 ± 0.013797 °C

Base MLP MAE:        0.150962 ± 0.011023 °C
Hybrid AI MAE:       0.127764 ± 0.012317 °C

RMSE improvement:   12.68%
MAE improvement:    15.37%
```

### 16.2 Seed-42 performance benchmark

The performance benchmark used the same 200 held-out cases and the seed-42 Base and Residual checkpoints.

| Method | RMSE (°C) | MAE (°C) | R² | Runtime per case (ms) |
|---|---:|---:|---:|---:|
| V5 Physics CUDA | — | — | — | 0.906417 |
| Base AI GPU | 0.205296 | 0.159564 | 0.999484 | 0.000962 |
| Hybrid AI GPU | 0.183997 | 0.142309 | 0.999585 | 0.001802 |

The V5 runtime is the average of previously recorded per-case simulation runtimes (181.283426 ms summed over 200 cases). Base AI and Hybrid AI times are median CUDA-event measurements of a batch of 200 predictions, divided by 200. AI timing used 50 warm-up runs and 500 measurement runs on an NVIDIA Tesla T4.

The Hybrid AI improved seed-42 RMSE by 10.37% and MAE by 10.81% relative to Base AI.

The physics accuracy columns are intentionally left blank because V5 physics supplies the reference targets; comparing the solver against its own generated targets would be a self-reference, not independent accuracy validation.

**Timing caveat:** The physics runtimes were recorded during dataset simulation, whereas AI timings were measured in a separate repeated inference benchmark. Ratios derived from these values are indicative reference-runtime comparisons, not controlled end-to-end application speedups. The AI timing is model inference only and excludes preprocessing and host-to-device transfer.

### 16.3 Stateful Digital Twin routing

```text
Dynamic routing test:
7 steps; 4 Hybrid AI; 3 V5 Physics
Hybrid utilization: 57.14%

Stateful test:
10 steps; 5 Hybrid AI; 5 V5 Physics
Hybrid utilization: 50.00%
```

The routing tests demonstrate automatic selection of Hybrid AI within the implemented validity conditions and V5 CUDA physics when the conditions are not satisfied.

### 16.4 CUDA solver

```text
Largest grid:             2048 × 2048
Halo runtime:             17.58 ms
Maximum CPU-GPU speedup:  1238.8×
Numerical RMSE:           < 2 × 10⁻⁶ °C
```

These CUDA solver results are reported separately from the AI inference benchmark.

---

## 17. Paper

This repository accompanies the paper:

**Vallinath S. and Binod Kumar,  
"GPU-Accelerated Physics-AI Thermal Digital Twins: A Scalable Scientific Computing Framework,"  
AI-SPC, 2026.**

The repository provides the implementation and experimental artifacts supporting the reported:

- Physics-AI prediction
- Residual learning
- CUDA acceleration
- CPU-GPU numerical consistency
- Roofline analysis
- CUDA stream evaluation
- Validity-aware Digital Twin model switching

---

## 18. Scope and Limitations

The current implementation focuses on scalar maximum-temperature prediction by AI, with the V5 CUDA physics solver retaining full spatial thermal-field computation and serving as the fallback path.

The current work demonstrates:

- Physics-generated dataset training and held-out evaluation.
- Five model initializations on a fixed train/test split.
- Base-plus-residual Hybrid AI prediction.
- Validity-aware automatic selection between Hybrid AI and V5 CUDA physics.
- Stateful routing tests over the reported 7-step and 10-step trajectories.

The current work does not yet provide:

- Independent experimental battery validation against measured cell or pack thermal data.
- AI prediction of the complete spatial thermal field.
- Long-duration closed-loop Digital Twin validation.
- Full integration with live BMS measurements in a deployed battery system.
- Multi-GPU validation.

These are directions for future work. The reported AI accuracy is agreement with V5 simulation-generated reference targets and should not be interpreted as experimental battery prediction accuracy.

The framework may be extended to other thermal-management applications, including data-center cooling, high-performance computing thermal management, liquid-based thermal management, and other large-scale thermal Digital Twins. These are proposed extensions and are not validated by the current experiments.

---

## 19. License

This repository is provided for research and academic use.

Please refer to the repository license for the applicable terms.
