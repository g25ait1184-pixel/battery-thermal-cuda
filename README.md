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

### Residual MLP Architecture

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
6. V5 GPU performance benchmark
```

### 7.5 Menu options

| Option | Execution mode | Description |
|---:|---|---|
| 1 | Normal V1–V5 execution | Runs the standard battery thermal simulations and configured GPU benchmarks. |
| 2 | Digital Twin – Automatic validity-aware routing | Automatically selects Hybrid AI within the configured validity conditions and V5 CUDA physics otherwise. |
| 3 | Digital Twin – Physics / Base AI / Hybrid comparison | Compares predictions from V5 physics, Base AI, and Hybrid AI. |
| 4 | Digital Twin – Stateful closed-loop test | Exercises stateful Digital Twin execution and automatic switching across operating conditions. |
| 5 | Run all Digital Twin tests | Executes the configured Digital Twin test suite. |
| 6 | V5 GPU performance benchmark | Runs the SOC/SOH-aware V5 thermal solver benchmark and reports CPU, GPU, tiled, and halo execution times, speedups, throughput, bandwidth, and CPU–GPU validation metrics across the configured grid sizes. |

Select the required option when prompted by the application.

### 7.6 V5 GPU performance benchmark

Build the application using the normal execution command in Section 7.3, then run:

```bash
./thermal
```

Select option **6 – V5 GPU performance benchmark**. This option is intended to evaluate the V5 SOC/SOH-aware electro-thermal solver across the configured spatial resolutions and compare the CPU reference with the CUDA global-memory, shared-memory tiled, and halo implementations. The recorded benchmark results are summarized in Section 9. The values shown there are from the captured V5 experiment log; rerunning the benchmark may produce different timings depending on GPU load, clock state, and runtime environment.

### 7.7 Nsight execution build

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

### 7.8 Measure total executable wall-clock time

To measure the elapsed time for a complete interactive application run, use:

```bash
/usr/bin/time -v ./thermal
```

Select the required menu option and allow that run to finish. The reported **Elapsed (wall clock) time** includes CPU work, CUDA operations, data transfers, synchronization, and application overhead occurring within the process. For an interactive run, it also includes time spent waiting for menu input; enter the selection promptly and use the same procedure for each comparison.

To save the timing report:

```bash
mkdir -p results
/usr/bin/time -v -o results/end_to_end_timing.txt ./thermal
```

For repeatable comparisons:

1. Use the same executable build, GPU, input conditions, and menu option.
2. Run a warm-up execution before recording measurements, where practical.
3. Repeat the measurement at least five times.
4. Report the mean and standard deviation, or median and range.
5. State whether compilation, model loading, dataset loading, warm-up, and file I/O are included.

**Important:** The stage-level measurements in this README are not a single controlled end-to-end benchmark. Populate the end-to-end timing results from the wall-clock measurement above; do not infer them by adding measurements collected under different benchmark procedures.

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

The seed-42 checkpoint pair is also used for the performance comparison in Section 17.

---

## 9. V5 GPU Performance Benchmark

The V5 GPU performance benchmark evaluates the SOC/SOH-aware electro-thermal model using the CPU reference and three CUDA implementations: global memory, shared-memory tiled execution, and halo-based data reuse. Four grid resolutions were evaluated: 256 × 256, 512 × 512, 1024 × 1024, and 2048 × 2048. Each configuration uses 20 benchmark runs, with 100 simulation time steps per run. The reported execution times follow the timing aggregation implemented in the benchmark routine; they are not labelled as averages or cumulative totals here.

### 9.1 Runtime and speedup across spatial resolutions

| Grid | CPU (ms) | GPU global (ms) | Tiled (ms) | Halo (ms) | GPU speedup | Tiled speedup | Halo speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 × 256 | 374.99 | 0.84 | 0.89 | 0.96 | 444.25× | 423.50× | 391.57× |
| 512 × 512 | 1356.82 | 2.04 | 2.20 | 2.18 | 665.73× | 616.37× | 623.18× |
| 1024 × 1024 | 5426.49 | 8.35 | 5.81 | 4.96 | 650.16× | 933.93× | 1094.78× |
| 2048 × 2048 | 22422.88 | 24.41 | 20.44 | 18.47 | 918.59× | 1097.01× | 1214.18× |

*20 runs, 100 time steps per run; execution times in milliseconds. Speedup is relative to CPU.*

At the largest evaluated grid, the halo implementation records the lowest runtime among the compared GPU implementations and the highest speedup, 1214.18×, relative to the CPU implementation.

### 9.2 CPU–GPU numerical validation

| Grid | RMSE (°C) | Maximum absolute error (°C) |
|---:|---:|---:|
| 256 × 256 | 1.60 × 10⁻⁶ | 1.53 × 10⁻⁵ |
| 512 × 512 | 1.75 × 10⁻⁶ | 2.29 × 10⁻⁵ |
| 1024 × 1024 | 1.81 × 10⁻⁶ | 1.53 × 10⁻⁵ |
| 2048 × 2048 | 1.84 × 10⁻⁶ | 1.53 × 10⁻⁵ |

The measured CPU–GPU RMSE remains below 2 × 10⁻⁶ °C across all evaluated V5 grid sizes. These values indicate numerical agreement between the implementations under the tested simulation configuration; they do not constitute experimental validation against physical battery measurements.

---

## 10. Roofline Analysis

For the V5 thermal stencil, the measured arithmetic intensity at the largest evaluated grid is approximately:

```text
0.29 FLOP/byte
```

At 2048 × 2048, the V5 benchmark reports:

| Implementation | GFLOPS | Bandwidth (GB/s) |
|---|---:|---:|
| Global Memory | 122.09 | 418.59 |
| Shared-memory tiled | 138.06 | 473.35 |
| Halo | 147.78 | 506.67 |

The roofline measurements characterize the computational and memory behaviour of the V5 CUDA thermal solver. They are specific to the recorded hardware and benchmark configuration.

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

The reported speedup values are reproduced from the experiment results. Since batch time increases with stream count, interpret these values using the benchmark's original throughput/speedup calculation rather than as a reduction in total batch latency. Confirm the speedup calculation and workload normalization before claiming a latency improvement.

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

## 16. Execution Timing: Stage-Level Results and End-to-End Measurement

The timing results in this repository were collected from separate experiments using different workloads and measurement procedures. They are therefore reported as **stage-level or workload-specific measurements**, not as one combined application runtime. In particular, the per-prediction AI values below are batch-amortized measurements and should not be interpreted as isolated single-request latency.

### 16.1 Available stage-level measurements

| Component / workload | Reported timing | Measurement context |
|---|---:|---|
| V5 CUDA physics dataset generation | 0.906417 ms/case | Arithmetic average derived from 181.283426 ms of recorded physics simulation time across 200 dataset-generation cases |
| Base AI GPU inference | 0.000962 ms/prediction | Median CUDA-event time for a batch of 200 predictions, divided by 200; batch-amortized time per prediction |
| Hybrid AI GPU inference | 0.001802 ms/prediction | Median CUDA-event time for a batch of 200 predictions, divided by 200; batch-amortized time per prediction |
| V5 CUDA Halo solver | 18.47 ms | Recorded V5 solver benchmark result at 2048 × 2048; a separate spatial-solver workload |

The Base AI and Hybrid AI inference measurements were performed on an NVIDIA Tesla T4 using 50 warm-up runs and 500 measurement runs. CUDA events were used to time batch inference, with each measured batch containing 200 predictions. The reported per-prediction values are calculated by dividing the median batch time by 200. They represent amortized batch execution time, not the latency of an independently submitted single prediction. These timings exclude preprocessing and host-to-device transfer.

The V5 physics value is an average derived from the recorded dataset-generation simulation time. It was not measured in the same repeated CUDA-event inference procedure used for the AI models. The 18.47 ms Halo value belongs to the separate V5 spatial thermal-solver benchmark at a 2048 × 2048 grid. It is not an AI inference time and must not be treated as a directly comparable per-case timing.

### 16.2 End-to-end timing status

A measured end-to-end runtime is available for the ten-step stateful Digital Twin evaluation. The complete ten-step execution took **13.4902 ms** and comprised five Hybrid AI steps and five V5 CUDA physics steps.

| Execution scenario | End-to-end wall time | Execution breakdown | Notes |
|---|---:|---|---|
| Ten-step stateful Digital Twin evaluation | **13.4902 ms** | 5 Hybrid AI steps + 5 V5 physics steps | 50% Hybrid AI utilization |
| Normal V1–V5 execution | To be measured | Record selected workloads | Record GPU and system |
| Automatic validity-aware routing (other trajectories) | To be measured | Record input trajectory and routing decisions | Record GPU and system |
| Physics / Base AI / Hybrid comparison | To be measured | Record model-loading and warm-up policy | Record GPU and system |

For the reported ten-step stateful run, the average measured step runtimes were **0.0190 ms per Hybrid AI step** and **2.6790 ms per V5 physics step**. These are per-step averages for this particular evaluation, not replacements for the separate batch-inference and solver microbenchmark measurements in Section 16.1.

The ten-step total is consistent with the reported step averages:
`(5 × 0.0190 ms) + (5 × 2.6790 ms) = 13.4900 ms` (approximately 13.4902 ms; the small difference is due to rounding of the displayed per-step averages).

The result demonstrates lower measured per-step execution time for Hybrid AI than for the V5 physics path in this tested trajectory. It should not be interpreted as a universal application speedup: the two paths perform different amounts and types of computation, and the result applies to this specific stateful test and its measurement configuration.

A complete end-to-end measurement for every application mode—including all startup, model-loading, input preparation, initialization, synchronization, routing, and output/file-handling costs—has not yet been reported. Use the wall-clock procedure in Section 7.8 for the remaining scenarios. Record the selected menu option, hardware, input trajectory, and whether startup, model loading, warm-up, and output/file operations are included.

Do not calculate end-to-end runtime by adding stage-level measurements collected under different workloads, timing methods, or inclusion boundaries.

---

## 17. Performance Summary

### 17.1 Five-seed AI accuracy

```text
Base MLP RMSE:       0.192780 ± 0.011577 °C
Hybrid AI RMSE:      0.168340 ± 0.013797 °C

Base MLP MAE:        0.150962 ± 0.011023 °C
Hybrid AI MAE:       0.127764 ± 0.012317 °C

RMSE improvement:   12.68%
MAE improvement:    15.37%
```

### 17.2 Seed-42 performance benchmark

The performance benchmark used the same 200 held-out cases and the seed-42 Base and Residual checkpoints.

| Method | RMSE (°C) | MAE (°C) | R² | Runtime per case (ms) |
|---|---:|---:|---:|---:|
| V5 Physics CUDA | — | — | — | 0.906417 |
| Base AI GPU | 0.205296 | 0.159564 | 0.999484 | 0.000962 |
| Hybrid AI GPU | 0.183997 | 0.142309 | 0.999585 | 0.001802 |

The V5 runtime is the arithmetic average of the recorded dataset-generation simulation times (181.283426 ms across 200 cases). Base AI and Hybrid AI values are the median CUDA-event times for batches of 200 predictions, divided by 200. AI timing used 50 warm-up runs and 500 measurement runs on an NVIDIA Tesla T4. The AI values are batch-amortized time per prediction, not isolated single-prediction latency.

The Hybrid AI improved seed-42 RMSE by 10.37% and MAE by 10.81% relative to Base AI.

The physics accuracy columns are intentionally left blank because V5 physics supplies the reference targets; comparing the solver against its own generated targets would be a self-reference, not independent accuracy validation.

**Timing caveat:** The physics runtimes were recorded during dataset simulation, whereas AI timings were measured in a separate repeated inference benchmark. Ratios derived from these values are indicative reference-runtime comparisons, not controlled end-to-end application speedups. The AI timing is batch inference only and excludes preprocessing and host-to-device transfer. Its per-prediction value is amortized across a batch of 200 predictions.

### 17.3 Stateful Digital Twin routing and execution timing

The Digital Twin was evaluated using dynamic-routing and stateful trajectories.

| Test | Total steps | Hybrid AI steps | V5 physics steps | Hybrid utilization | Total execution time |
|---|---:|---:|---:|---:|---:|
| Dynamic routing test | 7 | 4 | 3 | 57.14% | Not reported here |
| Stateful test | 10 | 5 | 5 | 50.00% | **13.4902 ms** |

For the ten-step stateful evaluation:

```text
Total execution time:       13.4902 ms
Hybrid AI steps:            5
V5 physics steps:           5
Hybrid AI utilization:      50%

Average Hybrid AI step:     0.0190 ms
Average V5 physics step:    2.6790 ms
```

The measured Hybrid AI step runtime was lower than the measured V5 physics step runtime for this tested trajectory. The two paths have different computational responsibilities: Hybrid AI supplies a scalar maximum-temperature estimate, whereas V5 CUDA physics computes and maintains the spatial thermal field. Therefore, this comparison demonstrates the runtime difference between the implemented execution paths for this test; it is not a like-for-like comparison of identical computational outputs and should not be generalized as a universal speedup.

The routing tests demonstrate automatic selection of Hybrid AI within the implemented validity conditions and V5 CUDA physics when the conditions are not satisfied. The physical temperature field continues to be maintained by the physics path; AI supplies scalar `Tmax` estimates.

### 17.4 V5 GPU performance benchmark

```text
Largest grid:             2048 × 2048
CPU runtime:              22422.88 ms
GPU global runtime:       24.41 ms
Tiled runtime:            20.44 ms
Halo runtime:             18.47 ms
Halo speedup:             1214.18×
CPU–GPU RMSE:             1.84 × 10⁻⁶ °C
```

These V5 benchmark results are reported separately from the AI inference benchmark and the CUDA stream experiment. The benchmark configuration uses 20 runs and 100 simulation time steps per run.

---

## 18. Paper

The repository provides the implementation and experimental artifacts supporting the reported:

- Physics-AI prediction
- Residual learning
- CUDA acceleration
- CPU-GPU numerical consistency
- Roofline analysis
- CUDA stream evaluation
- Validity-aware Digital Twin model switching

---

## 19. Scope and Limitations

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
- Controlled end-to-end timing measurements for every application mode and the complete deployment workflow. A ten-step stateful Digital Twin execution has been timed separately (Section 16.2).

These are directions for future work. The reported AI accuracy is agreement with V5 simulation-generated reference targets and should not be interpreted as experimental battery prediction accuracy.

The framework may be extended to other thermal-management applications, including data-center cooling, high-performance computing thermal management, liquid-based thermal management, and other large-scale thermal Digital Twins. These are proposed extensions and are not validated by the current experiments.

---

## 20. License

This repository is provided for research and academic use.

Please refer to the repository license for the applicable terms.
