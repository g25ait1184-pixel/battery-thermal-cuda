# GPU-Accelerated Physics-AI Thermal Digital Twin

This repository contains the implementation and experimental artifacts associated with the paper:

**"GPU-Accelerated Physics-AI Thermal Digital Twins: A Scalable Scientific Computing Framework"**

The work presents an integrated framework combining high-fidelity electro-thermal physics modeling, Physics-AI residual learning, validity-aware model switching, and CUDA-accelerated scientific computing for battery thermal Digital Twins.

---

## 1. Overview

High-fidelity electro-thermal simulations can become computationally expensive when evaluated over large spatial grids and repeated operating conditions.

This work addresses this challenge through a multi-fidelity Physics-AI framework that combines:

- High-fidelity electro-thermal physics modeling
- SOC/SOH-aware thermal modeling
- Neural-network-based thermal prediction
- Residual learning for prediction refinement
- Validity-aware Physics-AI model switching
- Physics-based fallback outside the validated AI domain
- CUDA-accelerated thermal stencil computation
- Shared-memory tiling
- Halo-based data reuse
- CUDA stream concurrency
- CPU-GPU numerical consistency analysis
- Roofline-based performance analysis

The physics model serves as the high-fidelity reference and fallback model, while the Physics-AI model provides a computationally efficient approximation within its validated operating region.

---

## 2. Framework

The overall framework is:

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
- SOC-dependent behavior
- SOH-dependent behavior

The physics model is used to generate the reference thermal response for AI training and evaluation and remains available as the trusted fallback outside the validated AI operating region.

---

## 4. Physics-AI Model

The base AI model predicts the maximum thermal response from four operating variables:

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

- `T_AI` is the base AI prediction
- `g_phi(x)` is the learned residual correction
- `T_PAI` is the final Physics-AI prediction
- `x` represents the operating conditions

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

## 7. Physics-AI Results

The primary evaluation using seed 42 gives:

| Model | RMSE (°C) | MAE (°C) | R² |
|---|---:|---:|---:|
| Base AI | 0.1750 | 0.1394 | 0.999625 |
| Base AI + Residual | 0.1409 | 0.1081 | 0.999757 |

The five-seed evaluation gives:

| Model | RMSE (°C) |
|---|---:|
| Base AI | 0.1715 ± 0.0104 |
| Base AI + Residual | 0.1246 ± 0.0113 |

The results demonstrate the improvement obtained by residual learning over the base AI predictor.

---

## 8. CUDA Performance

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

## 9. Roofline Analysis

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

The roofline analysis is used to characterize the computational and memory behavior of the CUDA thermal solver.

---

## 10. CUDA Stream Concurrency

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

## 11. Validity-Aware Digital Twin

The validated Physics-AI operating region is:

| Parameter | Validated Range |
|---|---|
| SOC | 0.20–1.00 |
| SOH | 0.70–1.00 |
| Current | 10–30 A |
| Temperature | 15–45 °C |

Within this validated region, the Physics-AI model is used.

When the operating state moves outside the validated AI region, the Digital Twin switches to the high-fidelity physics model.

A 10-step trajectory demonstrates the switching mechanism:

```text
Steps 1–5 : Physics-AI
Steps 6–10: Physics fallback

Physics-AI utilization: 50%
```

This demonstrates state-dependent model selection while retaining the physics solver as a fallback outside the validated AI domain.

---

## 12. Repository Structure

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

## 13. Key Experimental Artifacts

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
- Residual Physics-AI results
- CUDA performance results
- Roofline analysis data
- Digital Twin evaluation results

---

## 14. Reproducibility

The primary AI experiments use:

```text
Random seed: 42
```

Additional random seeds are used for multi-seed robustness evaluation.

The CUDA experiments use consistent physical and numerical configurations between the CPU and GPU implementations to enable direct numerical and performance comparison.

### Requirements

Reproduction requires:

- NVIDIA CUDA-capable GPU
- CUDA Toolkit
- C/C++ compiler with `nvcc`
- Python 3.x
- Required Python packages used by the AI and analysis scripts

The exact execution commands depend on the corresponding source files and scripts included in the repository.

---

## 15. Dataset

The thermal reference simulations use a high-fidelity SOC/SOH-aware electro-thermal model.

A dynamic heat-generation profile containing 1,698 samples is used from the NASA Battery Data Set.

The AI dataset consists of 1,000 operating cases generated from the physics model.

The AI cases cover:

| Parameter | Range |
|---|---|
| SOC | 0.20–1.00 |
| SOH | 0.70–1.00 |
| Current | 10–30 A |
| Initial Temperature | 15–45 °C |

---

## 16. Performance Summary

### AI Prediction

```text
Base AI RMSE:             0.1750 °C
Base AI + Residual RMSE:  0.1409 °C

Base AI MAE:              0.1394 °C
Base AI + Residual MAE:   0.1081 °C
```

### Multi-Seed Evaluation

```text
Base AI:
0.1715 ± 0.0104 °C

Base AI + Residual:
0.1246 ± 0.0113 °C
```

### CUDA Solver

```text
Largest grid:             2048 × 2048
Halo runtime:             17.58 ms
Maximum CPU-GPU speedup:  1238.8×
Numerical RMSE:           < 2 × 10⁻⁶ °C
```

### CUDA Streams

```text
Maximum aggregate speedup:
1.92× with 8 streams
```

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

The current implementation focuses on maximum-temperature prediction and validity-aware model switching.

The current work does not yet provide:

- Recursive next-state thermal prediction
- Complete spatial thermal-field AI prediction
- Realistic vehicle drive-cycle validation
- Closed-loop long-duration Digital Twin validation
- Actual BMS measurement integration
- Multi-GPU validation

These are considered directions for future work.

The proposed framework can also be extended to other thermal-management applications, including:

- AI data-center cooling
- High-performance computing thermal management
- Liquid-based thermal management
- Other large-scale thermal Digital Twins

These applications are proposed as future extensions and are not directly validated by the current experiments.

---

## 19. License

This repository is provided for research and academic use.

Please refer to the repository license for the applicable terms.
