# battery-thermal-cuda

GPU-Accelerated Battery Electro-Thermal Simulation and Thermal Propagation Framework using CUDA, with Global Memory, Shared Memory, and Halo optimization.

## Current Release

**v1.0.0 — Complete V1→V5 Validated Thermal Solver**

This release consolidates the completed V1→V5 battery electro-thermal model progression and the corresponding CPU/GPU, tiled, and halo validation results. The V1→V5 execution completed successfully for 256×256, 512×512, 1024×1024, and 2048×2048 grids.

### V1→V5 Model Evolution

| Model | Thermal formulation |
|---|---|
| **V1** | NASA heat only |
| **V2** | NASA heat + Joule heating |
| **V3** | NASA heat + temperature-dependent resistance |
| **V4** | NASA heat + temperature-dependent resistance + entropy heat |
| **V5** | SOC/SOH-aware electro-thermal model |

The execution uses **1,698 NASA heat samples** with dynamic heat input. The supplied results show CPU-vs-GPU validation errors remaining in the micro-/sub-micro-degree range across the tested grids.

## Release Roadmap

```text
v0.1.0  → NASA Heat V1 baseline
v1.0.0  → Complete V1→V5 validated thermal solver
v1.1.0  → CUDA Streams integration
v1.2.0  → Performance validation
v1.3.0  → V5 simulation dataset generation
v2.0.0  → AI thermal predictor
v2.1.0  → Physics + AI residual correction
v3.0.0  → Battery Thermal Digital Twin
v3.1.0  → Final integrated benchmark
```

---

## Overview

`battery-thermal-cuda` is a CUDA-based framework for simulating heat propagation inside batteries using finite-difference stencil methods. The battery is represented as a two-dimensional thermal domain, where each computational cell stores the local temperature.

The framework investigates different GPU memory optimization techniques and compares them against CPU execution. Dynamic heat sources are derived from the NASA battery dataset, enabling realistic time-varying thermal behavior.

The project supports:

* CPU thermal solver
* Global Memory GPU kernel
* Shared Memory tiled kernel
* Halo-region optimized kernel
* Constant and dynamic heat sources
* NASA battery dataset integration
* Performance benchmarking
* GFLOPS and bandwidth analysis
* Roofline analysis
* Nsight Compute profiling
* Thermal map generation

The framework serves as a foundation for future:

* Electro-thermal battery models
* Battery Digital Twins
* Multi-GPU halo exchange
* SOC and SOH estimation
* AI surrogate models

---

# Physics Model

## Battery Heat Generation

Heat generation is modeled using Joule heating:

```math
Q = I^2R
```

where

* (I) = Battery current (A)
* (R) = Internal resistance (Ω)
* (Q) = Generated heat (W)

For dynamic simulations, current values are obtained from the NASA battery dataset:

```math
Q(t)=I(t)^2R
```

---

## Transient Heat Equation

Heat diffusion inside the battery is governed by

```math
\frac{\partial T}{\partial t}
=
\alpha
\left(
\frac{\partial^2T}{\partial x^2}
+
\frac{\partial^2T}{\partial y^2}
\right)
+
Q
```

where

* (T) = Temperature
* (\alpha) = Thermal diffusivity
* (Q) = Heat source

Using finite differences, the update equation becomes

```math
T_{i,j}^{n+1}
=
T_{i,j}^{n}
+
\alpha
\left(
T_{i-1,j}
+
T_{i+1,j}
+
T_{i,j-1}
+
T_{i,j+1}
-
4T_{i,j}
\right)
+
\beta Q
```

---

# Battery Thermal Domain Representation

The battery is discretized into an (N \times N) thermal grid.

```text
        Battery Cross Section

        ┌────────────────────┐
        │                    │
        │    Thermal Grid    │
        │                    │
        │      256×256       │
        │                    │
        └────────────────────┘
```

Each grid cell represents a small region of the battery.

## Grid Resolution

| Grid Size | Number of Cells |
| --------- | --------------: |
| 256×256   |          65,536 |
| 512×512   |         262,144 |
| 1024×1024 |       1,048,576 |
| 2048×2048 |       4,194,304 |

Larger grids provide finer thermal resolution and improved accuracy.

---

# Heat Stencil

Each grid point exchanges heat with four neighboring cells.

```text
        Top
         ↑
Left ← Center → Right
         ↓
      Bottom
```

---

# GPU Mapping

Each grid cell is mapped to one CUDA thread.

```text
1 Battery Region
       ↓
1 Grid Cell
       ↓
1 CUDA Thread
       ↓
Temperature Update
```

For a 2048×2048 grid:

* Total cells = 4,194,304
* Total CUDA threads ≈ 4.2 million

---

# GPU Kernels

## Global Memory GPU

Neighbor values are accessed directly from global memory.

```text
Global Memory
      ↓
    Thread
      ↓
   Compute
```

---

## Shared Memory GPU

Tiles are loaded into shared memory to reduce global memory accesses.

```text
Global Memory
      ↓
 Shared Tile
      ↓
   Threads
      ↓
   Compute
```

---

## Halo GPU

Halo cells are loaded around each tile to maximize memory locality and bandwidth utilization.

```text
Global Memory
       ↓
 Shared Tile + Halo
       ↓
      Threads
       ↓
      Compute
```

---

# Project Structure

```text
battery-thermal-cuda
│
├── include
│   ├── config.h
│   ├── benchmark.h
│   ├── kernels.h
│   ├── metrics.h
│   ├── thermal_map.h
│   ├── utils.h
│   └── cuda_events.h
│
├── src
│   ├── main.cu
│   ├── config.cu
│   ├── benchmark_cpu.cu
│   ├── benchmark_gpu.cu
│   ├── benchmark_tiled.cu
│   ├── benchmark_halo.cu
│   ├── metrics.cu
│   ├── thermal_map.cu
│   ├── utils.cu
│   └── cuda_events.cu
│
├── kernels
│   ├── gpu_stencil.cu
│   ├── tiled_stencil.cu
│   └── halo_stencil.cu
│
├── data
│   └── battery_dataset.csv
│
├── results
│   ├── results.csv
│   ├── center_temp.csv
│   └── thermal_map.csv
│
├── plots
│   ├── speedup.png
│   ├── bandwidth.png
│   ├── roofline.png
│   └── thermal_map.png
│
├── README.md
├── LICENSE
└── .gitignore
```

---

# Build

Compile:

```%%bash

cd /content/project/project

nvcc \
-Wno-deprecated-gpu-targets \
-DHEAT_MODE=DYNAMIC_Q_MODE \
-DEXEC_MODE=NORMAL_EXECUTION \
-Iinclude \
src/*.cu \
kernels/*.cu \
-o thermal
```

---

# Run

```bash
./thermal
```

---

# Nsight Compute Profiling

Compile:

```%%bash

cd /content/project/project

nvcc \
-Wno-deprecated-gpu-targets \
-DHEAT_MODE=DYNAMIC_Q_MODE \
-DEXEC_MODE=NSIGHT_EXECUTION \
-Iinclude \
src/*.cu \
kernels/*.cu \
-o thermal
```

Profile:

```ncu \
--set full \
--launch-count 1 \
./thermal
```

Metrics analyzed:

* Occupancy
* Warp efficiency
* Memory throughput
* L1 cache utilization
* L2 cache utilization
* Shared memory utilization

---

# Performance Metrics

The framework reports:

* Runtime
* Speedup
* GFLOPS
* Effective memory bandwidth

Typical speedup:

| Grid Size | GPU Speedup | Halo Speedup |
| --------- | ----------: | -----------: |
| 256×256   |        107× |          84× |
| 512×512   |        160× |          97× |
| 1024×1024 |        170× |         203× |
| 2048×2048 |        202× |         227× |

---

# Roofline Characteristics

Arithmetic Intensity

```text
AI ≈ 0.29 FLOP/Byte
```

The stencil kernel is primarily memory-bandwidth limited, making shared-memory and halo optimizations highly effective.

# Performance Results

The framework reports runtime, speedup, computational throughput (GFLOPS), and effective memory bandwidth for CPU, Global GPU, Shared GPU, and Halo GPU implementations.

## Performance Comparison

| Grid Size | CPU (ms) | GPU (ms) | Shared GPU (ms) | Halo GPU (ms) | GPU Speedup | Shared Speedup | Halo Speedup | GPU GFLOPS | Shared GFLOPS | Halo GFLOPS | GPU BW (GB/s) | Shared BW (GB/s) | Halo BW (GB/s) |
| --------- | -------: | -------: | --------------: | ------------: | ----------: | -------------: | -----------: | ---------: | ------------: | ----------: | ------------: | ---------------: | -------------: |
| 256×256   |    51.55 |     0.48 |            0.53 |          0.62 |     107.76× |         96.44× |       83.68× |      94.42 |         84.49 |       73.32 |        323.72 |           289.69 |         251.37 |
| 512×512   |   236.54 |     1.48 |            1.87 |          2.43 |     160.28× |        126.34× |       97.47× |     123.37 |         97.25 |       75.02 |        422.98 |           333.41 |         257.21 |
| 1024×1024 |   906.35 |     5.34 |            6.13 |          4.46 |     169.79× |        147.79× |      203.44× |     136.97 |        119.22 |      164.11 |        469.61 |           408.74 |         562.68 |
| 2048×2048 |  3593.75 |    17.75 |           16.32 |         15.82 |     202.44× |        220.14× |      227.21× |     165.07 |        179.50 |      185.27 |        565.95 |           615.42 |         635.20 |

---

## Performance Observations

### CPU vs GPU

The CUDA implementation provides significant acceleration over the CPU implementation. For the largest problem size (2048×2048), the global GPU kernel achieves more than **202× speedup**.

### Shared Memory Optimization

Shared-memory tiling improves memory reuse and reduces global memory traffic. For larger grids, Shared GPU performance exceeds the global-memory implementation.

### Halo Optimization

The halo kernel achieves the best overall performance by maximizing memory locality and shared-memory utilization.

At a grid size of **2048×2048**, the Halo GPU kernel achieves:

* **227× speedup**
* **185 GFLOPS**
* **635 GB/s memory bandwidth**

### Computational Throughput

Peak computational performance:

| Kernel     | Peak GFLOPS |
| ---------- | ----------: |
| Global GPU |      165.07 |
| Shared GPU |      179.50 |
| Halo GPU   |      185.27 |

### Effective Memory Bandwidth

Peak memory bandwidth:

| Kernel     | Peak Bandwidth (GB/s) |
| ---------- | --------------------: |
| Global GPU |                565.95 |
| Shared GPU |                615.42 |
| Halo GPU   |                635.20 |

---
## Nsight Compute Profiling (Halo Kernel)

The Halo kernel was profiled using **NVIDIA Nsight Compute**. The profiling results indicate high occupancy, excellent warp execution efficiency, and effective cache utilization.

| Metric                    |                       Halo |
| ------------------------- | -------------------------: |
| Runtime                   |                  337.89 μs |
| Achieved Occupancy        |                     87.9 % |
| DRAM Throughput           |               39.67 % Peak |
| Memory Throughput         |               59.76 % Peak |
| L2 Cache Hit Rate         |                     76.6 % |
| Warp Execution Efficiency |                     99.9 % |
| Registers / Thread        |                         20 |
| Shared Memory / Block     |                     1.3 KB |
| Dominant Stall Reason     | No Eligible Warps (74.48%) |

The results show that the Halo kernel achieves high occupancy and near-perfect warp execution efficiency while benefiting from effective L2 cache reuse. The primary performance bottleneck is the lack of eligible warps available to hide latency, indicating that further improvements may be obtained through increased instruction-level parallelism or additional memory access overlap.

## Summary

The experimental results demonstrate that:

* GPU acceleration provides over two orders of magnitude speedup compared to CPU execution.
* Shared-memory tiling improves data locality and computational throughput.
* Halo optimization delivers the highest GFLOPS and memory bandwidth.
* Performance scales with increasing grid sizes, making the framework suitable for large-scale battery thermal simulations.
* The developed framework provides a foundation for future multi-GPU battery digital twins and electro-thermal simulations.

---

# Future Work and Next Development Stages

Temperature-dependent resistance:

```math
R(T)=R_0(1+\alpha(T-T_0))
```

Planned extensions and roadmap:

* Electro-thermal coupling
* SOC estimation
* SOH estimation
* Multi-GPU halo exchange
* CUDA streams
* NVLink optimization
* Battery Digital Twin
* AI surrogate models
* Tensor Core acceleration

---


# V1→V5 Electro-Thermal Model Evolution and Results

This release contains the complete **V1→V5 battery electro-thermal model progression**. All five models were executed using the NASA battery heat profile with **1,698 heat samples**, dynamic heat input, and normal execution mode.

## Model Evolution

| Model | Model description | Main addition |
|---|---|---|
| **V1** | NASA heat only | Dynamic NASA heat input |
| **V2** | NASA heat + Joule heating | Electrical resistive heat generation |
| **V3** | NASA heat + temperature-dependent resistance | Resistance varies with temperature |
| **V4** | NASA heat + temperature-dependent resistance + entropy heat | Entropic heat contribution |
| **V5** | SOC/SOH-aware electro-thermal model | SOC/SOH-dependent electro-thermal behavior |

### Common Simulation Configuration

- **SOC:** 1
- **SOH:** 1
- **SOC Resistance Coefficient:** 0.2
- **SOH Resistance Coefficient:** 0.5
- **SOC Heat Coefficient:** 0.1
- **SOH Heat Coefficient:** 0.2
- **Heat Mode:** `DYNAMIC Q`
- **Heat profile:** 1,698 samples
- **Resistance:** 0.01 Ω
- **Grid sizes:** 256×256, 512×512, 1024×1024, 2048×2048
- **Execution mode:** `NORMAL`

## V1 — NASA Heat Only

V1 uses the NASA heat profile directly as the thermal source. The supplied run completed successfully for all four grid sizes.

### V1 CPU vs GPU Validation

| Grid | RMSE (°C) | Max Error (°C) | Mean Error (°C) | Relative Error |
|---|---:|---:|---:|---:|
| 256×256 | 7.22e-08 | 3.81e-06 | 1.80e-09 | 2.58e-09 |
| 512×512 | 7.56e-08 | 7.63e-06 | 2.08e-09 | 2.69e-09 |
| 1024×1024 | 2.67e-08 | 3.81e-06 | 3.24e-10 | 9.50e-10 |
| 2048×2048 | 1.65e-08 | 7.63e-06 | 6.46e-11 | 5.86e-10 |

### V1 Performance

| Grid | CPU (ms) | GPU (ms) | Tiled (ms) | Halo (ms) | GPU Speedup | Tiled Speedup | Halo Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| 256×256 | 228.64 | 8.78 | 0.88 | 0.96 | 26.03× | 259.85× | 238.97× |
| 512×512 | 1084.98 | 1.94 | 2.16 | 2.18 | 559.34× | 503.10× | 497.75× |
| 1024×1024 | 4217.30 | 7.80 | 7.45 | 5.09 | 540.64× | 565.77× | 828.69× |
| 2048×2048 | 16767.11 | 21.69 | 19.65 | 18.17 | 772.87× | 853.10× | 923.01× |

## V2 — NASA Heat + Joule Heating

V2 extends the thermal model by adding **Joule heating** to the NASA heat source.

### V2 CPU vs GPU Validation

| Grid | RMSE (°C) | Max Error (°C) | Mean Error (°C) | Relative Error |
|---|---:|---:|---:|---:|
| 256×256 | 4.45e-07 | 1.53e-05 | 5.86e-08 | 1.40e-08 |
| 512×512 | 3.43e-07 | 1.53e-05 | 2.83e-08 | 1.07e-08 |
| 1024×1024 | 3.45e-07 | 2.29e-05 | 2.45e-08 | 1.08e-08 |
| 2048×2048 | 2.97e-07 | 1.53e-05 | 1.82e-08 | 9.27e-09 |

### V2 Performance

| Grid | CPU (ms) | GPU (ms) | Tiled (ms) | Halo (ms) | GPU Speedup | Tiled Speedup | Halo Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| 256×256 | 228.30 | 0.80 | 0.87 | 0.96 | 284.61× | 263.24× | 237.95× |
| 512×512 | 1094.45 | 1.99 | 2.14 | 2.16 | 551.07× | 511.91× | 505.75× |
| 1024×1024 | 4176.53 | 7.18 | 6.33 | 4.56 | 581.45× | 660.24× | 915.96× |
| 2048×2048 | 16882.72 | 22.77 | 20.67 | 19.82 | 741.39× | 816.66× | 851.81× |

## V3 — Temperature-Dependent Resistance

V3 extends V2 with **temperature-dependent resistance**.

### V3 CPU vs GPU Validation

| Grid | RMSE (°C) | Max Error (°C) | Mean Error (°C) | Relative Error |
|---|---:|---:|---:|---:|
| 256×256 | 1.60e-06 | 1.53e-05 | 1.26e-06 | 5.02e-08 |
| 512×512 | 1.75e-06 | 2.29e-05 | 1.54e-06 | 5.47e-08 |
| 1024×1024 | 1.81e-06 | 1.53e-05 | 1.67e-06 | 5.63e-08 |
| 2048×2048 | 1.84e-06 | 1.53e-05 | 1.74e-06 | 5.72e-08 |

### V3 Performance

| Grid | CPU (ms) | GPU (ms) | Tiled (ms) | Halo (ms) | GPU Speedup | Tiled Speedup | Halo Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| 256×256 | 300.63 | 0.82 | 0.88 | 0.96 | 368.53× | 341.16× | 314.27× |
| 512×512 | 951.53 | 1.95 | 2.16 | 2.39 | 487.06× | 439.65× | 398.47× |
| 1024×1024 | 3988.23 | 7.86 | 6.63 | 5.11 | 507.57× | 601.39× | 780.62× |
| 2048×2048 | 15973.15 | 23.30 | 21.20 | 20.72 | 685.52× | 753.30× | 770.93× |

## V4 — Entropy Heat

V4 extends V3 by adding **entropy heat** to the temperature-dependent electro-thermal model.

### V4 CPU vs GPU Validation

| Grid | RMSE (°C) | Max Error (°C) | Mean Error (°C) | Relative Error |
|---|---:|---:|---:|---:|
| 256×256 | 1.60e-06 | 1.53e-05 | 1.26e-06 | 5.02e-08 |
| 512×512 | 1.75e-06 | 2.29e-05 | 1.54e-06 | 5.47e-08 |
| 1024×1024 | 1.81e-06 | 1.53e-05 | 1.67e-06 | 5.63e-08 |
| 2048×2048 | 1.84e-06 | 1.53e-05 | 1.74e-06 | 5.72e-08 |

### V4 Performance

| Grid | CPU (ms) | GPU (ms) | Tiled (ms) | Halo (ms) | GPU Speedup | Tiled Speedup | Halo Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| 256×256 | 303.87 | 0.81 | 0.87 | 0.96 | 375.62× | 348.75× | 317.74× |
| 512×512 | 963.29 | 1.95 | 2.17 | 2.55 | 494.51× | 444.70× | 377.33× |
| 1024×1024 | 4021.11 | 7.35 | 6.12 | 5.00 | 546.81× | 656.81× | 804.79× |
| 2048×2048 | 15565.53 | 23.05 | 21.00 | 20.26 | 675.30× | 741.24× | 768.22× |

## V5 — SOC/SOH-Aware Electro-Thermal Model

V5 is the current physics-model endpoint of this release and implements the **SOC/SOH-aware electro-thermal model**.

### V5 CPU vs GPU Validation

| Grid | RMSE (°C) | Max Error (°C) | Mean Error (°C) | Relative Error |
|---|---:|---:|---:|---:|
| 256×256 | 1.60e-06 | 1.53e-05 | 1.26e-06 | 5.02e-08 |
| 512×512 | 1.75e-06 | 2.29e-05 | 1.54e-06 | 5.47e-08 |
| 1024×1024 | 1.81e-06 | 1.53e-05 | 1.67e-06 | 5.63e-08 |
| 2048×2048 | 1.84e-06 | 1.53e-05 | 1.74e-06 | 5.72e-08 |

### V5 Performance

| Grid | CPU (ms) | GPU (ms) | Tiled (ms) | Halo (ms) | GPU Speedup | Tiled Speedup | Halo Speedup | AI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 256×256 | 311.83 | 0.81 | 0.89 | 0.96 | 386.67× | 350.68× | 326.13× | 0.29 |
| 512×512 | 1412.12 | 1.94 | 2.20 | 2.55 | 728.73× | 643.15× | 554.83× | 0.29 |
| 1024×1024 | 5708.67 | 7.80 | 6.63 | 5.07 | 731.69× | 860.48× | 1125.42× | 0.29 |
| 2048×2048 | 22778.58 | 22.74 | 21.79 | 20.77 | 1001.81× | 1045.27× | **1096.88×** | 0.29 |

> **Note:** The final column above is the supplied **Arithmetic Intensity** value; it is labeled `AI` only to keep the table compact.

### V5 Throughput and Bandwidth

At 2048×2048:

| Metric | Global GPU | Tiled GPU | Halo GPU |
|---|---:|---:|---:|
| GFLOPS | 128.87 | 134.47 | **141.10** |
| Bandwidth (GB/s) | 441.86 | 461.02 | **483.79** |
| Arithmetic Intensity (FLOP/Byte) | 0.29 | 0.29 | 0.29 |
| Updates/sec | 18,410,666,313.14 | — | — |
| Latency (ms) | 0.23 | — | — |

The supplied V5 results therefore show the highest reported speedup, GFLOPS, and bandwidth for the Halo implementation at the largest tested grid.

# License
MIT License

---

# Author

**Vallinath S**

PGD-AI, IIT Jodhpur

📧 [g25ait1184@iitj.ac.in](mailto:g25ait1184@iitj.ac.in)

### Areas of Interest

* CUDA Programming
* GPU Computing
* Battery Management Systems
* Embedded Systems
* Battery Digital Twin
* AI for Energy Storage Systems