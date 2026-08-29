# battery-thermal-cuda

GPU-Accelerated Battery Electro-Thermal Simulation and Thermal Propagation Framework using CUDA, with Global Memory, Shared Memory, and Halo optimization.

## Current Release

**v1.0.0 â€” Complete V1â†’V5 Validated Thermal Solver**

This release consolidates the completed V1â†’V5 battery electro-thermal model progression and the corresponding CPU/GPU, tiled, and halo validation results. The V1â†’V5 execution completed successfully for 256Ã—256, 512Ã—512, 1024Ã—1024, and 2048Ã—2048 grids.

### V1â†’V5 Model Evolution

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
v0.1.0  â†’ NASA Heat V1 baseline
v1.0.0  â†’ Complete V1â†’V5 validated thermal solver
v1.1.0  â†’ CUDA Streams integration
v1.2.0  â†’ Performance validation
v1.3.0  â†’ V5 simulation dataset generation
v2.0.0  â†’ AI thermal predictor
v2.1.0  â†’ Physics + AI residual correction
v3.0.0  â†’ Battery Thermal Digital Twin
v3.1.0  â†’ Final integrated benchmark
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
* (R) = Internal resistance (Î©)
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

        â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
        â”‚                    â”‚
        â”‚    Thermal Grid    â”‚
        â”‚                    â”‚
        â”‚      256Ã—256       â”‚
        â”‚                    â”‚
        â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

Each grid cell represents a small region of the battery.

## Grid Resolution

| Grid Size | Number of Cells |
| --------- | --------------: |
| 256Ã—256   |          65,536 |
| 512Ã—512   |         262,144 |
| 1024Ã—1024 |       1,048,576 |
| 2048Ã—2048 |       4,194,304 |

Larger grids provide finer thermal resolution and improved accuracy.

---

# Heat Stencil

Each grid point exchanges heat with four neighboring cells.

```text
        Top
         â†‘
Left â† Center â†’ Right
         â†“
      Bottom
```

---

# GPU Mapping

Each grid cell is mapped to one CUDA thread.

```text
1 Battery Region
       â†“
1 Grid Cell
       â†“
1 CUDA Thread
       â†“
Temperature Update
```

For a 2048Ã—2048 grid:

* Total cells = 4,194,304
* Total CUDA threads â‰ˆ 4.2 million

---

# GPU Kernels

## Global Memory GPU

Neighbor values are accessed directly from global memory.

```text
Global Memory
      â†“
    Thread
      â†“
   Compute
```

---

## Shared Memory GPU

Tiles are loaded into shared memory to reduce global memory accesses.

```text
Global Memory
      â†“
 Shared Tile
      â†“
   Threads
      â†“
   Compute
```

---

## Halo GPU

Halo cells are loaded around each tile to maximize memory locality and bandwidth utilization.

```text
Global Memory
       â†“
 Shared Tile + Halo
       â†“
      Threads
       â†“
      Compute
```

---

# Project Structure

```text
battery-thermal-cuda
â”‚
â”œâ”€â”€ include
â”‚   â”œâ”€â”€ config.h
â”‚   â”œâ”€â”€ benchmark.h
â”‚   â”œâ”€â”€ kernels.h
â”‚   â”œâ”€â”€ metrics.h
â”‚   â”œâ”€â”€ thermal_map.h
â”‚   â”œâ”€â”€ utils.h
â”‚   â””â”€â”€ cuda_events.h
â”‚
â”œâ”€â”€ src
â”‚   â”œâ”€â”€ main.cu
â”‚   â”œâ”€â”€ config.cu
â”‚   â”œâ”€â”€ benchmark_cpu.cu
â”‚   â”œâ”€â”€ benchmark_gpu.cu
â”‚   â”œâ”€â”€ benchmark_tiled.cu
â”‚   â”œâ”€â”€ benchmark_halo.cu
â”‚   â”œâ”€â”€ metrics.cu
â”‚   â”œâ”€â”€ thermal_map.cu
â”‚   â”œâ”€â”€ utils.cu
â”‚   â””â”€â”€ cuda_events.cu
â”‚
â”œâ”€â”€ kernels
â”‚   â”œâ”€â”€ gpu_stencil.cu
â”‚   â”œâ”€â”€ tiled_stencil.cu
â”‚   â””â”€â”€ halo_stencil.cu
â”‚
â”œâ”€â”€ data
â”‚   â””â”€â”€ battery_dataset.csv
â”‚
â”œâ”€â”€ results
â”‚   â”œâ”€â”€ results.csv
â”‚   â”œâ”€â”€ center_temp.csv
â”‚   â””â”€â”€ thermal_map.csv
â”‚
â”œâ”€â”€ plots
â”‚   â”œâ”€â”€ speedup.png
â”‚   â”œâ”€â”€ bandwidth.png
â”‚   â”œâ”€â”€ roofline.png
â”‚   â””â”€â”€ thermal_map.png
â”‚
â”œâ”€â”€ README.md
â”œâ”€â”€ LICENSE
â””â”€â”€ .gitignore
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
| 256Ã—256   |        107Ã— |          84Ã— |
| 512Ã—512   |        160Ã— |          97Ã— |
| 1024Ã—1024 |        170Ã— |         203Ã— |
| 2048Ã—2048 |        202Ã— |         227Ã— |

---

# Roofline Characteristics

Arithmetic Intensity

```text
AI â‰ˆ 0.29 FLOP/Byte
```

The stencil kernel is primarily memory-bandwidth limited, making shared-memory and halo optimizations highly effective.

# Performance Results

The framework reports runtime, speedup, computational throughput (GFLOPS), and effective memory bandwidth for CPU, Global GPU, Shared GPU, and Halo GPU implementations.

## Performance Comparison

| Grid Size | CPU (ms) | GPU (ms) | Shared GPU (ms) | Halo GPU (ms) | GPU Speedup | Shared Speedup | Halo Speedup | GPU GFLOPS | Shared GFLOPS | Halo GFLOPS | GPU BW (GB/s) | Shared BW (GB/s) | Halo BW (GB/s) |
| --------- | -------: | -------: | --------------: | ------------: | ----------: | -------------: | -----------: | ---------: | ------------: | ----------: | ------------: | ---------------: | -------------: |
| 256Ã—256   |    51.55 |     0.48 |            0.53 |          0.62 |     107.76Ã— |         96.44Ã— |       83.68Ã— |      94.42 |         84.49 |       73.32 |        323.72 |           289.69 |         251.37 |
| 512Ã—512   |   236.54 |     1.48 |            1.87 |          2.43 |     160.28Ã— |        126.34Ã— |       97.47Ã— |     123.37 |         97.25 |       75.02 |        422.98 |           333.41 |         257.21 |
| 1024Ã—1024 |   906.35 |     5.34 |            6.13 |          4.46 |     169.79Ã— |        147.79Ã— |      203.44Ã— |     136.97 |        119.22 |      164.11 |        469.61 |           408.74 |         562.68 |
| 2048Ã—2048 |  3593.75 |    17.75 |           16.32 |         15.82 |     202.44Ã— |        220.14Ã— |      227.21Ã— |     165.07 |        179.50 |      185.27 |        565.95 |           615.42 |         635.20 |

---

## Performance Observations

### CPU vs GPU

The CUDA implementation provides significant acceleration over the CPU implementation. For the largest problem size (2048Ã—2048), the global GPU kernel achieves more than **202Ã— speedup**.

### Shared Memory Optimization

Shared-memory tiling improves memory reuse and reduces global memory traffic. For larger grids, Shared GPU performance exceeds the global-memory implementation.

### Halo Optimization

The halo kernel achieves the best overall performance by maximizing memory locality and shared-memory utilization.

At a grid size of **2048Ã—2048**, the Halo GPU kernel achieves:

* **227Ã— speedup**
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
| Runtime                   |                  337.89 Î¼s |
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


# V1â†’V5 Electro-Thermal Model Evolution and Results

This release contains the complete **V1â†’V5 battery electro-thermal model progression**. All five models were executed using the NASA battery heat profile with **1,698 heat samples**, dynamic heat input, and normal execution mode.

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
- **Resistance:** 0.01 Î©
- **Grid sizes:** 256Ã—256, 512Ã—512, 1024Ã—1024, 2048Ã—2048
- **Execution mode:** `NORMAL`

## V1 â€” NASA Heat Only

V1 uses the NASA heat profile directly as the thermal source. The supplied run completed successfully for all four grid sizes.

### V1 CPU vs GPU Validation

| Grid | RMSE (Â°C) | Max Error (Â°C) | Mean Error (Â°C) | Relative Error |
|---|---:|---:|---:|---:|
| 256Ã—256 | 7.22e-08 | 3.81e-06 | 1.80e-09 | 2.58e-09 |
| 512Ã—512 | 7.56e-08 | 7.63e-06 | 2.08e-09 | 2.69e-09 |
| 1024Ã—1024 | 2.67e-08 | 3.81e-06 | 3.24e-10 | 9.50e-10 |
| 2048Ã—2048 | 1.65e-08 | 7.63e-06 | 6.46e-11 | 5.86e-10 |

### V1 Performance

| Grid | CPU (ms) | GPU (ms) | Tiled (ms) | Halo (ms) | GPU Speedup | Tiled Speedup | Halo Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| 256Ã—256 | 228.64 | 8.78 | 0.88 | 0.96 | 26.03Ã— | 259.85Ã— | 238.97Ã— |
| 512Ã—512 | 1084.98 | 1.94 | 2.16 | 2.18 | 559.34Ã— | 503.10Ã— | 497.75Ã— |
| 1024Ã—1024 | 4217.30 | 7.80 | 7.45 | 5.09 | 540.64Ã— | 565.77Ã— | 828.69Ã— |
| 2048Ã—2048 | 16767.11 | 21.69 | 19.65 | 18.17 | 772.87Ã— | 853.10Ã— | 923.01Ã— |

## V2 â€” NASA Heat + Joule Heating

V2 extends the thermal model by adding **Joule heating** to the NASA heat source.

### V2 CPU vs GPU Validation

| Grid | RMSE (Â°C) | Max Error (Â°C) | Mean Error (Â°C) | Relative Error |
|---|---:|---:|---:|---:|
| 256Ã—256 | 4.45e-07 | 1.53e-05 | 5.86e-08 | 1.40e-08 |
| 512Ã—512 | 3.43e-07 | 1.53e-05 | 2.83e-08 | 1.07e-08 |
| 1024Ã—1024 | 3.45e-07 | 2.29e-05 | 2.45e-08 | 1.08e-08 |
| 2048Ã—2048 | 2.97e-07 | 1.53e-05 | 1.82e-08 | 9.27e-09 |

### V2 Performance

| Grid | CPU (ms) | GPU (ms) | Tiled (ms) | Halo (ms) | GPU Speedup | Tiled Speedup | Halo Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| 256Ã—256 | 228.30 | 0.80 | 0.87 | 0.96 | 284.61Ã— | 263.24Ã— | 237.95Ã— |
| 512Ã—512 | 1094.45 | 1.99 | 2.14 | 2.16 | 551.07Ã— | 511.91Ã— | 505.75Ã— |
| 1024Ã—1024 | 4176.53 | 7.18 | 6.33 | 4.56 | 581.45Ã— | 660.24Ã— | 915.96Ã— |
| 2048Ã—2048 | 16882.72 | 22.77 | 20.67 | 19.82 | 741.39Ã— | 816.66Ã— | 851.81Ã— |

## V3 â€” Temperature-Dependent Resistance

V3 extends V2 with **temperature-dependent resistance**.

### V3 CPU vs GPU Validation

| Grid | RMSE (Â°C) | Max Error (Â°C) | Mean Error (Â°C) | Relative Error |
|---|---:|---:|---:|---:|
| 256Ã—256 | 1.60e-06 | 1.53e-05 | 1.26e-06 | 5.02e-08 |
| 512Ã—512 | 1.75e-06 | 2.29e-05 | 1.54e-06 | 5.47e-08 |
| 1024Ã—1024 | 1.81e-06 | 1.53e-05 | 1.67e-06 | 5.63e-08 |
| 2048Ã—2048 | 1.84e-06 | 1.53e-05 | 1.74e-06 | 5.72e-08 |

### V3 Performance

| Grid | CPU (ms) | GPU (ms) | Tiled (ms) | Halo (ms) | GPU Speedup | Tiled Speedup | Halo Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| 256Ã—256 | 300.63 | 0.82 | 0.88 | 0.96 | 368.53Ã— | 341.16Ã— | 314.27Ã— |
| 512Ã—512 | 951.53 | 1.95 | 2.16 | 2.39 | 487.06Ã— | 439.65Ã— | 398.47Ã— |
| 1024Ã—1024 | 3988.23 | 7.86 | 6.63 | 5.11 | 507.57Ã— | 601.39Ã— | 780.62Ã— |
| 2048Ã—2048 | 15973.15 | 23.30 | 21.20 | 20.72 | 685.52Ã— | 753.30Ã— | 770.93Ã— |

## V4 â€” Entropy Heat

V4 extends V3 by adding **entropy heat** to the temperature-dependent electro-thermal model.

### V4 CPU vs GPU Validation

| Grid | RMSE (Â°C) | Max Error (Â°C) | Mean Error (Â°C) | Relative Error |
|---|---:|---:|---:|---:|
| 256Ã—256 | 1.60e-06 | 1.53e-05 | 1.26e-06 | 5.02e-08 |
| 512Ã—512 | 1.75e-06 | 2.29e-05 | 1.54e-06 | 5.47e-08 |
| 1024Ã—1024 | 1.81e-06 | 1.53e-05 | 1.67e-06 | 5.63e-08 |
| 2048Ã—2048 | 1.84e-06 | 1.53e-05 | 1.74e-06 | 5.72e-08 |

### V4 Performance

| Grid | CPU (ms) | GPU (ms) | Tiled (ms) | Halo (ms) | GPU Speedup | Tiled Speedup | Halo Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| 256Ã—256 | 303.87 | 0.81 | 0.87 | 0.96 | 375.62Ã— | 348.75Ã— | 317.74Ã— |
| 512Ã—512 | 963.29 | 1.95 | 2.17 | 2.55 | 494.51Ã— | 444.70Ã— | 377.33Ã— |
| 1024Ã—1024 | 4021.11 | 7.35 | 6.12 | 5.00 | 546.81Ã— | 656.81Ã— | 804.79Ã— |
| 2048Ã—2048 | 15565.53 | 23.05 | 21.00 | 20.26 | 675.30Ã— | 741.24Ã— | 768.22Ã— |

## V5 â€” SOC/SOH-Aware Electro-Thermal Model

V5 is the current physics-model endpoint of this release and implements the **SOC/SOH-aware electro-thermal model**.

### V5 CPU vs GPU Validation

| Grid | RMSE (Â°C) | Max Error (Â°C) | Mean Error (Â°C) | Relative Error |
|---|---:|---:|---:|---:|
| 256Ã—256 | 1.60e-06 | 1.53e-05 | 1.26e-06 | 5.02e-08 |
| 512Ã—512 | 1.75e-06 | 2.29e-05 | 1.54e-06 | 5.47e-08 |
| 1024Ã—1024 | 1.81e-06 | 1.53e-05 | 1.67e-06 | 5.63e-08 |
| 2048Ã—2048 | 1.84e-06 | 1.53e-05 | 1.74e-06 | 5.72e-08 |

### V5 Performance

| Grid | CPU (ms) | GPU (ms) | Tiled (ms) | Halo (ms) | GPU Speedup | Tiled Speedup | Halo Speedup | AI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 256Ã—256 | 311.83 | 0.81 | 0.89 | 0.96 | 386.67Ã— | 350.68Ã— | 326.13Ã— | 0.29 |
| 512Ã—512 | 1412.12 | 1.94 | 2.20 | 2.55 | 728.73Ã— | 643.15Ã— | 554.83Ã— | 0.29 |
| 1024Ã—1024 | 5708.67 | 7.80 | 6.63 | 5.07 | 731.69Ã— | 860.48Ã— | 1125.42Ã— | 0.29 |
| 2048Ã—2048 | 22778.58 | 22.74 | 21.79 | 20.77 | 1001.81Ã— | 1045.27Ã— | **1096.88Ã—** | 0.29 |

> **Note:** The final column above is the supplied **Arithmetic Intensity** value; it is labeled `AI` only to keep the table compact.

### V5 Throughput and Bandwidth

At 2048Ã—2048:

| Metric | Global GPU | Tiled GPU | Halo GPU |
|---|---:|---:|---:|
| GFLOPS | 128.87 | 134.47 | **141.10** |
| Bandwidth (GB/s) | 441.86 | 461.02 | **483.79** |
| Arithmetic Intensity (FLOP/Byte) | 0.29 | 0.29 | 0.29 |
| Updates/sec | 18,410,666,313.14 | â€” | â€” |
| Latency (ms) | 0.23 | â€” | â€” |

The supplied V5 results therefore show the highest reported speedup, GFLOPS, and bandwidth for the Halo implementation at the largest tested grid.


# License
MIT License

---

# Author

**Vallinath S**

PGD-AI, IIT Jodhpur

ðŸ“§ [g25ait1184@iitj.ac.in](mailto:g25ait1184@iitj.ac.in)

### Areas of Interest

* CUDA Programming
* GPU Computing
* Battery Management Systems
* Embedded Systems
* Battery Digital Twin
* AI for Energy Storage Systems