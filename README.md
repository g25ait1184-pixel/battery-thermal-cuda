# battery-thermal-cuda

GPU-Accelerated Battery Thermal Propagation Simulator using CUDA with Global Memory, Shared Memory, and Halo Optimization.

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

# Future Work

Temperature-dependent resistance:

```math
R(T)=R_0(1+\alpha(T-T_0))
```

Planned extensions:

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