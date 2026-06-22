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

## Summary

The experimental results demonstrate that:

* GPU acceleration provides over two orders of magnitude speedup compared to CPU execution.
* Shared-memory tiling improves data locality and computational throughput.
* Halo optimization delivers the highest GFLOPS and memory bandwidth.
* Performance scales with increasing grid sizes, making the framework suitable for large-scale battery thermal simulations.
* The developed framework provides a foundation for future multi-GPU battery digital twins and electro-thermal simulations.
