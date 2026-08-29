# V1→V5 Validation Results

## Scope

This file records the complete execution output for the battery electro-thermal simulation models **V1 through V5**.

- Heat input: **NASA battery dataset**
- Heat samples: **1,698**
- Heat mode: **DYNAMIC Q**
- Execution mode: **NORMAL**
- Tested grids: **256×256, 512×512, 1024×1024, 2048×2048**
- Implementations benchmarked: **CPU, Global GPU, Tiled GPU, Halo GPU**

## Model Progression

| Model | Formulation |
|---|---|
| **V1** | NASA heat only |
| **V2** | NASA heat + Joule heating |
| **V3** | NASA heat + temperature-dependent resistance |
| **V4** | NASA heat + temperature-dependent resistance + entropy heat |
| **V5** | SOC/SOH-aware electro-thermal model |

## V5 Validation Summary

| Grid | RMSE (°C) | Max Error (°C) | Mean Error (°C) | Relative Error |
|---|---:|---:|---:|---:|
| 256×256 | 1.60e-06 | 1.53e-05 | 1.26e-06 | 5.02e-08 |
| 512×512 | 1.75e-06 | 2.29e-05 | 1.54e-06 | 5.47e-08 |
| 1024×1024 | 1.81e-06 | 1.53e-05 | 1.67e-06 | 5.63e-08 |
| 2048×2048 | 1.84e-06 | 1.53e-05 | 1.74e-06 | 5.72e-08 |

## V5 Performance — 2048×2048

| Metric | Global GPU | Tiled GPU | Halo GPU |
|---|---:|---:|---:|
| Speedup | 1001.81× | 1045.27× | **1096.88×** |
| GFLOPS | 128.87 | 134.47 | **141.10** |
| Bandwidth (GB/s) | 441.86 | 461.02 | **483.79** |
| Arithmetic Intensity | 0.29 | 0.29 | 0.29 |
| Updates/sec | 18,410,666,313.14 | — | — |
| Latency (ms) | 0.23 | — | — |

## Generated V5 Result Files

The V5 execution generated:

- `results_MODEL_V5.csv`
- `thermal_metrics_MODEL_V5_CPU.csv`
- `thermal_metrics_MODEL_V5_GPU.csv`
- `thermal_metrics_MODEL_V5_Tiled.csv`
- `thermal_metrics_MODEL_V5_Halo.csv`
- `validation_MODEL_V5.csv`
- thermal map output
- center-temperature output

---

## Complete Execution Record

```text
=====================================
Running MODEL_V1
=====================================
NASA heat only
Loaded 1698 heat samples

Heat Mode : DYNAMIC Q
Heat Profile Size = 1698
Execution Mode : NORMAL

Grid,CPU(ms),GPU(ms),Tiled(ms),Halo(ms),GPU Speedup,Tiled Speedup,Halo Speedup,GPU GFLOPS,Tiled GFLOPS,Halo GFLOPS,GPU BW(GB/s),Tiled BW(GB/s),Halo BW(GB/s),ArithmeticIntensity,Updates/sec,Latency(ms)

=====================================
Grid Size : 256 x 256
Center Cell : 128
Heat Profile Size : 1698
Resistance (Ohm) : 0.01
Heat Source Range : 1.22243e-08 to 0.022309
=====================================
256,228.64,8.78,0.88,0.96,26.03,259.85,238.97,5.14,51.33,47.20,17.63,175.98,161.84,0.29,734571995.52,0.09

CPU vs GPU Validation
Model             : MODEL_V1
Grid              : 256
RMSE (C)          : 7.22e-08
Max Error (C)     : 3.81e-06
Mean Error (C)    : 1.80e-09
Relative Error    : 2.58e-09

=====================================
Grid Size : 512 x 512
Center Cell : 256
Heat Profile Size : 1698
Resistance (Ohm) : 0.01
Heat Source Range : 0.00 to 0.02
=====================================
512,1084.98,1.94,2.16,2.18,559.34,503.10,497.75,93.86,84.42,83.53,321.81,289.45,286.38,0.29,13408775742.24,0.02

CPU vs GPU Validation
Model             : MODEL_V1
Grid              : 512
RMSE (C)          : 7.56e-08
Max Error (C)     : 7.63e-06
Mean Error (C)    : 2.08e-09
Relative Error    : 2.69e-09

=====================================
Grid Size : 1024 x 1024
Center Cell : 512
Heat Profile Size : 1698
Resistance (Ohm) : 0.01
Heat Source Range : 0.00 to 0.02
=====================================
1024,4217.30,7.80,7.45,5.09,540.64,565.77,828.69,93.73,98.09,143.67,321.36,336.29,492.57,0.29,13389850967.84,0.08

CPU vs GPU Validation
Model             : MODEL_V1
Grid              : 1024
RMSE (C)          : 2.67e-08
Max Error (C)     : 3.81e-06
Mean Error (C)    : 3.24e-10
Relative Error    : 9.50e-10

=====================================
Grid Size : 2048 x 2048
Center Cell : 1024
Heat Profile Size : 1698
Resistance (Ohm) : 0.01
Heat Source Range : 0.00 to 0.02
=====================================
2048,16767.11,21.69,19.65,18.17,772.87,853.10,923.01,135.07,149.09,161.31,463.10,511.17,553.06,0.29,19295715904.04,0.22

CPU vs GPU Validation
Model             : MODEL_V1
Grid              : 2048
RMSE (C)          : 1.65e-08
Max Error (C)     : 7.63e-06
Mean Error (C)    : 6.46e-11
Relative Error    : 5.86e-10

/content/drive/MyDrive/CUDA_Project/project/results/results_MODEL_V1.csv created

/content/drive/MyDrive/CUDA_Project/project/results/thermal_metrics_MODEL_V1_CPU.csv created

/content/drive/MyDrive/CUDA_Project/project/results/thermal_metrics_MODEL_V1_GPU.csv created

/content/drive/MyDrive/CUDA_Project/project/results/thermal_metrics_MODEL_V1_Tiled.csv created

/content/drive/MyDrive/CUDA_Project/project/results/thermal_metrics_MODEL_V1_Halo.csv created
/content/drive/MyDrive/CUDA_Project/project/results/validation_MODEL_V1.csv created

thermal_map file created

center_temp file created

Finished MODEL_V1

=====================================
Running MODEL_V2
=====================================
NASA heat + Joule heating
Loaded 1698 heat samples

Heat Mode : DYNAMIC Q
Heat Profile Size = 1698
Execution Mode : NORMAL

Grid,CPU(ms),GPU(ms),Tiled(ms),Halo(ms),GPU Speedup,Tiled Speedup,Halo Speedup,GPU GFLOPS,Tiled GFLOPS,Halo GFLOPS,GPU BW(GB/s),Tiled BW(GB/s),Halo BW(GB/s),ArithmeticIntensity,Updates/sec,Latency(ms)

=====================================
Grid Size : 256 x 256
Center Cell : 128
Heat Profile Size : 1698
Resistance (Ohm) : 0.01
Heat Source Range : 0.00 to 0.02
=====================================
256,228.30,0.80,0.87,0.96,284.61,263.24,237.95,96.52,89.27,80.69,193.03,178.54,161.38,0.50,8042960895.33,0.01

CPU vs GPU Validation
Model             : MODEL_V2
Grid              : 256
RMSE (C)          : 4.45e-07
Max Error (C)     : 1.53e-05
Mean Error (C)    : 5.86e-08
Relative Error    : 1.40e-08

=====================================
Grid Size : 512 x 512
Center Cell : 256
Heat Profile Size : 1698
Resistance (Ohm) : 0.01
Heat Source Range : 0.00 to 0.02
=====================================
512,1094.45,1.99,2.14,2.16,551.07,511.91,505.75,157.16,145.99,144.23,314.31,291.98,288.46,0.50,13096349629.02,0.02

CPU vs GPU Validation
Model             : MODEL_V2
Grid              : 512
RMSE (C)          : 3.43e-07
Max Error (C)     : 1.53e-05
Mean Error (C)    : 2.83e-08
Relative Error    : 1.07e-08

=====================================
Grid Size : 1024 x 1024
Center Cell : 512
Heat Profile Size : 1698
Resistance (Ohm) : 0.01
Heat Source Range : 0.00 to 0.02
=====================================
1024,4176.53,7.18,6.33,4.56,581.45,660.24,915.96,174.49,198.14,274.88,348.99,396.28,549.76,0.50,14541094186.33,0.07

CPU vs GPU Validation
Model             : MODEL_V2
Grid              : 1024
RMSE (C)          : 3.45e-07
Max Error (C)     : 2.29e-05
Mean Error (C)    : 2.45e-08
Relative Error    : 1.08e-08

=====================================
Grid Size : 2048 x 2048
Center Cell : 1024
Heat Profile Size : 1698
Resistance (Ohm) : 0.01
Heat Source Range : 0.00 to 0.02
=====================================
2048,16882.72,22.77,20.67,19.82,741.39,816.66,851.81,220.60,242.99,253.45,441.19,485.99,506.90,0.50,18382934555.85,0.23

CPU vs GPU Validation
Model             : MODEL_V2
Grid              : 2048
RMSE (C)          : 2.97e-07
Max Error (C)     : 1.53e-05
Mean Error (C)    : 1.82e-08
Relative Error    : 9.27e-09

/content/drive/MyDrive/CUDA_Project/project/results/results_MODEL_V2.csv created

/content/drive/MyDrive/CUDA_Project/project/results/thermal_metrics_MODEL_V2_CPU.csv created

/content/drive/MyDrive/CUDA_Project/project/results/thermal_metrics_MODEL_V2_GPU.csv created

/content/drive/MyDrive/CUDA_Project/project/results/thermal_metrics_MODEL_V2_Tiled.csv created

/content/drive/MyDrive/CUDA_Project/project/results/thermal_metrics_MODEL_V2_Halo.csv created
/content/drive/MyDrive/CUDA_Project/project/results/validation_MODEL_V2.csv created

thermal_map file created

center_temp file created

Finished MODEL_V2

=====================================
Running MODEL_V3
=====================================
NASA heat + temperature-dependent resistance
Loaded 1698 heat samples

Heat Mode : DYNAMIC Q
Heat Profile Size = 1698
Execution Mode : NORMAL

Grid,CPU(ms),GPU(ms),Tiled(ms),Halo(ms),GPU Speedup,Tiled Speedup,Halo Speedup,GPU GFLOPS,Tiled GFLOPS,Halo GFLOPS,GPU BW(GB/s),Tiled BW(GB/s),Halo BW(GB/s),ArithmeticIntensity,Updates/sec,Latency(ms)

=====================================
Grid Size : 256 x 256
Center Cell : 128
Heat Profile Size : 1698
Resistance (Ohm) : 0.01
Heat Source Range : 0.00 to 0.02
=====================================
256,300.63,0.82,0.88,0.96,368.53,341.16,314.27,142.36,131.78,121.40,189.81,175.71,161.86,0.75,7908760655.69,0.01

CPU vs GPU Validation
Model             : MODEL_V3
Grid              : 256
RMSE (C)          : 1.60e-06
Max Error (C)     : 1.53e-05
Mean Error (C)    : 1.26e-06
Relative Error    : 5.02e-08

=====================================
Grid Size : 512 x 512
Center Cell : 256
Heat Profile Size : 1698
Resistance (Ohm) : 0.01
Heat Source Range : 0.00 to 0.02
=====================================
512,951.53,1.95,2.16,2.39,487.06,439.65,398.47,239.65,216.32,196.06,319.53,288.43,261.41,0.75,13313773029.28,0.02

CPU vs GPU Validation
Model             : MODEL_V3
Grid              : 512
RMSE (C)          : 1.75e-06
Max Error (C)     : 2.29e-05
Mean Error (C)    : 1.54e-06
Relative Error    : 5.47e-08

=====================================
Grid Size : 1024 x 1024
Center Cell : 512
Heat Profile Size : 1698
Resistance (Ohm) : 0.01
Heat Source Range : 0.00 to 0.02
=====================================
1024,3988.23,7.86,6.63,5.11,507.57,601.39,780.62,239.27,283.50,367.99,319.03,378.00,490.65,0.75,13292789295.03,0.08

CPU vs GPU Validation
Model             : MODEL_V3
Grid              : 1024
RMSE (C)          : 1.81e-06
Max Error (C)     : 1.53e-05
Mean Error (C)    : 1.67e-06
Relative Error    : 5.63e-08

=====================================
Grid Size : 2048 x 2048
Center Cell : 1024
Heat Profile Size : 1698
Resistance (Ohm) : 0.01
Heat Source Range : 0.00 to 0.02
=====================================
2048,15973.15,23.30,21.20,20.72,685.52,753.30,770.93,323.38,355.35,363.67,431.18,473.80,484.89,0.75,17965695487.41,0.23

CPU vs GPU Validation
Model             : MODEL_V3
Grid              : 2048
RMSE (C)          : 1.84e-06
Max Error (C)     : 1.53e-05
Mean Error (C)    : 1.74e-06
Relative Error    : 5.72e-08

/content/drive/MyDrive/CUDA_Project/project/results/results_MODEL_V3.csv created

/content/drive/MyDrive/CUDA_Project/project/results/thermal_metrics_MODEL_V3_CPU.csv created

/content/drive/MyDrive/CUDA_Project/project/results/thermal_metrics_MODEL_V3_GPU.csv created

/content/drive/MyDrive/CUDA_Project/project/results/thermal_metrics_MODEL_V3_Tiled.csv created

/content/drive/MyDrive/CUDA_Project/project/results/thermal_metrics_MODEL_V3_Halo.csv created
/content/drive/MyDrive/CUDA_Project/project/results/validation_MODEL_V3.csv created

thermal_map file created

center_temp file created

Finished MODEL_V3

=====================================
Running MODEL_V4
=====================================
NASA heat + temperature-dependent resistance + entropy heat
Loaded 1698 heat samples

Heat Mode : DYNAMIC Q
Heat Profile Size = 1698
Execution Mode : NORMAL

Grid,CPU(ms),GPU(ms),Tiled(ms),Halo(ms),GPU Speedup,Tiled Speedup,Halo Speedup,GPU GFLOPS,Tiled GFLOPS,Halo GFLOPS,GPU BW(GB/s),Tiled BW(GB/s),Halo BW(GB/s),ArithmeticIntensity,Updates/sec,Latency(ms)

=====================================
Grid Size : 256 x 256
Center Cell : 128
Heat Profile Size : 1698
Resistance (Ohm) : 0.01
Heat Source Range : 0.00 to 0.02
=====================================
256,303.87,0.81,0.87,0.96,375.62,348.75,317.74,199.37,185.11,168.65,191.40,177.71,161.91,1.04,7974831037.46,0.01

CPU vs GPU Validation
Model             : MODEL_V4
Grid              : 256
RMSE (C)          : 1.60e-06
Max Error (C)     : 1.53e-05
Mean Error (C)    : 1.26e-06
Relative Error    : 5.02e-08

=====================================
Grid Size : 512 x 512
Center Cell : 256
Heat Profile Size : 1698
Resistance (Ohm) : 0.01
Heat Source Range : 0.00 to 0.02
=====================================
512,963.29,1.95,2.17,2.55,494.51,444.70,377.33,333.81,300.19,254.71,320.46,288.18,244.52,1.04,13352298629.02,0.02

CPU vs GPU Validation
Model             : MODEL_V4
Grid              : 512
RMSE (C)          : 1.75e-06
Max Error (C)     : 2.29e-05
Mean Error (C)    : 1.54e-06
Relative Error    : 5.47e-08

=====================================
Grid Size : 1024 x 1024
Center Cell : 512
Heat Profile Size : 1698
Resistance (Ohm) : 0.01
Heat Source Range : 0.00 to 0.02
=====================================
1024,4021.11,7.35,6.12,5.00,546.81,656.81,804.79,355.08,426.52,522.61,340.88,409.46,501.71,1.04,14203372887.77,0.07

CPU vs GPU Validation
Model             : MODEL_V4
Grid              : 1024
RMSE (C)          : 1.81e-06
Max Error (C)     : 1.53e-05
Mean Error (C)    : 1.67e-06
Relative Error    : 5.63e-08

=====================================
Grid Size : 2048 x 2048
Center Cell : 1024
Heat Profile Size : 1698
Resistance (Ohm) : 0.01
Heat Source Range : 0.00 to 0.02
=====================================
2048,15565.53,23.05,21.00,20.26,675.30,741.24,768.22,454.03,498.36,516.50,435.87,478.43,495.84,1.04,18161224185.83,0.23

CPU vs GPU Validation
Model             : MODEL_V4
Grid              : 2048
RMSE (C)          : 1.84e-06
Max Error (C)     : 1.53e-05
Mean Error (C)    : 1.74e-06
Relative Error    : 5.72e-08

/content/drive/MyDrive/CUDA_Project/project/results/results_MODEL_V4.csv created

/content/drive/MyDrive/CUDA_Project/project/results/thermal_metrics_MODEL_V4_CPU.csv created

/content/drive/MyDrive/CUDA_Project/project/results/thermal_metrics_MODEL_V4_GPU.csv created

/content/drive/MyDrive/CUDA_Project/project/results/thermal_metrics_MODEL_V4_Tiled.csv created

/content/drive/MyDrive/CUDA_Project/project/results/thermal_metrics_MODEL_V4_Halo.csv created
/content/drive/MyDrive/CUDA_Project/project/results/validation_MODEL_V4.csv created

thermal_map file created

center_temp file created

Finished MODEL_V4

=====================================
Running MODEL_V5
=====================================
SOC/SOH-aware electro-thermal model
Loaded 1698 heat samples

Heat Mode : DYNAMIC Q
Heat Profile Size = 1698
Execution Mode : NORMAL

Grid,CPU(ms),GPU(ms),Tiled(ms),Halo(ms),GPU Speedup,Tiled Speedup,Halo Speedup,GPU GFLOPS,Tiled GFLOPS,Halo GFLOPS,GPU BW(GB/s),Tiled BW(GB/s),Halo BW(GB/s),ArithmeticIntensity,Updates/sec,Latency(ms)

=====================================
Grid Size : 256 x 256
Center Cell : 128
Heat Profile Size : 1698
Resistance (Ohm) : 0.01
Heat Source Range : 0.00 to 0.02
=====================================
256,311.83,0.81,0.89,0.96,386.67,350.68,326.13,56.00,50.79,47.23,192.01,174.13,161.94,0.29,8000210260.41,0.01

CPU vs GPU Validation
Model             : MODEL_V5
Grid              : 256
RMSE (C)          : 1.60e-06
Max Error (C)     : 1.53e-05
Mean Error (C)    : 1.26e-06
Relative Error    : 5.02e-08

=====================================
Grid Size : 512 x 512
Center Cell : 256
Heat Profile Size : 1698
Resistance (Ohm) : 0.01
Heat Source Range : 0.00 to 0.02
=====================================
512,1412.12,1.94,2.20,2.55,728.73,643.15,554.83,93.96,82.92,71.54,322.14,284.31,245.27,0.29,13422515337.06,0.02

CPU vs GPU Validation
Model             : MODEL_V5
Grid              : 512
RMSE (C)          : 1.75e-06
Max Error (C)     : 2.29e-05
Mean Error (C)    : 1.54e-06
Relative Error    : 5.47e-08

=====================================
Grid Size : 1024 x 1024
Center Cell : 512
Heat Profile Size : 1698
Resistance (Ohm) : 0.01
Heat Source Range : 0.00 to 0.02
=====================================
1024,5708.67,7.80,6.63,5.07,731.69,860.48,1125.42,93.71,110.21,144.14,321.30,377.85,494.19,0.29,13387368706.67,0.08

CPU vs GPU Validation
Model             : MODEL_V5
Grid              : 1024
RMSE (C)          : 1.81e-06
Max Error (C)     : 1.53e-05
Mean Error (C)    : 1.67e-06
Relative Error    : 5.63e-08

=====================================
Grid Size : 2048 x 2048
Center Cell : 1024
Heat Profile Size : 1698
Resistance (Ohm) : 0.01
Heat Source Range : 0.00 to 0.02
=====================================
2048,22778.58,22.74,21.79,20.77,1001.81,1045.27,1096.88,128.87,134.47,141.10,441.86,461.02,483.79,0.29,18410666313.14,0.23

CPU vs GPU Validation
Model             : MODEL_V5
Grid              : 2048
RMSE (C)          : 1.84e-06
Max Error (C)     : 1.53e-05
Mean Error (C)    : 1.74e-06
Relative Error    : 5.72e-08

/content/drive/MyDrive/CUDA_Project/project/results/results_MODEL_V5.csv created

/content/drive/MyDrive/CUDA_Project/project/results/thermal_metrics_MODEL_V5_CPU.csv created

/content/drive/MyDrive/CUDA_Project/project/results/thermal_metrics_MODEL_V5_GPU.csv created

/content/drive/MyDrive/CUDA_Project/project/results/thermal_metrics_MODEL_V5_Tiled.csv created

/content/drive/MyDrive/CUDA_Project/project/results/thermal_metrics_MODEL_V5_Halo.csv created
/content/drive/MyDrive/CUDA_Project/project/results/validation_MODEL_V5.csv created

thermal_map file created

center_temp file created

Finished MODEL_V5
```

---

This record corresponds to the supplied V1→V5 execution output. The run concludes with **All Battery Models Completed — V1 → V5**.

