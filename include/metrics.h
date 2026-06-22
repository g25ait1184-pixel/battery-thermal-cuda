#pragma once

struct Metrics
{
    double cpu_ms;

    double gpu_ms;

    double tiled_ms;

    double halo_ms;

    double gpu_speedup;

    double tiled_speedup;

    double halo_speedup;

    double gpu_gflops;

    double tiled_gflops;

    double halo_gflops;

    double gpu_bw;

    double tiled_bw;

    double halo_bw;
};

// GFLOPS

double calculateGFLOPS(
    int N,
    int steps,
    double ms
);

// Bandwidth

double calculateBandwidth(
    int N,
    int steps,
    double ms
);

// Metrics

Metrics computeMetrics(
    int N,
    int steps,
    double cpu_ms,
    double gpu_ms,
    double tiled_ms,
    double halo_ms
);