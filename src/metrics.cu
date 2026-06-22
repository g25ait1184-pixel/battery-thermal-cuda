#include "metrics.h"

double calculateGFLOPS(
    int N,
    int steps,
    double time_ms)
{
    double flops =
        7.0 *
        (N-2) *
        (N-2) *
        steps;

    return
        flops /
        (time_ms/1000.0) /
        1e9;
}


double calculateBandwidth(
    int N,
    int steps,
    double time_ms)
{
    double bytes =
        24.0 *
        (N-2) *
        (N-2) *
        steps;

    return
        bytes /
        (time_ms/1000.0) /
        1e9;
}


Metrics computeMetrics(
    int N,
    int steps,
    double cpu_ms,
    double gpu_ms,
    double tiled_ms,
    double halo_ms)
{
    Metrics m;

    m.cpu_ms = cpu_ms;
    m.gpu_ms = gpu_ms;
    m.tiled_ms = tiled_ms;
    m.halo_ms = halo_ms;

    //------------------------
    // Speedup
    //------------------------

    m.gpu_speedup =
        cpu_ms / gpu_ms;

    m.tiled_speedup =
        cpu_ms / tiled_ms;

    m.halo_speedup =
        cpu_ms / halo_ms;

    //------------------------
    // GFLOPS
    //------------------------

    m.gpu_gflops =
        calculateGFLOPS(
            N,
            steps,
            gpu_ms
        );

    m.tiled_gflops =
        calculateGFLOPS(
            N,
            steps,
            tiled_ms
        );

    m.halo_gflops =
        calculateGFLOPS(
            N,
            steps,
            halo_ms
        );

    //------------------------
    // Bandwidth
    //------------------------

    m.gpu_bw =
        calculateBandwidth(
            N,
            steps,
            gpu_ms
        );

    m.tiled_bw =
        calculateBandwidth(
            N,
            steps,
            tiled_ms
        );

    m.halo_bw =
        calculateBandwidth(
            N,
            steps,
            halo_ms
        );

    return m;
}