#include "metrics.h"

using namespace std;

//----------------------------------------------------
// GFLOPS
//----------------------------------------------------

double calculateGFLOPS(
    int N,
    int steps,
    double time_ms)
{
    if(time_ms <= 0.0)
    {
        return 0.0;
    }

    // 5 additions + 1 multiplication + 1 addition
    double flops =
        7.0 *
        (N - 2) *
        (N - 2) *
        steps;

    return
        flops /
        (time_ms / 1000.0) /
        1e9;
}

//----------------------------------------------------
// Effective Bandwidth
//----------------------------------------------------

double calculateBandwidth(
    int N,
    int steps,
    double time_ms)
{
    if(time_ms <= 0.0)
    {
        return 0.0;
    }

    // 5 reads + 1 write = 6 floats = 24 bytes
    double bytes =
        24.0 *
        (N - 2) *
        (N - 2) *
        steps;

    return
        bytes /
        (time_ms / 1000.0) /
        1e9;
}

//----------------------------------------------------
// Metrics
//----------------------------------------------------

Metrics computeMetrics(
    int N,
    int steps,
    double cpu_ms,
    double gpu_ms,
    double tiled_ms,
    double halo_ms)
{
    Metrics m;

    //------------------------------------------------
    // Runtime
    //------------------------------------------------

    m.cpu_ms =
        cpu_ms;

    m.gpu_ms =
        gpu_ms;

    m.tiled_ms =
        tiled_ms;

    m.halo_ms =
        halo_ms;


    //------------------------------------------------
    // Speedup
    //------------------------------------------------

    m.gpu_speedup =
        (gpu_ms > 0.0)
        ?
        cpu_ms / gpu_ms
        :
        0.0;

    m.tiled_speedup =
        (tiled_ms > 0.0)
        ?
        cpu_ms / tiled_ms
        :
        0.0;

    m.halo_speedup =
        (halo_ms > 0.0)
        ?
        cpu_ms / halo_ms
        :
        0.0;


    //------------------------------------------------
    // GFLOPS
    //------------------------------------------------

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


    //------------------------------------------------
    // Effective Memory Bandwidth
    //------------------------------------------------

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