#include "metrics.h"

using namespace std;

//----------------------------------------------------
// FLOPs Per Cell for Each Battery Model
//----------------------------------------------------

double getFlopsPerCell(
    BatteryModelVersion model)
{
    switch(model)
    {
        //------------------------------------------------
        // Baseline stencil
        //------------------------------------------------

        case MODEL_V1:

            return 7.0;

        //------------------------------------------------
        // Stencil + Joule Heat
        //------------------------------------------------

        case MODEL_V2:

            return 12.0;

        //------------------------------------------------
        // Stencil + Joule Heat +
        // Temperature-dependent Resistance
        //------------------------------------------------

        case MODEL_V3:

            return 18.0;

        //------------------------------------------------
        // Stencil + Joule Heat +
        // Temperature-dependent Resistance +
        // Entropic Heat
        //------------------------------------------------

        case MODEL_V4:

            return 25.0;

        default:

            return 7.0;
    }
}

//----------------------------------------------------
// GFLOPS
//----------------------------------------------------

double calculateGFLOPS(
    BatteryModelVersion model,
    int N,
    int steps,
    double time_ms)
{
    if(time_ms <= 0.0)
        return 0.0;

    double flopsPerCell =
        getFlopsPerCell(model);

    double flops =
        flopsPerCell *
        (N-2) *
        (N-2) *
        steps;

    return
        flops /
        (time_ms/1000.0) /
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
        return 0.0;

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

//----------------------------------------------------
// Arithmetic Intensity
//----------------------------------------------------

double calculateArithmeticIntensity(
    BatteryModelVersion model,
    int N,
    int steps)
{
    double flops =
        getFlopsPerCell(model) *
        (N-2) *
        (N-2) *
        steps;

    double bytes =
        24.0 *
        (N-2) *
        (N-2) *
        steps;

    return
        flops /
        bytes;
}

//----------------------------------------------------
// Cell Updates / Second
//----------------------------------------------------

double calculateUpdatesPerSecond(
    int N,
    int steps,
    double gpu_ms)
{
    if(gpu_ms <= 0.0)
        return 0.0;

    double cells =
        (double)
        (N-2) *
        (N-2) *
        steps;

    return
        cells /
        (gpu_ms/1000.0);
}

//----------------------------------------------------
// Kernel Latency
//----------------------------------------------------

double calculateLatency(
    int steps,
    double gpu_ms)
{
    if(steps <= 0)
        return 0.0;

    return
        gpu_ms /
        steps;
}

//----------------------------------------------------
// Metrics
//----------------------------------------------------

Metrics computeMetrics(
    BatteryModelVersion model,
    int N,
    int steps,
    double cpu_ms,
    double gpu_ms,
    double tiled_ms,
    double halo_ms)
{
    Metrics m;

    //--------------------------------------------
    // Runtime
    //--------------------------------------------

    m.cpu_ms   = cpu_ms;
    m.gpu_ms   = gpu_ms;
    m.tiled_ms = tiled_ms;
    m.halo_ms  = halo_ms;

    //--------------------------------------------
    // Speedup
    //--------------------------------------------

    m.gpu_speedup =
        (gpu_ms>0.0)?
        cpu_ms/gpu_ms:0.0;

    m.tiled_speedup =
        (tiled_ms>0.0)?
        cpu_ms/tiled_ms:0.0;

    m.halo_speedup =
        (halo_ms>0.0)?
        cpu_ms/halo_ms:0.0;

    //--------------------------------------------
    // GFLOPS
    //--------------------------------------------

    m.gpu_gflops =
        calculateGFLOPS(
            model,
            N,
            steps,
            gpu_ms);

    m.tiled_gflops =
        calculateGFLOPS(
            model,
            N,
            steps,
            tiled_ms);

    m.halo_gflops =
        calculateGFLOPS(
            model,
            N,
            steps,
            halo_ms);

    //--------------------------------------------
    // Bandwidth
    //--------------------------------------------

    m.gpu_bw =
        calculateBandwidth(
            N,
            steps,
            gpu_ms);

    m.tiled_bw =
        calculateBandwidth(
            N,
            steps,
            tiled_ms);

    m.halo_bw =
        calculateBandwidth(
            N,
            steps,
            halo_ms);

    //--------------------------------------------
    // Arithmetic Intensity
    //--------------------------------------------

    m.arithmetic_intensity =
        calculateArithmeticIntensity(
            model,
            N,
            steps);

    //--------------------------------------------
    // Updates/sec
    //--------------------------------------------

    m.updates_per_sec =
        calculateUpdatesPerSecond(
            N,
            steps,
            gpu_ms);

    //--------------------------------------------
    // Latency
    //--------------------------------------------

    m.latency_ms =
        calculateLatency(
            steps,
            gpu_ms);

    return m;
}