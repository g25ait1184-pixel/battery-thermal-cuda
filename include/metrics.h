#pragma once

#include "battery_model.h"

//----------------------------------------------------
// Metrics Structure
//----------------------------------------------------

struct Metrics
{
    //------------------------------------------------
    // Runtime
    //------------------------------------------------

    double cpu_ms;
    double gpu_ms;
    double tiled_ms;
    double halo_ms;

    //------------------------------------------------
    // Speedup
    //------------------------------------------------

    double gpu_speedup;
    double tiled_speedup;
    double halo_speedup;

    //------------------------------------------------
    // GFLOPS
    //------------------------------------------------

    double gpu_gflops;
    double tiled_gflops;
    double halo_gflops;

    //------------------------------------------------
    // Memory Bandwidth
    //------------------------------------------------

    double gpu_bw;
    double tiled_bw;
    double halo_bw;

    //------------------------------------------------
    // New Research Metrics
    //------------------------------------------------

    // Arithmetic Intensity (FLOPs / Byte)
    double arithmetic_intensity;

    // Cell Updates Per Second
    double updates_per_sec;

    // Kernel Latency Per Iteration
    double latency_ms;
};

//----------------------------------------------------
// FLOPs Per Cell
//----------------------------------------------------

double getFlopsPerCell(
    BatteryModelVersion model);

//----------------------------------------------------
// GFLOPS
//----------------------------------------------------

double calculateGFLOPS(
    BatteryModelVersion model,
    int N,
    int steps,
    double ms);

//----------------------------------------------------
// Effective Memory Bandwidth
//----------------------------------------------------

double calculateBandwidth(
    int N,
    int steps,
    double ms);

//----------------------------------------------------
// Arithmetic Intensity
//----------------------------------------------------

double calculateArithmeticIntensity(
    BatteryModelVersion model,
    int N,
    int steps);

//----------------------------------------------------
// Cell Updates Per Second
//----------------------------------------------------

double calculateUpdatesPerSecond(
    int N,
    int steps,
    double gpu_ms);

//----------------------------------------------------
// Kernel Latency
//----------------------------------------------------

double calculateLatency(
    int steps,
    double gpu_ms);

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
    double halo_ms);