#pragma once

#include <vector>
#include <cuda_runtime.h>

#include "battery_model.h"

//======================================================
// CUDA Streams Solver
//======================================================
//
// CUDA Streams are used to execute independent battery
// thermal simulations concurrently.
//
// The V5 physics is unchanged.
//
// Each simulation:
//     T0 -> T1 -> T2 -> ... -> Tn
//
// remains sequential within its own stream.
//
// Parallelism is obtained across independent simulations.
//======================================================


//------------------------------------------------------
// Stream Configuration
//------------------------------------------------------

struct StreamConfig
{
    int numStreams;
    int N;
    int steps;
};


//------------------------------------------------------
// Run one thermal simulation on one CUDA stream
//------------------------------------------------------

void runStreamCase(
    float* d_T,
    float* d_Tnew,
    int N,
    float alpha,
    int steps,
    float beta,
    const BatteryParameters& params,
    const std::vector<float>& heatProfile,
    cudaStream_t stream);


//------------------------------------------------------
// Run multiple independent thermal simulations using
// multiple CUDA streams.
//
// initialT is the same initial temperature field used
// by the existing CPU/GPU/Tiled/Halo benchmarks.
//------------------------------------------------------

double runCUDAStreams(
    int numStreams,
    int N,
    float alpha,
    int steps,
    float beta,
    const BatteryParameters& params,
    const std::vector<float>& initialT,
    const std::vector<float>& heatProfile,
    std::vector<float>& finalT);