#pragma once

#include <vector>
#include "battery_model.h"

//======================================================
// GPU KERNEL DECLARATIONS
//======================================================

//------------------------------------------------------
// Global Memory GPU Stencil
//------------------------------------------------------

__global__
void gpuStencil(
    float* T,
    float* Tnew,
    int N,
    float alpha,
    float beta,
    float Q
);


//------------------------------------------------------
// Shared Memory / Tiled GPU Stencil
//------------------------------------------------------

__global__
void tiledStencil(
    float* T,
    float* Tnew,
    int N,
    float alpha,
    float beta,
    float Q
);


//------------------------------------------------------
// Halo GPU Stencil
//------------------------------------------------------

__global__
void haloStencil(
    float* T,
    float* Tnew,
    int N,
    float alpha,
    float beta,
    float Q
);


//======================================================
// GPU SOLVER INTERFACES
//======================================================

//------------------------------------------------------
// Global GPU Solver
//------------------------------------------------------

void runGPU(
    float* d_T,
    float* d_Tnew,
    int N,
    float alpha,
    int steps,
    float beta,
    const BatteryParameters& params,
    const std::vector<float>& heatProfile
);


//------------------------------------------------------
// Tiled / Shared Memory GPU Solver
//------------------------------------------------------

void runTiledGPU(
    float* d_T,
    float* d_Tnew,
    int N,
    float alpha,
    int steps,
    float beta,
    const BatteryParameters& params,
    const std::vector<float>& heatProfile
);


//------------------------------------------------------
// Halo GPU Solver
//------------------------------------------------------

void runHaloGPU(
    float* d_T,
    float* d_Tnew,
    int N,
    float alpha,
    int steps,
    float beta,
    const BatteryParameters& params,
    const std::vector<float>& heatProfile
);