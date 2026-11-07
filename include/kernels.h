
#pragma once

#include <vector>
#include "battery_model.h"

//======================================================
// GPU KERNEL DECLARATIONS
//======================================================

//------------------------------------------------------
// Global Memory GPU Stencil
// V1–V5 thermal kernel.
// V5 supports SOC/SOH-dependent thermal modelling.
//------------------------------------------------------

__global__
void gpuStencil(
    float* T,
    float* Tnew,
    int N,

    float alpha,
    float beta,

    float nasaHeat,

    int modelVersion,

    float current,
    float baseResistance,

    float referenceTemperature,

    float resistanceCoeff,
    float diffusivityCoeff,

    float entropyCoeff,

    // V5 SOC/SOH parameters
    float soc,
    float soh,

    float socResistanceCoeff,
    float sohResistanceCoeff,

    float socHeatCoeff,
    float sohHeatCoeff
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

    float nasaHeat,

    int modelVersion,

    float current,
    float baseResistance,

    float referenceTemperature,

    float resistanceCoeff,
    float diffusivityCoeff,

    float entropyCoeff,

    // V5 SOC/SOH parameters
    float soc,
    float soh,

    float socResistanceCoeff,
    float sohResistanceCoeff,

    float socHeatCoeff,
    float sohHeatCoeff
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

    float nasaHeat,

    int modelVersion,

    float current,
    float baseResistance,

    float referenceTemperature,

    float resistanceCoeff,
    float diffusivityCoeff,

    float entropyCoeff,

    // V5 SOC/SOH parameters
    float soc,
    float soh,

    float socResistanceCoeff,
    float sohResistanceCoeff,

    float socHeatCoeff,
    float sohHeatCoeff
);


//======================================================
// GPU SOLVER INTERFACES
//======================================================


//------------------------------------------------------
// Global Memory GPU Solver
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
