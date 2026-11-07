#include <vector>
#include <algorithm>
#include <iostream>
#include <cuda_runtime.h>

#include "battery_model.h"

using namespace std;

//======================================================
// GPU THERMAL STENCIL KERNEL
//======================================================

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
    // V5 SOC / SOH parameters
    float soc,
    float soh,
    float socResistanceCoeff,
    float sohResistanceCoeff,
    float socHeatCoeff,
    float sohHeatCoeff)
{
    // Global thread coordinates
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;

    // Update interior cells only. Boundary cells retain their
    // initialized values in both ping-pong buffers.
    if (i <= 0 || i >= N - 1 || j <= 0 || j >= N - 1) {
        return;
    }

    const int idx = i * N + j;
    const float center = T[idx];

    // Temperature-dependent thermal diffusivity
    const float alphaLocal = computeDiffusivity(
        alpha,
        center,
        referenceTemperature,
        diffusivityCoeff);

    // Calculate resistance according to model version.
    // V1-V4: temperature-dependent resistance.
    // V5: temperature + SOC + SOH-dependent resistance.
    float resistance = 0.0f;

    if (modelVersion == MODEL_V5) {
        resistance = computeV5Resistance(
            baseResistance,
            center,
            referenceTemperature,
            resistanceCoeff,
            soc,
            soh,
            socResistanceCoeff,
            sohResistanceCoeff);
    } else {
        resistance = computeResistance(
            baseResistance,
            center,
            referenceTemperature,
            resistanceCoeff);
    }

    // Total heat dispatcher. For V5, resistance is already
    // SOC/SOH-corrected by computeV5Resistance().
    const float heat = computeTotalHeat(
        static_cast<BatteryModelVersion>(modelVersion),
        nasaHeat,
        current,
        resistance,
        center,
        soc,
        entropyCoeff,
        socResistanceCoeff,
        sohResistanceCoeff,
        socHeatCoeff,
        sohHeatCoeff,
        soh);

    // 2-D five-point Laplacian
    const float laplacian =
        T[idx - N] +
        T[idx + N] +
        T[idx - 1] +
        T[idx + 1] -
        4.0f * center;

    // Explicit thermal update
    Tnew[idx] =
        center +
        alphaLocal * laplacian +
        beta * heat;
}

//======================================================
// GLOBAL-MEMORY GPU SOLVER
//======================================================

void runGPU(
    float* d_T,
    float* d_Tnew,
    int N,
    float alpha,
    int steps,
    float beta,
    const BatteryParameters& params,
    const vector<float>& heatProfile)
{
    // Validate solver inputs
    if (d_T == nullptr || d_Tnew == nullptr) {
        cerr << "ERROR: runGPU received a null CUDA buffer.\n";
        return;
    }

    if (N <= 2) {
        cerr << "ERROR: runGPU grid size must be greater than 2.\n";
        return;
    }

    if (steps <= 0) {
        cerr << "ERROR: runGPU requires steps greater than zero.\n";
        return;
    }

    if (heatProfile.empty()) {
        cerr << "ERROR: runGPU received an empty heat profile.\n";
        return;
    }

    // CUDA launch configuration
    const dim3 block(16, 16);
    const dim3 grid(
        (N + block.x - 1) / block.x,
        (N + block.y - 1) / block.y);

    // Time integration with ping-pong buffers
    for (int s = 0; s < steps; ++s) {
        const float nasaHeat = heatProfile[
            static_cast<size_t>(s) % heatProfile.size()];

        gpuStencil<<<grid, block>>>(
            d_T,
            d_Tnew,
            N,
            alpha,
            beta,
            nasaHeat,
            params.modelVersion,
            params.current,
            params.baseResistance,
            params.referenceTemperature,
            params.resistanceCoeff,
            params.diffusivityCoeff,
            params.entropyCoeff,
            params.soc,
            params.soh,
            params.socResistanceCoeff,
            params.sohResistanceCoeff,
            params.socHeatCoeff,
            params.sohHeatCoeff);

        // Detect launch/configuration errors immediately.
        const cudaError_t launchError = cudaGetLastError();
        if (launchError != cudaSuccess) {
            cerr << "ERROR: gpuStencil launch failed at step "
                 << s << ": " << cudaGetErrorString(launchError) << '\n';
            return;
        }

        // Swap input/output buffers for the next time step.
        swap(d_T, d_Tnew);
    }

    // Wait for all submitted stencil work to finish.
    const cudaError_t syncError = cudaDeviceSynchronize();
    if (syncError != cudaSuccess) {
        cerr << "ERROR: runGPU synchronization failed: "
             << cudaGetErrorString(syncError) << '\n';
    }

    // NOTE:
    // The pointers are swapped locally. The caller's pointers are not
    // updated. If the caller needs the final field in its original d_T
    // allocation, use an even number of steps or change this interface
    // to return/pass back the final buffer pointer.
}
