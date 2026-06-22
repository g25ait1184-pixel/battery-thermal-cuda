#pragma once

#include <vector>

using namespace std;

// Global GPU

__global__
void gpuStencil(
    float *T,
    float *Tnew,
    int N,
    float alpha,
    float beta,
    float Q
);

// Shared GPU

__global__
void tiledStencil(
    float *T,
    float *Tnew,
    int N,
    float alpha,
    float beta,
    float Q
);

// Halo GPU

__global__
void haloStencil(
    float *T,
    float *Tnew,
    int N,
    float alpha,
    float beta,
    float Q
);



//------------------------------------------------
// Global GPU
//------------------------------------------------

__global__
void gpuStencil(
    float *T,
    float *Tnew,
    int N,
    float alpha,
    float beta,
    float Q
);

void runGPU(
    float *d_T,
    float *d_Tnew,
    int N,
    float alpha,
    int steps,
    float beta,
    const vector<float>& heatProfile
);


//------------------------------------------------
// Shared GPU
//------------------------------------------------

__global__
void tiledStencil(
    float *T,
    float *Tnew,
    int N,
    float alpha,
    float beta,
    float Q
);

void runTiledGPU(
    float *d_T,
    float *d_Tnew,
    int N,
    float alpha,
    int steps,
    float beta,
    const vector<float>& heatProfile
);


//------------------------------------------------
// Halo GPU
//------------------------------------------------

__global__
void haloStencil(
    float *T,
    float *Tnew,
    int N,
    float alpha,
    float beta,
    float Q
);

void runHaloGPU(
    float *d_T,
    float *d_Tnew,
    int N,
    float alpha,
    int steps,
    float beta,
    const vector<float>& heatProfile
);