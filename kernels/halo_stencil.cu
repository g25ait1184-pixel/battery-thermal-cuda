#include <vector>
#include <cuda_runtime.h>

#include "battery_model.h"

using namespace std;

#define TILE 16


//------------------------------------------------------
// Halo Shared Memory GPU Thermal Solver
//------------------------------------------------------

__global__
void haloStencil(
    float *T,
    float *Tnew,
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

    //==================================================
    // V5 - SOC / SOH Parameters
    //==================================================

    float soc,
    float soh,

    float socResistanceCoeff,
    float sohResistanceCoeff,

    float socHeatCoeff,
    float sohHeatCoeff)
{
    //--------------------------------------------------
    // Shared memory tile
    //--------------------------------------------------

    __shared__
    float tile[TILE + 2][TILE + 2];


    //--------------------------------------------------
    // Thread coordinates
    //--------------------------------------------------

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int j =
        blockIdx.x * TILE + tx;

    int i =
        blockIdx.y * TILE + ty;


    //--------------------------------------------------
    // Load center cell
    //--------------------------------------------------

    if(i < N && j < N)
    {
        tile[ty + 1][tx + 1] =
            T[i * N + j];
    }


    //--------------------------------------------------
    // Left Halo
    //--------------------------------------------------

    if(tx == 0 &&
       i < N &&
       j > 0)
    {
        tile[ty + 1][0] =
            T[i * N + j - 1];
    }


    //--------------------------------------------------
    // Right Halo
    //--------------------------------------------------

    if(tx == TILE - 1 &&
       i < N &&
       j < N - 1)
    {
        tile[ty + 1][TILE + 1] =
            T[i * N + j + 1];
    }


    //--------------------------------------------------
    // Top Halo
    //--------------------------------------------------

    if(ty == 0 &&
       i > 0 &&
       j < N)
    {
        tile[0][tx + 1] =
            T[(i - 1) * N + j];
    }


    //--------------------------------------------------
    // Bottom Halo
    //--------------------------------------------------

    if(ty == TILE - 1 &&
       i < N - 1 &&
       j < N)
    {
        tile[TILE + 1][tx + 1] =
            T[(i + 1) * N + j];
    }


    //--------------------------------------------------
    // Synchronize shared memory
    //--------------------------------------------------

    __syncthreads();


    //--------------------------------------------------
    // Interior cells
    //--------------------------------------------------

    if(i > 0 &&
       i < N - 1 &&
       j > 0 &&
       j < N - 1)
    {
        int idx =
            i * N + j;


        //------------------------------------------------
        // Current temperature
        //------------------------------------------------

        float center =
            tile[ty + 1][tx + 1];


        //------------------------------------------------
        // Temperature-dependent diffusivity
        //------------------------------------------------

        float alphaLocal =
            computeDiffusivity(
                alpha,
                center,
                referenceTemperature,
                diffusivityCoeff
            );


        //------------------------------------------------
        // Temperature-dependent resistance
        //------------------------------------------------

        float resistance =
            computeResistance(
                baseResistance,
                center,
                referenceTemperature,
                resistanceCoeff
            );


        //------------------------------------------------
        // Total heat generation
        //
        // V1:
        // NASA heat
        //
        // V2:
        // NASA + Joule heat
        //
        // V3:
        // NASA + temperature-dependent resistance
        //
        // V4:
        // V3 + entropy heat
        //
        // V5:
        // V4 + SOC/SOH effects
        //------------------------------------------------

        float heat =
            computeTotalHeat(
                (BatteryModelVersion)modelVersion,

                nasaHeat,

                current,

                resistance,

                center,

                // SOC
                soc,

                // Entropy coefficient
                entropyCoeff,

                // SOC resistance coefficient
                socResistanceCoeff,

                // SOH resistance coefficient
                sohResistanceCoeff,

                // SOC heat coefficient
                socHeatCoeff,

                // SOH heat coefficient
                sohHeatCoeff,

                // SOH
                soh
            );


        //------------------------------------------------
        // Laplacian using shared-memory halo
        //------------------------------------------------

        float laplacian =
            tile[ty][tx + 1]
            +
            tile[ty + 2][tx + 1]
            +
            tile[ty + 1][tx]
            +
            tile[ty + 1][tx + 2]
            -
            4.0f * center;


        //------------------------------------------------
        // Temperature update
        //------------------------------------------------

        Tnew[idx] =
            center
            +
            alphaLocal * laplacian
            +
            beta * heat;
    }
}


//------------------------------------------------------
// Run Halo GPU Solver
//------------------------------------------------------

void runHaloGPU(
    float *d_T,
    float *d_Tnew,
    int N,

    float alpha,
    int steps,
    float beta,

    const BatteryParameters& params,

    const vector<float>& heatProfile)
{
    //--------------------------------------------------
    // Safety check
    //--------------------------------------------------

    if(heatProfile.empty())
    {
        return;
    }


    //--------------------------------------------------
    // CUDA block
    //--------------------------------------------------

    dim3 block(
        TILE,
        TILE
    );


    //--------------------------------------------------
    // CUDA grid
    //--------------------------------------------------

    dim3 grid(
        (N + TILE - 1) / TILE,
        (N + TILE - 1) / TILE
    );


    //--------------------------------------------------
    // Time stepping
    //--------------------------------------------------

    for(int s = 0;
        s < steps;
        s++)
    {
        //------------------------------------------------
        // NASA / experimental heat
        //------------------------------------------------

        float nasaHeat =
            heatProfile[
                s % heatProfile.size()
            ];


        //------------------------------------------------
        // Launch Halo kernel
        //------------------------------------------------

        haloStencil<<<
            grid,
            block
        >>>(
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

            //------------------------------------------------
            // V5 SOC
            //------------------------------------------------

            params.soc,

            //------------------------------------------------
            // V5 SOH
            //------------------------------------------------

            params.soh,

            //------------------------------------------------
            // SOC resistance coefficient
            //------------------------------------------------

            params.socResistanceCoeff,

            //------------------------------------------------
            // SOH resistance coefficient
            //------------------------------------------------

            params.sohResistanceCoeff,

            //------------------------------------------------
            // SOC heat coefficient
            //------------------------------------------------

            params.socHeatCoeff,

            //------------------------------------------------
            // SOH heat coefficient
            //------------------------------------------------

            params.sohHeatCoeff
        );


        //------------------------------------------------
        // Check kernel launch
        //------------------------------------------------

        cudaError_t err =
            cudaGetLastError();

        if(err != cudaSuccess)
        {
            return;
        }


        //------------------------------------------------
        // Ping-pong buffers
        //------------------------------------------------

        swap(
            d_T,
            d_Tnew
        );
    }


    //--------------------------------------------------
    // Synchronize GPU
    //--------------------------------------------------

    cudaDeviceSynchronize();
}