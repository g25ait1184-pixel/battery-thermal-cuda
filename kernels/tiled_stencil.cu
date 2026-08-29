#include <vector>
#include <cuda_runtime.h>

#include "battery_model.h"

using namespace std;

#define TILE 16


//------------------------------------------------------
// Shared Memory Tiled GPU Thermal Solver
//------------------------------------------------------

__global__
void tiledStencil(
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
    // Thread coordinates
    //--------------------------------------------------

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int j = blockIdx.x * TILE + tx;
    int i = blockIdx.y * TILE + ty;


    //--------------------------------------------------
    // Shared memory tile
    //--------------------------------------------------

    __shared__ float tile[TILE][TILE];


    //--------------------------------------------------
    // Load current temperature into shared memory
    //--------------------------------------------------

    if(i < N && j < N)
    {
        tile[ty][tx] = T[i * N + j];
    }


    //--------------------------------------------------
    // Synchronize all threads
    //--------------------------------------------------

    __syncthreads();


    //--------------------------------------------------
    // Process interior cells only
    //--------------------------------------------------

    if(i > 0 &&
       i < N - 1 &&
       j > 0 &&
       j < N - 1)
    {
        int idx = i * N + j;


        //------------------------------------------------
        // Current temperature
        //------------------------------------------------

        float center =
            tile[ty][tx];


        //------------------------------------------------
        // Neighbor temperatures
        //------------------------------------------------

        float top =
            (ty == 0)
            ?
            T[idx - N]
            :
            tile[ty - 1][tx];


        float bottom =
            (ty == TILE - 1)
            ?
            T[idx + N]
            :
            tile[ty + 1][tx];


        float left =
            (tx == 0)
            ?
            T[idx - 1]
            :
            tile[ty][tx - 1];


        float right =
            (tx == TILE - 1)
            ?
            T[idx + 1]
            :
            tile[ty][tx + 1];


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
        // V5 SOC/SOH resistance coupling
        //
        // Lower SOC -> higher resistance
        // Lower SOH -> higher resistance
        //------------------------------------------------

        if(modelVersion == MODEL_V5)
        {
            resistance *=
                (1.0f +
                 socResistanceCoeff *
                 (1.0f - soc));

            resistance *=
                (1.0f +
                 sohResistanceCoeff *
                 (1.0f - soh));
        }


        //------------------------------------------------
        // Total battery heat generation
        //
        // IMPORTANT:
        // Argument order MUST match battery_model.h
        //------------------------------------------------

        float heat =
            computeTotalHeat(
                (BatteryModelVersion)modelVersion,

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

                soh
            );


        //------------------------------------------------
        // 2D thermal Laplacian
        //------------------------------------------------

        float laplacian =
            top +
            bottom +
            left +
            right -
            4.0f * center;


        //------------------------------------------------
        // Temperature update
        //------------------------------------------------

        Tnew[idx] =
            center +
            alphaLocal * laplacian +
            beta * heat;
    }
}


//------------------------------------------------------
// Run Tiled GPU Solver
//------------------------------------------------------

void runTiledGPU(
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
    // CUDA block configuration
    //--------------------------------------------------

    dim3 block(
        TILE,
        TILE
    );


    //--------------------------------------------------
    // CUDA grid configuration
    //--------------------------------------------------

    dim3 grid(
        (N + TILE - 1) / TILE,
        (N + TILE - 1) / TILE
    );


    //--------------------------------------------------
    // Protect against empty heat profile
    //--------------------------------------------------

    if(heatProfile.empty())
    {
        return;
    }


    //--------------------------------------------------
    // Time stepping
    //--------------------------------------------------

    for(int s = 0;
        s < steps;
        s++)
    {
        //------------------------------------------------
        // NASA / experimental heat input
        //------------------------------------------------

        float nasaHeat =
            heatProfile[
                s % heatProfile.size()
            ];


        //------------------------------------------------
        // Launch tiled kernel
        //------------------------------------------------

        tiledStencil<<<
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

            //--------------------------------------------
            // V5 SOC / SOH
            //--------------------------------------------

            params.soc,
            params.soh,

            //--------------------------------------------
            // V5 resistance coefficients
            //--------------------------------------------

            params.socResistanceCoeff,
            params.sohResistanceCoeff,

            //--------------------------------------------
            // V5 heat coefficients
            //--------------------------------------------

            params.socHeatCoeff,
            params.sohHeatCoeff
        );


        //------------------------------------------------
        // Ping-pong buffers
        //------------------------------------------------

        swap(
            d_T,
            d_Tnew
        );
    }


    //--------------------------------------------------
    // Ensure GPU execution is complete
    //--------------------------------------------------

    cudaDeviceSynchronize();
}