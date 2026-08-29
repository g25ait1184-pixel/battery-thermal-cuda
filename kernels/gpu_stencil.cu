#include <vector>
#include <cuda_runtime.h>

#include "battery_model.h"

using namespace std;


//------------------------------------------------------
// GPU Thermal Solver
//------------------------------------------------------

__global__
void gpuStencil(
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
    // Global thread coordinates
    //--------------------------------------------------

    int j =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    int i =
        blockIdx.y * blockDim.y +
        threadIdx.y;


    //--------------------------------------------------
    // Interior cells only
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
            T[idx];


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

                // Entropy heat coefficient
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
        // Laplacian
        //------------------------------------------------

        float laplacian =
            T[idx - N]
            +
            T[idx + N]
            +
            T[idx - 1]
            +
            T[idx + 1]
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


//======================================================
// GPU Solver
//======================================================

void runGPU(
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
        16,
        16
    );


    //--------------------------------------------------
    // CUDA grid configuration
    //--------------------------------------------------

    dim3 grid(
        (N + block.x - 1) / block.x,
        (N + block.y - 1) / block.y
    );


    //--------------------------------------------------
    // Safety check
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
        // NASA / experimental heat
        //------------------------------------------------

        float nasaHeat =
            heatProfile[
                s % heatProfile.size()
            ];


        //------------------------------------------------
        // Launch GPU stencil
        //------------------------------------------------

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