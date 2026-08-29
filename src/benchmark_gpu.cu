#include "config.h"
#include "cuda_events.h"
#include "kernels.h"
#include "battery_model.h"

#include <cuda_runtime.h>
#include <vector>

using namespace std;


//------------------------------------------------------
// GPU Benchmark
//------------------------------------------------------

double benchmarkGPU(
    Config& cfg,
    int N,
    vector<float>& T,
    vector<float>& heatProfile,
    vector<float>& finalT)
{
    //--------------------------------------------------
    // Memory size
    //--------------------------------------------------

    size_t bytes =
        static_cast<size_t>(N) *
        static_cast<size_t>(N) *
        sizeof(float);


    double gpu_total = 0.0;


    //--------------------------------------------------
    // Clear final output
    //--------------------------------------------------

    finalT.clear();


    //--------------------------------------------------
    // Battery Parameters
    //--------------------------------------------------

    BatteryParameters params{};


    //--------------------------------------------------
    // Model version
    //--------------------------------------------------

    params.modelVersion =
        static_cast<int>(cfg.modelVersion);


    //--------------------------------------------------
    // Electrical parameters
    //--------------------------------------------------

    params.current =
        cfg.current;

    params.baseResistance =
        cfg.resistance;


    //--------------------------------------------------
    // Reference temperature
    //--------------------------------------------------

    params.referenceTemperature =
        cfg.referenceTemperature;


    //--------------------------------------------------
    // Temperature-dependent parameters
    //--------------------------------------------------

    params.resistanceCoeff =
        cfg.resistanceCoeff;

    params.diffusivityCoeff =
        cfg.diffusivityCoeff;


    //--------------------------------------------------
    // Entropy / reversible heat
    //--------------------------------------------------

    params.entropyCoeff =
        cfg.entropyCoeff;


    //--------------------------------------------------
    // V5 - SOC / SOH state
    //--------------------------------------------------

    params.soc =
        cfg.initialSOC;

    params.soh =
        cfg.initialSOH;


    //--------------------------------------------------
    // V5 - SOC/SOH resistance coefficients
    //--------------------------------------------------

    params.socResistanceCoeff =
        cfg.socResistanceCoeff;

    params.sohResistanceCoeff =
        cfg.sohResistanceCoeff;


    //--------------------------------------------------
    // V5 - SOC/SOH heat coefficients
    //--------------------------------------------------

    params.socHeatCoeff =
        cfg.socHeatCoeff;

    params.sohHeatCoeff =
        cfg.sohHeatCoeff;


    //--------------------------------------------------
    // Benchmark runs
    //--------------------------------------------------

    if(cfg.RUNS <= 0)
    {
        return 0.0;
    }


    for(int r = 0;
        r < cfg.RUNS;
        r++)
    {
        //--------------------------------------------------
        // Device pointers
        //--------------------------------------------------

        float* d_T =
            nullptr;

        float* d_Tnew =
            nullptr;


        //--------------------------------------------------
        // Allocate GPU memory
        //--------------------------------------------------

        cudaMalloc(
            &d_T,
            bytes
        );

        cudaMalloc(
            &d_Tnew,
            bytes
        );


        //--------------------------------------------------
        // Copy initial temperature field
        //--------------------------------------------------

        cudaMemcpy(
            d_T,
            T.data(),
            bytes,
            cudaMemcpyHostToDevice
        );


        //--------------------------------------------------
        // Initialize second buffer
        //--------------------------------------------------

        cudaMemcpy(
            d_Tnew,
            T.data(),
            bytes,
            cudaMemcpyHostToDevice
        );


        //--------------------------------------------------
        // Start CUDA timing
        //--------------------------------------------------

        cudaEventRecord(
            startEvent
        );


        //--------------------------------------------------
        // Run GPU solver
        //--------------------------------------------------

        runGPU(
            d_T,
            d_Tnew,
            N,
            cfg.alpha,
            cfg.steps,
            cfg.beta,
            params,
            heatProfile
        );


        //--------------------------------------------------
        // Stop CUDA timing
        //--------------------------------------------------

        cudaEventRecord(
            stopEvent
        );

        cudaEventSynchronize(
            stopEvent
        );


        //--------------------------------------------------
        // Measure GPU execution time
        //--------------------------------------------------

        float ms = 0.0f;

        cudaEventElapsedTime(
            &ms,
            startEvent,
            stopEvent
        );

        gpu_total +=
            static_cast<double>(ms);


        //--------------------------------------------------
        // Capture final temperature field
        //--------------------------------------------------

        if(r == cfg.RUNS - 1)
        {
            finalT.resize(
                static_cast<size_t>(N) *
                static_cast<size_t>(N)
            );


            //------------------------------------------------
            // runGPU() swaps the local device pointers
            // after every timestep.
            //
            // Even steps -> final data in original d_T
            // Odd  steps -> final data in original d_Tnew
            //------------------------------------------------

            float* finalDeviceBuffer =
                nullptr;


            if(cfg.steps % 2 == 0)
            {
                finalDeviceBuffer =
                    d_T;
            }
            else
            {
                finalDeviceBuffer =
                    d_Tnew;
            }


            //------------------------------------------------
            // Copy final temperature field
            //------------------------------------------------

            cudaMemcpy(
                finalT.data(),
                finalDeviceBuffer,
                bytes,
                cudaMemcpyDeviceToHost
            );
        }


        //--------------------------------------------------
        // Free GPU memory
        //--------------------------------------------------

        cudaFree(
            d_T
        );

        cudaFree(
            d_Tnew
        );
    }


    //--------------------------------------------------
    // Average GPU execution time
    //--------------------------------------------------

    return
        gpu_total /
        static_cast<double>(cfg.RUNS);
}