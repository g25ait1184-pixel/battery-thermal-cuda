#include <vector>
#include <cuda_runtime.h>

#include "config.h"
#include "cuda_events.h"
#include "kernels.h"
#include "battery_model.h"

using namespace std;


//======================================================
// Tiled / Shared-Memory GPU Benchmark
//======================================================

double benchmarkTiled(
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


    //--------------------------------------------------
    // Total execution time
    //--------------------------------------------------

    double tiled_total = 0.0;


    //--------------------------------------------------
    // Prepare output thermal field
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
    // Validate benchmark configuration
    //--------------------------------------------------

    if(cfg.RUNS <= 0)
    {
        return 0.0;
    }


    //--------------------------------------------------
    // Benchmark runs
    //--------------------------------------------------

    for(int r = 0;
        r < cfg.RUNS;
        r++)
    {
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
        // Initialize second ping-pong buffer
        //--------------------------------------------------

        cudaMemcpy(
            d_Tnew,
            T.data(),
            bytes,
            cudaMemcpyHostToDevice
        );


        //--------------------------------------------------
        // Start CUDA timer
        //--------------------------------------------------

        cudaEventRecord(
            startEvent
        );


        //--------------------------------------------------
        // Run Tiled GPU solver
        //--------------------------------------------------

        runTiledGPU(
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
        // Stop CUDA timer
        //--------------------------------------------------

        cudaEventRecord(
            stopEvent
        );

        cudaEventSynchronize(
            stopEvent
        );


        //--------------------------------------------------
        // Measure elapsed time
        //--------------------------------------------------

        float ms = 0.0f;

        cudaEventElapsedTime(
            &ms,
            startEvent,
            stopEvent
        );

        tiled_total +=
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
            // runTiledGPU() swaps its local pointers.
            //
            // Caller pointers remain unchanged.
            //
            // Even steps -> d_T
            // Odd  steps -> d_Tnew
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
            // Copy final thermal field to host
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
    // Return average execution time
    //--------------------------------------------------

    return
        tiled_total /
        static_cast<double>(cfg.RUNS);
}