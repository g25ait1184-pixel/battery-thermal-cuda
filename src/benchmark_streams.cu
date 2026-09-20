#include "config.h"
#include "stream_solver.h"
#include "battery_model.h"

#include <cuda_runtime.h>

#include <vector>
#include <cstdio>

using namespace std;


//======================================================
// CUDA Error Checking
//======================================================

static inline bool checkCuda(
    cudaError_t error,
    const char* operation)
{
    if(error != cudaSuccess)
    {
        fprintf(
            stderr,
            "CUDA error during %s: %s\n",
            operation,
            cudaGetErrorString(error)
        );

        return false;
    }

    return true;
}


//======================================================
// Benchmark CUDA Streams
//======================================================
//
// Tests multiple independent V5 thermal simulations:
//
//     1 stream
//     2 streams
//     4 streams
//     8 streams
//
// IMPORTANT:
//
// The physics and numerical parameters remain identical.
// Only the execution concurrency changes.
//
// This makes the experiment suitable for studying:
//
//     stream count
//          vs.
//     execution time
//
//======================================================

double benchmarkStreams(
    Config& cfg,
    int N,
    vector<float>& T,
    vector<float>& heatProfile,
    vector<float>& finalT)
{
    //--------------------------------------------------
    // Validate configuration
    //--------------------------------------------------

    if(cfg.RUNS <= 0)
    {
        return 0.0;
    }

    if(N <= 2)
    {
        return 0.0;
    }

    if(T.empty())
    {
        return 0.0;
    }

    if(heatProfile.empty())
    {
        return 0.0;
    }


    //--------------------------------------------------
    // Battery parameters
    //--------------------------------------------------

    BatteryParameters params;


    //--------------------------------------------------
    // Model version
    //--------------------------------------------------

    params.modelVersion =
        cfg.modelVersion;


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
    // V5 SOC / SOH state
    //--------------------------------------------------

    params.soc =
        cfg.initialSOC;

    params.soh =
        cfg.initialSOH;


    //--------------------------------------------------
    // V5 SOC / SOH resistance parameters
    //--------------------------------------------------

    params.socResistanceCoeff =
        cfg.socResistanceCoeff;

    params.sohResistanceCoeff =
        cfg.sohResistanceCoeff;


    //--------------------------------------------------
    // V5 SOC / SOH heat parameters
    //--------------------------------------------------

    params.socHeatCoeff =
        cfg.socHeatCoeff;

    params.sohHeatCoeff =
        cfg.sohHeatCoeff;


    //--------------------------------------------------
    // Stream configurations
    //--------------------------------------------------
    //
    // We explicitly test:
    //
    // 1
    // 2
    // 4
    // 8
    //
    //--------------------------------------------------

    const int streamCounts[] =
    {
        1,
        2,
        4,
        8
    };

    const int numConfigurations =
        sizeof(streamCounts) /
        sizeof(streamCounts[0]);


    //--------------------------------------------------
    // Print header
    //--------------------------------------------------

    printf(
        "\n"
        "============================================================\n"
        " V5 CUDA Streams Benchmark\n"
        "============================================================\n"
    );

    printf(
        "Grid size      : %d x %d\n",
        N,
        N
    );

    printf(
        "Timesteps      : %d\n",
        cfg.steps
    );

    printf(
        "Model          : V%d\n",
        cfg.modelVersion
    );

    printf(
        "Initial SOC    : %.4f\n",
        params.soc
    );

    printf(
        "Initial SOH    : %.4f\n",
        params.soh
    );

    printf(
        "Benchmark runs : %d\n",
        cfg.RUNS
    );

    printf(
        "============================================================\n"
    );

    printf(
        "%-12s %-18s %-18s %-18s\n",
        "Streams",
        "Avg Time (ms)",
        "Speedup",
        "Efficiency (%)"
    );

    printf(
        "------------------------------------------------------------\n"
    );


    //--------------------------------------------------
    // Baseline runtime
    //--------------------------------------------------

    double baselineTime =
        0.0;


    //--------------------------------------------------
    // Run each stream configuration
    //--------------------------------------------------

    for(int c = 0;
        c < numConfigurations;
        c++)
    {
        int numStreams =
            streamCounts[c];


        //------------------------------------------------
        // Accumulated runtime
        //------------------------------------------------

        double totalTime =
            0.0;


        //------------------------------------------------
        // Keep final result from last run
        //------------------------------------------------

        vector<float> runFinalT;


        //------------------------------------------------
        // Benchmark repetitions
        //------------------------------------------------

        for(int r = 0;
            r < cfg.RUNS;
            r++)
        {
            //------------------------------------------------
            // Run CUDA stream solver
            //------------------------------------------------

            double elapsed =
                runCUDAStreams(
                    numStreams,

                    N,

                    cfg.alpha,

                    cfg.steps,

                    cfg.beta,

                    params,

                    T,

                    heatProfile,

                    runFinalT
                );


            totalTime +=
                elapsed;


            //------------------------------------------------
            // Keep final result from final benchmark run
            //------------------------------------------------

            if(r == cfg.RUNS - 1)
            {
                finalT =
                    runFinalT;
            }
        }


        //------------------------------------------------
        // Average runtime
        //------------------------------------------------

        double averageTime =
            totalTime /
            static_cast<double>(
                cfg.RUNS
            );


        //------------------------------------------------
        // Establish 1-stream baseline
        //------------------------------------------------

        if(numStreams == 1)
        {
            baselineTime =
                averageTime;
        }


        //------------------------------------------------
        // Calculate speedup
        //------------------------------------------------

        double speedup =
            0.0;

        if(averageTime > 0.0 &&
           baselineTime > 0.0)
        {
            speedup =
                baselineTime /
                averageTime;
        }


        //------------------------------------------------
        // Calculate parallel efficiency
        //------------------------------------------------
        //
        // Efficiency =
        //
        // Speedup / Number of Streams * 100
        //
        //------------------------------------------------

        double efficiency =
            0.0;

        if(numStreams > 0)
        {
            efficiency =
                (speedup /
                 static_cast<double>(
                     numStreams
                 )) *
                100.0;
        }


        //------------------------------------------------
        // Print result
        //------------------------------------------------

        printf(
            "%-12d %-18.4f %-18.4f %-18.2f\n",
            numStreams,
            averageTime,
            speedup,
            efficiency
        );
    }


    //--------------------------------------------------
    // Finish table
    //--------------------------------------------------

    printf(
        "============================================================\n"
    );


    return baselineTime;
}