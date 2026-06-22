#include <iostream>
#include <iomanip>
#include <vector>

#include "config.h"
#include "benchmark.h"
#include "metrics.h"
#include "thermal_map.h"
#include "utils.h"
#include "cuda_events.h"

using namespace std;

int main()
{
    //------------------------------------------------
    // Configuration
    //------------------------------------------------

    Config cfg =
        initConfig();

    //------------------------------------------------
    // Heat Profile
    //------------------------------------------------

    vector<float> heatProfile;

    if(cfg.heatMode == DYNAMIC_Q)
    {
        heatProfile =
            loadHeatProfile(
                cfg.resistance
            );

        if(heatProfile.empty())
        {
            cout
            << "Failed to load heat profile!"
            << endl;

            return -1;
        }

        cout
        << "\nHeat Mode : DYNAMIC Q"
        << endl;

        cout
        << "Heat Profile Size = "
        << heatProfile.size()
        << endl;
    }
    else
    {
        float Q =
            cfg.current
            *
            cfg.current
            *
            cfg.resistance;

        heatProfile.push_back(
            Q
        );

        cout
        << "\nHeat Mode : CONSTANT Q"
        << endl;

        cout
        << "Heat Source Q = "
        << Q
        << endl;
    }

    //------------------------------------------------
    // Execution Mode
    //------------------------------------------------

    if(cfg.execMode == NORMAL_MODE)
    {
        cout
        << "Execution Mode : NORMAL"
        << endl;
    }
    else
    {
        cout
        << "Execution Mode : NSIGHT"
        << endl;
    }

    //------------------------------------------------
    // CUDA Events
    //------------------------------------------------

    initCudaEvents();

    //------------------------------------------------
    // Results
    //------------------------------------------------

    vector<int>
    gridSizes;

    vector<Metrics>
    results;

    //------------------------------------------------
    // Header
    //------------------------------------------------

    cout
    << "\nGrid,CPU(ms),GPU(ms),Tiled(ms),Halo(ms),"
    << "GPU Speedup,Tiled Speedup,Halo Speedup,"
    << "GPU GFLOPS,Tiled GFLOPS,Halo GFLOPS,"
    << "GPU BW(GB/s),Tiled BW(GB/s),Halo BW(GB/s)"
    << endl;

    //------------------------------------------------
    // Benchmark Loop
    //------------------------------------------------

    for(auto N : cfg.sizes)
    {
        //--------------------------------------------
        // Initialize Grid
        //--------------------------------------------

        vector<float>
        T;

        vector<float>
        Tnew;

        initializeGrid(
            T,
            Tnew,
            N
        );

        //--------------------------------------------
        // Simulation Information
        //--------------------------------------------

        printSimulationInfo(
            N,
            heatProfile,
            cfg
        );

        //--------------------------------------------
        // CPU
        //--------------------------------------------

        double cpu_ms =
            benchmarkCPU(
                cfg,
                N,
                T,
                Tnew,
                heatProfile
            );

        //--------------------------------------------
        // GPU
        //--------------------------------------------

        double gpu_ms =
            benchmarkGPU(
                cfg,
                N,
                T,
                heatProfile
            );

        //--------------------------------------------
        // Shared GPU
        //--------------------------------------------

        double tiled_ms =
            benchmarkTiled(
                cfg,
                N,
                T,
                heatProfile
            );

        //--------------------------------------------
        // Halo GPU
        //--------------------------------------------

        double halo_ms =
            benchmarkHalo(
                cfg,
                N,
                T,
                heatProfile
            );

        //--------------------------------------------
        // Metrics
        //--------------------------------------------

        Metrics m =
            computeMetrics(
                N,
                cfg.steps,
                cpu_ms,
                gpu_ms,
                tiled_ms,
                halo_ms
            );

        //--------------------------------------------
        // Print Results
        //--------------------------------------------

        cout
        << fixed
        << setprecision(2)

        << N << ","
        << m.cpu_ms << ","
        << m.gpu_ms << ","
        << m.tiled_ms << ","
        << m.halo_ms << ","

        << m.gpu_speedup << ","
        << m.tiled_speedup << ","
        << m.halo_speedup << ","

        << m.gpu_gflops << ","
        << m.tiled_gflops << ","
        << m.halo_gflops << ","

        << m.gpu_bw << ","
        << m.tiled_bw << ","
        << m.halo_bw

        << endl;

        //--------------------------------------------
        // Store
        //--------------------------------------------

        gridSizes.push_back(
            N
        );

        results.push_back(
            m
        );
    }

    //------------------------------------------------
    // Save CSV
    //------------------------------------------------

    saveResultsCSV(
        gridSizes,
        results
    );

    //------------------------------------------------
    // Thermal Map
    //------------------------------------------------

    generateThermalMap(
        cfg,
        heatProfile
    );

    //------------------------------------------------
    // Cleanup
    //------------------------------------------------

    cleanupCudaEvents();

    cout
    << "\nSimulation Finished"
    << endl;

    return 0;
}