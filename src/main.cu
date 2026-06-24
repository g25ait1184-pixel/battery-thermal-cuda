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

    Config cfg = initConfig();

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
            cfg.current *
            cfg.current *
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
    else if(cfg.execMode == NSIGHT_MODE)
    {
        cout
        << "Execution Mode : NSIGHT"
        << endl;
    }
    else
    {
        cout
        << "Execution Mode : PROFILE"
        << endl;
    }

    //------------------------------------------------
    // CUDA Events
    //------------------------------------------------

    initCudaEvents();

    //------------------------------------------------
    // Results
    //------------------------------------------------

    vector<int> gridSizes;

    vector<Metrics> results;

    //------------------------------------------------
    // Header
    //------------------------------------------------

    if(cfg.execMode == NORMAL_MODE)
    {
        cout
        << "\nGrid,CPU(ms),GPU(ms),Tiled(ms),Halo(ms),"
        << "GPU Speedup,Tiled Speedup,Halo Speedup,"
        << "GPU GFLOPS,Tiled GFLOPS,Halo GFLOPS,"
        << "GPU BW(GB/s),Tiled BW(GB/s),Halo BW(GB/s)"
        << endl;
    }

    //------------------------------------------------
    // Benchmark Loop
    //------------------------------------------------

    for(auto N : cfg.sizes)
    {
        vector<float> T;
        vector<float> Tnew;

        initializeGrid(
            T,
            Tnew,
            N
        );

        printSimulationInfo(
            N,
            heatProfile,
            cfg
        );

        double cpu_ms   = 0.0;
        double gpu_ms   = 0.0;
        double tiled_ms = 0.0;
        double halo_ms  = 0.0;

        //--------------------------------------------
        // NORMAL MODE
        //--------------------------------------------

        if(cfg.execMode == NORMAL_MODE)
        {
            cpu_ms =
                benchmarkCPU(
                    cfg,
                    N,
                    T,
                    Tnew,
                    heatProfile
                );

            gpu_ms =
                benchmarkGPU(
                    cfg,
                    N,
                    T,
                    heatProfile
                );

            tiled_ms =
                benchmarkTiled(
                    cfg,
                    N,
                    T,
                    heatProfile
                );

            halo_ms =
                benchmarkHalo(
                    cfg,
                    N,
                    T,
                    heatProfile
                );
        }

        //--------------------------------------------
        // NSIGHT MODE
        //--------------------------------------------

        else if(cfg.execMode == NSIGHT_MODE)
        {
            gpu_ms =
                benchmarkGPU(
                    cfg,
                    N,
                    T,
                    heatProfile
                );

            tiled_ms =
                benchmarkTiled(
                    cfg,
                    N,
                    T,
                    heatProfile
                );

            halo_ms =
                benchmarkHalo(
                    cfg,
                    N,
                    T,
                    heatProfile
                );
        }

        //--------------------------------------------
        // PROFILE MODE
        //--------------------------------------------

        else
        {
            if(cfg.kernelMode == GPU_KERNEL)
            {
                gpu_ms =
                    benchmarkGPU(
                        cfg,
                        N,
                        T,
                        heatProfile
                    );
            }
            else if(cfg.kernelMode == TILED_KERNEL)
            {
                tiled_ms =
                    benchmarkTiled(
                        cfg,
                        N,
                        T,
                        heatProfile
                    );
            }
            else
            {
                halo_ms =
                    benchmarkHalo(
                        cfg,
                        N,
                        T,
                        heatProfile
                    );
            }
        }

        //--------------------------------------------
        // Metrics only for NORMAL mode
        //--------------------------------------------

        if(cfg.execMode == NORMAL_MODE)
        {
            Metrics m =
                computeMetrics(
                    N,
                    cfg.steps,
                    cpu_ms,
                    gpu_ms,
                    tiled_ms,
                    halo_ms
                );

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

            gridSizes.push_back(
                N
            );

            results.push_back(
                m
            );
        }
    }

    //------------------------------------------------
    // Save CSV
    //------------------------------------------------

    if(cfg.saveCSV)
    {
        saveResultsCSV(
            gridSizes,
            results
        );
    }

    //------------------------------------------------
    // Thermal Map
    //------------------------------------------------

    if(cfg.savePlots)
    {
        generateThermalMap(
            cfg,
            heatProfile
        );
    }

    //------------------------------------------------
    // Cleanup
    //------------------------------------------------

    cleanupCudaEvents();

    cout
    << "\nSimulation Finished"
    << endl;

    return 0;
}