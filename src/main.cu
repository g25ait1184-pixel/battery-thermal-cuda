#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

#include "config.h"
#include "benchmark.h"
#include "metrics.h"
#include "thermal_metrics.h"
#include "thermal_map.h"
#include "utils.h"
#include "cuda_events.h"
#include "battery_model.h"
#include "validation_metrics.h"

using namespace std;


//======================================================
// MAIN
//======================================================

int main()
{
    //--------------------------------------------------
    // Configuration
    //--------------------------------------------------

    Config cfg = initConfig();


    //--------------------------------------------------
    // Battery Models
    //
    // V1 : NASA heat
    // V2 : NASA + Joule heating
    // V3 : Temperature-dependent resistance
    // V4 : Temperature-dependent resistance + entropy
    // V5 : SOC/SOH-aware electro-thermal model
    //--------------------------------------------------

    BatteryModelVersion models[] =
    {
        MODEL_V1,
        MODEL_V2,
        MODEL_V3,
        MODEL_V4,
        MODEL_V5
    };


    const char* modelNames[] =
    {
        "MODEL_V1",
        "MODEL_V2",
        "MODEL_V3",
        "MODEL_V4",
        "MODEL_V5"
    };


    const int NUM_MODELS = 5;


    //--------------------------------------------------
    // Print V5 parameters
    //--------------------------------------------------

    cout << "\n=====================================\n";
    cout << "Battery Electro-Thermal Simulation";
    cout << "\n=====================================\n";

    cout << "\nSOC = "
         << cfg.soc
         << endl;

    cout << "SOH = "
         << cfg.soh
         << endl;

    cout << "SOC Resistance Coeff = "
         << cfg.socResistanceCoeff
         << endl;

    cout << "SOH Resistance Coeff = "
         << cfg.sohResistanceCoeff
         << endl;

    cout << "SOC Heat Coeff = "
         << cfg.socHeatCoeff
         << endl;

    cout << "SOH Heat Coeff = "
         << cfg.sohHeatCoeff
         << endl;


    //--------------------------------------------------
    // Run V1 -> V5
    //--------------------------------------------------

    for(int modelIndex = 0;
        modelIndex < NUM_MODELS;
        modelIndex++)
    {
        //--------------------------------------------------
        // Set current battery model
        //--------------------------------------------------

        cfg.modelVersion =
            models[modelIndex];


        cout
            << "\n=====================================\n";

        cout
            << "Running "
            << modelNames[modelIndex]
            << endl;

        cout
            << "=====================================\n";


        //--------------------------------------------------
        // Print model description
        //--------------------------------------------------

        switch(cfg.modelVersion)
        {
            case MODEL_V1:

                cout
                    << "NASA heat only"
                    << endl;

                break;


            case MODEL_V2:

                cout
                    << "NASA heat + Joule heating"
                    << endl;

                break;


            case MODEL_V3:

                cout
                    << "NASA heat + temperature-dependent resistance"
                    << endl;

                break;


            case MODEL_V4:

                cout
                    << "NASA heat + temperature-dependent resistance"
                    << " + entropy heat"
                    << endl;

                break;


            case MODEL_V5:

                cout
                    << "SOC/SOH-aware electro-thermal model"
                    << endl;

                break;


            default:

                cout
                    << "Unknown model"
                    << endl;

                break;
        }


        //--------------------------------------------------
        // Load Heat Profile
        //--------------------------------------------------

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


            heatProfile.push_back(Q);


            cout
                << "\nHeat Mode : CONSTANT Q"
                << endl;

            cout
                << "Heat Source Q = "
                << Q
                << endl;
        }


        //--------------------------------------------------
        // Execution Mode
        //--------------------------------------------------

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


        //--------------------------------------------------
        // CUDA Events
        //--------------------------------------------------

        initCudaEvents();


        //--------------------------------------------------
        // Performance Results
        //--------------------------------------------------

        vector<int> gridSizes;

        vector<Metrics>
            performanceResults;


        //--------------------------------------------------
        // Thermal Results
        //--------------------------------------------------

        vector<ThermalMetrics>
            cpuThermalResults;

        vector<ThermalMetrics>
            gpuThermalResults;

        vector<ThermalMetrics>
            tiledThermalResults;

        vector<ThermalMetrics>
            haloThermalResults;


        //--------------------------------------------------
        // CPU vs GPU Validation Results
        //--------------------------------------------------

        vector<ValidationMetrics>
            validationResults;


        vector<string>
            validationModelNames;


        //--------------------------------------------------
        // Header
        //--------------------------------------------------

        if(cfg.execMode == NORMAL_MODE)
        {
            cout
                << "\n"
                << "Grid,"
                << "CPU(ms),"
                << "GPU(ms),"
                << "Tiled(ms),"
                << "Halo(ms),"
                << "GPU Speedup,"
                << "Tiled Speedup,"
                << "Halo Speedup,"
                << "GPU GFLOPS,"
                << "Tiled GFLOPS,"
                << "Halo GFLOPS,"
                << "GPU BW(GB/s),"
                << "Tiled BW(GB/s),"
                << "Halo BW(GB/s),"
                << "ArithmeticIntensity,"
                << "Updates/sec,"
                << "Latency(ms)"
                << endl;
        }


        //--------------------------------------------------
        // Grid Loop
        //--------------------------------------------------

        for(auto N : cfg.sizes)
        {
            //--------------------------------------------------
            // Initialize temperature grids
            //--------------------------------------------------

            vector<float> T;

            vector<float> Tnew;

            initializeGrid(
                T,
                Tnew,
                N
            );


            //--------------------------------------------------
            // Print simulation information
            //--------------------------------------------------

            printSimulationInfo(
                N,
                heatProfile,
                cfg
            );


            //--------------------------------------------------
            // Runtime variables
            //--------------------------------------------------

            double cpu_ms =
                0.0;

            double gpu_ms =
                0.0;

            double tiled_ms =
                0.0;

            double halo_ms =
                0.0;


            //--------------------------------------------------
            // Final temperature fields
            //--------------------------------------------------

            vector<float>
                finalT_CPU;

            vector<float>
                finalT_GPU;

            vector<float>
                finalT_Tiled;

            vector<float>
                finalT_Halo;


            //==================================================
            // NORMAL MODE
            //==================================================

            if(cfg.execMode == NORMAL_MODE)
            {
                //--------------------------------------------------
                // CPU
                //--------------------------------------------------

                cpu_ms =
                    benchmarkCPU(
                        cfg,
                        N,
                        T,
                        Tnew,
                        heatProfile,
                        finalT_CPU
                    );


                //--------------------------------------------------
                // GPU
                //--------------------------------------------------

                gpu_ms =
                    benchmarkGPU(
                        cfg,
                        N,
                        T,
                        heatProfile,
                        finalT_GPU
                    );


                //--------------------------------------------------
                // Tiled GPU
                //--------------------------------------------------

                tiled_ms =
                    benchmarkTiled(
                        cfg,
                        N,
                        T,
                        heatProfile,
                        finalT_Tiled
                    );


                //--------------------------------------------------
                // Halo GPU
                //--------------------------------------------------

                halo_ms =
                    benchmarkHalo(
                        cfg,
                        N,
                        T,
                        heatProfile,
                        finalT_Halo
                    );
            }


            //==================================================
            // NSIGHT MODE
            //==================================================

            else if(cfg.execMode == NSIGHT_MODE)
            {
                //--------------------------------------------------
                // GPU
                //--------------------------------------------------

                gpu_ms =
                    benchmarkGPU(
                        cfg,
                        N,
                        T,
                        heatProfile,
                        finalT_GPU
                    );


                //--------------------------------------------------
                // Tiled
                //--------------------------------------------------

                tiled_ms =
                    benchmarkTiled(
                        cfg,
                        N,
                        T,
                        heatProfile,
                        finalT_Tiled
                    );


                //--------------------------------------------------
                // Halo
                //--------------------------------------------------

                halo_ms =
                    benchmarkHalo(
                        cfg,
                        N,
                        T,
                        heatProfile,
                        finalT_Halo
                    );
            }


            //==================================================
            // PROFILE MODE
            //==================================================

            else
            {
                if(cfg.kernelMode == GPU_KERNEL)
                {
                    gpu_ms =
                        benchmarkGPU(
                            cfg,
                            N,
                            T,
                            heatProfile,
                            finalT_GPU
                        );
                }
                else if(cfg.kernelMode == TILED_KERNEL)
                {
                    tiled_ms =
                        benchmarkTiled(
                            cfg,
                            N,
                            T,
                            heatProfile,
                            finalT_Tiled
                        );
                }
                else
                {
                    halo_ms =
                        benchmarkHalo(
                            cfg,
                            N,
                            T,
                            heatProfile,
                            finalT_Halo
                        );
                }
            }


            //==================================================
            // PERFORMANCE METRICS
            //==================================================

            if(cfg.execMode == NORMAL_MODE)
            {
                Metrics m =
                    computeMetrics(
                        cfg.modelVersion,
                        N,
                        cfg.steps,
                        cpu_ms,
                        gpu_ms,
                        tiled_ms,
                        halo_ms
                    );


                //--------------------------------------------------
                // Print performance
                //--------------------------------------------------

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
                    << m.halo_bw << ","

                    << m.arithmetic_intensity << ","
                    << m.updates_per_sec << ","
                    << m.latency_ms

                    << endl;


                //--------------------------------------------------
                // Store performance
                //--------------------------------------------------

                gridSizes.push_back(
                    N
                );

                performanceResults.push_back(
                    m
                );


                //==================================================
                // THERMAL METRICS
                //==================================================

                //--------------------------------------------------
                // CPU
                //--------------------------------------------------

                if(!finalT_CPU.empty())
                {
                    ThermalMetrics cpuThermal =
                        calculateThermalMetrics(
                            finalT_CPU,
                            N
                        );

                    cpuThermalResults.push_back(
                        cpuThermal
                    );
                }


                //--------------------------------------------------
                // GPU
                //--------------------------------------------------

                if(!finalT_GPU.empty())
                {
                    ThermalMetrics gpuThermal =
                        calculateThermalMetrics(
                            finalT_GPU,
                            N
                        );

                    gpuThermalResults.push_back(
                        gpuThermal
                    );
                }


                //--------------------------------------------------
                // Tiled
                //--------------------------------------------------

                if(!finalT_Tiled.empty())
                {
                    ThermalMetrics tiledThermal =
                        calculateThermalMetrics(
                            finalT_Tiled,
                            N
                        );

                    tiledThermalResults.push_back(
                        tiledThermal
                    );
                }


                //--------------------------------------------------
                // Halo
                //--------------------------------------------------

                if(!finalT_Halo.empty())
                {
                    ThermalMetrics haloThermal =
                        calculateThermalMetrics(
                            finalT_Halo,
                            N
                        );

                    haloThermalResults.push_back(
                        haloThermal
                    );
                }


                //==================================================
                // CPU vs GPU NUMERICAL VALIDATION
                //==================================================

                if(!finalT_CPU.empty() &&
                   !finalT_GPU.empty())
                {
                    ValidationMetrics validation =
                        calculateValidationMetrics(
                            finalT_CPU,
                            finalT_GPU
                        );


                    validationResults.push_back(
                        validation
                    );


                    validationModelNames.push_back(
                        string(modelNames[modelIndex])
                    );


                    cout
                        << "\nCPU vs GPU Validation"
                        << endl;

                    cout
                        << "Model             : "
                        << modelNames[modelIndex]
                        << endl;

                    cout
                        << "Grid              : "
                        << N
                        << endl;

                    cout
                        << "RMSE (C)          : "
                        << scientific
                        << validation.rmse
                        << endl;

                    cout
                        << "Max Error (C)     : "
                        << validation.maxAbsoluteError
                        << endl;

                    cout
                        << "Mean Error (C)    : "
                        << validation.meanAbsoluteError
                        << endl;

                    cout
                        << "Relative Error    : "
                        << validation.relativeError
                        << endl;

                    cout
                        << fixed;
                }
            }
        }


        //======================================================
        // SAVE PERFORMANCE CSV
        //======================================================

        if(cfg.saveCSV)
        {
            saveResultsCSV(
                gridSizes,
                performanceResults,
                modelNames[modelIndex]
            );
        }


        //======================================================
        // SAVE THERMAL METRICS
        //======================================================

        if(cfg.saveCSV)
        {
            //--------------------------------------------------
            // CPU
            //--------------------------------------------------

            if(!cpuThermalResults.empty())
            {
                saveThermalMetricsCSV(
                    gridSizes,
                    cpuThermalResults,
                    string(modelNames[modelIndex])
                    + "_CPU"
                );
            }


            //--------------------------------------------------
            // GPU
            //--------------------------------------------------

            if(!gpuThermalResults.empty())
            {
                saveThermalMetricsCSV(
                    gridSizes,
                    gpuThermalResults,
                    string(modelNames[modelIndex])
                    + "_GPU"
                );
            }


            //--------------------------------------------------
            // Tiled
            //--------------------------------------------------

            if(!tiledThermalResults.empty())
            {
                saveThermalMetricsCSV(
                    gridSizes,
                    tiledThermalResults,
                    string(modelNames[modelIndex])
                    + "_Tiled"
                );
            }


            //--------------------------------------------------
            // Halo
            //--------------------------------------------------

            if(!haloThermalResults.empty())
            {
                saveThermalMetricsCSV(
                    gridSizes,
                    haloThermalResults,
                    string(modelNames[modelIndex])
                    + "_Halo"
                );
            }


            //==================================================
            // CPU vs GPU VALIDATION CSV
            //==================================================

            if(!validationResults.empty())
            {
                string validationFilename =
                    string(RESULT_DIR) +
                    "validation_" +
                    modelNames[modelIndex] +
                    ".csv";


                saveValidationMetricsCSV(
                    validationFilename,
                    validationModelNames,
                    gridSizes,
                    validationResults
                );
            }
        }


        //======================================================
        // THERMAL MAP
        //======================================================

        if(cfg.savePlots)
        {
            generateThermalMap(
                cfg,
                heatProfile
            );
        }


        //======================================================
        // CLEANUP
        //======================================================

        cleanupCudaEvents();


        cout
            << "\nFinished "
            << modelNames[modelIndex]
            << endl;
    }


    //==========================================================
    // ALL MODELS COMPLETE
    //==========================================================

    cout
        << "\n=====================================\n";

    cout
        << "All Battery Models Completed"
        << endl;

    cout
        << "V1 -> V5"
        << endl;

    cout
        << "=====================================\n";


    return 0;
}