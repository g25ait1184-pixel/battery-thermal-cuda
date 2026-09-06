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
#include "stream_solver.h"
#include "generate_v5_dataset.h"


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

    //==================================================
    // V5 DATASET GENERATION
    //==================================================
    // Keep disabled while validating the corrected V5
    // physics, numerical accuracy, CUDA Streams, and
    // Nsight results. Enable only after the baseline
    // is frozen.
    constexpr bool ENABLE_V5_DATASET_GENERATION = true;


    //--------------------------------------------------
    // NSIGHT / CUDA STREAMS EXPERIMENT SETTINGS
    //--------------------------------------------------
    // Use a fixed, reproducible workload for the stream
    // scaling experiment. Only MODEL_V5 is profiled.
    //--------------------------------------------------

    if(cfg.execMode == NSIGHT_MODE)
    {
        cfg.RUNS = 1;
        cfg.steps = 100;

        cfg.sizes.clear();
        cfg.sizes.push_back(2048);

        cfg.saveCSV = false;
        cfg.savePlots = false;
    }


    //--------------------------------------------------
    // Battery Models
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

    cout << "\nInitial SOC = "
         << cfg.initialSOC
         << endl;

    cout << "Initial SOH = "
         << cfg.initialSOH
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
        // Model description
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
                << "Execution Mode : NSIGHT / CUDA STREAMS"
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
        // CPU vs GPU Validation
        //--------------------------------------------------

        vector<ValidationMetrics>
            validationResults;

        vector<string>
            validationModelNames;


        //--------------------------------------------------
        // CUDA Streams Results
        //--------------------------------------------------

        vector<int> streamGridSizes;

        vector<double> stream1Results;
        vector<double> stream2Results;
        vector<double> stream4Results;
        vector<double> stream8Results;


        //--------------------------------------------------
        // Normal-mode CSV header
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

            double cpu_ms = 0.0;
            double gpu_ms = 0.0;
            double tiled_ms = 0.0;
            double halo_ms = 0.0;


            //--------------------------------------------------
            // Final temperature fields
            //--------------------------------------------------

            vector<float> finalT_CPU;
            vector<float> finalT_GPU;
            vector<float> finalT_Tiled;
            vector<float> finalT_Halo;
            vector<float> finalT_Streams;


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
            //
            // Dedicated CUDA Streams profiling mode.
            //
            // Only MODEL_V5 is profiled here.
            //
            // Stream counts:
            //     1
            //     2
            //     4
            //     8
            //
            // This keeps the Nsight timeline clean.
            //
            //==================================================

            else if(cfg.execMode == NSIGHT_MODE)
            {
                //--------------------------------------------------
                // Only run CUDA Streams for V5
                //--------------------------------------------------
                if(cfg.modelVersion == MODEL_V5)
                {
                    cout
                        << "\n"
                        << "-------------------------------------\n";

                    cout
                        << "CUDA Streams Nsight Experiment"
                        << endl;

                    cout
                        << "Model : MODEL_V5"
                        << endl;

                    cout
                        << "Grid  : "
                        << N
                        << " x "
                        << N
                        << endl;

                    cout
                        << "Steps : "
                        << cfg.steps
                        << endl;

                    cout
                        << "-------------------------------------\n";


                    //--------------------------------------------------
                    // Battery Parameters
                    //--------------------------------------------------

                    BatteryParameters streamParams;

                    streamParams.modelVersion =
                        cfg.modelVersion;

                    streamParams.current =
                        cfg.current;

                    streamParams.baseResistance =
                        cfg.resistance;

                    streamParams.referenceTemperature =
                        cfg.referenceTemperature;

                    streamParams.resistanceCoeff =
                        cfg.resistanceCoeff;

                    streamParams.diffusivityCoeff =
                        cfg.diffusivityCoeff;

                    streamParams.entropyCoeff =
                        cfg.entropyCoeff;


                    //--------------------------------------------------
                    // V5 SOC / SOH
                    //--------------------------------------------------

                    streamParams.soc =
                        cfg.initialSOC;

                    streamParams.soh =
                        cfg.initialSOH;


                    streamParams.socResistanceCoeff =
                        cfg.socResistanceCoeff;

                    streamParams.sohResistanceCoeff =
                        cfg.sohResistanceCoeff;

                    streamParams.socHeatCoeff =
                        cfg.socHeatCoeff;

                    streamParams.sohHeatCoeff =
                        cfg.sohHeatCoeff;


                    //--------------------------------------------------
                    // Stream counts
                    //--------------------------------------------------

                    const int streamCounts[] =
                    {
                        1,
                        2,
                        4,
                        8
                    };


                    double streamTimes[4] =
                    {
                        0.0,
                        0.0,
                        0.0,
                        0.0
                    };


                    //--------------------------------------------------
                    // Run stream experiments
                    //--------------------------------------------------

                    for(int si = 0;
                        si < 4;
                        si++)
                    {
                        int numStreams =
                            streamCounts[si];

                        double totalStreamTime =
                            0.0;


                        //--------------------------------------------------
                        // Benchmark repetitions
                        //
                        // In NSIGHT mode cfg.RUNS is normally 1.
                        //--------------------------------------------------

                        for(int r = 0;
                            r < cfg.RUNS;
                            r++)
                        {
                            vector<float>
                                streamFinalT;


                            double elapsed =
                                runCUDAStreams(
                                    numStreams,
                                    N,
                                    cfg.alpha,
                                    cfg.steps,
                                    cfg.beta,
                                    streamParams,
                                    T,
                                    heatProfile,
                                    streamFinalT
                                );


                            totalStreamTime +=
                                elapsed;


                            if(r == cfg.RUNS - 1)
                            {
                                finalT_Streams =
                                    streamFinalT;
                            }
                        }


                        //--------------------------------------------------
                        // Average
                        //--------------------------------------------------

                        if(cfg.RUNS > 0)
                        {
                            streamTimes[si] =
                                totalStreamTime /
                                static_cast<double>(
                                    cfg.RUNS
                                );
                        }


                        //--------------------------------------------------
                        // Print result
                        //--------------------------------------------------

                        cout
                            << fixed
                            << setprecision(4)
                            << numStreams
                            << " Stream(s) : "
                            << streamTimes[si]
                            << " ms"
                            << endl;
                    }


                    //--------------------------------------------------
                    // Stream scaling
                    //--------------------------------------------------

                    double baseline =
                        streamTimes[0];


                    cout
                        << "\nCUDA Stream Scaling"
                        << endl;

                    cout
                        << "-------------------------------------\n";


                    for(int si = 0;
                        si < 4;
                        si++)
                    {
                        double speedup =
                            0.0;

                        double efficiency =
                            0.0;


                        if(streamTimes[si] > 0.0 &&
                           baseline > 0.0)
                        {
                            speedup =
                                baseline /
                                streamTimes[si];

                            efficiency =
                                speedup /
                                static_cast<double>(
                                    streamCounts[si]
                                ) *
                                100.0;
                        }


                        cout
                            << streamCounts[si]
                            << " Stream(s)"
                            << " | Time = "
                            << streamTimes[si]
                            << " ms"
                            << " | Speedup = "
                            << speedup
                            << "x"
                            << " | Efficiency = "
                            << efficiency
                            << "%"
                            << endl;
                    }


                    cout
                        << "-------------------------------------\n";


                    //--------------------------------------------------
                    // Store results
                    //--------------------------------------------------

                    streamGridSizes.push_back(
                        N
                    );

                    stream1Results.push_back(
                        streamTimes[0]
                    );

                    stream2Results.push_back(
                        streamTimes[1]
                    );

                    stream4Results.push_back(
                        streamTimes[2]
                    );

                    stream8Results.push_back(
                        streamTimes[3]
                    );
                }
                else
                {
                    //--------------------------------------------------
                    // Do not profile V1-V4 in NSIGHT stream mode
                    //--------------------------------------------------

                    cout
                        << "NSIGHT stream experiment skipped for "
                        << modelNames[modelIndex]
                        << endl;
                }
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
                        string(
                            modelNames[modelIndex]
                        )
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
            if(!cpuThermalResults.empty())
            {
                saveThermalMetricsCSV(
                    gridSizes,
                    cpuThermalResults,
                    string(modelNames[modelIndex])
                    + "_CPU"
                );
            }


            if(!gpuThermalResults.empty())
            {
                saveThermalMetricsCSV(
                    gridSizes,
                    gpuThermalResults,
                    string(modelNames[modelIndex])
                    + "_GPU"
                );
            }


            if(!tiledThermalResults.empty())
            {
                saveThermalMetricsCSV(
                    gridSizes,
                    tiledThermalResults,
                    string(modelNames[modelIndex])
                    + "_Tiled"
                );
            }


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


            //==================================================
            // CUDA STREAMS CSV
            //==================================================

            if(cfg.modelVersion == MODEL_V5 &&
               !streamGridSizes.empty())
            {
                string streamFilename =
                    string(RESULT_DIR) +
                    "cuda_streams_MODEL_V5.csv";


                ofstream streamFile(
                    streamFilename
                );


                if(streamFile.is_open())
                {
                    streamFile
                        << "Grid,"
                        << "Streams_1,"
                        << "Streams_2,"
                        << "Streams_4,"
                        << "Streams_8"
                        << "\n";


                    for(size_t i = 0;
                        i < streamGridSizes.size();
                        i++)
                    {
                        streamFile
                            << streamGridSizes[i]
                            << ","
                            << stream1Results[i]
                            << ","
                            << stream2Results[i]
                            << ","
                            << stream4Results[i]
                            << ","
                            << stream8Results[i]
                            << "\n";
                    }


                    streamFile.close();


                    cout
                        << "\nCUDA Streams results saved to: "
                        << streamFilename
                        << endl;
                }
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
        // V5 DATASET GENERATION
        //======================================================
        //
        // Uses the SAME heat profile and Config already used
        // by the validated V5 simulation. The generator calls
        // the existing runGPU() path and therefore uses the
        // corrected gpuStencil() / V5 physics.
        //
        // Disabled by default so normal, Streams, and Nsight
        // validation runs are not affected.
        //
        //======================================================

        if(cfg.modelVersion == MODEL_V5 &&
           cfg.execMode == NORMAL_MODE &&
           ENABLE_V5_DATASET_GENERATION)
        {
            cout
                << "\n=====================================\n"
                << "Starting V5 Dataset Generation"
                << "\n=====================================\n";

            generateV5Dataset(
                cfg,
                heatProfile
            );

            cout
                << "\nV5 Dataset Generation Completed"
                << endl;
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
        << "V5 CUDA Streams Benchmark Completed"
        << endl;

    cout
        << "=====================================\n";


    return 0;
}