#include <chrono>
#include <vector>
#include <algorithm>

#include "config.h"
#include "benchmark.h"
#include "utils.h"
#include "battery_model.h"

using namespace std;


//==========================================================
// Single CPU Stencil Step
//==========================================================

void cpuStencil(
    float *T,
    float *Tnew,
    int N,
    float alpha,
    float beta,
    float nasaHeat,
    const BatteryParameters& params)
{
    for(int i = 1; i < N - 1; i++)
    {
        for(int j = 1; j < N - 1; j++)
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
                    params.referenceTemperature,
                    params.diffusivityCoeff
                );


            //------------------------------------------------
            // Resistance
            //
            // V1-V4:
            //     R(T)
            //
            // V5:
            //     R(T) * SOC factor * SOH factor
            //
            // The V5 resistance correction is applied
            // exactly once.
            //------------------------------------------------

            float resistance;

            if(params.modelVersion == MODEL_V5)
            {
                resistance =
                    computeV5Resistance(
                        params.baseResistance,
                        center,
                        params.referenceTemperature,
                        params.resistanceCoeff,
                        params.soc,
                        params.soh,
                        params.socResistanceCoeff,
                        params.sohResistanceCoeff
                    );
            }
            else
            {
                resistance =
                    computeResistance(
                        params.baseResistance,
                        center,
                        params.referenceTemperature,
                        params.resistanceCoeff
                    );
            }


            //------------------------------------------------
            // Authoritative heat calculation
            //
            // CPU, GPU, Tiled and Halo all use the same
            // computeTotalHeat() dispatcher.
            //------------------------------------------------

            float heat =
                computeTotalHeat(
                    static_cast<BatteryModelVersion>(
                        params.modelVersion
                    ),

                    nasaHeat,

                    params.current,

                    resistance,

                    center,

                    params.soc,

                    params.entropyCoeff,

                    params.socResistanceCoeff,

                    params.sohResistanceCoeff,

                    params.socHeatCoeff,

                    params.sohHeatCoeff,

                    params.soh
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
}


//==========================================================
// CPU Solver
//==========================================================

void runCPU(
    float *T,
    float *Tnew,
    int N,
    float alpha,
    int steps,
    float beta,
    Config& cfg,
    vector<float>& heatProfile)
{
    //------------------------------------------------------
    // Battery parameters
    //------------------------------------------------------

    BatteryParameters params{};


    //------------------------------------------------------
    // Model version
    //------------------------------------------------------

    params.modelVersion =
        static_cast<int>(
            cfg.modelVersion
        );


    //------------------------------------------------------
    // Electrical parameters
    //------------------------------------------------------

    params.current =
        cfg.current;

    params.baseResistance =
        cfg.resistance;


    //------------------------------------------------------
    // Reference temperature
    //------------------------------------------------------

    params.referenceTemperature =
        cfg.referenceTemperature;


    //------------------------------------------------------
    // Temperature-dependent parameters
    //------------------------------------------------------

    params.resistanceCoeff =
        cfg.resistanceCoeff;

    params.diffusivityCoeff =
        cfg.diffusivityCoeff;


    //------------------------------------------------------
    // Entropy / reversible heat
    //------------------------------------------------------

    params.entropyCoeff =
        cfg.entropyCoeff;


    //------------------------------------------------------
    // V5 SOC / SOH state
    //------------------------------------------------------

    params.soc =
        cfg.initialSOC;

    params.soh =
        cfg.initialSOH;


    //------------------------------------------------------
    // V5 resistance coefficients
    //------------------------------------------------------

    params.socResistanceCoeff =
        cfg.socResistanceCoeff;

    params.sohResistanceCoeff =
        cfg.sohResistanceCoeff;


    //------------------------------------------------------
    // V5 heat coefficients
    //------------------------------------------------------

    params.socHeatCoeff =
        cfg.socHeatCoeff;

    params.sohHeatCoeff =
        cfg.sohHeatCoeff;


    //------------------------------------------------------
    // Validate heat profile
    //------------------------------------------------------

    if(heatProfile.empty())
    {
        return;
    }


    //------------------------------------------------------
    // Time stepping
    //------------------------------------------------------

    for(int s = 0;
        s < steps;
        s++)
    {
        //--------------------------------------------------
        // NASA / experimental heat
        //--------------------------------------------------

        float nasaHeat =
            heatProfile[
                s % heatProfile.size()
            ];


        //--------------------------------------------------
        // One stencil step
        //--------------------------------------------------

        cpuStencil(
            T,
            Tnew,
            N,
            alpha,
            beta,
            nasaHeat,
            params
        );


        //--------------------------------------------------
        // Ping-pong buffers
        //--------------------------------------------------

        swap(
            T,
            Tnew
        );
    }
}


//==========================================================
// CPU Benchmark
//==========================================================

double benchmarkCPU(
    Config& cfg,
    int N,
    vector<float>& T,
    vector<float>& Tnew,
    vector<float>& heatProfile,
    vector<float>& finalT)
{
    //------------------------------------------------------
    // Validate benchmark configuration
    //------------------------------------------------------

    if(cfg.RUNS <= 0)
    {
        return 0.0;
    }


    if(heatProfile.empty())
    {
        return 0.0;
    }


    //------------------------------------------------------
    // Total runtime
    //------------------------------------------------------

    double cpu_total =
        0.0;


    //------------------------------------------------------
    // Clear final result
    //------------------------------------------------------

    finalT.clear();


    //------------------------------------------------------
    // Benchmark runs
    //------------------------------------------------------

    for(int r = 0;
        r < cfg.RUNS;
        r++)
    {
        //--------------------------------------------------
        // Local copies
        //--------------------------------------------------

        vector<float> cpuT =
            T;

        vector<float> cpuTnew =
            Tnew;


        //--------------------------------------------------
        // Start timer
        //--------------------------------------------------

        auto start =
            chrono::high_resolution_clock::now();


        //--------------------------------------------------
        // Run CPU solver
        //--------------------------------------------------

        runCPU(
            cpuT.data(),
            cpuTnew.data(),
            N,
            cfg.alpha,
            cfg.steps,
            cfg.beta,
            cfg,
            heatProfile
        );


        //--------------------------------------------------
        // Stop timer
        //--------------------------------------------------

        auto stop =
            chrono::high_resolution_clock::now();


        //--------------------------------------------------
        // Runtime
        //--------------------------------------------------

        cpu_total +=
            chrono::duration<
                double,
                milli
            >(
                stop - start
            ).count();


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
            // Ping-pong buffer handling
            //
            // runCPU() swaps its local pointers.
            //
            // Caller pointers remain unchanged.
            //
            // Even steps -> cpuT
            // Odd  steps -> cpuTnew
            //------------------------------------------------

            if(cfg.steps % 2 == 0)
            {
                finalT =
                    cpuT;
            }
            else
            {
                finalT =
                    cpuTnew;
            }
        }
    }


    //------------------------------------------------------
    // Average runtime
    //------------------------------------------------------

    return
        cpu_total /
        static_cast<double>(
            cfg.RUNS
        );
}