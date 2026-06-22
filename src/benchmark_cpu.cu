#include <chrono>
#include <vector>
#include <algorithm>

#include "config.h"
#include "benchmark.h"
#include "utils.h"

using namespace std;

//--------------------------------------------------
// Single CPU Stencil Step
//--------------------------------------------------

void cpuStencil(
    float *T,
    float *Tnew,
    int N,
    float alpha,
    float beta,
    float Q)
{
    for(int i=1;i<N-1;i++)
    {
        for(int j=1;j<N-1;j++)
        {
            int idx =
                i*N + j;

            Tnew[idx] =
                T[idx]
                +
                alpha *
                (
                    T[idx-N]
                    +
                    T[idx+N]
                    +
                    T[idx-1]
                    +
                    T[idx+1]
                    -
                    4.0f*T[idx]
                )
                +
                beta * Q;
        }
    }
}

//--------------------------------------------------
// CPU Solver
//--------------------------------------------------

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
    for(int s=0;
        s<steps;
        s++)
    {
        float Q =
            getQ(
                cfg,
                s,
                heatProfile
            );

        cpuStencil(
            T,
            Tnew,
            N,
            alpha,
            beta,
            Q
        );

        swap(
            T,
            Tnew
        );
    }
}

//--------------------------------------------------
// CPU Benchmark
//--------------------------------------------------

double benchmarkCPU(
    Config& cfg,
    int N,
    vector<float>& T,
    vector<float>& Tnew,
    vector<float>& heatProfile)
{
    double cpu_total = 0.0;

    for(int r=0;
        r<cfg.RUNS;
        r++)
    {
        //--------------------------------------
        // Local copies
        //--------------------------------------

        vector<float> cpuT =
            T;

        vector<float> cpuTnew =
            Tnew;

        //--------------------------------------
        // Start timer
        //--------------------------------------

        auto start =
            chrono::high_resolution_clock::now();

        //--------------------------------------
        // Run CPU solver
        //--------------------------------------

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

        //--------------------------------------
        // Stop timer
        //--------------------------------------

        auto stop =
            chrono::high_resolution_clock::now();

        //--------------------------------------
        // Accumulate
        //--------------------------------------

        cpu_total +=
            chrono::duration<double,milli>(
                stop - start
            ).count();
    }

    //--------------------------------------
    // Average runtime
    //--------------------------------------

    return
        cpu_total /
        cfg.RUNS;
}