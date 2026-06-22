#include "config.h"
#include "cuda_events.h"
#include "kernels.h"

double benchmarkGPU(
    Config& cfg,
    int N,
    vector<float>& T,
    vector<float>& heatProfile)
{
    size_t bytes =
        N * N * sizeof(float);

    float gpu_total = 0.0f;

    for(int r=0;
        r<cfg.RUNS;
        r++)
    {
        float *d_T;
        float *d_Tnew;

        cudaMalloc(
            &d_T,
            bytes
        );

        cudaMalloc(
            &d_Tnew,
            bytes
        );

        cudaMemcpy(
            d_T,
            T.data(),
            bytes,
            cudaMemcpyHostToDevice
        );

        cudaEventRecord(
            startEvent
        );

        runGPU(
            d_T,
            d_Tnew,
            N,
            cfg.alpha,
            cfg.steps,
            cfg.beta,
            heatProfile
        );

        cudaEventRecord(
            stopEvent
        );

        cudaEventSynchronize(
            stopEvent
        );

        float ms;

        cudaEventElapsedTime(
            &ms,
            startEvent,
            stopEvent
        );

        gpu_total += ms;

        cudaFree(
            d_T
        );

        cudaFree(
            d_Tnew
        );
    }

    return
        gpu_total /
        cfg.RUNS;
}