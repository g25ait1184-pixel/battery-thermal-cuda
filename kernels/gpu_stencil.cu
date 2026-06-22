#include <vector>
#include <cuda_runtime.h>

using namespace std;

__global__
void gpuStencil(
    float *T,
    float *Tnew,
    int N,
    float alpha,
    float beta,
    float Q)
{
    int j =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    int i =
        blockIdx.y * blockDim.y +
        threadIdx.y;

    if(i > 0 &&
       i < N-1 &&
       j > 0 &&
       j < N-1)
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


void runGPU(
    float *d_T,
    float *d_Tnew,
    int N,
    float alpha,
    int steps,
    float beta,
    const vector<float>& heatProfile)
{
    dim3 block(
        16,
        16
    );

    dim3 grid(
        (N+15)/16,
        (N+15)/16
    );

    for(int s=0;s<steps;s++)
    {
        float Q =
            heatProfile[
                s %
                heatProfile.size()
            ];

        gpuStencil<<<
            grid,
            block
        >>>(
            d_T,
            d_Tnew,
            N,
            alpha,
            beta,
            Q
        );

        swap(
            d_T,
            d_Tnew
        );
    }

    cudaDeviceSynchronize();
}