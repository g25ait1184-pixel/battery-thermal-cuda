#include <vector>
#include <cuda_runtime.h>

using namespace std;

#define TILE 16

__global__
void haloStencil(
    float *T,
    float *Tnew,
    int N,
    float alpha,
    float beta,
    float Q)
{
    __shared__
    float tile[TILE+2][TILE+2];

    int tx =
        threadIdx.x;

    int ty =
        threadIdx.y;

    int j =
        blockIdx.x*TILE + tx;

    int i =
        blockIdx.y*TILE + ty;

    //------------------------------------
    // center
    //------------------------------------

    if(i < N &&
       j < N)
    {
        tile[ty+1][tx+1] =
            T[i*N+j];
    }

    //------------------------------------
    // halo
    //------------------------------------

    if(tx==0 && j>0)
        tile[ty+1][0] =
            T[i*N+j-1];

    if(tx==TILE-1 && j<N-1)
        tile[ty+1][TILE+1] =
            T[i*N+j+1];

    if(ty==0 && i>0)
        tile[0][tx+1] =
            T[(i-1)*N+j];

    if(ty==TILE-1 && i<N-1)
        tile[TILE+1][tx+1] =
            T[(i+1)*N+j];

    __syncthreads();

    if(i>0 &&
       i<N-1 &&
       j>0 &&
       j<N-1)
    {
        Tnew[i*N+j]
            =
            tile[ty+1][tx+1]
            +
            alpha *
            (
                tile[ty][tx+1]
                +
                tile[ty+2][tx+1]
                +
                tile[ty+1][tx]
                +
                tile[ty+1][tx+2]
                -
                4.0f*
                tile[ty+1][tx+1]
            )
            +
            beta*Q;
    }
}


void runHaloGPU(
    float *d_T,
    float *d_Tnew,
    int N,
    float alpha,
    int steps,
    float beta,
    const vector<float>& heatProfile)
{
    dim3 block(
        TILE,
        TILE
    );

    dim3 grid(
        (N+TILE-1)/TILE,
        (N+TILE-1)/TILE
    );

    for(int s=0;s<steps;s++)
    {
        float Q =
            heatProfile[
                s %
                heatProfile.size()
            ];

        haloStencil<<<
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