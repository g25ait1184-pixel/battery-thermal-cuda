#include <vector>
#include <cuda_runtime.h>

using namespace std;


#define TILE 16

__global__
void tiledStencil(
    float *T,
    float *Tnew,
    int N,
    float alpha,
    float beta,
    float Q)
{
    __shared__
    float tile[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int j =
        blockIdx.x * TILE + tx;

    int i =
        blockIdx.y * TILE + ty;

    //-----------------------------------
    // Load tile into shared memory
    //-----------------------------------

    if(i < N && j < N)
    {
        tile[ty][tx] =
            T[i*N + j];
    }

    __syncthreads();

    //-----------------------------------
    // Interior points
    //-----------------------------------

    if(
        i > 0 &&
        i < N-1 &&
        j > 0 &&
        j < N-1
    )
    {
        int idx =
            i*N + j;

        float center =
            tile[ty][tx];

        //-----------------------------------
        // Top
        //-----------------------------------

        float top =
            (ty == 0)
            ?
            T[idx-N]
            :
            tile[ty-1][tx];

        //-----------------------------------
        // Bottom
        //-----------------------------------

        float bottom =
            (ty == TILE-1)
            ?
            T[idx+N]
            :
            tile[ty+1][tx];

        //-----------------------------------
        // Left
        //-----------------------------------

        float left =
            (tx == 0)
            ?
            T[idx-1]
            :
            tile[ty][tx-1];

        //-----------------------------------
        // Right
        //-----------------------------------

        float right =
            (tx == TILE-1)
            ?
            T[idx+1]
            :
            tile[ty][tx+1];

        //-----------------------------------
        // Stencil update
        //-----------------------------------

        Tnew[idx]
            =
            center
            +
            alpha *
            (
                top
                +
                bottom
                +
                left
                +
                right
                -
                4.0f*center
            )
            +
            beta*Q;
    }
}

void runTiledGPU(
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

        tiledStencil<<<
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