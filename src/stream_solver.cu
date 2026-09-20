#include "stream_solver.h"
#include "kernels.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace std;


//======================================================
// CUDA Error Checking
//======================================================

static inline void checkCuda(
    cudaError_t error,
    const char* operation)
{
    if(error != cudaSuccess)
    {
        fprintf(
            stderr,
            "CUDA error during %s: %s\n",
            operation,
            cudaGetErrorString(error)
        );

        exit(EXIT_FAILURE);
    }
}


//======================================================
// Run One Thermal Simulation on One CUDA Stream
//======================================================
//
// The timesteps within one simulation remain dependent:
//
// T0 -> T1 -> T2 -> ... -> Tn
//
// Therefore they remain sequential on one stream.
//
// Parallelism is obtained by running independent
// simulations on different streams.
//======================================================

void runStreamCase(
    float* d_T,
    float* d_Tnew,
    int N,
    float alpha,
    int steps,
    float beta,
    const BatteryParameters& params,
    const vector<float>& heatProfile,
    cudaStream_t stream)
{
    //--------------------------------------------------
    // Validate heat profile
    //--------------------------------------------------

    if(heatProfile.empty())
    {
        return;
    }


    //--------------------------------------------------
    // CUDA block configuration
    //--------------------------------------------------

    dim3 block(
        16,
        16
    );


    //--------------------------------------------------
    // CUDA grid configuration
    //--------------------------------------------------

    dim3 grid(
        (N + block.x - 1) / block.x,
        (N + block.y - 1) / block.y
    );


    //--------------------------------------------------
    // Time stepping
    //--------------------------------------------------

    for(int s = 0;
        s < steps;
        s++)
    {
        //------------------------------------------------
        // Heat input
        //------------------------------------------------

        float nasaHeat =
            heatProfile[
                s % heatProfile.size()
            ];


        //------------------------------------------------
        // Launch existing V5 GPU stencil
        //
        // IMPORTANT:
        // This is the same gpuStencil() used by the
        // normal GPU benchmark.
        //------------------------------------------------

        gpuStencil<<<
            grid,
            block,
            0,
            stream
        >>>(
            d_T,
            d_Tnew,

            N,

            alpha,
            beta,

            nasaHeat,

            params.modelVersion,

            params.current,

            params.baseResistance,

            params.referenceTemperature,

            params.resistanceCoeff,

            params.diffusivityCoeff,

            params.entropyCoeff,

            //------------------------------------------------
            // V5 SOC / SOH
            //------------------------------------------------

            params.soc,
            params.soh,

            params.socResistanceCoeff,
            params.sohResistanceCoeff,

            params.socHeatCoeff,
            params.sohHeatCoeff
        );


        //------------------------------------------------
        // Check launch
        //------------------------------------------------

        checkCuda(
            cudaGetLastError(),
            "gpuStencil stream launch"
        );


        //------------------------------------------------
        // Ping-pong buffers
        //------------------------------------------------

        swap(
            d_T,
            d_Tnew
        );
    }
}


//======================================================
// CUDA Streams Solver
//======================================================
//
// Each stream represents an independent thermal case.
//
// Example:
//
//   Stream 0 -> Simulation 0
//   Stream 1 -> Simulation 1
//   Stream 2 -> Simulation 2
//   Stream 3 -> Simulation 3
//
// All simulations use the SAME initial temperature
// field and SAME V5 model parameters.
//
// This makes the stream experiment controlled and
// reproducible.
//======================================================

double runCUDAStreams(
    int numStreams,
    int N,
    float alpha,
    int steps,
    float beta,
    const BatteryParameters& params,
    const vector<float>& initialT,
    const vector<float>& heatProfile,
    vector<float>& finalT)
{
    //--------------------------------------------------
    // Validate parameters
    //--------------------------------------------------

    if(numStreams <= 0)
    {
        return 0.0;
    }


    if(N <= 2)
    {
        return 0.0;
    }


    if(steps <= 0)
    {
        return 0.0;
    }


    if(heatProfile.empty())
    {
        return 0.0;
    }


    //--------------------------------------------------
    // Grid size
    //--------------------------------------------------

    size_t elements =
        static_cast<size_t>(N) *
        static_cast<size_t>(N);


    //--------------------------------------------------
    // Verify initial temperature field
    //--------------------------------------------------

    if(initialT.size() != elements)
    {
        fprintf(
            stderr,
            "CUDA Streams error: initialT size "
            "does not equal N*N.\n"
        );

        return 0.0;
    }


    //--------------------------------------------------
    // Device memory size
    //--------------------------------------------------

    size_t bytes =
        elements *
        sizeof(float);


    //--------------------------------------------------
    // Create CUDA streams
    //--------------------------------------------------

    vector<cudaStream_t> streams(
        numStreams
    );


    for(int s = 0;
        s < numStreams;
        s++)
    {
        checkCuda(
            cudaStreamCreate(
                &streams[s]
            ),
            "cudaStreamCreate"
        );
    }


    //--------------------------------------------------
    // Allocate two buffers per stream
    //--------------------------------------------------

    vector<float*> d_T(
        numStreams,
        nullptr
    );

    vector<float*> d_Tnew(
        numStreams,
        nullptr
    );


    //--------------------------------------------------
    // Allocate device memory
    //--------------------------------------------------

    for(int s = 0;
        s < numStreams;
        s++)
    {
        checkCuda(
            cudaMalloc(
                &d_T[s],
                bytes
            ),
            "cudaMalloc d_T"
        );


        checkCuda(
            cudaMalloc(
                &d_Tnew[s],
                bytes
            ),
            "cudaMalloc d_Tnew"
        );
    }


    //--------------------------------------------------
    // Copy SAME initial field to every stream
    //--------------------------------------------------
    //
    // The H2D initialization is deliberately excluded
    // from the solver timing.
    //
    //--------------------------------------------------

    for(int s = 0;
        s < numStreams;
        s++)
    {
        checkCuda(
            cudaMemcpyAsync(
                d_T[s],
                initialT.data(),
                bytes,
                cudaMemcpyHostToDevice,
                streams[s]
            ),
            "cudaMemcpyAsync initial d_T"
        );


        checkCuda(
            cudaMemcpyAsync(
                d_Tnew[s],
                initialT.data(),
                bytes,
                cudaMemcpyHostToDevice,
                streams[s]
            ),
            "cudaMemcpyAsync initial d_Tnew"
        );
    }


    //--------------------------------------------------
    // Wait for initialization to finish
    //--------------------------------------------------

    for(int s = 0;
        s < numStreams;
        s++)
    {
        checkCuda(
            cudaStreamSynchronize(
                streams[s]
            ),
            "cudaStreamSynchronize initialization"
        );
    }


    //--------------------------------------------------
    // Start CPU wall-clock timer
    //
    // The timer surrounds the COMPLETE multi-stream
    // workload.
    //
    // This avoids the problem of using a single CUDA
    // event across multiple independent streams.
    //--------------------------------------------------

    auto start =
        chrono::high_resolution_clock::now();


    //--------------------------------------------------
    // Launch independent simulations
    //--------------------------------------------------

    for(int s = 0;
        s < numStreams;
        s++)
    {
        runStreamCase(
            d_T[s],
            d_Tnew[s],

            N,

            alpha,
            steps,
            beta,

            params,

            heatProfile,

            streams[s]
        );
    }


    //--------------------------------------------------
    // Synchronize ALL streams
    //
    // The workload is complete only when every stream
    // has finished.
    //--------------------------------------------------

    for(int s = 0;
        s < numStreams;
        s++)
    {
        checkCuda(
            cudaStreamSynchronize(
                streams[s]
            ),
            "cudaStreamSynchronize solver"
        );
    }


    //--------------------------------------------------
    // Stop timer
    //--------------------------------------------------

    auto stop =
        chrono::high_resolution_clock::now();


    //--------------------------------------------------
    // Calculate elapsed time
    //--------------------------------------------------

    double elapsedMs =
        chrono::duration<double, milli>(
            stop - start
        ).count();


    //--------------------------------------------------
    // Determine final buffer for stream 0
    //--------------------------------------------------
    //
    // runStreamCase() performs one swap after every
    // timestep.
    //
    // EVEN steps -> d_T
    // ODD  steps -> d_Tnew
    //
    //--------------------------------------------------

    float* finalDeviceBuffer =
        nullptr;


    if(steps % 2 == 0)
    {
        finalDeviceBuffer =
            d_T[0];
    }
    else
    {
        finalDeviceBuffer =
            d_Tnew[0];
    }


    //--------------------------------------------------
    // Copy final result from stream 0
    //
    // Stream 0 is the reference simulation.
    //--------------------------------------------------

    finalT.resize(
        elements
    );


    checkCuda(
        cudaMemcpy(
            finalT.data(),
            finalDeviceBuffer,
            bytes,
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy final temperature"
    );


    //--------------------------------------------------
    // Cleanup device memory
    //--------------------------------------------------

    for(int s = 0;
        s < numStreams;
        s++)
    {
        if(d_T[s] != nullptr)
        {
            checkCuda(
                cudaFree(
                    d_T[s]
                ),
                "cudaFree d_T"
            );
        }


        if(d_Tnew[s] != nullptr)
        {
            checkCuda(
                cudaFree(
                    d_Tnew[s]
                ),
                "cudaFree d_Tnew"
            );
        }
    }


    //--------------------------------------------------
    // Destroy CUDA streams
    //--------------------------------------------------

    for(int s = 0;
        s < numStreams;
        s++)
    {
        checkCuda(
            cudaStreamDestroy(
                streams[s]
            ),
            "cudaStreamDestroy"
        );
    }


    //--------------------------------------------------
    // Return total wall-clock execution time
    //--------------------------------------------------

    return elapsedMs;
}