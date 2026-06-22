#include <fstream>
#include <iostream>
#include "thermal_map.h"
#include "kernels.h"

void generateThermalMap(
    Config& cfg,
    vector<float>& heatProfile)
{
    int N =
        cfg.thermalMapSize;

    int steps =
        cfg.thermalMapSteps;

    size_t bytes =
        N*N*sizeof(float);

    vector<float>
    T(
        N*N,
        0.0f
    );

    int center =
        N/2;

    float peakTemp =
        100.0f;

    float sigma =
        20.0f;

    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            float dx =
                i-center;

            float dy =
                j-center;

            float r2 =
                dx*dx +
                dy*dy;

            T[i*N+j] =
                peakTemp *
                exp(
                    -r2 /
                    (2.0f*sigma*sigma)
                );
        }
    }

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

    runHaloGPU(
        d_T,
        d_Tnew,
        N,
        cfg.alpha,
        steps,
        cfg.beta,
        heatProfile
    );

    vector<float>
    result(
        N*N
    );

    cudaMemcpy(
        result.data(),
        d_T,
        bytes,
        cudaMemcpyDeviceToHost
    );

    //-------------------------------------------------
    // thermal map
    //-------------------------------------------------

  ofstream thermal(
    RESULT_DIR "thermal_map.csv"
);

    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            thermal
            << result[i*N+j];

            if(j != N-1)
                thermal << ",";
        }

        thermal << "\n";
    }

    thermal.close();
      cout
    << "\nthermal_map file created"
    << endl;

    //-------------------------------------------------
    // center temperature
    //-------------------------------------------------
ofstream centerFile(
    RESULT_DIR "center_temp.csv"
);

    centerFile
    << "CenterTemperature\n";

    centerFile
    << result[
        center*N +
        center
    ]
    << "\n";

    centerFile.close();
    cout
    << "\ncenter_temp file created"
    << endl;

    cudaFree(
        d_T
    );

    cudaFree(
        d_Tnew
    );
}