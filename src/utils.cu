#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>

#include "config.h"
#include "metrics.h"

using namespace std;


//--------------------------------------------------
// Load Heat Profile
//--------------------------------------------------

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

vector<float> loadHeatProfile(float resistance)
{
    vector<float> heatProfile;

    ifstream fin(
    DATA_DIR "battery_dataset.csv"
);

    if(!fin)
    {
        cout
        << "Cannot open battery_dataset.csv"
        << endl;

        return heatProfile;
    }

    string line;

    // Skip header
    getline(fin, line);

    while(getline(fin, line))
    {
        stringstream ss(line);

        vector<string> cols;

        string value;

        while(getline(ss, value, ','))
        {
            cols.push_back(value);
        }

        if(cols.size() < 2)
            continue;

        float current =
            stof(
                cols[1]      // Current_measured
            );

        float Q =
            current *
            current *
            resistance;

        heatProfile.push_back(
            Q
        );
    }

    fin.close();

    cout
    << "Loaded "
    << heatProfile.size()
    << " heat samples"
    << endl;

    return heatProfile;
}


//--------------------------------------------------
// Initialize Grid
//--------------------------------------------------

void initializeGrid(
    vector<float>& T,
    vector<float>& Tnew,
    int N)
{
    T.assign(
        N*N,
        25.0f
    );

    Tnew.assign(
        N*N,
        25.0f
    );

    int center =
        N/2;

    float peakTemp =
        100.0f;

    float sigma =
        N/12.0f;

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

            float hotspot =
                peakTemp *
                exp(
                    -r2 /
                    (2.0f*sigma*sigma)
                );

            T[i*N+j] =
                max(
                    25.0f,
                    hotspot
                );
        }
    }
}


//--------------------------------------------------
// Print Simulation Info
//--------------------------------------------------

void printSimulationInfo(
    int N,
    vector<float>& heatProfile,
    Config& cfg)
{
    int center =
        N/2;

    float minQ =
        heatProfile[0];

    float maxQ =
        heatProfile[0];

    for(int i=1;i<heatProfile.size();i++)
    {
        if(heatProfile[i] < minQ)
            minQ = heatProfile[i];

        if(heatProfile[i] > maxQ)
            maxQ = heatProfile[i];
    }

    cout
    << "\n=====================================\n";

    cout
    << "Grid Size : "
    << N
    << " x "
    << N
    << endl;

    cout
    << "Center Cell : "
    << center
    << endl;

    cout
    << "Heat Profile Size : "
    << heatProfile.size()
    << endl;

    cout
    << "Resistance (Ohm) : "
    << cfg.resistance
    << endl;

    cout
    << "Heat Source Range : "
    << minQ
    << " to "
    << maxQ
    << endl;

    cout
    << "=====================================\n";
}


//--------------------------------------------------
// Save Results CSV
//--------------------------------------------------

void saveResultsCSV(
    vector<int>& gridSizes,
    vector<Metrics>& results)
{
    ofstream fout(
    RESULT_DIR "results.csv"
);

    fout
    << "Grid,CPU,GPU,Tiled,Halo,"
    << "GPU_Speedup,Tiled_Speedup,Halo_Speedup,"
    << "GPU_GFLOPS,Tiled_GFLOPS,Halo_GFLOPS,"
    << "GPU_BW,Tiled_BW,Halo_BW\n";

    for(int i=0;i<gridSizes.size();i++)
    {
        Metrics m =
            results[i];

        fout
        << gridSizes[i] << ","
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
        << m.halo_bw
        << "\n";
    }

    fout.close();
      cout
    << "\nresults file created"
    << endl;
}

float getQ(
    Config& cfg,
    int step,
    vector<float>& heatProfile)
{
    if(cfg.heatMode ==
       CONSTANT_Q)
    {
        return
            cfg.current
            *
            cfg.current
            *
            cfg.resistance;
    }

    return
        heatProfile[
            step %
            heatProfile.size()
        ];
}