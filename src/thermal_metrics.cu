#include "thermal_metrics.h"

#include <fstream>
#include <iostream>
#include <limits>
#include <iomanip>
#include "config.h"
using namespace std;


//------------------------------------------------------
// Calculate Thermal Metrics
//------------------------------------------------------

ThermalMetrics calculateThermalMetrics(
    const vector<float>& T,
    int N)
{
    ThermalMetrics m;

    //--------------------------------------------------
    // Safety check
    //--------------------------------------------------

    if(T.empty() || N <= 0)
    {
        m.Tmin   = 0.0f;
        m.Tmax   = 0.0f;
        m.Tavg   = 0.0f;
        m.Tcenter = 0.0f;
        m.deltaT  = 0.0f;

        return m;
    }

    //--------------------------------------------------
    // Minimum / Maximum
    //--------------------------------------------------

    float Tmin =
        numeric_limits<float>::max();

    float Tmax =
        numeric_limits<float>::lowest();

    double sum = 0.0;

    for(size_t i = 0;
        i < T.size();
        i++)
    {
        float temp = T[i];

        if(temp < Tmin)
            Tmin = temp;

        if(temp > Tmax)
            Tmax = temp;

        sum += temp;
    }

    //--------------------------------------------------
    // Average Temperature
    //--------------------------------------------------

    float Tavg =
        static_cast<float>(
            sum / T.size()
        );

    //--------------------------------------------------
    // Center Temperature
    //--------------------------------------------------

    int center =
        N / 2;

    int centerIndex =
        center * N + center;

    float Tcenter = 0.0f;

    if(centerIndex >= 0 &&
       centerIndex < static_cast<int>(T.size()))
    {
        Tcenter =
            T[centerIndex];
    }

    //--------------------------------------------------
    // Temperature Difference
    //--------------------------------------------------

    float deltaT =
        Tmax - Tmin;

    //--------------------------------------------------
    // Store Results
    //--------------------------------------------------

    m.Tmin =
        Tmin;

    m.Tmax =
        Tmax;

    m.Tavg =
        Tavg;

    m.Tcenter =
        Tcenter;

    m.deltaT =
        deltaT;

    return m;
}


//------------------------------------------------------
// Save Thermal Metrics
//------------------------------------------------------

void saveThermalMetricsCSV(
    const vector<int>& gridSizes,
    const vector<ThermalMetrics>& results,
    const string& modelName)
{
    string filename =
        string(RESULT_DIR) +
        "thermal_metrics_" +
        modelName +
        ".csv";

    ofstream fout(filename);

    if(!fout)
    {
        cout
        << "Failed to create "
        << filename
        << endl;

        return;
    }

    //--------------------------------------------------
    // Header
    //--------------------------------------------------

    fout
    << "Grid,"
    << "Tmin(C),"
    << "Tmax(C),"
    << "Tavg(C),"
    << "Tcenter(C),"
    << "DeltaT(C)"
    << "\n";

    //--------------------------------------------------
    // Data
    //--------------------------------------------------

    fout
    << fixed
    << setprecision(4);

    for(size_t i = 0;
        i < gridSizes.size();
        i++)
    {
        const ThermalMetrics& m =
            results[i];

        fout
        << gridSizes[i] << ","
        << m.Tmin << ","
        << m.Tmax << ","
        << m.Tavg << ","
        << m.Tcenter << ","
        << m.deltaT
        << "\n";
    }

    fout.close();

    cout
    << "\n"
    << filename
    << " created"
    << endl;
}