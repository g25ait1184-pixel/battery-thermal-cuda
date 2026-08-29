#pragma once

#include <vector>
#include <string>

struct ThermalMetrics
{
    float Tmin;
    float Tmax;
    float Tavg;
    float Tcenter;
    float deltaT;
};

// Calculate thermal statistics from final temperature field
ThermalMetrics calculateThermalMetrics(
    const std::vector<float>& T,
    int N
);

// Save V1-V4 thermal results
void saveThermalMetricsCSV(
    const std::vector<int>& gridSizes,
    const std::vector<ThermalMetrics>& results,
    const std::string& modelName
);