#pragma once

#include "config.h"

#include <vector>
#include "metrics.h"
using namespace std;

// Heat Profile
vector<float> loadHeatProfile(float resistance);

// Initial Grid

void initializeGrid(
    vector<float>& T,
    vector<float>& Tnew,
    int N
);

// Print

void printSimulationInfo(
    int N,
    vector<float>& heatProfile,
    Config& cfg
);

// Heat Source

float getQ(
    Config& cfg,
    int step,
    vector<float>& heatProfile
);

// CSV

void saveResultsCSV(
    vector<int>& gridSizes,
    vector<Metrics>& results
);