#pragma once

#include "config.h"

#include <vector>

using namespace std;

// CPU

double benchmarkCPU(
    Config& cfg,
    int N,
    vector<float>& T,
    vector<float>& Tnew,
    vector<float>& heatProfile
);

// GPU

double benchmarkGPU(
    Config& cfg,
    int N,
    vector<float>& T,
    vector<float>& heatProfile
);

// Shared GPU

double benchmarkTiled(
    Config& cfg,
    int N,
    vector<float>& T,
    vector<float>& heatProfile
);

// Halo GPU

double benchmarkHalo(
    Config& cfg,
    int N,
    vector<float>& T,
    vector<float>& heatProfile
);