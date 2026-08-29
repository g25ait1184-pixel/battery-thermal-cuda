#pragma once

#include "config.h"

#include <vector>

using namespace std;

// CPU

double benchmarkCPU(
    Config& cfg,
    int N,
    std::vector<float>& T,
    std::vector<float>& Tnew,
    std::vector<float>& heatProfile,
    std::vector<float>& finalT
);

double benchmarkGPU(
    Config& cfg,
    int N,
    std::vector<float>& T,
    std::vector<float>& heatProfile,
    std::vector<float>& finalT
);

double benchmarkTiled(
    Config& cfg,
    int N,
    std::vector<float>& T,
    std::vector<float>& heatProfile,
    std::vector<float>& finalT
);

double benchmarkHalo(
    Config& cfg,
    int N,
    std::vector<float>& T,
    std::vector<float>& heatProfile,
    std::vector<float>& finalT
);