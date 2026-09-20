#pragma once

#include <vector>

#include "config.h"

//======================================================
// V5 Dataset Generation
//======================================================
//
// Generates the initial 100-case V5 pilot dataset
// using the same V5 GPU solver used by main.cu.
//
// No separate main() is defined here.
//======================================================

void generateV5Dataset(
    const Config& cfg,
    const std::vector<float>& heatProfile);