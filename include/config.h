#pragma once

#include <vector>
#include <string>
using namespace std;

#define BASE_DIR "/content/project/"

#define DATA_DIR    BASE_DIR "data/"
#define RESULT_DIR  BASE_DIR "results/"
#define PLOT_DIR    BASE_DIR "plots/"
//----------------------------------
// Heat Modes
//----------------------------------

#define CONSTANT_Q_MODE 0
#define DYNAMIC_Q_MODE  1

//----------------------------------
// Execution Modes
//----------------------------------

#define NORMAL_EXECUTION 0
#define NSIGHT_EXECUTION 1

#ifndef HEAT_MODE
#define HEAT_MODE DYNAMIC_Q_MODE
#endif

#ifndef EXEC_MODE
#define EXEC_MODE NORMAL_EXECUTION
#endif
//-------------------------------------
// Heat Source Mode
//-------------------------------------

enum HeatMode
{
    CONSTANT_Q,
    DYNAMIC_Q
};

//-------------------------------------
// Execution Mode
//-------------------------------------

enum ExecMode
{
    NORMAL_MODE,
    NSIGHT_MODE
};

//-------------------------------------
// Configuration
//-------------------------------------

struct Config
{
    // Physics

    float alpha = 0.1f;

    float beta = 0.01f;

    float resistance = 0.01f;

    float current = 20.0f;

    // Modes

    HeatMode heatMode;

    ExecMode execMode;

    // Benchmark

    int RUNS;

    int steps;

    vector<int> sizes;

    // Thermal Map

    int thermalMapSize = 256;

    int thermalMapSteps = 2000;
};

// function

Config initConfig();