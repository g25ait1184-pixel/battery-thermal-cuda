#pragma once

#include <vector>
#include <string>

#include "battery_model.h"

using namespace std;


//=====================================================
// Paths
//=====================================================

#define BASE_DIR   "/content/drive/MyDrive/CUDA_Project/project/"

#define DATA_DIR   BASE_DIR "data/"
#define RESULT_DIR BASE_DIR "results/"
#define PLOT_DIR   BASE_DIR "plots/"


//=====================================================
// Compile-Time Heat Modes
//=====================================================

#define CONSTANT_Q_MODE 0
#define DYNAMIC_Q_MODE  1

#ifndef HEAT_MODE
#define HEAT_MODE DYNAMIC_Q_MODE
#endif


//=====================================================
// Compile-Time Execution Modes
//=====================================================

#define NORMAL_EXECUTION         0
#define NSIGHT_EXECUTION         1
#define PROFILE_SINGLE_KERNEL    2

#ifndef EXEC_MODE
#define EXEC_MODE NORMAL_EXECUTION
#endif


//=====================================================
// Kernel Selection for Profile Mode
//=====================================================

#define PROFILE_GPU_KERNEL       0
#define PROFILE_TILED_KERNEL     1
#define PROFILE_HALO_KERNEL      2

#ifndef PROFILE_KERNEL
#define PROFILE_KERNEL PROFILE_HALO_KERNEL
#endif


//=====================================================
// Runtime Heat Mode
//=====================================================

enum HeatMode
{
    CONSTANT_Q,
    DYNAMIC_Q
};


//=====================================================
// Runtime Execution Mode
//=====================================================

enum ExecMode
{
    NORMAL_MODE,
    NSIGHT_MODE,
    PROFILE_MODE
};


//=====================================================
// Runtime Kernel Selection
//=====================================================

enum KernelMode
{
    GPU_KERNEL,
    TILED_KERNEL,
    HALO_KERNEL
};


//=====================================================
// Configuration Structure
//=====================================================

struct Config
{
    //=================================================
    // Basic Thermal Physics
    //=================================================

    float alpha = 0.1f;

    float beta = 0.01f;

    float resistance = 0.01f;

    float current = 20.0f;


    //=================================================
    // Execution Modes
    //=================================================

    HeatMode heatMode;

    ExecMode execMode;

    KernelMode kernelMode;


    //=================================================
    // Benchmark Configuration
    //=================================================

    int RUNS = 20;

    int steps = 100;

    vector<int> sizes;


    //=================================================
    // Output
    //=================================================

    bool saveCSV = true;

    bool savePlots = true;


    //=================================================
    // Thermal Map
    //=================================================

    int thermalMapSize = 256;

    int thermalMapSteps = 2000;


    //=================================================
    // Battery Model Version
    //=================================================

    BatteryModelVersion modelVersion;


    //=================================================
    // Battery Physics Parameters
    //=================================================

    //-----------------------------------------------
    // Reference Temperature
    //-----------------------------------------------

    float referenceTemperature = 25.0f;


    //-----------------------------------------------
    // Temperature-dependent Resistance
    //-----------------------------------------------

    float resistanceCoeff = 0.004f;


    //-----------------------------------------------
    // Temperature-dependent Diffusivity
    //-----------------------------------------------

    float diffusivityCoeff = 0.002f;


    //-----------------------------------------------
    // Entropy / Reversible Heat
    //-----------------------------------------------

    float entropyCoeff = 0.0001f;


    //=================================================
    // Electro-Thermal Enable Flags
    //=================================================

    bool enableTempResistance = true;

    bool enableTempDiffusivity = true;

    bool enableElectroThermal = true;


    //=================================================
    // V5 — SOC / SOH PARAMETERS
    //=================================================

    //-----------------------------------------------
    // Initial SOC
    //-----------------------------------------------
    //
    // 1.0 = 100%
    // 0.0 = 0%
    //

    float initialSOC = 1.0f;


    //-----------------------------------------------
    // Initial SOH
    //-----------------------------------------------
    //
    // 1.0 = 100%
    // 0.0 = fully degraded
    //

    float initialSOH = 1.0f;


    //=================================================
    // Runtime SOC / SOH
    //=================================================
    //
    // These are explicitly provided because the
    // V5 solver/benchmark uses cfg.soc and cfg.soh.
    //
    // They can later be updated dynamically during
    // the simulation.
    //

    float soc = 1.0f;

    float soh = 1.0f;


    //=================================================
    // SOC-Dependent Resistance
    //=================================================
    //
    // Resistance multiplier contribution:
    //
    // R_SOC = 1 + coeff * (1 - SOC)
    //
    // Higher resistance at lower SOC.
    //

    float socResistanceCoeff = 0.20f;


    //=================================================
    // SOH-Dependent Resistance
    //=================================================
    //
    // Resistance multiplier contribution:
    //
    // R_SOH = 1 + coeff * (1 - SOH)
    //
    // Resistance increases as battery ages.
    //

    float sohResistanceCoeff = 0.50f;


    //=================================================
    // SOC-Dependent Heat
    //=================================================
    //
    // Controls additional heat generated as a
    // function of SOC.
    //

    float socHeatCoeff = 0.10f;


    //=================================================
    // SOH-Dependent Heat
    //=================================================
    //
    // Controls additional heat generated as a
    // function of degradation.
    //

    float sohHeatCoeff = 0.20f;


    //=================================================
    // SOC Coupling Enable
    //=================================================

    bool enableSOCCoupling = true;


    //=================================================
    // SOH Coupling Enable
    //=================================================

    bool enableSOHCoupling = true;
};


//=====================================================
// Configuration Initialization
//=====================================================

Config initConfig();