#include "config.h"


Config initConfig()
{
    Config cfg;


    //--------------------------------------
    // Heat Source Selection
    //--------------------------------------

    cfg.heatMode =
        static_cast<HeatMode>(
            HEAT_MODE
        );


    //--------------------------------------
    // Execution Mode
    //--------------------------------------

#if EXEC_MODE == NORMAL_EXECUTION

    cfg.execMode = NORMAL_MODE;

#elif EXEC_MODE == NSIGHT_EXECUTION

    cfg.execMode = NSIGHT_MODE;

#else

    cfg.execMode = PROFILE_MODE;

#endif


    //--------------------------------------
    // Kernel Selection
    //--------------------------------------

#if PROFILE_KERNEL == PROFILE_GPU_KERNEL

    cfg.kernelMode = GPU_KERNEL;

#elif PROFILE_KERNEL == PROFILE_TILED_KERNEL

    cfg.kernelMode = TILED_KERNEL;

#else

    cfg.kernelMode = HALO_KERNEL;

#endif


    //--------------------------------------
    // Basic Physics
    //--------------------------------------

    cfg.alpha = 0.1f;

    cfg.beta = 0.01f;

    cfg.resistance = 0.01f;

    cfg.current = 20.0f;


    //--------------------------------------
    // NORMAL MODE
    //--------------------------------------

    if(cfg.execMode == NORMAL_MODE)
    {
        cfg.RUNS = 20;

        cfg.steps = 100;

        cfg.sizes =
        {
            256,
            512,
            1024,
            2048
        };

        cfg.saveCSV = true;

        cfg.savePlots = true;
    }


    //--------------------------------------
    // NSIGHT MODE
    //--------------------------------------

    else if(cfg.execMode == NSIGHT_MODE)
    {
        cfg.RUNS = 1;

        cfg.steps = 1;

        cfg.sizes =
        {
            2048
        };

        cfg.saveCSV = false;

        cfg.savePlots = false;
    }


    //--------------------------------------
    // PROFILE SINGLE KERNEL MODE
    //--------------------------------------

    else
    {
        cfg.RUNS = 1;

        cfg.steps = 1;

        cfg.sizes =
        {
            2048
        };

        cfg.saveCSV = false;

        cfg.savePlots = false;
    }


    //--------------------------------------
    // Thermal Model
    //--------------------------------------

    cfg.thermalMapSize = 256;

    cfg.thermalMapSteps = 2000;


    //--------------------------------------
    // Reference Temperature
    //--------------------------------------

    cfg.referenceTemperature = 25.0f;


    //--------------------------------------
    // Temperature Dependent Resistance
    //--------------------------------------

    cfg.resistanceCoeff = 0.004f;

    cfg.enableTempResistance = true;


    //--------------------------------------
    // Temperature Dependent Diffusivity
    //--------------------------------------

    cfg.diffusivityCoeff = 0.002f;

    cfg.enableTempDiffusivity = true;


    //--------------------------------------
    // Entropy / Reversible Heat
    //--------------------------------------

    cfg.entropyCoeff = 0.0001f;


    //--------------------------------------
    // Electro-Thermal Coupling
    //--------------------------------------

    cfg.enableElectroThermal = true;


    //--------------------------------------
    // SOC / SOH INITIAL CONDITIONS
    //--------------------------------------

    cfg.initialSOC = 1.0f;

    cfg.initialSOH = 1.0f;


    //--------------------------------------
    // SOC-Dependent Resistance
    //--------------------------------------
    //
    // Resistance modification:
    //
    // R = R(T) *
    //     [1 + socResistanceCoeff * (1-SOC)]
    //
    //--------------------------------------

    cfg.socResistanceCoeff = 0.20f;


    //--------------------------------------
    // SOH-Dependent Resistance
    //--------------------------------------
    //
    // Resistance modification:
    //
    // R = R(T) *
    //     [1 + sohResistanceCoeff * (1-SOH)]
    //
    //--------------------------------------

    cfg.sohResistanceCoeff = 0.50f;


    //--------------------------------------
    // SOC-Dependent Heat
    //--------------------------------------

    cfg.socHeatCoeff = 0.10f;


    //--------------------------------------
    // SOH-Dependent Heat
    //--------------------------------------

    cfg.sohHeatCoeff = 0.20f;


    //--------------------------------------
    // SOC / SOH Coupling Enable
    //--------------------------------------

    cfg.enableSOCCoupling = true;

    cfg.enableSOHCoupling = true;


    //--------------------------------------
    // MODEL VERSION
    //--------------------------------------
    //
    // V5:
    //
    // NASA Heat
    // + Joule Heating
    // + Temperature Dependent Resistance
    // + Entropy Heat
    // + SOC Dependent Effects
    // + SOH Dependent Effects
    //
    //--------------------------------------

    cfg.modelVersion = MODEL_V5;


    return cfg;
}