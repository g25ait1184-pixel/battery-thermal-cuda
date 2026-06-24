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
    // Physics
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
    // Thermal Map
    //--------------------------------------

    cfg.thermalMapSize = 256;

    cfg.thermalMapSteps = 2000;

    return cfg;
}