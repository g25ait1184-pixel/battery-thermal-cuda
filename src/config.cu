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

    // cfg.heatMode = CONSTANT_Q;

    //--------------------------------------
    // Execution Mode
    //--------------------------------------

    cfg.execMode =
    static_cast<ExecMode>(
        EXEC_MODE
    );

    // cfg.execMode = NSIGHT_MODE;

    //--------------------------------------
    // Physics
    //--------------------------------------

    cfg.alpha = 0.1f;

    cfg.beta = 0.01f;

    cfg.resistance = 0.01f;

    cfg.current = 20.0f;

    //--------------------------------------
    // Benchmark Configuration
    //--------------------------------------

    if(cfg.execMode == NSIGHT_MODE)
    {
        cfg.RUNS = 1;

        cfg.steps = 1;

        cfg.sizes =
        {
            2048
        };
    }
    else
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
    }

    //--------------------------------------
    // Thermal Map
    //--------------------------------------

    cfg.thermalMapSize = 256;

    cfg.thermalMapSteps = 2000;

    return cfg;
}