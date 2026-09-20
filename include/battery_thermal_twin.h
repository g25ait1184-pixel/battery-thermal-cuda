#pragma once

#include <vector>

#include "config.h"
#include "battery_model.h"
#include "twin_state.h"
#include "twin_config.h"


class BatteryThermalDigitalTwin
{
public:

    //--------------------------------------------------
    // Constructor / Destructor
    //--------------------------------------------------

    BatteryThermalDigitalTwin(
        const TwinConfig& config,
        const Config& systemConfig
    );

    ~BatteryThermalDigitalTwin();


    //--------------------------------------------------
    // Initialization
    //--------------------------------------------------

    bool initialize(
        const TwinInput& input,
        const std::vector<float>& heatProfile
    );


    //--------------------------------------------------
    // Synchronize external input
    //--------------------------------------------------

    void synchronize(
        const TwinInput& input
    );


    //--------------------------------------------------
    // Existing execution paths
    //--------------------------------------------------

    bool stepPhysics();

    bool stepAI();

    bool stepHybrid();


    //--------------------------------------------------
    // V6 Stateful Digital Twin
    //--------------------------------------------------

    bool step(
        const TwinInput& input
    );

    bool updateState();

    void resetState();


    //--------------------------------------------------
    // State / Output
    //--------------------------------------------------

    TwinState getState() const;

    TwinOutput getOutput() const;


    //--------------------------------------------------
    // Complete reset
    //--------------------------------------------------

    void reset();


private:

    //--------------------------------------------------
    // Configuration
    //--------------------------------------------------

    TwinConfig twinConfig_;

    Config systemConfig_;


    //--------------------------------------------------
    // Runtime input
    //--------------------------------------------------

    TwinInput input_;


    //--------------------------------------------------
    // Twin state
    //--------------------------------------------------

    TwinState state_;


    //--------------------------------------------------
    // Twin output
    //--------------------------------------------------

    TwinOutput output_;


    //--------------------------------------------------
    // Heat profile
    //--------------------------------------------------

    std::vector<float> heatProfile_;


    //--------------------------------------------------
    // CUDA thermal buffers
    //--------------------------------------------------

    float* d_T_;
    float* d_Tnew_;


    //--------------------------------------------------
    // Initialization state
    //--------------------------------------------------

    bool initialized_;


    //--------------------------------------------------
    // Internal helpers
    //--------------------------------------------------

    BatteryParameters createV5Parameters() const;

    bool allocateDeviceMemory();

    bool initializeThermalField();

    void updateDerivedState();

    void updateThermalState(
        float Tmin,
        float Tmax,
        float Tmean,
        float Tcenter
    );


    //--------------------------------------------------
    // V6 State-transition helpers
    //--------------------------------------------------

    void updateSOC();

    void updateSOH();

    void updateElectroThermalState();

    void updateThermalStateFromPrediction(
        float newTemperature
    );

    bool validateState() const;
};