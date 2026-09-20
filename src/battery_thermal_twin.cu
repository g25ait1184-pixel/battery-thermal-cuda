#include "battery_thermal_twin.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

using namespace std;


// ============================================================
// CONSTRUCTOR
// ============================================================

BatteryThermalDigitalTwin::
BatteryThermalDigitalTwin(
    const TwinConfig& config,
    const Config& systemConfig)
    :
    twinConfig_(config),
    systemConfig_(systemConfig),
    d_T_(nullptr),
    d_Tnew_(nullptr),
    aiInitialized_(false),
    initialized_(false)
{
    input_ = {};

    state_ = {};

    output_ = {};
}


// ============================================================
// DESTRUCTOR
// ============================================================

BatteryThermalDigitalTwin::
~BatteryThermalDigitalTwin()
{
    reset();

    if(aiInitialized_)
    {
        destroyTwinAI(aiModel_);

        aiInitialized_ =
            false;
    }
}


// ============================================================
// CREATE V5 PARAMETERS
// ============================================================

BatteryParameters
BatteryThermalDigitalTwin::
createV5Parameters() const
{
    BatteryParameters params{};

    params.modelVersion =
        MODEL_V5;

    params.current =
        input_.current_A;

    params.baseResistance =
        systemConfig_.resistance;

    params.referenceTemperature =
        systemConfig_.referenceTemperature;

    params.resistanceCoeff =
        systemConfig_.resistanceCoeff;

    params.diffusivityCoeff =
        systemConfig_.diffusivityCoeff;

    params.entropyCoeff =
        systemConfig_.entropyCoeff;

    params.soc =
        input_.soc;

    params.soh =
        input_.soh;

    params.socResistanceCoeff =
        systemConfig_.socResistanceCoeff;

    params.sohResistanceCoeff =
        systemConfig_.sohResistanceCoeff;

    params.socHeatCoeff =
        systemConfig_.socHeatCoeff;

    params.sohHeatCoeff =
        systemConfig_.sohHeatCoeff;

    return params;
}


// ============================================================
// ALLOCATE DEVICE MEMORY
// ============================================================

bool BatteryThermalDigitalTwin::
allocateDeviceMemory()
{
    const int N =
        twinConfig_.gridSize;

    if(N <= 2)
    {
        cerr
            << "ERROR: Invalid Digital Twin grid size: "
            << N
            << endl;

        return false;
    }

    const size_t elements =
        static_cast<size_t>(N) *
        static_cast<size_t>(N);

    const size_t bytes =
        elements * sizeof(float);


    cudaError_t err =
        cudaMalloc(
            &d_T_,
            bytes
        );

    if(err != cudaSuccess)
    {
        cerr
            << "ERROR: cudaMalloc(d_T): "
            << cudaGetErrorString(err)
            << endl;

        d_T_ =
            nullptr;

        return false;
    }


    err =
        cudaMalloc(
            &d_Tnew_,
            bytes
        );

    if(err != cudaSuccess)
    {
        cerr
            << "ERROR: cudaMalloc(d_Tnew): "
            << cudaGetErrorString(err)
            << endl;

        cudaFree(d_T_);

        d_T_ =
            nullptr;

        d_Tnew_ =
            nullptr;

        return false;
    }


    return true;
}


// ============================================================
// INITIALIZE THERMAL FIELD
// ============================================================

bool BatteryThermalDigitalTwin::
initializeThermalField()
{
    if(d_T_ == nullptr ||
       d_Tnew_ == nullptr)
    {
        return false;
    }


    const int N =
        twinConfig_.gridSize;

    const size_t elements =
        static_cast<size_t>(N) *
        static_cast<size_t>(N);

    const size_t bytes =
        elements * sizeof(float);


    vector<float> initialTemperature(
        elements,
        input_.temperature_C
    );


    cudaError_t err =
        cudaMemcpy(
            d_T_,
            initialTemperature.data(),
            bytes,
            cudaMemcpyHostToDevice
        );

    if(err != cudaSuccess)
    {
        cerr
            << "ERROR: cudaMemcpy(d_T): "
            << cudaGetErrorString(err)
            << endl;

        return false;
    }


    err =
        cudaMemcpy(
            d_Tnew_,
            initialTemperature.data(),
            bytes,
            cudaMemcpyHostToDevice
        );

    if(err != cudaSuccess)
    {
        cerr
            << "ERROR: cudaMemcpy(d_Tnew): "
            << cudaGetErrorString(err)
            << endl;

        return false;
    }


    return true;
}


// ============================================================
// INITIALIZE DIGITAL TWIN
// ============================================================

bool BatteryThermalDigitalTwin::
initialize(
    const TwinInput& input,
    const vector<float>& heatProfile)
{
    input_ =
        input;

    heatProfile_ =
        heatProfile;


    if(heatProfile_.empty())
    {
        cerr
            << "ERROR: Digital Twin heat profile is empty."
            << endl;

        return false;
    }


    // --------------------------------------------------------
    // Validate grid
    // --------------------------------------------------------

    if(twinConfig_.gridSize <= 2)
    {
        cerr
            << "ERROR: Digital Twin grid size must be > 2."
            << endl;

        return false;
    }


    // --------------------------------------------------------
    // Physics initialization
    // --------------------------------------------------------

    if(twinConfig_.enablePhysics ||
       twinConfig_.mode == TWIN_PHYSICS ||
       twinConfig_.mode == TWIN_HYBRID)
    {
        if(!allocateDeviceMemory())
        {
            return false;
        }


        if(!initializeThermalField())
        {
            reset();

            return false;
        }
    }


    // --------------------------------------------------------
    // AI initialization
    // --------------------------------------------------------

    if(twinConfig_.enableAI ||
       twinConfig_.mode == TWIN_AI ||
       twinConfig_.mode == TWIN_HYBRID)
    {
        if(!initializeTwinAI(aiModel_))
        {
            cerr
                << "ERROR: initializeTwinAI() failed."
                << endl;

            reset();

            return false;
        }

        aiInitialized_ =
            true;
    }


    // --------------------------------------------------------
    // Initialize state
    // --------------------------------------------------------

    state_.time_s =
        0.0f;

    state_.soc =
        input_.soc;

    state_.soh =
        input_.soh;

    state_.current_A =
        input_.current_A;

    state_.temperature_C =
        input_.temperature_C;

    state_.tMin_C =
        input_.temperature_C;

    state_.tMax_C =
        input_.temperature_C;

    state_.tMean_C =
        input_.temperature_C;

    state_.tCenter_C =
        input_.temperature_C;

    state_.thermalRange_C =
        0.0f;

    state_.resistance_ohm =
        systemConfig_.resistance;

    state_.heatGeneration_W =
        0.0f;


    // --------------------------------------------------------
    // Initialize output
    // --------------------------------------------------------

    output_.predicted_Tmax_C =
        input_.temperature_C;

    output_.predicted_Tmin_C =
        input_.temperature_C;

    output_.predicted_Tmean_C =
        input_.temperature_C;

    output_.thermalRange_C =
        0.0f;

    output_.predictionError_C =
        0.0f;

    output_.inferenceTime_ms =
        0.0f;

    output_.executionMode =
        static_cast<int>(
            twinConfig_.mode
        );

    output_.valid =
        true;


    // --------------------------------------------------------
    // Calculate derived V5 quantities
    // --------------------------------------------------------

    updateDerivedState();


    initialized_ =
        true;


    return true;
}


// ============================================================
// SYNCHRONIZE
// ============================================================

void BatteryThermalDigitalTwin::
synchronize(
    const TwinInput& input)
{
    input_ =
        input;


    state_.soc =
        input_.soc;

    state_.soh =
        input_.soh;

    state_.current_A =
        input_.current_A;


    state_.time_s +=
        input_.dt_s;


    // --------------------------------------------------------
    // For AI mode temperature comes from input.
    //
    // For physics mode the internally simulated temperature
    // remains the thermal state.
    // --------------------------------------------------------

    if(twinConfig_.mode == TWIN_AI)
    {
        state_.temperature_C =
            input_.temperature_C;
    }


    updateDerivedState();
}


// ============================================================
// UPDATE DERIVED V5 STATE
//
// Uses the actual V5 resistance function.
//
// No duplicated V5 resistance equation here.
// ============================================================

void BatteryThermalDigitalTwin::
updateDerivedState()
{
    BatteryParameters params =
        createV5Parameters();


    // --------------------------------------------------------
    // Use the same combined V5 resistance implementation
    // used by the battery model.
    // --------------------------------------------------------

    state_.resistance_ohm =
        computeSOCSOHResistance(
            params.baseResistance,
            state_.temperature_C,
            params.referenceTemperature,
            params.resistanceCoeff,
            state_.soc,
            state_.soh,
            params.socResistanceCoeff,
            params.sohResistanceCoeff
        );


    // --------------------------------------------------------
    // Joule heating
    //
    // Q = I^2 * R
    // --------------------------------------------------------

    state_.heatGeneration_W =
        params.current *
        params.current *
        state_.resistance_ohm;
}


// ============================================================
// UPDATE THERMAL STATE
// ============================================================

void BatteryThermalDigitalTwin::
updateThermalState(
    float Tmin,
    float Tmax,
    float Tmean,
    float Tcenter)
{
    state_.tMin_C =
        Tmin;

    state_.tMax_C =
        Tmax;

    state_.tMean_C =
        Tmean;

    state_.tCenter_C =
        Tcenter;

    state_.thermalRange_C =
        Tmax - Tmin;


    state_.temperature_C =
        Tmean;


    updateDerivedState();
}


// ============================================================
// PHYSICS STEP
// ============================================================

bool BatteryThermalDigitalTwin::
stepPhysics()
{
    if(!initialized_)
    {
        cerr
            << "ERROR: Digital Twin not initialized."
            << endl;

        return false;
    }


    if(d_T_ == nullptr ||
       d_Tnew_ == nullptr)
    {
        cerr
            << "ERROR: Digital Twin device memory unavailable."
            << endl;

        return false;
    }


    if(heatProfile_.empty())
    {
        cerr
            << "ERROR: Digital Twin heat profile empty."
            << endl;

        return false;
    }


    int steps =
        twinConfig_.physicsSteps;


    if(steps <= 0)
    {
        steps =
            100;
    }


    // --------------------------------------------------------
    // IMPORTANT:
    //
    // runGPU() swaps pointers locally.
    //
    // Use even number of steps so original d_T_
    // contains the final result.
    // --------------------------------------------------------

    if(steps % 2 != 0)
    {
        steps++;
    }


    BatteryParameters params =
        createV5Parameters();


    auto start =
        chrono::high_resolution_clock::now();


    runGPU(
        d_T_,
        d_Tnew_,
        twinConfig_.gridSize,
        systemConfig_.alpha,
        steps,
        systemConfig_.beta,
        params,
        heatProfile_
    );


    cudaError_t err =
        cudaGetLastError();

    if(err != cudaSuccess)
    {
        cerr
            << "ERROR: runGPU(): "
            << cudaGetErrorString(err)
            << endl;

        return false;
    }


    err =
        cudaDeviceSynchronize();

    if(err != cudaSuccess)
    {
        cerr
            << "ERROR: cudaDeviceSynchronize(): "
            << cudaGetErrorString(err)
            << endl;

        return false;
    }


    auto stop =
        chrono::high_resolution_clock::now();


    float runtime_ms =
        static_cast<float>(
            chrono::duration<double, milli>(
                stop - start
            ).count()
        );


    // --------------------------------------------------------
    // Copy final field
    // --------------------------------------------------------

    const int N =
        twinConfig_.gridSize;

    const size_t elements =
        static_cast<size_t>(N) *
        static_cast<size_t>(N);

    const size_t bytes =
        elements * sizeof(float);


    vector<float> finalT(
        elements
    );


    err =
        cudaMemcpy(
            finalT.data(),
            d_T_,
            bytes,
            cudaMemcpyDeviceToHost
        );

    if(err != cudaSuccess)
    {
        cerr
            << "ERROR: Final temperature copy: "
            << cudaGetErrorString(err)
            << endl;

        return false;
    }


    // --------------------------------------------------------
    // Thermal statistics
    // --------------------------------------------------------

    float Tmin =
        numeric_limits<float>::max();

    float Tmax =
        numeric_limits<float>::lowest();

    double sum =
        0.0;


    for(size_t i = 0;
        i < elements;
        ++i)
    {
        const float T =
            finalT[i];


        Tmin =
            min(Tmin, T);

        Tmax =
            max(Tmax, T);

        sum +=
            static_cast<double>(T);
    }


    float Tmean =
        static_cast<float>(
            sum /
            static_cast<double>(elements)
        );


    // --------------------------------------------------------
    // Center temperature
    // --------------------------------------------------------

    int center =
        N / 2;


    float Tcenter =
        finalT[
            center * N +
            center
        ];


    // --------------------------------------------------------
    // Update Twin state
    // --------------------------------------------------------

    updateThermalState(
        Tmin,
        Tmax,
        Tmean,
        Tcenter
    );


    // --------------------------------------------------------
    // Output
    // --------------------------------------------------------

    output_.predicted_Tmax_C =
        Tmax;

    output_.predicted_Tmin_C =
        Tmin;

    output_.predicted_Tmean_C =
        Tmean;

    output_.thermalRange_C =
        Tmax - Tmin;

    output_.predictionError_C =
        0.0f;

    output_.inferenceTime_ms =
        runtime_ms;

    output_.executionMode =
        TWIN_PHYSICS;

    output_.valid =
        true;


    return true;
}


// ============================================================
// AI STEP
//
// Current trained model predicts Tmax only.
// ============================================================

bool BatteryThermalDigitalTwin::
stepAI()
{
    if(!initialized_)
    {
        cerr
            << "ERROR: Digital Twin not initialized."
            << endl;

        return false;
    }


    if(!aiInitialized_)
    {
        cerr
            << "ERROR: AI model not initialized."
            << endl;

        return false;
    }


    float predictedTmax =
        0.0f;

    float inferenceTime =
        0.0f;


    bool success =
        runTwinAI(
            aiModel_,
            input_.soc,
            input_.soh,
            input_.current_A,
            input_.temperature_C,
            predictedTmax,
            inferenceTime
        );


    if(!success)
    {
        cerr
            << "ERROR: runTwinAI() failed."
            << endl;

        return false;
    }


    // --------------------------------------------------------
    // Current AI model predicts Tmax.
    //
    // It does NOT predict Tmin/Tmean.
    // Therefore do not fabricate them.
    // --------------------------------------------------------

    output_.predicted_Tmax_C =
        predictedTmax;

    output_.predicted_Tmin_C =
        state_.tMin_C;

    output_.predicted_Tmean_C =
        state_.tMean_C;

    output_.thermalRange_C =
        predictedTmax -
        state_.tMin_C;

    output_.predictionError_C =
        0.0f;

    output_.inferenceTime_ms =
        inferenceTime;

    output_.executionMode =
        TWIN_AI;

    output_.valid =
        true;


    state_.tMax_C =
        predictedTmax;

    state_.thermalRange_C =
        predictedTmax -
        state_.tMin_C;


    updateDerivedState();


    return true;
}


// ============================================================
// HYBRID STEP
//
// Current stage:
//     Base AI
//
// Residual model can be integrated after the base AI
// interface is verified.
// ============================================================

bool BatteryThermalDigitalTwin::
stepHybrid()
{
    if(!initialized_)
    {
        cerr
            << "ERROR: Digital Twin not initialized."
            << endl;

        return false;
    }


    if(!aiInitialized_)
    {
        cerr
            << "ERROR: AI model not initialized."
            << endl;

        return false;
    }


    if(!stepAI())
    {
        return false;
    }


    output_.executionMode =
        TWIN_HYBRID;


    return true;
}


// ============================================================
// GET STATE
// ============================================================

TwinState
BatteryThermalDigitalTwin::
getState() const
{
    return state_;
}


// ============================================================
// GET OUTPUT
// ============================================================

TwinOutput
BatteryThermalDigitalTwin::
getOutput() const
{
    return output_;
}


// ============================================================
// RESET
// ============================================================

void BatteryThermalDigitalTwin::
reset()
{
    if(d_T_ != nullptr)
    {
        cudaFree(d_T_);

        d_T_ =
            nullptr;
    }


    if(d_Tnew_ != nullptr)
    {
        cudaFree(d_Tnew_);

        d_Tnew_ =
            nullptr;
    }


    heatProfile_.clear();


    initialized_ =
        false;
}