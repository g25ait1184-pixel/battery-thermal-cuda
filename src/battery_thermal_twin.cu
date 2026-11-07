#include "battery_thermal_twin.h"
#include "kernels.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

using namespace std;

// ============================================================
// CONSTRUCTOR / DESTRUCTOR
// ============================================================

BatteryThermalDigitalTwin::BatteryThermalDigitalTwin(
    const TwinConfig& config,
    const Config& systemConfig)
    : twinConfig_(config),
      systemConfig_(systemConfig),
      d_T_(nullptr),
      d_Tnew_(nullptr),
      initialized_(false)
{
    input_ = {};
    state_ = {};
    output_ = {};
}

BatteryThermalDigitalTwin::~BatteryThermalDigitalTwin()
{
    reset();
}

// ============================================================
// V5 PARAMETER PACKING
// ============================================================

BatteryParameters BatteryThermalDigitalTwin::createV5Parameters() const
{
    BatteryParameters params{};

    params.modelVersion = MODEL_V5;
    params.current = input_.current_A;
    params.baseResistance = systemConfig_.resistance;
    params.referenceTemperature = systemConfig_.referenceTemperature;
    params.resistanceCoeff = systemConfig_.resistanceCoeff;
    params.diffusivityCoeff = systemConfig_.diffusivityCoeff;
    params.entropyCoeff = systemConfig_.entropyCoeff;
    params.soc = input_.soc;
    params.soh = input_.soh;
    params.socResistanceCoeff = systemConfig_.socResistanceCoeff;
    params.sohResistanceCoeff = systemConfig_.sohResistanceCoeff;
    params.socHeatCoeff = systemConfig_.socHeatCoeff;
    params.sohHeatCoeff = systemConfig_.sohHeatCoeff;

    return params;
}

// ============================================================
// CUDA MEMORY MANAGEMENT
// ============================================================

bool BatteryThermalDigitalTwin::allocateDeviceMemory()
{
    const int N = twinConfig_.gridSize;

    if (N <= 2) {
        cerr << "ERROR: Digital Twin grid size must be greater than 2.\n";
        return false;
    }

    const size_t elements = static_cast<size_t>(N) * static_cast<size_t>(N);
    const size_t bytes = elements * sizeof(float);

    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_T_), bytes);
    if (err != cudaSuccess) {
        cerr << "ERROR: cudaMalloc(d_T): " << cudaGetErrorString(err) << '\n';
        d_T_ = nullptr;
        return false;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&d_Tnew_), bytes);
    if (err != cudaSuccess) {
        cerr << "ERROR: cudaMalloc(d_Tnew): " << cudaGetErrorString(err) << '\n';
        cudaFree(d_T_);
        d_T_ = nullptr;
        d_Tnew_ = nullptr;
        return false;
    }

    return true;
}

bool BatteryThermalDigitalTwin::initializeThermalField()
{
    if (d_T_ == nullptr || d_Tnew_ == nullptr) {
        cerr << "ERROR: Thermal device buffers are not allocated.\n";
        return false;
    }

    const int N = twinConfig_.gridSize;
    const size_t elements = static_cast<size_t>(N) * static_cast<size_t>(N);
    const size_t bytes = elements * sizeof(float);

    vector<float> initialTemperature(elements, input_.temperature_C);

    cudaError_t err = cudaMemcpy(
        d_T_, initialTemperature.data(), bytes, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        cerr << "ERROR: cudaMemcpy to d_T: " << cudaGetErrorString(err) << '\n';
        return false;
    }

    err = cudaMemcpy(
        d_Tnew_, initialTemperature.data(), bytes, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        cerr << "ERROR: cudaMemcpy to d_Tnew: " << cudaGetErrorString(err) << '\n';
        return false;
    }

    return true;
}

// ============================================================
// INITIALIZATION
// ============================================================

bool BatteryThermalDigitalTwin::initialize(
    const TwinInput& input,
    const vector<float>& heatProfile)
{
    // Release any previous CUDA allocation before reinitializing.
    reset();

    input_ = input;
    heatProfile_ = heatProfile;

    if (heatProfile_.empty()) {
        cerr << "ERROR: Digital Twin heat profile is empty.\n";
        return false;
    }

    if (twinConfig_.gridSize <= 2) {
        cerr << "ERROR: Digital Twin grid size must be greater than 2.\n";
        return false;
    }

    const bool physicsRequested =
        twinConfig_.enablePhysics ||
        twinConfig_.mode == TWIN_PHYSICS ||
        twinConfig_.mode == TWIN_HYBRID;

    // This source/header set does not include an AI model interface.
    // Do not reference undeclared AI model members or unresolved AI symbols.
    if (twinConfig_.mode == TWIN_AI ||
        (twinConfig_.enableAI && !physicsRequested)) {
        cerr << "ERROR: AI mode requested, but no AI model implementation "
                "is included in this project build.\n";
        return false;
    }

    if (physicsRequested) {
        if (!allocateDeviceMemory() || !initializeThermalField()) {
            reset();
            return false;
        }
    }

    state_ = {};
    state_.time_s = 0.0f;
    state_.soc = input_.soc;
    state_.soh = input_.soh;
    state_.current_A = input_.current_A;
    state_.temperature_C = input_.temperature_C;
    state_.tMin_C = input_.temperature_C;
    state_.tMax_C = input_.temperature_C;
    state_.tMean_C = input_.temperature_C;
    state_.tCenter_C = input_.temperature_C;
    state_.thermalRange_C = 0.0f;
    state_.dt_s = input_.dt_s;

    updateDerivedState();

    output_ = {};
    output_.predicted_Tmax_C = input_.temperature_C;
    output_.predicted_Tmin_C = input_.temperature_C;
    output_.predicted_Tmean_C = input_.temperature_C;
    output_.thermalRange_C = 0.0f;
    output_.predictionError_C = 0.0f;
    output_.inferenceTime_ms = 0.0f;
    output_.executionMode = static_cast<int>(twinConfig_.mode);

    initialized_ = true;
    return validateState();
}

// ============================================================
// INPUT SYNCHRONIZATION / DERIVED STATE
// ============================================================

void BatteryThermalDigitalTwin::synchronize(const TwinInput& input)
{
    state_.previousSOC = state_.soc;
    state_.previousSOH = state_.soh;
    state_.previousTemperature_C = state_.temperature_C;

    input_ = input;

    state_.soc = input_.soc;
    state_.soh = input_.soh;
    state_.current_A = input_.current_A;
    state_.dt_s = input_.dt_s;
    state_.time_s += input_.dt_s;

    // In physics mode, temperature is advanced by the CUDA solver.
    // For non-physics updates, retain the externally measured temperature.
    if (twinConfig_.mode == TWIN_AI) {
        state_.temperature_C = input_.temperature_C;
    }

    updateDerivedState();
}

void BatteryThermalDigitalTwin::updateDerivedState()
{
    const BatteryParameters params = createV5Parameters();

    state_.resistance_ohm = computeV5Resistance(
        params.baseResistance,
        state_.temperature_C,
        params.referenceTemperature,
        params.resistanceCoeff,
        state_.soc,
        state_.soh,
        params.socResistanceCoeff,
        params.sohResistanceCoeff);

    state_.heatGeneration_W =
        params.current * params.current * state_.resistance_ohm;
}

// ============================================================
// THERMAL STATE UPDATE
// ============================================================

void BatteryThermalDigitalTwin::updateThermalState(
    float Tmin,
    float Tmax,
    float Tmean,
    float Tcenter)
{
    state_.tMin_C = Tmin;
    state_.tMax_C = Tmax;
    state_.tMean_C = Tmean;
    state_.tCenter_C = Tcenter;
    state_.thermalRange_C = Tmax - Tmin;
    state_.temperature_C = Tmean;

    updateDerivedState();
}

void BatteryThermalDigitalTwin::updateThermalStateFromPrediction(
    float newTemperature)
{
    state_.temperature_C = newTemperature;
    state_.tMin_C = newTemperature;
    state_.tMax_C = newTemperature;
    state_.tMean_C = newTemperature;
    state_.tCenter_C = newTemperature;
    state_.thermalRange_C = 0.0f;
    updateDerivedState();
}

// ============================================================
// PHYSICS STEP
// ============================================================

bool BatteryThermalDigitalTwin::stepPhysics()
{
    if (!initialized_) {
        cerr << "ERROR: Digital Twin is not initialized.\n";
        return false;
    }

    if (d_T_ == nullptr || d_Tnew_ == nullptr) {
        cerr << "ERROR: CUDA thermal buffers are unavailable.\n";
        return false;
    }

    if (heatProfile_.empty()) {
        cerr << "ERROR: Digital Twin heat profile is empty.\n";
        return false;
    }

    int steps = twinConfig_.physicsSteps;
    if (steps <= 0) {
        steps = 100;
    }

    // runGPU swaps its local input/output pointers. An even number
    // of iterations leaves the final field in the original d_T_ buffer.
    if ((steps % 2) != 0) {
        ++steps;
    }

    const BatteryParameters params = createV5Parameters();

    const auto start = chrono::high_resolution_clock::now();

    runGPU(
        d_T_,
        d_Tnew_,
        twinConfig_.gridSize,
        systemConfig_.alpha,
        steps,
        systemConfig_.beta,
        params,
        heatProfile_);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "ERROR: runGPU launch: " << cudaGetErrorString(err) << '\n';
        return false;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cerr << "ERROR: CUDA synchronization: "
             << cudaGetErrorString(err) << '\n';
        return false;
    }

    const auto stop = chrono::high_resolution_clock::now();
    const float runtimeMs = static_cast<float>(
        chrono::duration<double, milli>(stop - start).count());

    const int N = twinConfig_.gridSize;
    const size_t elements = static_cast<size_t>(N) * static_cast<size_t>(N);
    const size_t bytes = elements * sizeof(float);

    vector<float> finalTemperature(elements);

    err = cudaMemcpy(
        finalTemperature.data(), d_T_, bytes, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        cerr << "ERROR: Copying final temperature field: "
             << cudaGetErrorString(err) << '\n';
        return false;
    }

    float Tmin = numeric_limits<float>::max();
    float Tmax = numeric_limits<float>::lowest();
    double sum = 0.0;

    for (const float temperature : finalTemperature) {
        if (!isfinite(temperature)) {
            cerr << "ERROR: Non-finite temperature returned by GPU solver.\n";
            return false;
        }

        Tmin = min(Tmin, temperature);
        Tmax = max(Tmax, temperature);
        sum += static_cast<double>(temperature);
    }

    const float Tmean = static_cast<float>(
        sum / static_cast<double>(elements));

    const int center = N / 2;
    const float Tcenter = finalTemperature[
        static_cast<size_t>(center) * static_cast<size_t>(N) +
        static_cast<size_t>(center)];

    updateThermalState(Tmin, Tmax, Tmean, Tcenter);

    output_.predicted_Tmax_C = Tmax;
    output_.predicted_Tmin_C = Tmin;
    output_.predicted_Tmean_C = Tmean;
    output_.thermalRange_C = Tmax - Tmin;
    output_.predictionError_C = 0.0f;
    output_.inferenceTime_ms = runtimeMs;
    output_.executionMode = static_cast<int>(TWIN_PHYSICS);

    return validateState();
}

// ============================================================
// AI STEP
// ============================================================

bool BatteryThermalDigitalTwin::stepAI()
{
    cerr << "ERROR: stepAI() is unavailable: AI model headers, "
            "implementation, and model data are not part of the supplied "
            "project files.\n";
    return false;
}

// ============================================================
// HYBRID STEP
// ============================================================

bool BatteryThermalDigitalTwin::stepHybrid()
{
    // A true hybrid step requires both the physics solver and a working
    // AI prediction/residual model. Do not label a physics-only result
    // as hybrid.
    cerr << "ERROR: stepHybrid() requires an AI model implementation. "
            "The supplied project currently contains no AI model interface.\n";
    return false;
}

// ============================================================
// V6 STATE UPDATE INTERFACES
// ============================================================

void BatteryThermalDigitalTwin::updateSOC()
{
    // SOC is supplied by the BMS through TwinInput. No battery-capacity
    // integration model or current sign convention is declared here.
    state_.soc = input_.soc;
}

void BatteryThermalDigitalTwin::updateSOH()
{
    // SOH is supplied by the BMS through TwinInput. No ageing model is
    // declared in the supplied headers.
    state_.soh = input_.soh;
}

void BatteryThermalDigitalTwin::updateElectroThermalState()
{
    updateDerivedState();
}

bool BatteryThermalDigitalTwin::validateState() const
{
    return isfinite(state_.time_s) &&
           isfinite(state_.soc) &&
           isfinite(state_.soh) &&
           isfinite(state_.current_A) &&
           isfinite(state_.temperature_C) &&
           isfinite(state_.tMin_C) &&
           isfinite(state_.tMax_C) &&
           isfinite(state_.tMean_C) &&
           isfinite(state_.tCenter_C) &&
           isfinite(state_.resistance_ohm) &&
           isfinite(state_.heatGeneration_W);
}

bool BatteryThermalDigitalTwin::updateState()
{
    if (!initialized_) {
        cerr << "ERROR: Cannot update an uninitialized Digital Twin.\n";
        return false;
    }

    updateSOC();
    updateSOH();
    updateElectroThermalState();
    return validateState();
}

bool BatteryThermalDigitalTwin::step(const TwinInput& input)
{
    if (!initialized_) {
        cerr << "ERROR: Digital Twin is not initialized.\n";
        return false;
    }

    synchronize(input);

    switch (twinConfig_.mode) {
        case TWIN_PHYSICS:
            return stepPhysics();

        case TWIN_AI:
            return stepAI();

        case TWIN_HYBRID:
            return stepHybrid();

        default:
            cerr << "ERROR: Unsupported Digital Twin execution mode.\n";
            return false;
    }
}

void BatteryThermalDigitalTwin::resetState()
{
    state_ = {};
    output_ = {};
    input_ = {};
}

// ============================================================
// GET STATE / OUTPUT
// ============================================================

TwinState BatteryThermalDigitalTwin::getState() const
{
    return state_;
}

TwinOutput BatteryThermalDigitalTwin::getOutput() const
{
    return output_;
}

// ============================================================
// RESET / RESOURCE CLEANUP
// ============================================================

void BatteryThermalDigitalTwin::reset()
{
    if (d_T_ != nullptr) {
        cudaFree(d_T_);
        d_T_ = nullptr;
    }

    if (d_Tnew_ != nullptr) {
        cudaFree(d_Tnew_);
        d_Tnew_ = nullptr;
    }

    heatProfile_.clear();
    initialized_ = false;
    output_ = {};
}
