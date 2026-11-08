
#include "battery_thermal_twin.h"
#include "kernels.h"
#include "ai_inference.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <string>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

using namespace std;

namespace
{
#ifndef AI_MODEL_PARAMS_PATH
#define AI_MODEL_PARAMS_PATH "ai_model_params.json"
#endif

AIInference& getTwinAIModel()
{
    static AIInference model;
    return model;
}

bool ensureTwinAIModelLoaded()
{
    AIInference& model = getTwinAIModel();

    if (model.isLoaded())
        return true;

    return model.load(AI_MODEL_PARAMS_PATH);
}

std::array<float, 4> makeAIInput(const TwinInput& input)
{
    // Feature order must match train_residual.py and exported JSON.
    return {
        input.soc,
        input.soh,
        input.current_A,
        input.temperature_C
    };
}
}

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

    const size_t elements =
        static_cast<size_t>(N) * static_cast<size_t>(N);

    const size_t bytes = elements * sizeof(float);

    cudaError_t err = cudaMalloc(
        reinterpret_cast<void**>(&d_T_), bytes);

    if (err != cudaSuccess) {
        cerr << "ERROR: cudaMalloc(d_T): "
             << cudaGetErrorString(err) << '\n';

        d_T_ = nullptr;
        return false;
    }

    err = cudaMalloc(
        reinterpret_cast<void**>(&d_Tnew_), bytes);

    if (err != cudaSuccess) {
        cerr << "ERROR: cudaMalloc(d_Tnew): "
             << cudaGetErrorString(err) << '\n';

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

    const size_t elements =
        static_cast<size_t>(N) * static_cast<size_t>(N);

    const size_t bytes = elements * sizeof(float);

    vector<float> initialTemperature(
        elements, input_.temperature_C);

    cudaError_t err = cudaMemcpy(
        d_T_,
        initialTemperature.data(),
        bytes,
        cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        cerr << "ERROR: cudaMemcpy to d_T: "
             << cudaGetErrorString(err) << '\n';
        return false;
    }

    err = cudaMemcpy(
        d_Tnew_,
        initialTemperature.data(),
        bytes,
        cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        cerr << "ERROR: cudaMemcpy to d_Tnew: "
             << cudaGetErrorString(err) << '\n';
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
    reset();

    input_ = input;
    heatProfile_ = heatProfile;

    if (twinConfig_.gridSize <= 2) {
        cerr << "ERROR: Digital Twin grid size must be greater than 2.\n";
        return false;
    }

    const bool physicsRequested =
        twinConfig_.enablePhysics ||
        twinConfig_.mode == TWIN_PHYSICS ||
        twinConfig_.mode == TWIN_HYBRID;

    const bool aiRequested =
        twinConfig_.enableAI ||
        twinConfig_.mode == TWIN_AI ||
        twinConfig_.mode == TWIN_HYBRID;

    if (physicsRequested && heatProfile_.empty()) {
        cerr << "ERROR: Physics/Hybrid mode requires "
                "a non-empty heat profile.\n";
        return false;
    }

    if (aiRequested && !ensureTwinAIModelLoaded()) {
        cerr << "ERROR: Unable to load AI model parameters from "
             << AI_MODEL_PARAMS_PATH << '\n';
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

    // Keep BMS temperature as an AI input.
    // Do not overwrite the retained physical thermal state.
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
    // AI predicts Tmax only.
    // It cannot reconstruct the complete thermal field.
    // Preserve the trusted physics-derived state.
    (void)newTemperature;
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

    if (steps <= 0)
        steps = 100;

    // Keep the final field in the original d_T_ buffer.
    if ((steps % 2) != 0)
        ++steps;

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
        cerr << "ERROR: runGPU launch: "
             << cudaGetErrorString(err) << '\n';
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

    const size_t elements =
        static_cast<size_t>(N) * static_cast<size_t>(N);

    const size_t bytes = elements * sizeof(float);

    vector<float> finalTemperature(elements);

    err = cudaMemcpy(
        finalTemperature.data(),
        d_T_,
        bytes,
        cudaMemcpyDeviceToHost);

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
// AI STEP - BASE MLP INFERENCE
// ============================================================

bool BatteryThermalDigitalTwin::stepAI()
{
    if (!initialized_) {
        cerr << "ERROR: Digital Twin is not initialized.\n";
        return false;
    }

    if (!ensureTwinAIModelLoaded()) {
        cerr << "ERROR: AI model parameters could not be loaded.\n";
        return false;
    }

    AIInference& model = getTwinAIModel();

    const array<float, 4> features = makeAIInput(input_);

    if (!model.isWithinValidatedDomain(features)) {
        cerr << "ERROR: AI input is outside the validated operating domain. "
                "Use Hybrid mode for automatic V5 physics fallback.\n";
        return false;
    }

    const auto start = chrono::high_resolution_clock::now();

    float predictedTmaxC = 0.0f;

    if (!model.predictBase(features, predictedTmaxC)) {
        cerr << "ERROR: Base MLP inference failed.\n";
        return false;
    }

    const auto stop = chrono::high_resolution_clock::now();

    const float runtimeMs = static_cast<float>(
        chrono::duration<double, milli>(stop - start).count());

    // Base AI predicts Tmax only.
    output_.predicted_Tmax_C = predictedTmaxC;
    output_.predicted_Tmin_C = numeric_limits<float>::quiet_NaN();
    output_.predicted_Tmean_C = numeric_limits<float>::quiet_NaN();

    output_.thermalRange_C = numeric_limits<float>::quiet_NaN();
    output_.predictionError_C = numeric_limits<float>::quiet_NaN();

    output_.inferenceTime_ms = runtimeMs;
    output_.executionMode = static_cast<int>(TWIN_AI);

    return validateState();
}

// ============================================================
// HYBRID STEP - BASE MLP + RESIDUAL MLP / PHYSICS FALLBACK
// ============================================================

bool BatteryThermalDigitalTwin::stepHybrid()
{
    if (!initialized_) {
        cerr << "ERROR: Digital Twin is not initialized.\n";
        return false;
    }

    if (!ensureTwinAIModelLoaded()) {
        cerr << "ERROR: AI model parameters could not be loaded.\n";
        return false;
    }

    AIInference& model = getTwinAIModel();

    const array<float, 4> features = makeAIInput(input_);

    // Input-domain validity check.
    if (twinConfig_.enableValidation &&
        !model.isWithinValidatedDomain(features)) {

        cerr << "WARNING: Input outside validated AI domain; "
                "switching to V5 CUDA physics.\n";

        return stepPhysics();
    }

    const auto start = chrono::high_resolution_clock::now();

    float basePredictionC = 0.0f;
    float residualCorrectionC = 0.0f;
    float hybridPredictionC = 0.0f;

    if (!model.predictHybrid(
            features,
            basePredictionC,
            residualCorrectionC,
            hybridPredictionC)) {

        cerr << "ERROR: Hybrid MLP inference failed.\n";
        return false;
    }

    const auto stop = chrono::high_resolution_clock::now();

    const float runtimeMs = static_cast<float>(
        chrono::duration<double, milli>(stop - start).count());

    // Hybrid AI predicts Tmax only.
    // Do not overwrite the physical thermal state.
    output_.predicted_Tmax_C = hybridPredictionC;

    output_.predicted_Tmin_C = numeric_limits<float>::quiet_NaN();
    output_.predicted_Tmean_C = numeric_limits<float>::quiet_NaN();

    output_.thermalRange_C = numeric_limits<float>::quiet_NaN();
    output_.predictionError_C = numeric_limits<float>::quiet_NaN();

    output_.inferenceTime_ms = runtimeMs;
    output_.executionMode = static_cast<int>(TWIN_HYBRID);

    return validateState();
}

// ============================================================
// V6 STATE UPDATE INTERFACES
// ============================================================

void BatteryThermalDigitalTwin::updateSOC()
{
    // SOC is supplied by the BMS through TwinInput.
    state_.soc = input_.soc;
}

void BatteryThermalDigitalTwin::updateSOH()
{
    // SOH is supplied by the BMS through TwinInput.
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

// ============================================================
// STATEFUL STEP
// ============================================================

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
            // IMPORTANT:
            // Apply both input validity and thermal-state validity.
            return stepAutomatic();

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

// ============================================================
// AI OPERATING-DOMAIN VALIDATION
// ============================================================

bool BatteryThermalDigitalTwin::isAIValid() const
{
    // These limits must match the validated training domain.
    return std::isfinite(input_.soc) &&
           std::isfinite(input_.soh) &&
           std::isfinite(input_.current_A) &&
           std::isfinite(input_.temperature_C) &&

           input_.soc >= 0.20f &&
           input_.soc <= 1.00f &&

           input_.soh >= 0.70f &&
           input_.soh <= 1.00f &&

           input_.current_A >= 10.0f &&
           input_.current_A <= 30.0f &&

           input_.temperature_C >= 15.0f &&
           input_.temperature_C <= 45.0f;
}

// ============================================================
// AUTOMATIC VALIDITY-AWARE EXECUTION
// ============================================================

bool BatteryThermalDigitalTwin::stepAutomatic()
{
    if (!initialized_) {
        cerr << "ERROR: Digital Twin is not initialized.\n";
        return false;
    }

    // Check current BMS operating conditions.
    const bool inputValid = isAIValid();

    // Check the retained physical thermal state.
    const bool thermalStateValid =
        std::isfinite(state_.tMax_C) &&
        state_.tMax_C <= 45.0f;

    // Execute Hybrid AI only when both checks pass.
    if (inputValid && thermalStateValid) {
        return stepHybrid();
    }

    cerr << "WARNING: AI operating conditions invalid; "
            "switching to V5 CUDA physics.\n";

    return stepPhysics();
}
