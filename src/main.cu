#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>
#include "config.h"
#include "benchmark.h"
#include "metrics.h"
#include "thermal_metrics.h"
#include "thermal_map.h"
#include "utils.h"
#include "cuda_events.h"
#include "battery_model.h"
#include "validation_metrics.h"
#include "stream_solver.h"
#include "generate_v5_dataset.h"
#include "battery_thermal_twin.h"
#include "twin_config.h"


using namespace std;

// Runtime selection is used; no recompilation is needed to switch modes.
//======================================================
// MAIN
//======================================================
void runAutomaticDigitalTwinTest(
    const Config& cfg,
    const std::vector<float>& heatProfile)
{
    cout << "\n";
    cout << "=====================================\n";
    cout << "DYNAMIC DIGITAL TWIN TEST\n";
    cout << "=====================================\n";


    // --------------------------------------------------------
    // Twin configuration
    // --------------------------------------------------------

    TwinConfig twinConfig;

    // Hybrid AI is the validated AI path
    twinConfig.mode = TWIN_HYBRID;

    twinConfig.enablePhysics = true;
    twinConfig.enableAI = true;
    twinConfig.enableResidual = true;
    twinConfig.enableValidation = true;

    twinConfig.gridSize = 256;
    twinConfig.physicsSteps = 100;


    // --------------------------------------------------------
    // Create Digital Twin
    // --------------------------------------------------------

    BatteryThermalDigitalTwin twin(
        twinConfig,
        cfg
    );


    // --------------------------------------------------------
    // Initial input
    // --------------------------------------------------------

    TwinInput input;

    input.soc = 0.80f;
    input.soh = 0.90f;
    input.current_A = 20.0f;
    input.temperature_C = 30.0f;
    input.dt_s = 1.0f;


    // --------------------------------------------------------
    // Initialize Digital Twin
    // --------------------------------------------------------

    if(!twin.initialize(
        input,
        heatProfile))
    {
        cerr
            << "ERROR: Digital Twin initialization failed."
            << endl;

        return;
    }


    // --------------------------------------------------------
    // Dynamic operating profile
    //
    // Expected:
    //
    // 20 A -> HYBRID
    // 22 A -> HYBRID
    // 28 A -> HYBRID
    // 30 A -> HYBRID
    // 35 A -> PHYSICS
    // 40 A -> PHYSICS
    // 25 A -> HYBRID
    // --------------------------------------------------------

    const std::vector<float> currentProfile =
    {
        20.0f,
        22.0f,
        28.0f,
        30.0f,
        35.0f,
        40.0f,
        25.0f
    };


    // --------------------------------------------------------
    // Statistics
    // --------------------------------------------------------

    int hybridCount = 0;
    int physicsCount = 0;

    double totalRuntime_ms = 0.0;


    // --------------------------------------------------------
    // Table header
    // --------------------------------------------------------

    cout << "\n";

    cout << left
         << setw(6)  << "Step"
         << setw(8)  << "SOC"
         << setw(8)  << "SOH"
         << setw(12) << "Current"
         << setw(10) << "Temp"
         << setw(15) << "Decision"
         << setw(15) << "Tmax"
         << setw(15) << "Runtime"
         << "\n";

    cout << string(91, '-')
         << "\n";


    // --------------------------------------------------------
    // Execute dynamic operating profile
    // --------------------------------------------------------

    for(size_t i = 0;
        i < currentProfile.size();
        ++i)
    {
        // ----------------------------------------------------
        // Update current battery operating point
        // ----------------------------------------------------

        input.current_A =
            currentProfile[i];


        // ----------------------------------------------------
        // Synchronize new BMS input
        // ----------------------------------------------------

        twin.synchronize(input);


        // ----------------------------------------------------
        // Check AI validity BEFORE execution
        // ----------------------------------------------------

        const bool aiValid =
            twin.isAIValid();


        // ----------------------------------------------------
        // Automatic execution
        // ----------------------------------------------------

        const bool success =
            twin.stepAutomatic();


        if(!success)
        {
            cerr
                << "\nERROR: Digital Twin failed at step "
                << (i + 1)
                << endl;

            return;
        }


        // ----------------------------------------------------
        // Get result
        // ----------------------------------------------------

        TwinOutput output =
            twin.getOutput();

        TwinState state =
            twin.getState();


        // ----------------------------------------------------
        // Determine execution mode
        // ----------------------------------------------------

        const char* decision;

        if(output.executionMode == TWIN_HYBRID)
        {
            decision = "HYBRID";
            hybridCount++;
        }
        else if(output.executionMode == TWIN_PHYSICS)
        {
            decision = "PHYSICS";
            physicsCount++;
        }
        else if(output.executionMode == TWIN_AI)
        {
            // Kept for backward compatibility
            decision = "AI";
            hybridCount++;
        }
        else
        {
            decision = "UNKNOWN";
        }


        // ----------------------------------------------------
        // Accumulate runtime
        // ----------------------------------------------------

        totalRuntime_ms +=
            static_cast<double>(
                output.inferenceTime_ms
            );


        // ----------------------------------------------------
        // Print result
        // ----------------------------------------------------

        cout << left
             << setw(6)
             << (i + 1)

             << setw(8)
             << fixed
             << setprecision(2)
             << state.soc

             << setw(8)
             << fixed
             << setprecision(2)
             << state.soh

             << setw(12)
             << fixed
             << setprecision(2)
             << state.current_A

             << setw(10)
             << fixed
             << setprecision(2)
             << state.temperature_C

             << setw(15)
             << decision

             << setw(15)
             << fixed
             << setprecision(4)
             << output.predicted_Tmax_C

             << setw(15)
             << fixed
             << setprecision(6)
             << output.inferenceTime_ms

             << "\n";


        // ----------------------------------------------------
        // Detailed validity information
        // ----------------------------------------------------

        cout << "       AI Validity: "
             << (aiValid ? "VALID" : "INVALID")
             << "\n";
    }


    // --------------------------------------------------------
    // Calculate Hybrid AI utilization
    // --------------------------------------------------------

    const int totalSteps =
        static_cast<int>(
            currentProfile.size()
        );


    double hybridUtilization =
        0.0;


    if(totalSteps > 0)
    {
        hybridUtilization =
            100.0 *
            static_cast<double>(hybridCount) /
            static_cast<double>(totalSteps);
    }


    // --------------------------------------------------------
    // Summary
    // --------------------------------------------------------

    cout << "\n";

    cout << "=====================================\n";
    cout << "DYNAMIC DIGITAL TWIN SUMMARY\n";
    cout << "=====================================\n";

    cout << "Total Steps       : "
         << totalSteps
         << "\n";

    cout << "Hybrid AI Steps   : "
         << hybridCount
         << "\n";

    cout << "Physics Steps     : "
         << physicsCount
         << "\n";

    cout << "Hybrid Utilization: "
         << fixed
         << setprecision(2)
         << hybridUtilization
         << " %\n";

    cout << "Total Runtime     : "
         << fixed
         << setprecision(6)
         << totalRuntime_ms
         << " ms\n";


    // --------------------------------------------------------
    // Verification
    // --------------------------------------------------------

    const bool switchingVerified =
        (hybridCount > 0 &&
         physicsCount > 0);


    cout << "\n";

    if(switchingVerified)
    {
        cout
            << "HYBRID / PHYSICS SWITCHING : VERIFIED\n";
    }
    else
    {
        cout
            << "HYBRID / PHYSICS SWITCHING : NOT VERIFIED\n";
    }


    cout << "=====================================\n";
    cout << "DYNAMIC DIGITAL TWIN COMPLETE\n";
    cout << "=====================================\n";
}
void runDigitalTwinTest(
    const Config& cfg,
    const std::vector<float>& heatProfile)
{
    std::cout
        << "\n=====================================\n"
        << "DIGITAL TWIN TEST\n"
        << "=====================================\n";


    // ============================================================
    // Twin configuration
    // ============================================================

    TwinConfig twinConfig;

    twinConfig.enablePhysics   = true;
    twinConfig.enableAI        = true;
    twinConfig.enableResidual  = true;
    twinConfig.enableValidation = true;

    twinConfig.mode = TWIN_HYBRID;

    twinConfig.gridSize = 256;
    twinConfig.physicsSteps = 100;


    // ============================================================
    // Input
    // ============================================================

    TwinInput input;

    input.current_A =
        cfg.current;

    input.soc =
        cfg.soc;

    input.soh =
        cfg.soh;

    input.temperature_C =
        30.0f;

    input.dt_s =
        1.0f;


    // ============================================================
    // Create Digital Twin
    // ============================================================

    BatteryThermalDigitalTwin twin(
        twinConfig,
        cfg
    );


    if(!twin.initialize(
            input,
            heatProfile))
    {
        std::cerr
            << "Digital Twin initialization failed!"
            << std::endl;

        return;
    }


    // ============================================================
    // STEP 1 : V5 CUDA PHYSICS
    // ============================================================

    std::cout
        << "\n-------------------------------------\n"
        << "STEP 1 : V5 CUDA PHYSICS\n"
        << "-------------------------------------\n";


    auto physicsStart =
        std::chrono::high_resolution_clock::now();


    if(!twin.stepPhysics())
    {
        std::cerr
            << "Physics execution failed!"
            << std::endl;

        return;
    }


    auto physicsEnd =
        std::chrono::high_resolution_clock::now();


    double physicsTime =
        std::chrono::duration<double, std::milli>(
            physicsEnd - physicsStart
        ).count();


    TwinState physicsState =
        twin.getState();


    float physicsTmax =
        physicsState.tMax_C;


    std::cout
        << "Physics Tmax : "
        << physicsTmax
        << " C"
        << std::endl;

    std::cout
        << "Physics Tmin : "
        << physicsState.tMin_C
        << " C"
        << std::endl;

    std::cout
        << "Physics Tmean: "
        << physicsState.tMean_C
        << " C"
        << std::endl;

    std::cout
        << "Physics Time : "
        << physicsTime
        << " ms"
        << std::endl;


    // ============================================================
    // STEP 2 : BASE AI
    // ============================================================

    std::cout
        << "\n-------------------------------------\n"
        << "STEP 2 : BASE AI\n"
        << "-------------------------------------\n";


    // Restore SAME operating condition
    twin.synchronize(input);


    auto baseStart =
        std::chrono::high_resolution_clock::now();


    if(!twin.stepAI())
    {
        std::cerr
            << "Base AI execution failed!"
            << std::endl;

        return;
    }


    auto baseEnd =
        std::chrono::high_resolution_clock::now();


    double baseWallTime =
        std::chrono::duration<double, std::milli>(
            baseEnd - baseStart
        ).count();


    TwinOutput baseOutput =
        twin.getOutput();


    float baseTmax =
        baseOutput.predicted_Tmax_C;


    float baseInferenceTime =
        baseOutput.inferenceTime_ms;


    std::cout
        << "Base AI Tmax      : "
        << baseTmax
        << " C"
        << std::endl;

    std::cout
        << "Base AI Inference : "
        << baseInferenceTime
        << " ms"
        << std::endl;

    std::cout
        << "Base AI Wall Time : "
        << baseWallTime
        << " ms"
        << std::endl;


    // ============================================================
    // STEP 3 : HYBRID AI
    //
    // Internally:
    //
    //      Base AI
    //        ↓
    //      Base Tmax
    //        ↓
    //      Residual AI
    //        ↓
    //      Delta Tmax
    //        ↓
    //      Hybrid Tmax
    //
    //      Hybrid = Base + Residual
    // ============================================================

    std::cout
        << "\n-------------------------------------\n"
        << "STEP 3 : HYBRID AI\n"
        << "-------------------------------------\n";


    // Restore SAME operating condition
    twin.synchronize(input);


    auto hybridStart =
        std::chrono::high_resolution_clock::now();


    if(!twin.stepHybrid())
    {
        std::cerr
            << "Hybrid AI execution failed!"
            << std::endl;

        return;
    }


    auto hybridEnd =
        std::chrono::high_resolution_clock::now();


    double hybridWallTime =
        std::chrono::duration<double, std::milli>(
            hybridEnd - hybridStart
        ).count();


    TwinOutput hybridOutput =
        twin.getOutput();


    float hybridTmax =
        hybridOutput.predicted_Tmax_C;


    float hybridInferenceTime =
        hybridOutput.inferenceTime_ms;


    std::cout
        << "Hybrid AI Tmax      : "
        << hybridTmax
        << " C"
        << std::endl;

    std::cout
        << "Hybrid AI Inference : "
        << hybridInferenceTime
        << " ms"
        << std::endl;

    std::cout
        << "Hybrid AI Wall Time : "
        << hybridWallTime
        << " ms"
        << std::endl;


    // ============================================================
    // RESIDUAL CORRECTION
    //
    // Since:
    //
    //     Hybrid = Base + Residual
    //
    // residual can be obtained as:
    //
    //     Residual = Hybrid - Base
    //
    // This avoids adding a new Digital Twin API.
    // ============================================================

    float residualT =
        hybridTmax - baseTmax;


    std::cout
        << "\n-------------------------------------\n"
        << "STEP 4 : RESIDUAL AI RESULT\n"
        << "-------------------------------------\n";

    std::cout
        << "Base Tmax           : "
        << baseTmax
        << " C"
        << std::endl;

    std::cout
        << "Residual DeltaT     : "
        << residualT
        << " C"
        << std::endl;

    std::cout
        << "Hybrid Tmax         : "
        << hybridTmax
        << " C"
        << std::endl;


    // ============================================================
    // ACCURACY COMPARISON
    // ============================================================

    double baseError =
        std::abs(
            static_cast<double>(physicsTmax) -
            static_cast<double>(baseTmax)
        );


    double hybridError =
        std::abs(
            static_cast<double>(physicsTmax) -
            static_cast<double>(hybridTmax)
        );


    double accuracyImprovement =
        0.0;


    if(baseError > 1.0e-12)
    {
        accuracyImprovement =
            100.0 *
            (baseError - hybridError) /
            baseError;
    }


    // ============================================================
    // SPEEDUP
    // ============================================================

    double baseSpeedup =
        0.0;

    double hybridSpeedup =
        0.0;


    if(baseWallTime > 0.0)
    {
        baseSpeedup =
            physicsTime /
            baseWallTime;
    }


    if(hybridWallTime > 0.0)
    {
        hybridSpeedup =
            physicsTime /
            hybridWallTime;
    }


    // ============================================================
    // FINAL COMPARISON TABLE
    // ============================================================

    std::cout
        << "\n=====================================\n"
        << "DIGITAL TWIN COMPARISON\n"
        << "=====================================\n";


    std::cout
        << std::left
        << std::setw(22) << "Method"
        << std::setw(18) << "Tmax (C)"
        << std::setw(20) << "Error vs Physics"
        << std::setw(18) << "Runtime (ms)"
        << "\n";


    std::cout
        << std::string(78, '-')
        << "\n";


    std::cout
        << std::left
        << std::setw(22)
        << "V5 CUDA Physics"
        << std::setw(18)
        << physicsTmax
        << std::setw(20)
        << 0.0
        << std::setw(18)
        << physicsTime
        << "\n";


    std::cout
        << std::left
        << std::setw(22)
        << "Base AI"
        << std::setw(18)
        << baseTmax
        << std::setw(20)
        << baseError
        << std::setw(18)
        << baseWallTime
        << "\n";


    std::cout
        << std::left
        << std::setw(22)
        << "Residual AI (DeltaT)"
        << std::setw(18)
        << residualT
        << std::setw(20)
        << "Correction"
        << std::setw(18)
        << hybridInferenceTime - baseInferenceTime
        << "\n";


    std::cout
        << std::left
        << std::setw(22)
        << "Hybrid AI"
        << std::setw(18)
        << hybridTmax
        << std::setw(20)
        << hybridError
        << std::setw(18)
        << hybridWallTime
        << "\n";


    // ============================================================
    // ACCURACY
    // ============================================================

    std::cout
        << "\n-------------------------------------\n"
        << "ACCURACY\n"
        << "-------------------------------------\n";


    std::cout
        << "Physics Tmax       : "
        << physicsTmax
        << " C"
        << std::endl;


    std::cout
        << "Base AI Tmax       : "
        << baseTmax
        << " C"
        << std::endl;


    std::cout
        << "Hybrid AI Tmax     : "
        << hybridTmax
        << " C"
        << std::endl;


    std::cout
        << "Base AI Error      : "
        << baseError
        << " C"
        << std::endl;


    std::cout
        << "Hybrid AI Error    : "
        << hybridError
        << " C"
        << std::endl;


    std::cout
        << "Accuracy Improvement: "
        << accuracyImprovement
        << " %"
        << std::endl;


    // ============================================================
    // PERFORMANCE
    // ============================================================

    std::cout
        << "\n-------------------------------------\n"
        << "PERFORMANCE\n"
        << "-------------------------------------\n";


    std::cout
        << "Physics Runtime    : "
        << physicsTime
        << " ms"
        << std::endl;


    std::cout
        << "Base AI Runtime    : "
        << baseWallTime
        << " ms"
        << std::endl;


    std::cout
        << "Hybrid AI Runtime  : "
        << hybridWallTime
        << " ms"
        << std::endl;


    std::cout
        << "Base AI Speedup    : "
        << baseSpeedup
        << " x"
        << std::endl;


    std::cout
        << "Hybrid AI Speedup  : "
        << hybridSpeedup
        << " x"
        << std::endl;


    // ============================================================
    // VALIDATION
    // ============================================================

    twin.synchronize(input);

    bool aiValid =
        twin.isAIValid();


    std::cout
        << "\n-------------------------------------\n"
        << "AI VALIDATION\n"
        << "-------------------------------------\n";


    std::cout
        << "SOC         : "
        << input.soc
        << std::endl;

    std::cout
        << "SOH         : "
        << input.soh
        << std::endl;

    std::cout
        << "Current     : "
        << input.current_A
        << " A"
        << std::endl;

    std::cout
        << "Temperature : "
        << input.temperature_C
        << " C"
        << std::endl;

    std::cout
        << "AI Region   : "
        << (aiValid ? "VALID" : "INVALID")
        << std::endl;


    // ============================================================
    // FINAL VERIFICATION
    // ============================================================

    std::cout
        << "\n-------------------------------------\n"
        << "FINAL VERIFICATION\n"
        << "-------------------------------------\n";


    if(hybridError < baseError)
    {
        std::cout
            << "Base AI             : VERIFIED"
            << std::endl;

        std::cout
            << "Residual Correction : VERIFIED"
            << std::endl;

        std::cout
            << "Hybrid Improvement  : VERIFIED"
            << std::endl;
    }
    else
    {
        std::cout
            << "Residual Correction : NOT IMPROVED"
            << std::endl;
    }


    std::cout
        << "\n=====================================\n"
        << "DIGITAL TWIN TEST COMPLETE\n"
        << "=====================================\n";
}

//======================================================
// STATEFUL DIGITAL TWIN TEST
//======================================================
// One call to twin.step(input) represents one external/BMS timestep.
// The current AI/Hybrid model predicts Tmax only; V5 CUDA physics
// remains the authoritative thermal-state transition when fallback
// is required. AI/Hybrid predictions do not overwrite the full
// thermal field.
//======================================================
void runStatefulDigitalTwinTest(
    const Config& cfg,
    const std::vector<float>& heatProfile)
{
    cout << "\n";
    cout << "=====================================================\n";
    cout << "STATEFUL BATTERY THERMAL DIGITAL TWIN TEST\n";
    cout << "=====================================================\n";

    TwinConfig twinConfig;

    twinConfig.mode = TWIN_HYBRID;
    twinConfig.enablePhysics = true;
    twinConfig.enableAI = true;
    twinConfig.enableResidual = true;
    twinConfig.enableValidation = true;
    twinConfig.enableStateEvolution = true;
    twinConfig.gridSize = 256;
    twinConfig.physicsSteps = 100;

    BatteryThermalDigitalTwin twin(
        twinConfig,
        cfg
    );

    TwinInput input;
    input.soc = 0.80f;
    input.soh = 0.90f;
    input.current_A = 20.0f;
    input.temperature_C = 30.0f;
    input.dt_s = 1.0f;

    if(!twin.initialize(input, heatProfile))
    {
        cerr << "ERROR: Stateful Digital Twin initialization failed."
             << endl;
        return;
    }

    // Validated AI region is:
    // SOC      : 0.20 - 1.00
    // SOH      : 0.70 - 1.00
    // Current  : 10 - 30 A
    // Temp     : 15 - 45 C
    //
    // 35 A and 40 A intentionally force the V5 CUDA physics fallback.
    const std::vector<float> currentProfile =
    {
        20.0f,
        22.0f,
        25.0f,
        28.0f,
        30.0f,
        35.0f,
        40.0f,
        25.0f,
        20.0f,
        15.0f
    };

    cout << "\n";
    cout << left
         << setw(6)  << "Step"
         << setw(10) << "Time(s)"
         << setw(8)  << "SOC"
         << setw(8)  << "SOH"
         << setw(12) << "Current"
         << setw(12) << "BMS Temp"
         << setw(12) << "Tmin"
         << setw(12) << "Tmax"
         << setw(12) << "Tmean"
         << setw(15) << "Resistance"
         << setw(12) << "Heat(W)"
         << setw(12) << "Mode"
         << "\n";

    cout << string(131, '-') << "\n";

    int hybridCount = 0;
    int physicsCount = 0;
    double totalRuntime_ms = 0.0;

    for(size_t i = 0; i < currentProfile.size(); ++i)
    {
        input.current_A = currentProfile[i];

        const auto start =
            std::chrono::high_resolution_clock::now();

        if(!twin.step(input))
        {
            cerr << "ERROR: Stateful Digital Twin failed at step "
                 << (i + 1) << endl;
            return;
        }

        const auto end =
            std::chrono::high_resolution_clock::now();

        const double wallTime_ms =
            std::chrono::duration<double, std::milli>(end - start).count();

        TwinState state = twin.getState();
        TwinOutput output = twin.getOutput();

        const bool aiValid = twin.isAIValid();

        const char* mode = "UNKNOWN";

        if(output.executionMode == TWIN_HYBRID)
        {
            mode = "HYBRID";
            hybridCount++;
        }
        else if(output.executionMode == TWIN_PHYSICS)
        {
            mode = "PHYSICS";
            physicsCount++;
        }
        else if(output.executionMode == TWIN_AI)
        {
            mode = "AI";
            hybridCount++;
        }

        totalRuntime_ms += wallTime_ms;

        cout << left
             << setw(6)
             << (i + 1)
             << setw(10)
             << fixed << setprecision(2)
             << state.time_s
             << setw(8)
             << setprecision(3)
             << state.soc
             << setw(8)
             << state.soh
             << setw(12)
             << setprecision(2)
             << state.current_A
             << setw(12)
             << input.temperature_C
             << setw(12)
             << setprecision(3)
             << state.tMin_C
             << setw(12)
             << state.tMax_C
             << setw(12)
             << state.tMean_C
             << setw(15)
             << setprecision(6)
             << state.resistance_ohm
             << setw(12)
             << setprecision(4)
             << state.heatGeneration_W
             << setw(12)
             << mode
             << "\n";

        cout << "       AI Validity: "
             << (aiValid ? "VALID" : "INVALID")
             << " | Prediction Tmax: "
             << fixed << setprecision(4)
             << output.predicted_Tmax_C
             << " C"
             << " | Wall: "
             << setprecision(6)
             << wallTime_ms
             << " ms\n";
    }

    TwinState finalState = twin.getState();

    const int totalSteps =
        static_cast<int>(currentProfile.size());

    const double hybridUtilization =
        (totalSteps > 0)
        ? 100.0 * static_cast<double>(hybridCount) /
          static_cast<double>(totalSteps)
        : 0.0;

    cout << "\n";
    cout << "=====================================================\n";
    cout << "FINAL DIGITAL TWIN STATE\n";
    cout << "=====================================================\n";

    cout << "Time              : " << finalState.time_s << " s\n";
    cout << "SOC               : " << finalState.soc << "\n";
    cout << "SOH               : " << finalState.soh << "\n";
    cout << "Current           : " << finalState.current_A << " A\n";
    cout << "Temperature       : " << finalState.temperature_C << " C\n";
    cout << "Tmin              : " << finalState.tMin_C << " C\n";
    cout << "Tmax              : " << finalState.tMax_C << " C\n";
    cout << "Tmean             : " << finalState.tMean_C << " C\n";
    cout << "Thermal Range     : " << finalState.thermalRange_C << " C\n";
    cout << "Resistance        : " << finalState.resistance_ohm << " Ohm\n";
    cout << "Heat Generation   : " << finalState.heatGeneration_W << " W\n";
    cout << "Execution Mode    : " << twin.getOutput().executionMode << "\n";
    cout << "Inference Time    : " << twin.getOutput().inferenceTime_ms << " ms\n";

    cout << "\n=====================================================\n";
    cout << "STATEFUL DIGITAL TWIN SUMMARY\n";
    cout << "=====================================================\n";
    cout << "Total Steps       : " << totalSteps << "\n";
    cout << "Hybrid AI Steps   : " << hybridCount << "\n";
    cout << "Physics Steps     : " << physicsCount << "\n";
    cout << "Hybrid Utilization: " << fixed << setprecision(2)
         << hybridUtilization << " %\n";
    cout << "Total Wall Runtime: " << fixed << setprecision(6)
         << totalRuntime_ms << " ms\n";

    const bool switchingVerified =
        (hybridCount > 0 && physicsCount > 0);

    cout << "\n";
    cout << (switchingVerified
             ? "HYBRID / PHYSICS STATEFUL SWITCHING : VERIFIED\n"
             : "HYBRID / PHYSICS STATEFUL SWITCHING : NOT VERIFIED\n");

    cout << "=====================================================\n";
    cout << "STATEFUL DIGITAL TWIN TEST COMPLETE\n";
    cout << "=====================================================\n";
}


//======================================================
// MAIN
//======================================================
int main(int argc, char* argv[])
{
    //--------------------------------------------------
    // Configuration
    //--------------------------------------------------

    Config cfg = initConfig();

// Runtime menu:
    // 1 = Existing V1-V5 workflow (unchanged)
    // 2 = Validity-aware automatic Digital Twin routing
    // 3 = Physics vs Base AI vs Hybrid comparison
    // 4 = Stateful closed-loop Digital Twin test
    // 5 = Run all Digital Twin tests
    int selection = 0;

    cout << "\n=====================================\n";
    cout << "BATTERY THERMAL SIMULATION\n";
    cout << "=====================================\n";
    cout << "1. Normal V1-V5 execution\n";
    cout << "2. Digital Twin - Automatic validity-aware routing\n";
    cout << "3. Digital Twin - Physics / Base AI / Hybrid comparison\n";
    cout << "4. Digital Twin - Stateful closed-loop test\n";
    cout << "5. Run all Digital Twin tests\n";
    cout << "Select mode: ";

    // Colab/non-interactive execution: pass the selection as an argument,
    // for example: ./thermal 2. If no argument is supplied, retain the
    // interactive prompt for terminals that support stdin.
    if(argc >= 2)
    {
        try
        {
            size_t parsedChars = 0;
            selection = std::stoi(argv[1], &parsedChars);

            if(parsedChars != std::string(argv[1]).length())
            {
                cerr << "Invalid argument. Use ./thermal 1, 2, 3, 4, or 5."
                     << endl;
                return 1;
            }
        }
        catch(const std::exception&)
        {
            cerr << "Invalid argument. Use ./thermal 1, 2, 3, 4, or 5."
                 << endl;
            return 1;
        }
    }
    else
    {
        if(!(cin >> selection))
        {
            cerr << "No menu selection received. In Colab, run with an argument, "
                 << "for example: ./thermal 2" << endl;
            return 1;
        }
    }

    if(selection < 1 || selection > 5)
    {
        cerr << "Invalid selection. Please enter a number from 1 to 5." << endl;
        return -1;
    }

    if(selection >= 2)
    {
        // Build the same heat profile used by the existing V1-V5 flow.
        vector<float> heatProfile;

        if(cfg.heatMode == DYNAMIC_Q)
        {
            heatProfile = loadHeatProfile(cfg.resistance);
        }
        else
        {
            const float Q =
                cfg.current * cfg.current * cfg.resistance;
            heatProfile.push_back(Q);
        }

        if(heatProfile.empty())
        {
            cerr << "ERROR: Digital Twin heat profile is empty." << endl;
            return -1;
        }

        cout << "Loaded " << heatProfile.size()
             << " heat samples for Digital Twin." << endl;

        switch(selection)
        {
            case 2:
                runAutomaticDigitalTwinTest(cfg, heatProfile);
                break;

            case 3:
                runDigitalTwinTest(cfg, heatProfile);
                break;

            case 4:
                runStatefulDigitalTwinTest(cfg, heatProfile);
                break;

            case 5:
                runAutomaticDigitalTwinTest(cfg, heatProfile);
                runDigitalTwinTest(cfg, heatProfile);
                runStatefulDigitalTwinTest(cfg, heatProfile);
                break;

            default:
                cerr << "Unsupported Digital Twin selection." << endl;
                return -1;
        }

        return 0;
    }

    // Selection 1 falls through to the original V1-V5 execution below.
    // No existing V1-V5 solver, benchmark, dataset, or CUDA kernel code
    // has been changed.


    //==================================================
    // V5 DATASET GENERATION
    //==================================================
    constexpr bool ENABLE_V5_DATASET_GENERATION = true;

    //--------------------------------------------------
    // NSIGHT / CUDA STREAMS EXPERIMENT SETTINGS
    //--------------------------------------------------

    if(cfg.execMode == NSIGHT_MODE)
    {
        cfg.RUNS = 1;
        cfg.steps = 100;

        cfg.sizes.clear();
        cfg.sizes.push_back(2048);

        cfg.saveCSV = false;
        cfg.savePlots = false;
    }

    //--------------------------------------------------
    // Battery Models
    //--------------------------------------------------

    BatteryModelVersion models[] =
    {
        MODEL_V1,
        MODEL_V2,
        MODEL_V3,
        MODEL_V4,
        MODEL_V5
    };

    const char* modelNames[] =
    {
        "MODEL_V1",
        "MODEL_V2",
        "MODEL_V3",
        "MODEL_V4",
        "MODEL_V5"
    };

    const int NUM_MODELS = 5;

    //--------------------------------------------------
    // Print V5 parameters
    //--------------------------------------------------

    cout << "\n=====================================\n";
    cout << "Battery Electro-Thermal Simulation";
    cout << "\n=====================================\n";

    cout << "\nInitial SOC = " << cfg.initialSOC << endl;
    cout << "Initial SOH = " << cfg.initialSOH << endl;
    cout << "SOC Resistance Coeff = " << cfg.socResistanceCoeff << endl;
    cout << "SOH Resistance Coeff = " << cfg.sohResistanceCoeff << endl;
    cout << "SOC Heat Coeff = " << cfg.socHeatCoeff << endl;
    cout << "SOH Heat Coeff = " << cfg.sohHeatCoeff << endl;

    //--------------------------------------------------
    // Run V1 -> V5
    //--------------------------------------------------

    for(int modelIndex = 0;
        modelIndex < NUM_MODELS;
        modelIndex++)
    {
        cfg.modelVersion = models[modelIndex];

        cout << "\n=====================================\n";
        cout << "Running " << modelNames[modelIndex] << endl;
        cout << "=====================================\n";

        switch(cfg.modelVersion)
        {
            case MODEL_V1:
                cout << "NASA heat only" << endl;
                break;

            case MODEL_V2:
                cout << "NASA heat + Joule heating" << endl;
                break;

            case MODEL_V3:
                cout << "NASA heat + temperature-dependent resistance" << endl;
                break;

            case MODEL_V4:
                cout << "NASA heat + temperature-dependent resistance"
                     << " + entropy heat" << endl;
                break;

            case MODEL_V5:
                cout << "SOC/SOH-aware electro-thermal model" << endl;
                break;

            default:
                cout << "Unknown model" << endl;
                break;
        }

        vector<float> heatProfile;

        if(cfg.heatMode == DYNAMIC_Q)
        {
            heatProfile = loadHeatProfile(cfg.resistance);

            if(heatProfile.empty())
            {
                cout << "Failed to load heat profile!" << endl;
                return -1;
            }

            cout << "\nHeat Mode : DYNAMIC Q" << endl;
            cout << "Heat Profile Size = " << heatProfile.size() << endl;
        }
        else
        {
            const float Q =
                cfg.current *
                cfg.current *
                cfg.resistance;

            heatProfile.push_back(Q);

            cout << "\nHeat Mode : CONSTANT Q" << endl;
            cout << "Heat Source Q = " << Q << endl;
        }

        if(cfg.execMode == NORMAL_MODE)
        {
            cout << "Execution Mode : NORMAL" << endl;
        }
        else if(cfg.execMode == NSIGHT_MODE)
        {
            cout << "Execution Mode : NSIGHT / CUDA STREAMS" << endl;
        }
        else
        {
            cout << "Execution Mode : PROFILE" << endl;
        }

        initCudaEvents();

        vector<int> gridSizes;
        vector<Metrics> performanceResults;

        vector<ThermalMetrics> cpuThermalResults;
        vector<ThermalMetrics> gpuThermalResults;
        vector<ThermalMetrics> tiledThermalResults;
        vector<ThermalMetrics> haloThermalResults;

        vector<ValidationMetrics> validationResults;
        vector<string> validationModelNames;

        vector<int> streamGridSizes;
        vector<double> stream1Results;
        vector<double> stream2Results;
        vector<double> stream4Results;
        vector<double> stream8Results;

        if(cfg.execMode == NORMAL_MODE)
        {
            cout << "\n"
                 << "Grid,"
                 << "CPU(ms),"
                 << "GPU(ms),"
                 << "Tiled(ms),"
                 << "Halo(ms),"
                 << "GPU Speedup,"
                 << "Tiled Speedup,"
                 << "Halo Speedup,"
                 << "GPU GFLOPS,"
                 << "Tiled GFLOPS,"
                 << "Halo GFLOPS,"
                 << "GPU BW(GB/s),"
                 << "Tiled BW(GB/s),"
                 << "Halo BW(GB/s),"
                 << "ArithmeticIntensity,"
                 << "Updates/sec,"
                 << "Latency(ms)"
                 << endl;
        }

        for(auto N : cfg.sizes)
        {
            vector<float> T;
            vector<float> Tnew;

            initializeGrid(T, Tnew, N);

            printSimulationInfo(N, heatProfile, cfg);

            double cpu_ms = 0.0;
            double gpu_ms = 0.0;
            double tiled_ms = 0.0;
            double halo_ms = 0.0;

            vector<float> finalT_CPU;
            vector<float> finalT_GPU;
            vector<float> finalT_Tiled;
            vector<float> finalT_Halo;
            vector<float> finalT_Streams;

            if(cfg.execMode == NORMAL_MODE)
            {
                cpu_ms = benchmarkCPU(
                    cfg, N, T, Tnew, heatProfile, finalT_CPU);

                gpu_ms = benchmarkGPU(
                    cfg, N, T, heatProfile, finalT_GPU);

                tiled_ms = benchmarkTiled(
                    cfg, N, T, heatProfile, finalT_Tiled);

                halo_ms = benchmarkHalo(
                    cfg, N, T, heatProfile, finalT_Halo);
            }
            else if(cfg.execMode == NSIGHT_MODE)
            {
                if(cfg.modelVersion == MODEL_V5)
                {
                    cout << "\n-------------------------------------\n";
                    cout << "CUDA Streams Nsight Experiment" << endl;
                    cout << "Model : MODEL_V5" << endl;
                    cout << "Grid  : " << N << " x " << N << endl;
                    cout << "Steps : " << cfg.steps << endl;
                    cout << "-------------------------------------\n";

                    BatteryParameters streamParams;
                    streamParams.modelVersion = cfg.modelVersion;
                    streamParams.current = cfg.current;
                    streamParams.baseResistance = cfg.resistance;
                    streamParams.referenceTemperature = cfg.referenceTemperature;
                    streamParams.resistanceCoeff = cfg.resistanceCoeff;
                    streamParams.diffusivityCoeff = cfg.diffusivityCoeff;
                    streamParams.entropyCoeff = cfg.entropyCoeff;
                    streamParams.soc = cfg.initialSOC;
                    streamParams.soh = cfg.initialSOH;
                    streamParams.socResistanceCoeff = cfg.socResistanceCoeff;
                    streamParams.sohResistanceCoeff = cfg.sohResistanceCoeff;
                    streamParams.socHeatCoeff = cfg.socHeatCoeff;
                    streamParams.sohHeatCoeff = cfg.sohHeatCoeff;

                    const int streamCounts[] = {1, 2, 4, 8};
                    double streamTimes[4] = {0.0, 0.0, 0.0, 0.0};

                    for(int si = 0; si < 4; si++)
                    {
                        const int numStreams = streamCounts[si];
                        double totalStreamTime = 0.0;

                        for(int r = 0; r < cfg.RUNS; r++)
                        {
                            vector<float> streamFinalT;

                            const double elapsed = runCUDAStreams(
                                numStreams,
                                N,
                                cfg.alpha,
                                cfg.steps,
                                cfg.beta,
                                streamParams,
                                T,
                                heatProfile,
                                streamFinalT
                            );

                            totalStreamTime += elapsed;

                            if(r == cfg.RUNS - 1)
                                finalT_Streams = streamFinalT;
                        }

                        if(cfg.RUNS > 0)
                        {
                            streamTimes[si] =
                                totalStreamTime /
                                static_cast<double>(cfg.RUNS);
                        }

                        cout << fixed << setprecision(4)
                             << numStreams << " Stream(s) : "
                             << streamTimes[si] << " ms" << endl;
                    }

                    const double baseline = streamTimes[0];

                    cout << "\nCUDA Stream Scaling" << endl;
                    cout << "-------------------------------------\n";

                    for(int si = 0; si < 4; si++)
                    {
                        double speedup = 0.0;
                        double efficiency = 0.0;

                        if(streamTimes[si] > 0.0 && baseline > 0.0)
                        {
                            speedup = baseline / streamTimes[si];
                            efficiency =
                                speedup /
                                static_cast<double>(streamCounts[si]) *
                                100.0;
                        }

                        cout << streamCounts[si] << " Stream(s)"
                             << " | Time = " << streamTimes[si] << " ms"
                             << " | Speedup = " << speedup << "x"
                             << " | Efficiency = " << efficiency << "%"
                             << endl;
                    }

                    cout << "-------------------------------------\n";

                    streamGridSizes.push_back(N);
                    stream1Results.push_back(streamTimes[0]);
                    stream2Results.push_back(streamTimes[1]);
                    stream4Results.push_back(streamTimes[2]);
                    stream8Results.push_back(streamTimes[3]);
                }
                else
                {
                    cout << "NSIGHT stream experiment skipped for "
                         << modelNames[modelIndex] << endl;
                }
            }
            else
            {
                if(cfg.kernelMode == GPU_KERNEL)
                {
                    gpu_ms = benchmarkGPU(
                        cfg, N, T, heatProfile, finalT_GPU);
                }
                else if(cfg.kernelMode == TILED_KERNEL)
                {
                    tiled_ms = benchmarkTiled(
                        cfg, N, T, heatProfile, finalT_Tiled);
                }
                else
                {
                    halo_ms = benchmarkHalo(
                        cfg, N, T, heatProfile, finalT_Halo);
                }
            }

            if(cfg.execMode == NORMAL_MODE)
            {
                Metrics m = computeMetrics(
                    cfg.modelVersion,
                    N,
                    cfg.steps,
                    cpu_ms,
                    gpu_ms,
                    tiled_ms,
                    halo_ms
                );

                cout << fixed << setprecision(2)
                     << N << ","
                     << m.cpu_ms << ","
                     << m.gpu_ms << ","
                     << m.tiled_ms << ","
                     << m.halo_ms << ","
                     << m.gpu_speedup << ","
                     << m.tiled_speedup << ","
                     << m.halo_speedup << ","
                     << m.gpu_gflops << ","
                     << m.tiled_gflops << ","
                     << m.halo_gflops << ","
                     << m.gpu_bw << ","
                     << m.tiled_bw << ","
                     << m.halo_bw << ","
                     << m.arithmetic_intensity << ","
                     << m.updates_per_sec << ","
                     << m.latency_ms
                     << endl;

                gridSizes.push_back(N);
                performanceResults.push_back(m);

                if(!finalT_CPU.empty())
                    cpuThermalResults.push_back(
                        calculateThermalMetrics(finalT_CPU, N));

                if(!finalT_GPU.empty())
                    gpuThermalResults.push_back(
                        calculateThermalMetrics(finalT_GPU, N));

                if(!finalT_Tiled.empty())
                    tiledThermalResults.push_back(
                        calculateThermalMetrics(finalT_Tiled, N));

                if(!finalT_Halo.empty())
                    haloThermalResults.push_back(
                        calculateThermalMetrics(finalT_Halo, N));

                if(!finalT_CPU.empty() && !finalT_GPU.empty())
                {
                    ValidationMetrics validation =
                        calculateValidationMetrics(finalT_CPU, finalT_GPU);

                    validationResults.push_back(validation);
                    validationModelNames.push_back(
                        string(modelNames[modelIndex]));

                    cout << "\nCPU vs GPU Validation" << endl;
                    cout << "Model             : " << modelNames[modelIndex] << endl;
                    cout << "Grid              : " << N << endl;
                    cout << "RMSE (C)          : " << scientific
                         << validation.rmse << endl;
                    cout << "Max Error (C)     : "
                         << validation.maxAbsoluteError << endl;
                    cout << "Mean Error (C)    : "
                         << validation.meanAbsoluteError << endl;
                    cout << "Relative Error    : "
                         << validation.relativeError << endl;
                    cout << fixed;
                }
            }
        }

        if(cfg.saveCSV)
        {
            saveResultsCSV(
                gridSizes,
                performanceResults,
                modelNames[modelIndex]
            );

            if(!cpuThermalResults.empty())
                saveThermalMetricsCSV(
                    gridSizes,
                    cpuThermalResults,
                    string(modelNames[modelIndex]) + "_CPU");

            if(!gpuThermalResults.empty())
                saveThermalMetricsCSV(
                    gridSizes,
                    gpuThermalResults,
                    string(modelNames[modelIndex]) + "_GPU");

            if(!tiledThermalResults.empty())
                saveThermalMetricsCSV(
                    gridSizes,
                    tiledThermalResults,
                    string(modelNames[modelIndex]) + "_Tiled");

            if(!haloThermalResults.empty())
                saveThermalMetricsCSV(
                    gridSizes,
                    haloThermalResults,
                    string(modelNames[modelIndex]) + "_Halo");

            if(!validationResults.empty())
            {
                const string validationFilename =
                    string(RESULT_DIR) +
                    "validation_" +
                    modelNames[modelIndex] +
                    ".csv";

                saveValidationMetricsCSV(
                    validationFilename,
                    validationModelNames,
                    gridSizes,
                    validationResults
                );
            }

            if(cfg.modelVersion == MODEL_V5 &&
               !streamGridSizes.empty())
            {
                const string streamFilename =
                    string(RESULT_DIR) +
                    "cuda_streams_MODEL_V5.csv";

                ofstream streamFile(streamFilename);

                if(streamFile.is_open())
                {
                    streamFile << "Grid,Streams_1,Streams_2,Streams_4,Streams_8\n";

                    for(size_t i = 0; i < streamGridSizes.size(); i++)
                    {
                        streamFile
                            << streamGridSizes[i] << ","
                            << stream1Results[i] << ","
                            << stream2Results[i] << ","
                            << stream4Results[i] << ","
                            << stream8Results[i] << "\n";
                    }

                    streamFile.close();

                    cout << "\nCUDA Streams results saved to: "
                         << streamFilename << endl;
                }
            }
        }

        if(cfg.savePlots)
        {
            generateThermalMap(cfg, heatProfile);
        }

        if(cfg.modelVersion == MODEL_V5 &&
           cfg.execMode == NORMAL_MODE &&
           ENABLE_V5_DATASET_GENERATION)
        {
            cout << "\n=====================================\n"
                 << "Starting V5 Dataset Generation"
                 << "\n=====================================\n";

            generateV5Dataset(cfg, heatProfile);

            cout << "\nV5 Dataset Generation Completed" << endl;
        }

        cleanupCudaEvents();

        cout << "\nFinished "
             << modelNames[modelIndex]
             << endl;
    }

    cout << "\n=====================================\n";
    cout << "All Battery Models Completed" << endl;
    cout << "V1 -> V5" << endl;
    cout << "V5 CUDA Streams Benchmark Completed" << endl;
    cout << "=====================================\n";

    return 0;
}
