#include "generate_v5_dataset.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "battery_model.h"
#include "kernels.h"

using namespace std;
namespace fs = std::filesystem;


//======================================================
// DATASET CONFIGURATION
//======================================================

// Generate 1000 V5 cases
static constexpr int DATASET_CASES = 1000;

// Dataset grid
// Keep 256 x 256 for fast dataset generation.
// Production physics benchmark remains 2048 x 2048.
static constexpr int DATASET_N = 256;

// Keep this even because runGPU() performs one
// local pointer swap per timestep.
static constexpr int DATASET_STEPS = 100;


//======================================================
// DATASET PATHS
//======================================================

// IMPORTANT:
// Keep the original 100-case pilot untouched.
static const string DATASET_ROOT =
    string(BASE_DIR) + "dataset/v5_1000/";

static const string FIELD_DIR =
    DATASET_ROOT + "fields/";

static const string VALIDATION_DIR =
    DATASET_ROOT + "validation/";


//======================================================
// CUDA ERROR CHECK
//======================================================

static bool checkCuda(
    cudaError_t err,
    const char* operation)
{
    if(err != cudaSuccess)
    {
        cerr
            << "CUDA ERROR: "
            << operation
            << " : "
            << cudaGetErrorString(err)
            << endl;

        return false;
    }

    return true;
}


//======================================================
// CREATE V5 PARAMETERS
//======================================================

static BatteryParameters createV5Parameters(
    const Config& cfg,
    float soc,
    float soh,
    float current)
{
    BatteryParameters params{};

    params.modelVersion =
        MODEL_V5;

    params.current =
        current;

    params.baseResistance =
        cfg.resistance;

    params.referenceTemperature =
        cfg.referenceTemperature;

    params.resistanceCoeff =
        cfg.resistanceCoeff;

    params.diffusivityCoeff =
        cfg.diffusivityCoeff;

    params.entropyCoeff =
        cfg.entropyCoeff;

    params.soc =
        soc;

    params.soh =
        soh;

    params.socResistanceCoeff =
        cfg.socResistanceCoeff;

    params.sohResistanceCoeff =
        cfg.sohResistanceCoeff;

    params.socHeatCoeff =
        cfg.socHeatCoeff;

    params.sohHeatCoeff =
        cfg.sohHeatCoeff;

    return params;
}


//======================================================
// WRITE METADATA HEADER
//======================================================

static void writeMetadataHeader(
    ofstream& file)
{
    file
        << "case_id,"
        << "model,"
        << "grid_size,"
        << "steps,"
        << "soc,"
        << "soh,"
        << "current_A,"
        << "initial_temperature_C,"
        << "reference_temperature_C,"
        << "base_resistance_ohm,"
        << "resistance_coeff,"
        << "diffusivity_coeff,"
        << "entropy_coeff,"
        << "soc_resistance_coeff,"
        << "soh_resistance_coeff,"
        << "soc_heat_coeff,"
        << "soh_heat_coeff,"
        << "heat_profile_size,"
        << "runtime_ms,"
        << "valid"
        << "\n";
}


//======================================================
// WRITE THERMAL METRICS HEADER
//======================================================

static void writeMetricsHeader(
    ofstream& file)
{
    file
        << "case_id,"
        << "T_min_C,"
        << "T_max_C,"
        << "T_mean_C,"
        << "T_center_C,"
        << "T_range_C"
        << "\n";
}


//======================================================
// CALCULATE THERMAL METRICS
//======================================================

static void calculateMetrics(
    const vector<float>& T,
    int N,
    float& Tmin,
    float& Tmax,
    float& Tmean,
    float& Tcenter)
{
    Tmin =
        *min_element(
            T.begin(),
            T.end());

    Tmax =
        *max_element(
            T.begin(),
            T.end());

    double sum = 0.0;

    for(float value : T)
    {
        sum +=
            static_cast<double>(value);
    }

    Tmean =
        static_cast<float>(
            sum /
            static_cast<double>(
                T.size()));

    const int center =
        (N / 2) * N +
        (N / 2);

    Tcenter =
        T[center];
}


//======================================================
// VALIDATE THERMAL FIELD
//======================================================

static bool validateField(
    const vector<float>& T)
{
    if(T.empty())
        return false;

    for(float value : T)
    {
        if(!std::isfinite(value))
            return false;
    }

    return true;
}


//======================================================
// SAVE BINARY THERMAL FIELD
//======================================================

static bool saveThermalField(
    const string& filename,
    const vector<float>& T)
{
    ofstream file(
        filename,
        ios::binary);

    if(!file)
        return false;

    file.write(
        reinterpret_cast<const char*>(
            T.data()),
        static_cast<streamsize>(
            T.size() * sizeof(float)));

    return file.good();
}


//======================================================
// CASE DEFINITION
//======================================================

struct DatasetCase
{
    float soc;
    float soh;
    float current;
    float initialTemperature;
};


//======================================================
// GENERATE ONE V5 CASE
//======================================================

static bool generateCase(
    const Config& cfg,
    int caseID,
    const DatasetCase& datasetCase,
    const vector<float>& heatProfile,
    ofstream& metadataFile,
    ofstream& metricsFile)
{
    const int N =
        DATASET_N;

    const size_t elements =
        static_cast<size_t>(N) *
        static_cast<size_t>(N);

    const size_t bytes =
        elements *
        sizeof(float);


    //--------------------------------------------------
    // Initial thermal field
    //--------------------------------------------------

    vector<float> initialT(
        elements,
        datasetCase.initialTemperature);

    vector<float> finalT(
        elements);


    //--------------------------------------------------
    // Create V5 parameters
    //--------------------------------------------------

    BatteryParameters params =
        createV5Parameters(
            cfg,
            datasetCase.soc,
            datasetCase.soh,
            datasetCase.current);


    //--------------------------------------------------
    // Allocate device buffers
    //--------------------------------------------------

    float* d_T = nullptr;
    float* d_Tnew = nullptr;

    if(!checkCuda(
        cudaMalloc(
            &d_T,
            bytes),
        "cudaMalloc(d_T)"))
    {
        return false;
    }

    if(!checkCuda(
        cudaMalloc(
            &d_Tnew,
            bytes),
        "cudaMalloc(d_Tnew)"))
    {
        cudaFree(d_T);
        return false;
    }


    //--------------------------------------------------
    // Initialize both buffers
    //--------------------------------------------------

    if(!checkCuda(
        cudaMemcpy(
            d_T,
            initialT.data(),
            bytes,
            cudaMemcpyHostToDevice),
        "cudaMemcpy d_T"))
    {
        cudaFree(d_T);
        cudaFree(d_Tnew);
        return false;
    }

    if(!checkCuda(
        cudaMemcpy(
            d_Tnew,
            initialT.data(),
            bytes,
            cudaMemcpyHostToDevice),
        "cudaMemcpy d_Tnew"))
    {
        cudaFree(d_T);
        cudaFree(d_Tnew);
        return false;
    }


    //--------------------------------------------------
    // Run existing V5 GPU solver
    //--------------------------------------------------

    auto start =
        chrono::high_resolution_clock::now();

    runGPU(
        d_T,
        d_Tnew,
        N,
        cfg.alpha,
        DATASET_STEPS,
        cfg.beta,
        params,
        heatProfile);

    cudaError_t err =
        cudaGetLastError();

    if(err != cudaSuccess)
    {
        cerr
            << "Kernel error in case "
            << caseID
            << ": "
            << cudaGetErrorString(err)
            << endl;

        cudaFree(d_T);
        cudaFree(d_Tnew);

        return false;
    }

    err =
        cudaDeviceSynchronize();

    if(err != cudaSuccess)
    {
        cerr
            << "Synchronization error in case "
            << caseID
            << ": "
            << cudaGetErrorString(err)
            << endl;

        cudaFree(d_T);
        cudaFree(d_Tnew);

        return false;
    }

    auto stop =
        chrono::high_resolution_clock::now();

    double runtime_ms =
        chrono::duration<double, milli>(
            stop - start).count();


    //--------------------------------------------------
    // IMPORTANT
    //
    // DATASET_STEPS = 100.
    //
    // runGPU() performs one swap per timestep.
    // 100 is even, therefore the original d_T
    // contains the final field.
    //--------------------------------------------------

    if(!checkCuda(
        cudaMemcpy(
            finalT.data(),
            d_T,
            bytes,
            cudaMemcpyDeviceToHost),
        "copy final field"))
    {
        cudaFree(d_T);
        cudaFree(d_Tnew);

        return false;
    }


    //--------------------------------------------------
    // Validate field
    //--------------------------------------------------

    bool valid =
        validateField(finalT);


    //--------------------------------------------------
    // Metadata
    //--------------------------------------------------

    metadataFile
        << caseID << ","
        << "V5,"
        << N << ","
        << DATASET_STEPS << ","
        << setprecision(9)
        << datasetCase.soc << ","
        << datasetCase.soh << ","
        << datasetCase.current << ","
        << datasetCase.initialTemperature << ","
        << cfg.referenceTemperature << ","
        << cfg.resistance << ","
        << cfg.resistanceCoeff << ","
        << cfg.diffusivityCoeff << ","
        << cfg.entropyCoeff << ","
        << cfg.socResistanceCoeff << ","
        << cfg.sohResistanceCoeff << ","
        << cfg.socHeatCoeff << ","
        << cfg.sohHeatCoeff << ","
        << heatProfile.size() << ","
        << runtime_ms << ","
        << (valid ? 1 : 0)
        << "\n";


    //--------------------------------------------------
    // Save valid field and metrics
    //--------------------------------------------------

    if(valid)
    {
        float Tmin;
        float Tmax;
        float Tmean;
        float Tcenter;

        calculateMetrics(
            finalT,
            N,
            Tmin,
            Tmax,
            Tmean,
            Tcenter);

        metricsFile
            << caseID << ","
            << setprecision(9)
            << Tmin << ","
            << Tmax << ","
            << Tmean << ","
            << Tcenter << ","
            << Tmax - Tmin
            << "\n";


        //------------------------------------------------
        // Save final field
        //------------------------------------------------

        string fieldFile =
            FIELD_DIR +
            "case_" +
            to_string(caseID) +
            "_final.bin";

        if(!saveThermalField(
            fieldFile,
            finalT))
        {
            cerr
                << "WARNING: field save failed for case "
                << caseID
                << endl;
        }
    }


    //--------------------------------------------------
    // Cleanup
    //--------------------------------------------------

    cudaFree(d_T);
    cudaFree(d_Tnew);


    return valid;
}


//======================================================
// PUBLIC DATASET GENERATOR
//======================================================

void generateV5Dataset(
    const Config& cfg,
    const vector<float>& heatProfile)
{
    cout
        << "\n"
        << "=====================================\n"
        << "V5 HIGH-QUALITY DATASET GENERATION\n"
        << "1000-CASE DATASET\n"
        << "=====================================\n";


    //--------------------------------------------------
    // Safety checks
    //--------------------------------------------------

    if(heatProfile.empty())
    {
        cerr
            << "ERROR: V5 dataset generation skipped.\n"
            << "Heat profile is empty.\n";

        return;
    }


    //--------------------------------------------------
    // Create directories
    //--------------------------------------------------

    fs::create_directories(
        DATASET_ROOT);

    fs::create_directories(
        FIELD_DIR);

    fs::create_directories(
        VALIDATION_DIR);


    //--------------------------------------------------
    // Create metadata files
    //--------------------------------------------------

    ofstream metadataFile(
        DATASET_ROOT +
        "metadata.csv");

    ofstream metricsFile(
        DATASET_ROOT +
        "thermal_metrics.csv");

    if(!metadataFile ||
       !metricsFile)
    {
        cerr
            << "ERROR: Cannot create dataset CSV files.\n";

        return;
    }


    writeMetadataHeader(
        metadataFile);

    writeMetricsHeader(
        metricsFile);


    //==================================================
    // RANDOMIZED PARAMETER SPACE
    //==================================================

    const float SOC_MIN = 0.20f;
    const float SOC_MAX = 1.00f;

    const float SOH_MIN = 0.70f;
    const float SOH_MAX = 1.00f;

    const float CURRENT_MIN = 10.0f;
    const float CURRENT_MAX = 30.0f;

    const float TEMP_MIN = 15.0f;
    const float TEMP_MAX = 45.0f;


    //--------------------------------------------------
    // Deterministic random generator
    //--------------------------------------------------

    // Fixed seed = reproducible dataset
    mt19937 generator(42);

    uniform_real_distribution<float> socDistribution(
        SOC_MIN,
        SOC_MAX);

    uniform_real_distribution<float> sohDistribution(
        SOH_MIN,
        SOH_MAX);

    uniform_real_distribution<float> currentDistribution(
        CURRENT_MIN,
        CURRENT_MAX);

    uniform_real_distribution<float> temperatureDistribution(
        TEMP_MIN,
        TEMP_MAX);


    //--------------------------------------------------
    // Generate randomized dataset cases
    //--------------------------------------------------

    vector<DatasetCase> cases;

    cases.reserve(
        DATASET_CASES);

    for(int i = 0;
        i < DATASET_CASES;
        ++i)
    {
        DatasetCase c;

        c.soc =
            socDistribution(generator);

        c.soh =
            sohDistribution(generator);

        c.current =
            currentDistribution(generator);

        c.initialTemperature =
            temperatureDistribution(generator);

        cases.push_back(c);
    }


    //--------------------------------------------------
    // Generate cases
    //--------------------------------------------------

    const int totalCases =
        min(
            DATASET_CASES,
            static_cast<int>(
                cases.size()));

    int successfulCases = 0;


    for(int index = 0;
        index < totalCases;
        ++index)
    {
        const DatasetCase& c =
            cases[index];

        const int caseID =
            index + 1;


        cout
            << "\n-------------------------------------\n"
            << "Case "
            << caseID
            << " / "
            << totalCases
            << "\n"
            << "SOC      : "
            << c.soc
            << "\n"
            << "SOH      : "
            << c.soh
            << "\n"
            << "Current  : "
            << c.current
            << " A\n"
            << "Initial T: "
            << c.initialTemperature
            << " C\n";


        bool success =
            generateCase(
                cfg,
                caseID,
                c,
                heatProfile,
                metadataFile,
                metricsFile);


        if(success)
        {
            successfulCases++;

            cout
                << "Status   : PASS\n";
        }
        else
        {
            cout
                << "Status   : FAIL\n";
        }
    }


    //--------------------------------------------------
    // Close files
    //--------------------------------------------------

    metadataFile.close();
    metricsFile.close();


    //--------------------------------------------------
    // Summary
    //--------------------------------------------------

    cout
        << "\n"
        << "=====================================\n"
        << "V5 DATASET GENERATION COMPLETE\n"
        << "=====================================\n"
        << "Requested cases : "
        << totalCases
        << "\n"
        << "Successful      : "
        << successfulCases
        << "\n"
        << "Failed          : "
        << totalCases - successfulCases
        << "\n"
        << "Grid            : "
        << DATASET_N
        << " x "
        << DATASET_N
        << "\n"
        << "Timesteps       : "
        << DATASET_STEPS
        << "\n"
        << "Heat samples    : "
        << heatProfile.size()
        << "\n"
        << "SOC range       : "
        << SOC_MIN
        << " - "
        << SOC_MAX
        << "\n"
        << "SOH range       : "
        << SOH_MIN
        << " - "
        << SOH_MAX
        << "\n"
        << "Current range   : "
        << CURRENT_MIN
        << " - "
        << CURRENT_MAX
        << " A\n"
        << "Temperature     : "
        << TEMP_MIN
        << " - "
        << TEMP_MAX
        << " C\n"
        << "Output          : "
        << DATASET_ROOT
        << "\n"
        << "=====================================\n";
}