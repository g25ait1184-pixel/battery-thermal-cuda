#pragma once


//------------------------------------------------------
// Battery Model Version
//------------------------------------------------------

enum BatteryModelVersion
{
    MODEL_V1 = 1,
    MODEL_V2,
    MODEL_V3,
    MODEL_V4,
    MODEL_V5
};


//------------------------------------------------------
// Battery Parameters
//------------------------------------------------------

struct BatteryParameters
{
    //------------------------------------------
    // Model selection
    //------------------------------------------

    int modelVersion;


    //------------------------------------------
    // Electrical parameters
    //------------------------------------------

    float current;

    float baseResistance;


    //------------------------------------------
    // Reference temperature
    //------------------------------------------

    float referenceTemperature;


    //------------------------------------------
    // Temperature coefficients
    //------------------------------------------

    float resistanceCoeff;

    float diffusivityCoeff;


    //------------------------------------------
    // Entropy / reversible heat
    //------------------------------------------

    float entropyCoeff;


    //------------------------------------------
    // V5 SOC / SOH state
    //------------------------------------------

    float soc;

    float soh;


    //------------------------------------------
    // V5 SOC-dependent resistance
    //------------------------------------------

    float socResistanceCoeff;


    //------------------------------------------
    // V5 SOH-dependent resistance
    //------------------------------------------

    float sohResistanceCoeff;


    //------------------------------------------
    // V5 SOC-dependent heat
    //------------------------------------------

    float socHeatCoeff;


    //------------------------------------------
    // V5 SOH-dependent heat
    //------------------------------------------

    float sohHeatCoeff;
};


//======================================================
// Function Declarations
//======================================================


//------------------------------------------------------
// Temperature-Dependent Resistance
//------------------------------------------------------

__host__ __device__
float computeResistance(
    float R0,
    float T,
    float Tref,
    float coeff
);


//------------------------------------------------------
// Temperature-Dependent Diffusivity
//------------------------------------------------------

__host__ __device__
float computeDiffusivity(
    float alpha0,
    float T,
    float Tref,
    float coeff
);


//------------------------------------------------------
// Joule Heating
//------------------------------------------------------

__host__ __device__
float computeJouleHeat(
    float current,
    float resistance
);


//------------------------------------------------------
// Entropy / Reversible Heat
//------------------------------------------------------

__host__ __device__
float computeEntropyHeat(
    float current,
    float temperature,
    float entropyCoeff
);


//------------------------------------------------------
// Temperature Resistance Model
//------------------------------------------------------

__host__ __device__
float computeTemperatureResistance(
    float resistance0,
    float temperature,
    float tref,
    float beta
);


//======================================================
// V5 SOC/SOH-Aware Resistance
//
// R_V5 = R(T)
//        * [1 + kSOC(1-SOC)]
//        * [1 + kSOH(1-SOH)]
//
// This is the ONLY function that applies the
// SOC/SOH resistance correction.
//======================================================

__host__ __device__
float computeV5Resistance(
    float R0,
    float temperature,
    float Tref,
    float temperatureCoeff,
    float soc,
    float soh,
    float socResistanceCoeff,
    float sohResistanceCoeff
);


//======================================================
// V5 Authoritative Total Heat
//
// Q_V5 = Q_NASA
//      + Q_Joule
//      + Q_entropy
//      + Q_SOC
//      + Q_SOH
//
// The resistance passed to this function must already
// be the V5 SOC/SOH-aware resistance.
//
// NO resistance multiplier is applied here.
//======================================================

__host__ __device__
float computeV5TotalHeat(
    float nasaHeat,
    float current,
    float resistance,
    float temperature,
    float soc,
    float soh,
    float entropyCoeff,
    float socHeatCoeff,
    float sohHeatCoeff
);


//======================================================
// Total Heat Dispatcher
//
// All CPU/GPU/Tiled/Halo implementations should use
// this function so that the thermal physics remains
// identical across implementations.
//======================================================

__host__ __device__
float computeTotalHeat(
    BatteryModelVersion version,
    float nasaHeat,
    float current,
    float resistance,
    float temperature,
    float soc,
    float entropyCoeff,
    float socResistanceCoeff,
    float sohResistanceCoeff,
    float socHeatCoeff,
    float sohHeatCoeff,
    float soh
);