#include "battery_model.h"

#include <math.h>


//======================================================
// Temperature-Dependent Resistance
//======================================================

__host__ __device__
float computeResistance(
    float R0,
    float T,
    float Tref,
    float coeff)
{
    float deltaT = T - Tref;

    float R =
        R0 *
        (1.0f + coeff * deltaT);

    // Prevent non-physical negative resistance
    if(R < 0.0f)
        R = 0.0f;

    return R;
}


//======================================================
// Temperature-Dependent Diffusivity
//======================================================

__host__ __device__
float computeDiffusivity(
    float alpha0,
    float T,
    float Tref,
    float coeff)
{
    float deltaT = T - Tref;

    float alpha =
        alpha0 *
        (1.0f + coeff * deltaT);

    // Prevent non-physical negative diffusivity
    if(alpha < 0.0f)
        alpha = 0.0f;

    return alpha;
}


//======================================================
// Joule Heating
//======================================================

__host__ __device__
float computeJouleHeat(
    float current,
    float resistance)
{
    return
        current *
        current *
        resistance;
}


//======================================================
// Entropy / Reversible Heat
//======================================================

__host__ __device__
float computeEntropyHeat(
    float current,
    float temperature,
    float entropyCoeff)
{
    return
        current *
        temperature *
        entropyCoeff;
}


//======================================================
// Temperature Resistance Model
//======================================================

__host__ __device__
float computeTemperatureResistance(
    float resistance0,
    float temperature,
    float tref,
    float beta)
{
    float R =
        resistance0 *
        (1.0f +
         beta *
         (temperature - tref));

    if(R < 0.0f)
        R = 0.0f;

    return R;
}


//======================================================
// V5 SOC/SOH-Aware Resistance
//
// R_V5 = R(T)
//        * [1 + kSOC(1-SOC)]
//        * [1 + kSOH(1-SOH)]
//
// IMPORTANT:
// This is the ONLY place where the SOC/SOH
// resistance multiplier is applied.
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
    float sohResistanceCoeff)
{
    //--------------------------------------------------
    // Temperature-dependent base resistance
    //--------------------------------------------------

    float resistance =
        computeResistance(
            R0,
            temperature,
            Tref,
            temperatureCoeff
        );


    //--------------------------------------------------
    // SOC resistance factor
    //
    // SOC = 1.0 -> factor = 1.0
    // SOC < 1.0 -> resistance increases
    //--------------------------------------------------

    float socFactor =
        1.0f +
        socResistanceCoeff *
        (1.0f - soc);


    //--------------------------------------------------
    // SOH resistance factor
    //
    // SOH = 1.0 -> factor = 1.0
    // SOH < 1.0 -> resistance increases
    //--------------------------------------------------

    float sohFactor =
        1.0f +
        sohResistanceCoeff *
        (1.0f - soh);


    //--------------------------------------------------
    // Combined V5 resistance
    //--------------------------------------------------

    resistance *=
        socFactor *
        sohFactor;


    //--------------------------------------------------
    // Physical protection
    //--------------------------------------------------

    if(resistance < 0.0f)
        resistance = 0.0f;

    return resistance;
}


//======================================================
// V5 Total Heat
//
// AUTHORITATIVE V5 EQUATION:
//
// Q_V5 = Q_NASA
//      + Q_Joule
//      + Q_entropy
//      + Q_SOC
//      + Q_SOH
//
// where:
//
// R_V5 = R(T)
//        * SOC_factor
//        * SOH_factor
//
// Q_Joule = I^2 * R_V5
//
// Q_SOC = Q_Joule
//         * kSOC_heat
//         * (1-SOC)
//
// Q_SOH = Q_Joule
//         * kSOH_heat
//         * (1-SOH)
//
// IMPORTANT:
// The resistance multiplier is NOT applied again here.
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
    float sohHeatCoeff)
{
    //--------------------------------------------------
    // Joule heat using already-coupled V5 resistance
    //--------------------------------------------------

    float jouleHeat =
        computeJouleHeat(
            current,
            resistance
        );


    //--------------------------------------------------
    // Entropy / reversible heat
    //--------------------------------------------------

    float entropyHeat =
        computeEntropyHeat(
            current,
            temperature,
            entropyCoeff
        );


    //--------------------------------------------------
    // SOC-dependent heat
    //--------------------------------------------------

    float socHeat =
        jouleHeat *
        socHeatCoeff *
        (1.0f - soc);


    //--------------------------------------------------
    // SOH-dependent heat
    //--------------------------------------------------

    float sohHeat =
        jouleHeat *
        sohHeatCoeff *
        (1.0f - soh);


    //--------------------------------------------------
    // Authoritative V5 total heat
    //--------------------------------------------------

    return
        nasaHeat
        +
        jouleHeat
        +
        entropyHeat
        +
        socHeat
        +
        sohHeat;
}


//======================================================
// Total Heat Dispatcher
//
// CPU / GPU / Tiled / Halo must all use this
// dispatcher so that the physics is identical.
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
    float soh)
{
    //--------------------------------------------------
    // V5
    //--------------------------------------------------

    if(version == MODEL_V5)
    {
        //------------------------------------------------
        // IMPORTANT:
        //
        // resistance passed here must already be the
        // V5 SOC/SOH-aware resistance.
        //
        // Do NOT apply resistance scaling here again.
        //------------------------------------------------

        return
            computeV5TotalHeat(
                nasaHeat,
                current,
                resistance,
                temperature,
                soc,
                soh,
                entropyCoeff,
                socHeatCoeff,
                sohHeatCoeff
            );
    }


    //--------------------------------------------------
    // V1-V4
    //--------------------------------------------------

    float jouleHeat =
        computeJouleHeat(
            current,
            resistance
        );


    float entropyHeat =
        computeEntropyHeat(
            current,
            temperature,
            entropyCoeff
        );


    //--------------------------------------------------
    // Model selection
    //--------------------------------------------------

    switch(version)
    {
        //------------------------------------------------
        // V1
        // NASA heat only
        //------------------------------------------------

        case MODEL_V1:

            return
                nasaHeat;


        //------------------------------------------------
        // V2
        // NASA + Joule heating
        //------------------------------------------------

        case MODEL_V2:

            return
                nasaHeat
                +
                jouleHeat;


        //------------------------------------------------
        // V3
        // NASA + Joule + entropy
        //------------------------------------------------

        case MODEL_V3:

            return
                nasaHeat
                +
                jouleHeat
                +
                entropyHeat;


        //------------------------------------------------
        // V4
        // NASA + Joule + entropy
        // with temperature-dependent resistance
        //
        // Resistance is calculated by caller.
        //------------------------------------------------

        case MODEL_V4:

            return
                nasaHeat
                +
                jouleHeat
                +
                entropyHeat;


        //------------------------------------------------
        // Safe fallback
        //------------------------------------------------

        default:

            return
                nasaHeat;
    }
}