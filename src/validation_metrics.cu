#include "validation_metrics.h"

#include <fstream>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <algorithm>

using namespace std;


//======================================================
// Calculate Validation Metrics
//======================================================

ValidationMetrics calculateValidationMetrics(
    const vector<float>& reference,
    const vector<float>& test)
{
    ValidationMetrics result;

    //--------------------------------------------------
    // Initialize
    //--------------------------------------------------

    result.rmse = 0.0;
    result.maxAbsoluteError = 0.0;
    result.meanAbsoluteError = 0.0;
    result.relativeError = 0.0;


    //--------------------------------------------------
    // Safety check
    //--------------------------------------------------

    if(reference.empty() ||
       test.empty() ||
       reference.size() != test.size())
    {
        return result;
    }


    //--------------------------------------------------
    // Accumulators
    //--------------------------------------------------

    double squaredError = 0.0;

    double absoluteError = 0.0;

    double referenceNorm = 0.0;

    double maxError = 0.0;


    //--------------------------------------------------
    // Compare CPU and GPU temperature fields
    //--------------------------------------------------

    for(size_t i = 0;
        i < reference.size();
        i++)
    {
        double ref =
            static_cast<double>(
                reference[i]
            );

        double value =
            static_cast<double>(
                test[i]
            );

        double error =
            value - ref;

        double absError =
            fabs(error);


        //------------------------------------------------
        // Squared error
        //------------------------------------------------

        squaredError +=
            error * error;


        //------------------------------------------------
        // Absolute error
        //------------------------------------------------

        absoluteError +=
            absError;


        //------------------------------------------------
        // Maximum absolute error
        //------------------------------------------------

        if(absError > maxError)
        {
            maxError =
                absError;
        }


        //------------------------------------------------
        // Reference norm
        //------------------------------------------------

        referenceNorm +=
            ref * ref;
    }


    //--------------------------------------------------
    // RMSE
    //--------------------------------------------------

    result.rmse =
        sqrt(
            squaredError /
            static_cast<double>(
                reference.size()
            )
        );


    //--------------------------------------------------
    // Mean Absolute Error
    //--------------------------------------------------

    result.meanAbsoluteError =
        absoluteError /
        static_cast<double>(
            reference.size()
        );


    //--------------------------------------------------
    // Maximum Absolute Error
    //--------------------------------------------------

    result.maxAbsoluteError =
        maxError;


    //--------------------------------------------------
    // Relative L2 Error
    //--------------------------------------------------

    if(referenceNorm > 1.0e-12)
    {
        result.relativeError =
            sqrt(squaredError) /
            sqrt(referenceNorm);
    }
    else
    {
        result.relativeError =
            0.0;
    }


    return result;
}


//======================================================
// Save Validation CSV
//======================================================

void saveValidationMetricsCSV(
    const string& filename,
    const vector<string>& modelNames,
    const vector<int>& gridSizes,
    const vector<ValidationMetrics>& results)
{
    //--------------------------------------------------
    // Open file
    //--------------------------------------------------

    ofstream fout(
        filename
    );


    //--------------------------------------------------
    // File check
    //--------------------------------------------------

    if(!fout)
    {
        cout
        << "Failed to create "
        << filename
        << endl;

        return;
    }


    //--------------------------------------------------
    // Header
    //--------------------------------------------------

    fout
    << "Model,"
    << "Grid,"
    << "RMSE(C),"
    << "MaxAbsoluteError(C),"
    << "MeanAbsoluteError(C),"
    << "RelativeError"
    << "\n";


    //--------------------------------------------------
    // Precision
    //--------------------------------------------------

    fout
    << fixed
    << setprecision(8);


    //--------------------------------------------------
    // Number of valid rows
    //--------------------------------------------------

    size_t count =
        min(
            modelNames.size(),
            min(
                gridSizes.size(),
                results.size()
            )
        );


    //--------------------------------------------------
    // Write results
    //--------------------------------------------------

    for(size_t i = 0;
        i < count;
        i++)
    {
        fout
        << modelNames[i]
        << ","
        << gridSizes[i]
        << ","
        << results[i].rmse
        << ","
        << results[i].maxAbsoluteError
        << ","
        << results[i].meanAbsoluteError
        << ","
        << results[i].relativeError
        << "\n";
    }


    //--------------------------------------------------
    // Close
    //--------------------------------------------------

    fout.close();


    cout
    << filename
    << " created"
    << endl;
}