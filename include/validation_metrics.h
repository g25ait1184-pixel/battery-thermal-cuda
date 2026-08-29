#pragma once

#include <vector>
#include <string>

//======================================================
// CPU vs GPU Validation Metrics
//======================================================

struct ValidationMetrics
{
    double rmse;
    double maxAbsoluteError;
    double relativeError;
    double meanAbsoluteError;
};

//======================================================
// Calculate CPU-GPU error
//======================================================

ValidationMetrics calculateValidationMetrics(
    const std::vector<float>& reference,
    const std::vector<float>& test
);

//======================================================
// Save validation results
//======================================================

void saveValidationMetricsCSV(
    const std::string& filename,
    const std::vector<std::string>& modelNames,
    const std::vector<int>& gridSizes,
    const std::vector<ValidationMetrics>& results
);