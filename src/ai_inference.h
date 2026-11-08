#pragma once

#include <array>
#include <string>
#include <vector>

class AIInference
{
public:
    bool load(const std::string& jsonPath);
    bool isLoaded() const;

    bool predictBase(const std::array<float, 4>& input, float& predictionC) const;
    bool predictHybrid(const std::array<float, 4>& input,
                       float& basePredictionC,
                       float& residualCorrectionC,
                       float& hybridPredictionC) const;

    bool isWithinValidatedDomain(const std::array<float, 4>& input) const;

private:
    struct Layer
    {
        int inputSize = 0;
        int outputSize = 0;
        std::vector<float> weights; // PyTorch Linear layout: [output][input]
        std::vector<float> bias;
    };

    bool loaded_ = false;
    std::array<float, 4> xMean_{};
    std::array<float, 4> xScale_{};
    float yMean_ = 0.0f;
    float yScale_ = 1.0f;
    std::vector<Layer> baseLayers_;
    std::vector<Layer> residualLayers_;

    static bool readNumberArray(const std::string& json,
                                const std::string& key,
                                std::vector<float>& values);
    static bool readNetwork(const std::string& json,
                            const std::string& networkKey,
                            const std::array<int, 5>& dimensions,
                            std::vector<Layer>& layers);
    static bool forward(const std::vector<Layer>& layers,
                        const std::array<float, 4>& input,
                        float& output);
};
