#include "ai_inference.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>

namespace
{
std::string extractJsonArray(const std::string& json, const std::string& key)
{
    const std::string token = "\"" + key + "\"";
    const std::size_t keyPos = json.find(token);
    if (keyPos == std::string::npos) return {};
    const std::size_t colon = json.find(':', keyPos + token.size());
    if (colon == std::string::npos) return {};
    const std::size_t start = json.find('[', colon + 1);
    if (start == std::string::npos) return {};

    int depth = 0;
    for (std::size_t i = start; i < json.size(); ++i) {
        if (json[i] == '[') ++depth;
        else if (json[i] == ']') {
            --depth;
            if (depth == 0) return json.substr(start, i - start + 1);
        }
    }
    return {};
}

bool parseNumbers(const std::string& text, std::vector<float>& values)
{
    values.clear();
    const char* p = text.c_str();
    const char* end = p + text.size();
    while (p < end) {
        if ((*p >= '0' && *p <= '9') || *p == '-' || *p == '+') {
            char* next = nullptr;
            const double v = std::strtod(p, &next);
            if (next == p) { ++p; continue; }
            values.push_back(static_cast<float>(v));
            p = next;
        } else {
            ++p;
        }
    }
    return !values.empty();
}

std::string extractObject(const std::string& json, const std::string& key)
{
    const std::string token = "\"" + key + "\"";
    const std::size_t keyPos = json.find(token);
    if (keyPos == std::string::npos) return {};
    const std::size_t colon = json.find(':', keyPos + token.size());
    if (colon == std::string::npos) return {};
    const std::size_t start = json.find('{', colon + 1);
    if (start == std::string::npos) return {};
    int depth = 0;
    for (std::size_t i = start; i < json.size(); ++i) {
        if (json[i] == '{') ++depth;
        else if (json[i] == '}') {
            --depth;
            if (depth == 0) return json.substr(start, i - start + 1);
        }
    }
    return {};
}
}

bool AIInference::readNumberArray(const std::string& json,
                                  const std::string& key,
                                  std::vector<float>& values)
{
    const std::string arrayText = extractJsonArray(json, key);
    return !arrayText.empty() && parseNumbers(arrayText, values);
}

bool AIInference::readNetwork(const std::string& json,
                              const std::string& networkKey,
                              const std::array<int, 5>& dimensions,
                              std::vector<Layer>& layers)
{
    const std::string network = extractObject(json, networkKey);
    if (network.empty()) return false;

    layers.clear();
    // Parse each layer object in order. The exporter writes four layer objects.
    std::size_t searchPos = 0;
    for (int i = 0; i < 4; ++i) {
        const std::size_t weightKey = network.find("\"weight\"", searchPos);
        if (weightKey == std::string::npos) return false;
        const std::size_t weightColon = network.find(':', weightKey);
        const std::size_t weightStart = network.find('[', weightColon);
        if (weightStart == std::string::npos) return false;
        int depth = 0;
        std::size_t weightEnd = std::string::npos;
        for (std::size_t j = weightStart; j < network.size(); ++j) {
            if (network[j] == '[') ++depth;
            else if (network[j] == ']' && --depth == 0) { weightEnd = j; break; }
        }
        if (weightEnd == std::string::npos) return false;

        std::vector<float> weights;
        if (!parseNumbers(network.substr(weightStart, weightEnd - weightStart + 1), weights))
            return false;

        const std::size_t biasKey = network.find("\"bias\"", weightEnd);
        if (biasKey == std::string::npos) return false;
        const std::size_t biasColon = network.find(':', biasKey);
        const std::size_t biasStart = network.find('[', biasColon);
        const std::size_t biasEnd = network.find(']', biasStart);
        if (biasStart == std::string::npos || biasEnd == std::string::npos) return false;

        std::vector<float> bias;
        if (!parseNumbers(network.substr(biasStart, biasEnd - biasStart + 1), bias))
            return false;

        Layer layer;
        layer.inputSize = dimensions[i];
        layer.outputSize = dimensions[i + 1];
        layer.weights = std::move(weights);
        layer.bias = std::move(bias);
        if (layer.weights.size() != static_cast<std::size_t>(layer.inputSize * layer.outputSize) ||
            layer.bias.size() != static_cast<std::size_t>(layer.outputSize))
            return false;

        layers.push_back(std::move(layer));
        searchPos = biasEnd + 1;
    }
    return true;
}

bool AIInference::load(const std::string& jsonPath)
{
    loaded_ = false;
    std::ifstream file(jsonPath);
    if (!file) {
        std::cerr << "ERROR: Cannot open AI model parameters: " << jsonPath << '\n';
        return false;
    }
    const std::string json((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());

    std::vector<float> xMean, xScale;
    const std::string preprocessing = extractObject(json, "preprocessing");
    auto readScalar = [&preprocessing](const std::string& key, float& value) -> bool {
        const std::string token = "\"" + key + "\"";
        const std::size_t keyPos = preprocessing.find(token);
        if (keyPos == std::string::npos) return false;
        const std::size_t colon = preprocessing.find(':', keyPos + token.size());
        if (colon == std::string::npos) return false;
        const char* begin = preprocessing.c_str() + colon + 1;
        char* end = nullptr;
        value = std::strtof(begin, &end);
        return end != begin && std::isfinite(value);
    };
    if (preprocessing.empty() ||
        !readNumberArray(preprocessing, "x_mean", xMean) ||
        !readNumberArray(preprocessing, "x_scale", xScale) ||
        !readScalar("y_mean", yMean_) || !readScalar("y_scale", yScale_) ||
        xMean.size() != 4 || xScale.size() != 4) {
        std::cerr << "ERROR: Invalid scaler data in AI model JSON.\n";
        return false;
    }

    for (int i = 0; i < 4; ++i) {
        if (!std::isfinite(xMean[i]) || !std::isfinite(xScale[i]) || xScale[i] == 0.0f)
            return false;
        xMean_[i] = xMean[i];
        xScale_[i] = xScale[i];
    }
    if (!std::isfinite(yMean_) || !std::isfinite(yScale_) || yScale_ == 0.0f)
        return false;

    if (!readNetwork(json, "base_mlp", {4, 32, 32, 16, 1}, baseLayers_) ||
        !readNetwork(json, "residual_mlp", {4, 16, 16, 8, 1}, residualLayers_)) {
        std::cerr << "ERROR: Invalid MLP weights/biases in AI model JSON.\n";
        return false;
    }

    loaded_ = true;
    return true;
}

bool AIInference::isLoaded() const { return loaded_; }

bool AIInference::forward(const std::vector<Layer>& layers,
                          const std::array<float, 4>& input,
                          float& output)
{
    if (layers.size() != 4) return false;
    std::vector<float> current(input.begin(), input.end());
    for (std::size_t l = 0; l < layers.size(); ++l) {
        const Layer& layer = layers[l];
        if (current.size() != static_cast<std::size_t>(layer.inputSize)) return false;
        std::vector<float> next(static_cast<std::size_t>(layer.outputSize), 0.0f);
        for (int o = 0; o < layer.outputSize; ++o) {
            double sum = layer.bias[o];
            for (int i = 0; i < layer.inputSize; ++i)
                sum += static_cast<double>(layer.weights[o * layer.inputSize + i]) * current[i];
            float value = static_cast<float>(sum);
            if (l < 3 && value < 0.0f) value = 0.0f; // ReLU on hidden layers
            if (!std::isfinite(value)) return false;
            next[o] = value;
        }
        current.swap(next);
    }
    if (current.size() != 1) return false;
    output = current[0];
    return std::isfinite(output);
}

bool AIInference::predictBase(const std::array<float, 4>& input, float& predictionC) const
{
    if (!loaded_) return false;
    std::array<float, 4> normalized{};
    for (int i = 0; i < 4; ++i) {
        if (!std::isfinite(input[i])) return false;
        normalized[i] = (input[i] - xMean_[i]) / xScale_[i];
    }
    float standardized = 0.0f;
    if (!forward(baseLayers_, normalized, standardized)) return false;
    predictionC = standardized * yScale_ + yMean_;
    return std::isfinite(predictionC);
}

bool AIInference::predictHybrid(const std::array<float, 4>& input,
                                float& basePredictionC,
                                float& residualCorrectionC,
                                float& hybridPredictionC) const
{
    if (!loaded_) return false;
    std::array<float, 4> normalized{};
    for (int i = 0; i < 4; ++i) {
        if (!std::isfinite(input[i])) return false;
        normalized[i] = (input[i] - xMean_[i]) / xScale_[i];
    }
    float standardizedBase = 0.0f;
    if (!forward(baseLayers_, normalized, standardizedBase) ||
        !forward(residualLayers_, normalized, residualCorrectionC)) return false;
    basePredictionC = standardizedBase * yScale_ + yMean_;
    hybridPredictionC = basePredictionC + residualCorrectionC;
    return std::isfinite(basePredictionC) && std::isfinite(residualCorrectionC) &&
           std::isfinite(hybridPredictionC);
}

bool AIInference::isWithinValidatedDomain(const std::array<float, 4>& x) const
{
    // Validated operating domain used in the V5 AI experiments.
    return std::isfinite(x[0]) && std::isfinite(x[1]) &&
           std::isfinite(x[2]) && std::isfinite(x[3]) &&
           x[0] >= 0.20f && x[0] <= 1.00f &&
           x[1] >= 0.70f && x[1] <= 1.00f &&
           x[2] >= 10.0f && x[2] <= 30.0f &&
           x[3] >= 15.0f && x[3] <= 45.0f;
}
