#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ex_llama {
// JEVH0001: little-endian uint32 hidden/classes, then FP32 mu, sd,
// row-major linear.weight and linear.bias. No pickle or executable content.
class JevHead {
public:
    uint32_t hidden = 0;
    uint32_t classes = 0;
    std::vector<float> mu, sd, weights, bias;

    static JevHead load(const std::string &path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) throw std::runtime_error("decision head file unavailable");
        const auto size = file.tellg();
        if (size < 16 || size > 128 * 1024 * 1024)
            throw std::runtime_error("invalid decision head file size");
        file.seekg(0);
        char magic[8]; file.read(magic, 8);
        if (std::memcmp(magic, "JEVH0001", 8))
            throw std::runtime_error("unsupported decision head format");
        JevHead head;
        head.hidden = read_u32(file); head.classes = read_u32(file);
        if (!head.hidden || head.hidden > 65536 || head.classes != 256)
            throw std::runtime_error("invalid decision head dimensions");
        const size_t count = 2 * size_t(head.hidden) + size_t(head.hidden) * head.classes + head.classes;
        if (size_t(size) != 16 + count * 4)
            throw std::runtime_error("decision head length mismatch");
        head.mu = read_floats(file, head.hidden);
        head.sd = read_floats(file, head.hidden);
        head.weights = read_floats(file, size_t(head.hidden) * head.classes);
        head.bias = read_floats(file, head.classes);
        for (float value : head.sd)
            if (value <= 0) throw std::runtime_error("decision head standard deviation must be positive");
        return head;
    }

    std::vector<double> probabilities(const std::vector<float> &features, uint32_t count) const {
        if (features.size() != hidden) throw std::runtime_error("classifier hidden dimension mismatch");
        if (count < 2 || count > classes) throw std::runtime_error("classifier requires 2..256 options");
        std::vector<float> normalized(hidden);
        for (size_t j = 0; j < hidden; ++j) {
            if (!std::isfinite(features[j])) throw std::runtime_error("non-finite classifier hidden state");
            normalized[j] = (features[j] - mu[j]) / sd[j];
            if (!std::isfinite(normalized[j])) throw std::runtime_error("non-finite standardized hidden state");
        }
        std::vector<double> output(count);
        for (size_t i = 0; i < count; ++i) {
            float logit = bias[i];
            for (size_t j = 0; j < hidden; ++j)
                logit += weights[i * hidden + j] * normalized[j];
            if (!std::isfinite(logit)) throw std::runtime_error("non-finite classifier logit");
            output[i] = logit;
        }
        const double maximum = *std::max_element(output.begin(), output.end());
        double total = 0;
        for (double &value : output) { value = std::exp(value - maximum); total += value; }
        for (double &value : output) value /= total;
        return output;
    }

private:
    static uint32_t read_u32(std::istream &input) {
        unsigned char bytes[4];
        if (!input.read(reinterpret_cast<char *>(bytes), 4)) throw std::runtime_error("truncated decision head");
        return uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8) |
               (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);
    }
    static std::vector<float> read_floats(std::istream &input, size_t count) {
        static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "IEEE FP32 required");
        std::vector<float> result(count);
        for (float &value : result) {
            const uint32_t bits = read_u32(input);
            std::memcpy(&value, &bits, sizeof(value));
            if (!std::isfinite(value)) throw std::runtime_error("non-finite decision head parameter");
        }
        return result;
    }
};
}
