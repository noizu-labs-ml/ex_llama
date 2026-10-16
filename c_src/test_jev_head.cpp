#include "jev_head.h"
#include <cassert>
#include <cstdio>
#include <functional>
#include <iostream>

static void u32(std::ofstream &file, uint32_t value) {
    for (int i = 0; i < 4; ++i) file.put((value >> (8 * i)) & 255);
}
static void f32(std::ofstream &file, float value) {
    uint32_t bits; std::memcpy(&bits, &value, 4); u32(file, bits);
}
static void fixture(const char *path, float deviation = 2.0f) {
    std::ofstream file(path, std::ios::binary);
    file.write("JEVH0001", 8); u32(file, 2); u32(file, 256);
    f32(file, 1); f32(file, 0); f32(file, deviation); f32(file, 1);
    for (int i = 0; i < 256; ++i) { f32(file, i == 0 ? 1 : 0); f32(file, i == 1 ? 1 : 0); }
    for (int i = 0; i < 256; ++i) f32(file, i == 255 ? 10000 : 0);
}
static void rejects(std::function<void()> call) {
    bool rejected = false;
    try { call(); } catch (const std::runtime_error &) { rejected = true; }
    assert(rejected);
}
int main() {
    const char *path = "/tmp/ex_llama_jev_head_fixture.bin";
    fixture(path);
    const auto head = ex_llama::JevHead::load(path);
    const auto probabilities = head.probabilities({3, 0}, 2);
    assert(probabilities.size() == 2);
    assert(std::abs(probabilities[0] - 0.7310585786300049) < 1e-12);
    assert(std::abs(probabilities[0] + probabilities[1] - 1) < 1e-12);
    // The enormous unused slot must not affect a two-choice softmax.
    assert(head.probabilities({3, 0}, 256)[255] == 1.0);
    rejects([&] { head.probabilities({3}, 2); });
    rejects([&] { head.probabilities({3, 0}, 1); });
    rejects([&] { head.probabilities({3, 0}, 257); });
    rejects([&] { head.probabilities({std::numeric_limits<float>::infinity(), 0}, 2); });
    fixture(path, 0);
    rejects([&] { ex_llama::JevHead::load(path); });
    fixture(path, std::numeric_limits<float>::quiet_NaN());
    rejects([&] { ex_llama::JevHead::load(path); });
    { std::ofstream file(path, std::ios::binary); file.write("wrong", 5); }
    rejects([&] { ex_llama::JevHead::load(path); });
    std::remove(path);
    std::cout << "Jev head normalization, masking, stable softmax and invalid-input checks passed\n";
}
