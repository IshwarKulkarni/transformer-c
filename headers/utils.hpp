#ifndef UTILS_HPP
#define UTILS_HPP

#include "logger.hpp"
#include "matrix.cuh"
#include <chrono>
#include <fstream>
#include <iterator>
#include <vector>

template <typename FloatT> Matrix<FloatT> read_csv(const std::string& filename)
{
    std::ifstream file(filename, std::ios::in);
    uint32_t m, n;
    file >> m >> n;
    std::vector<FloatT> data(m * n);
    std::copy(std::istream_iterator<float>(file), std::istream_iterator<float>(), data.begin());
    Matrix<FloatT> matrix(m, n, data.data());
    return matrix;
}

struct Timer
{
    std::string name;
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;
    Timer(const std::string& name) : name(name), t1(std::chrono::high_resolution_clock::now()) {}
    ~Timer()
    {
        auto span_count = stop();
        LOG(name, "took", span_count, "seconds");
    }
    float64 get_duration() const
    {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float64> time_span =
            std::chrono::duration_cast<std::chrono::duration<float64>>(now - t1);
        return time_span.count();
    }
    float64 stop()
    {
        t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float64> time_span =
            std::chrono::duration_cast<std::chrono::duration<float64>>(t2 - t1);
        return time_span.count();
    }
};

#endif
