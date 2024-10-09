#ifndef UTILS_HPP
#define UTILS_HPP

#include <chrono>
#include <fstream>
#include <iterator>
#include <vector>
#include "matrix.cuh"

template <typename FloatT>
Matrix<FloatT> read_csv(const std::string& filename)
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
    Timer(const std::string& name) : 
        name(name),
        t1(std::chrono::high_resolution_clock::now()){}
    ~Timer()
    {
        t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        std::cout << name << " Time | " << time_span.count() << " | seconds." << std::endl;
    }
};

#endif
