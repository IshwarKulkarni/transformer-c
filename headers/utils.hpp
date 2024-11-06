#ifndef UTILS_HPP
#define UTILS_HPP

#include <chrono>
#include "errors.hpp"
#include "logger.hpp"
#include "matrix.cuh"
#include "types"

inline std::string exec_cli(const char* cmd)
{
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe)
    {
        throw_rte_with_backtrace("popen() failed!");
    }
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr)
    {
        result += buffer.data();
    }
    return result;
}

inline bool endswith(const std::string& str, const std::string& tail)
{
    if (str.size() < tail.size()) return false;
    return str.compare(str.size() - tail.size(), tail.size(), tail) == 0;
}

inline bool startswith(const std::string& str, const std::string& head)
{
    if (str.size() < head.size()) return false;
    return str.compare(0, head.size(), head) == 0;
}
struct Timer
{
    std::string name;
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;
    bool stopped = false;
    Timer(const std::string& name) : name(name), t1(std::chrono::high_resolution_clock::now()) {}
    ~Timer()
    {
        if (!stopped)
        {
            stop(true);
        }
    }
    float64 get_duration() const
    {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float64> time_span =
            std::chrono::duration_cast<std::chrono::duration<float64>>(now - t1);
        return time_span.count();
    }
    float64 stop(bool log = false)
    {
        t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float64> time_span =
            std::chrono::duration_cast<std::chrono::duration<float64>>(t2 - t1);
        stopped = true;
        auto span_count = time_span.count();
        if (log) LOG(name, " took ", span_count, "s.");
        return span_count;
    }
};

struct CudaEventTimer
{
    std::string name;
    cudaEvent_t start, end;
    bool stopped = false;
    CudaEventTimer(const std::string& name) : name(name)
    {
        cudaErrCheck(cudaEventCreate(&start));
        cudaErrCheck(cudaEventCreate(&end));
        cudaErrCheck(cudaEventRecord(start, 0));
    }
    ~CudaEventTimer()
    {
        float32 time = stop();
        if (!stopped) LOG(name, " took ", time, " seconds ");
    }
    float32 stop()
    {
        cudaErrCheck(cudaEventRecord(end, 0));
        cudaErrCheck(cudaEventSynchronize(end));
        stopped = true;
        float32 elapsed = 0;
        cudaErrCheck(cudaEventElapsedTime(&elapsed, start, end));
        elapsed /= 1000;  // convert to seconds
        return elapsed;
    }
};

#endif
