#ifndef UTILS_HPP
#define UTILS_HPP

#include <cuda_runtime_api.h>
#include <chrono>
#include "errors.hpp"
#include "logger.hpp"
#include "types"

#define cudaErrCheck(err) cudaErrCheck_((err), __FILE__, __LINE__)

inline void cudaErrCheck_(cudaError_t code, const char* file, uint32 line, bool abort = true)
{
    if (code == cudaSuccess) return;
    LOG(BOLD, RED, "CUDA ERROR: ", code, ", `", cudaGetErrorString(code), "` at ",
        Log::Location{file, line});
    if (abort) throw_rte_with_backtrace("CUDA ERROR")
}

template <typename T>
class Optional
{
    T value;
    bool valid = false;

 public:
    inline __host__ __device__ Optional() : valid(false) {}
    inline __host__ __device__ Optional(const T& val) : value(val), valid(true) {}

    inline __host__ __device__ bool is_valid() const { return valid; }
    inline __host__ __device__ T& get()
    {
        if (valid) return value;
        throw_rte_with_backtrace("Accessing unavaible Optional");
    }
    inline __host__ __device__ const T& get() const
    {
        return value;
        throw_rte_with_backtrace("Accessing unavaible Optional");
    }
    inline __host__ __device__ operator bool() const { return valid; }
    inline __host__ __device__ T& operator*() { return value; }
    inline __host__ __device__ const T& operator*() const { return value; }
    inline __host__ __device__ T* operator->()
    {
        if (valid) return &value;
        throw_rte_with_backtrace("Accessing unavaible Optional");
        return nullptr;
    }
    inline __host__ __device__ const T* operator->() const
    {
        if (valid) return &value;
        throw_rte_with_backtrace("Accessing unavaible Optional");
        return nullptr;
    }
};

inline std::string convertMemorySting(size_t raw_bytes)
{
    double bytes = static_cast<double>(raw_bytes);
    if (bytes < 1024)
    {
        return std::to_string(bytes) + "B";
    }
    else if (bytes < 1024 * 1024)
    {
        return std::to_string(bytes / 1024) + "KB";
    }
    else if (bytes < 1024 * 1024 * 1024)
    {
        return std::to_string(bytes / (1024 * 1024)) + "MB";
    }
    return std::to_string(bytes / (1024 * 1024 * 1024)) + "GB";
}

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

    float64 elapsed() const
    {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float64> time_span =
            std::chrono::duration_cast<std::chrono::duration<float64>>(now - t1);
        return time_span.count();
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
        bool was_stopped = stopped;
        float32 time = stop();
        if (!was_stopped) LOG(name, " took ", time, " seconds ");
    }
    // return seconds
    float64 stop()
    {
        cudaErrCheck(cudaEventRecord(end, 0));
        cudaErrCheck(cudaEventSynchronize(end));
        stopped = true;
        float32 elapsed = 0;
        cudaErrCheck(cudaEventElapsedTime(&elapsed, start, end));
        return float64(elapsed) / 1000.0;  // convert to seconds
    }
};

// Poor man's TQDM
inline std::ostream& progress_bar(uint32 cur, uint32 limit)
{
    using namespace std;
    uint32 pct = static_cast<uint32>((float32(cur) * 100) / static_cast<float32>(limit));
    string bar = "[" + std::string(pct, '=') + ">" + std::string(100 - pct, ' ') + "]";
    static Timer timer("Progress");
    float64 rate = float32(cur) / timer.get_duration();
    float64 eta = float32(limit - cur) / rate;
    cout << "\r" << setw(3) << cur << '/' << limit << " " << setw(4) << pct << "% " << bar << " "
         << setprecision(4) << setw(5) << rate << "it/s | " << setw(6) << eta << "s.";
    return cout;
}

inline std::string num_to_si(float64 num, bool use_pow2 = false)
{
    const char* suffixes_si[] = {"", "K", "M", "G", "T", "P", "E"};
    const char* suffixes_p2[] = {"", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei"};
    const char** suffixes = use_pow2 ? suffixes_p2 : suffixes_si;
    uint32 i = 0;
    const float32 base = use_pow2 ? 1024 : 1000;
    while (num >= base)
    {
        num /= base;
        i++;
    }
    float64 frac = uint64(num) - num;
    char out[32];
    if (std::abs(frac) < 1e-5)
        snprintf(out, sizeof(out), "%lld%s", uint64(num), suffixes[i]);
    else
        snprintf(out, sizeof(out), "%3.2f%s", num, suffixes[i]);
    return out;
}

#endif
