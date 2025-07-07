/*
 * Author: Ishwar Kulkarni
 * This file is distributed under the MIT license.
 * See: https://mit-license.org
 */

#ifndef UTILS_HPP
#define UTILS_HPP

#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <vector>
#include "errors.hpp"
#include "logger.hpp"
#include "types"

#define cudaErrCheck(err) cudaErrCheck_((err), __FILE__, __LINE__)

inline void cudaErrCheck_(cudaError_t code, const char* file, uint32 line, bool abort = true)
{
    if (code == cudaSuccess) return;
    LOG(BOLD, RED, "CUDA ERROR: ", code, ", `", cudaGetErrorString(code), "` at ", file, ":", line);
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

    inline __host__ __device__ T value_or(T val)
    {
        if (valid) return value;
        return val;
    }

    inline __host__ __device__ const T& get_or(T val) const
    {
        if (valid) return value;
        return val;
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
    setlocale(LC_NUMERIC, "");
    double bytes = static_cast<double>(raw_bytes);
    char buffer[128];
    if (bytes < 1024)
    {
        snprintf(buffer, sizeof(buffer), "%'3.2f B", bytes);
    }
    else if (bytes < 1024 * 1024)
    {
        snprintf(buffer, sizeof(buffer), "%'3.2f KB", bytes / 1024);
    }
    else if (bytes < 1024 * 1024 * 1024)
    {
        snprintf(buffer, sizeof(buffer), "%'3.2f MB", bytes / (1024 * 1024));
    }
    else
    {
        snprintf(buffer, sizeof(buffer), "%'3.2f GB", bytes / (1024 * 1024 * 1024));
    }
    return std::string(buffer);
}

inline float32 levenshteinDistance(const std::string& s1, const std::string& s2)
{
    const uint32 len1 = s1.size();
    const uint32 len2 = s2.size();
    std::vector<std::vector<uint32>> dp(len1 + 1, std::vector<uint32>(len2 + 1));

    // Initialize first row &&column
    for (uint32 i = 0; i <= len1; i++) dp[i][0] = i;
    for (uint32 j = 0; j <= len2; j++) dp[0][j] = j;

    // Fill dp table
    for (uint32 i = 1; i <= len1; i++)
    {
        for (uint32 j = 1; j <= len2; j++)
        {
            if (s1[i - 1] == s2[j - 1])
            {
                dp[i][j] = dp[i - 1][j - 1];
            }
            else
            {
                dp[i][j] = 1 + std::min({dp[i - 1][j],        // deletion
                                         dp[i][j - 1],        // insertion
                                         dp[i - 1][j - 1]});  // substitution
            }
        }
    }

    // Return normalized distance between 0 &&1
    return static_cast<float32>(dp[len1][len2]) / std::max(len1, len2);
}

template <typename Iter>
std::pair<std::string, float32> get_closest_match(Iter beg, Iter end, const std::string& str)
{
    auto closest = *std::min_element(beg, end, [&](const auto& a, const auto& b) {
        return levenshteinDistance(a, str) < levenshteinDistance(b, str);
    });
    float32 dist = levenshteinDistance(closest, str);
    return std::make_pair(closest, dist);
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
    using clock = std::chrono::high_resolution_clock;
    using time_point = clock::time_point;
    using duration = std::chrono::duration<float64>;
    static constexpr time_point epoch = time_point();
    time_point t1{epoch};
    time_point t2{epoch};
    time_point checkpoint{epoch};
    bool stopped = false;
    Timer(const std::string& name) : name(name), t1(clock::now()) {}
    ~Timer()
    {
        if (!stopped)
        {
            stop(true);
        }
    }
    float64 get_duration() const
    {
        auto now = clock::now();
        duration time_span = std::chrono::duration_cast<duration>(now - t1);
        return time_span.count();
    }
    duration stop(bool log = false)
    {
        t2 = clock::now();
        duration time_span = std::chrono::duration_cast<duration>(t2 - t1);
        stopped = true;
        if (log) LOG(name, " took ", time_span);
        return time_span;
    }

    duration elapsed(time_point since = time_point()) const
    {
        if (since == time_point()) since = t1;
        auto now = clock::now();
        duration time_span = std::chrono::duration_cast<duration>(now - since);
        return time_span;
    }

    // get time, &&checkpoint the time in m_checkpoints
    duration check()
    {
        duration time = elapsed(checkpoint);
        checkpoint = clock::now();
        return time;
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
         << setprecision(4) << setw(5) << rate << "it/s | " << setw(6) << eta << "s."
         << std::string(20, ' ');
    if (cur >= limit) cout << '\n';
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
