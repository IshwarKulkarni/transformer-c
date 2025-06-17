/*
 * Author: Ishwar Kulkarni
 * This file is distributed under the MIT license.
 * See: https://mit-license.org
 */

#ifndef LOGGER_H
#define LOGGER_H

#include <cuda_runtime_api.h>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <ostream>
#include <set>

namespace Log {

#define BLUE "\033[1;34m"
#define GREEN "\033[1;32m"
#define RED "\033[0;31m"
#define YELLOW "\033[1;33m"
#define MAGENTA "\033[1;35m"
#define CYAN "\033[1;36m"
#define WHITE "\033[1;37m"
#define BLACK "\033[1;30m"
#define ORANGE "\033[1;31m"
#define GRAY "\033[1;30m"
#define RESET "\033[0m"
#define BOLD "\033[1m"

struct Location
{
    const char* file;
    const uint32_t line;
};

enum class LogLevel  // not used
{
    LOG_LEVEL = 0,
    DEBUG_LEVEL = 1,
    INFO_LEVEL = 2,
    WARN_LEVEL = 3,
    ERROR_LEVEL = 4,
};

class Logger
{
    // add mutex for threads here.
    std::mutex ostream_mtx;
    char time_str[256];

 public:
    void inline tee(const std::string& filename)
    {
        if (!m_tee_file.is_open())
        {
            m_tee_file.open(filename);
        }
    }

    template <typename First>
    inline std::ostream& log_v(std::ostream& strm, const First& arg1)
    {
        std::lock_guard<std::mutex> guard(ostream_mtx);
        return (strm << arg1);
    }

    inline std::ostream& log_v(std::ostream& strm) { return strm; }

    template <typename... Args>
    inline std::ostream& log_v(std::ostream& strm, const Location& arg1, const Args&... args)
    {
        if (disabled_files.count(arg1.file)) return strm;
        char file_loc[48];
        std::string file;
        static constexpr int max_len = sizeof(file_loc) - 12;
        unsigned int len = strlen(arg1.file);
        const char* prev_slash = arg1.file;
        const char* end = arg1.file + len - 1;
        if (len > max_len)  //  clip from slash that keeps the length below max_len;
            for (int i = 0; i < max_len && end != arg1.file; i++, end--)
                if (*end == '/') prev_slash = end + 1;
        if (file[0] == '/') file = file.substr(1);
        snprintf(file_loc, sizeof(file_loc), " %*.*s:%-4d ", max_len, max_len, prev_slash,
                 arg1.line);
        return this->log_v(strm << file_loc, args...);
    }

    template <typename First, typename... Args>
    inline std::ostream& log_v(std::ostream& strm, const First& arg1, const Args&... args)
    {
        return this->log_v((strm << arg1), args...);
    }

    void flush()
    {
        if (m_tee_file.is_open())
        {
            m_tee_file.flush();
        }
    }

    template <typename... Args>
    inline void log(const Args&... args)
    {
        std::cout << get_time_str() << RESET << " ";
        if (m_tee_file.is_open())
        {
            m_tee_file << get_time_str() << RESET << " ";
            this->log_v(m_tee_file, args...) << RESET << std::endl;
        }
        this->log_v(std::cout, args...) << RESET << std::endl;
    }

    inline static Logger& get()
    {
        static Logger instance;
        return instance;
    }

    const char* get_time_str()
    {
        time_t rawtime;
        struct tm* timeinfo;
        time(&rawtime);
        timeinfo = localtime(&rawtime);
        strftime(time_str, sizeof(time_str), GRAY "%Y-%m-%d %H:%M:%S", timeinfo);
        return time_str;
    }

    void disable(const std::string& filename) { disabled_files.insert(filename); }

    void enable(const std::string& filename) { disabled_files.erase(filename); }

    ~Logger() = default;

 private:
    std::ofstream m_tee_file;
    std::set<std::string> disabled_files;
    Logger() = default;
};

}  // namespace Log

inline std::ostream& operator<<(std::ostream& strm, const dim3& dim)
{
    return strm << "(" << dim.x << ", " << dim.y << ", " << dim.z << ")";
}

// Claude generated:
inline std::ostream& operator<<(std::ostream& strm, const std::chrono::duration<double>& duration)
{
    double seconds = duration.count();
    auto prec = strm.precision();
    auto flags = strm.flags();

    auto return_strm = [&]() -> std::ostream& {
        strm.precision(prec);
        strm.flags(flags);
        return strm;
    };

    // Handle microseconds (under 1ms)
    if (seconds < 0.001)
    {
        strm << std::fixed << std::setprecision(1) << (seconds * 1000000) << "us";
        return return_strm();
    }

    // Handle milliseconds (under 1s)
    if (seconds < 1.0)
    {
        strm << std::fixed << std::setprecision(2) << (seconds * 1000) << "ms";
        return return_strm();
    }

    // Handle seconds (under 2 mins)
    if (seconds < 120)
    {
        strm << std::fixed << std::setprecision(2) << seconds << "s";
        return return_strm();
    }

    // Handle minutes (under 1 hour)
    if (seconds < 3600)
    {
        int mins = static_cast<int>(seconds) / 60;
        double secs = seconds - (mins * 60);
        strm << mins << "m:" << std::fixed << std::setprecision(1) << secs << "s";
        return return_strm();
    }

    // Handle hours
    int hours = static_cast<int>(seconds) / 3600;
    int mins = (static_cast<int>(seconds) % 3600) / 60;
    double secs = seconds - (hours * 3600) - (mins * 60);

    strm << std::setfill('0') << std::setw(2) << hours << ":" << std::setfill('0') << std::setw(2)
         << mins << ":" << std::fixed << std::setprecision(1) << secs << "s";
    return return_strm();
}

#ifdef LOG_NODE_CREATION_ON
#define LOG_NODE_CREATION(...) \
    Log::Logger::get().log(Log::Location{__FILE__, __LINE__}, " NODE CREATION: ", __VA_ARGS__)
#else
#define LOG_NODE_CREATION(...)
#endif

#ifdef LOG_ALLOC_ON
#define LOG_ALLOC(...) \
    Log::Logger::get().log(Log::Location{__FILE__, __LINE__}, GREEN, " ALLOC: ", RESET, __VA_ARGS__)
#define LOG_FREE(...) \
    Log::Logger::get().log(Log::Location{__FILE__, __LINE__}, RED, " FREE: ", RESET, __VA_ARGS__)
#else
#define LOG_ALLOC(...)
#define LOG_FREE(...)
#endif

#ifdef LOG_NODE_TRACE_ON
#define LOG_NODE_TRACE(...) \
    Log::Logger::get().log(Log::Location{__FILE__, __LINE__}, " NODE TRACE #, " __VA_ARGS__)
#else
#define LOG_NODE_TRACE(...)
#endif

#ifdef LOG_PARAM_UPDATE_ON
#define LOG_PARAM_UPDATE(...) \
    Log::Logger::get().log(Log::Location{__FILE__, __LINE__}, " PARAM UPDATE #, " __VA_ARGS__)
#else
#define LOG_PARAM_UPDATE(...)
#endif

#ifdef LOG_MATRIX_OPS_ON
#define LOG_MATRIX_OPS(...) \
    Log::Logger::get().log(Log::Location{__FILE__, __LINE__}, " OP #, ", __VA_ARGS__)
#else
#define LOG_MATRIX_OPS(...)
#endif

#ifdef LOG_MATRIX_CREATE_ON
#define LOG_MATRIX_CREATE(...) \
    Log::Logger::get().log(Log::Location{__FILE__, __LINE__}, "Created #, ", __VA_ARGS__)
#else
#define LOG_MATRIX_CREATE(...)
#endif

#ifdef LOG_KERNEL_SIZE_ON
#define LOG_KERNEL_SIZE(...) \
    Log::Logger::get().log(Log::Location{__FILE__, __LINE__}, " Kernel Size #, ", __VA_ARGS__)
#else
#define LOG_KERNEL_SIZE(...)
#endif

#define R_JUST(x, n) std::setw(n), std::setfill(' '), std::right, x, " "
#define L_JUST(x, n) std::setw(n), std::setfill(' '), std::left, x, " "
#define DISABLE_LOG_FOR_FILE Log::Logger::get().disable(__FILE__);
#define ENABLE_LOG_FOR_FILE Log::Logger::get().enable(__FILE__);
#define LOG_NOLOC(...) Log::Logger::get().log(__VA_ARGS__)
#define LOG(...) Log::Logger::get().log(Log::Location{__FILE__, __LINE__}, __VA_ARGS__)
#define LOG_SYNC(...)                                                           \
    do                                                                          \
    {                                                                           \
        cudaErrCheck(cudaDeviceSynchronize());                                  \
        Log::Logger::get().log(Log::Location{__FILE__, __LINE__}, __VA_ARGS__); \
    } while (0);

#endif  // LOGGER_H
