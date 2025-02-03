#ifndef LOGGER_H
#define LOGGER_H

#include <cuda_runtime_api.h>
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
        if (!file.is_open())
        {
            file.open(filename);
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
        snprintf(file_loc, sizeof(file_loc), " %28s:%d\t", arg1.file, arg1.line);
        return this->log_v(strm << file_loc, args...);
    }

    template <typename First, typename... Args>
    inline std::ostream& log_v(std::ostream& strm, const First& arg1, const Args&... args)
    {
        return this->log_v((strm << arg1), args...);
    }

    template <typename... Args>
    inline void log(const Args&... args)
    {
        std::cout << get_time_str() << RESET << " ";
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
    std::ofstream file;
    std::set<std::string> disabled_files;
    Logger() = default;
};

}  // namespace Log

inline std::ostream& operator<<(std::ostream& strm, const dim3& dim)
{
    return strm << "(" << dim.x << ", " << dim.y << ", " << dim.z << ")";
}

// right justify fill with spaces:
#define R_JUST(x, n) std::setw(n), std::setfill(' '), std::right, x, " "
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
