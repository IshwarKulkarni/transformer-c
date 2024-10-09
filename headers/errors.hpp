#ifndef ERRORS_HPP
#define ERRORS_HPP

#include <cxxabi.h>
#include <execinfo.h>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include "logger.hpp"

inline char* demangle_symbol1(const char* abiName)
{
    int status;
    std::size_t sz = 0;
    char* buffer = static_cast<char*>(std::malloc(sz));

    char* ret = abi::__cxa_demangle(abiName, buffer, &sz, &status);
    return ret;
}

inline std::pair<std::string, std::string> split(const std::string& str, const std::string& delim)
{
    size_t pos = str.find(delim);
    if (pos == std::string::npos)
    {
        return {str, ""};
    }
    return {str.substr(0, pos), str.substr(pos + delim.size())};
}

inline std::string get_substr(const std::string& str, const std::string& start,
                              const std::string& end)
{
    size_t s = str.find(start);
    size_t e = str.find(end, s);
    if (s != std::string::npos && e != std::string::npos)
    {
        return str.substr(s + start.size(), e - s - start.size());
    }
    return std::string();
}

inline void print_backtrace()
{
    void* array[255];
    long int size = backtrace(array, 255);
    char** strings = backtrace_symbols(array, size);
    if (size == 0)
    {
        printf("\n No backtrace available");
        return;
    }

    for (long int i = size - 1; i >= 0; --i)
    {
        std::string string(strings[i]);
        size_t start = string.find("(");
        size_t end = string.find("+0x", start);
        auto addr = get_substr(std::string(strings[i]), "[0x", "]");
        auto func_name = get_substr(string, "(", "+0x");
        if (func_name.size())
        {
            std::string mangled = string.substr(start + 1, end - start - 1);
            int status;
            char* demangled = abi::__cxa_demangle(mangled.c_str(), NULL, NULL, &status);
            if (demangled)
            {
                const auto& pair = split(demangled, "(");
                std::string func = pair.first;
                std::string args = pair.second;
                if (func.size() == 0)
                {
                    printf("\n Frame-%ld: | %s\n", i, demangled);
                }
                else
                {
                    const char* color = (i % 2) ? GREEN : BLUE;
                    printf("\n Frame-%ld: %s %s %s  (%s", i, color, func.c_str(), RESET,
                           args.c_str());
                }
            }
            free(demangled);
        }
        else
        {
            printf("\n Frame-%ld: | 0x%s | %s", i, addr.c_str(), strings[i]);
        }
    }
    free(strings);
}

class runtime_error_with_backtrace : public std::runtime_error
{
 public:
    runtime_error_with_backtrace(const std::string& error) : std::runtime_error(error)
    {
        printf("Exception: %s\n Trace:", error.c_str());
        print_backtrace();
    }
};

#endif
