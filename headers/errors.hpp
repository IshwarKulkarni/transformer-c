
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

inline std::string get_substr(std::string& str, const std::string& start, const std::string& end)
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

    for (long int i = 1; i < size; ++i)
    {
        std::string string(strings[i]);
        size_t start = string.find("(");
        size_t end = string.find("+0x", start);

        auto func_name = get_substr(string, "(", "+0x");

        if (func_name.size())
        {
            std::string mangled = string.substr(start + 1, end - start - 1);
            int status;
            char* demangled = abi::__cxa_demangle(mangled.c_str(), NULL, NULL, &status);
            if (demangled)
            {
                printf("\n Frame-%ld: %s", i, demangled);
            }
            free(demangled);
        }
        else
        {
            printf("\n Frame-%ld: %s", i, strings[i]);
        }
    }
    free(strings);
}

class runtime_error_with_backtrace : public std::runtime_error
{
 public:
    runtime_error_with_backtrace(const std::string& error) : std::runtime_error(error)
    {
        printf("%sException: %s%s\n Trace:", RED, error.c_str(), RESET);
        print_backtrace();
    }
};
